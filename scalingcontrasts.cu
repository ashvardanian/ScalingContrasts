/**
 *  @file scalingcontrasts.cu
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Benchmarks of the primitives in @c scalingcontrasts.cuh, each printing CSV rows.
 *
 *      scalingcontrasts oracle [--rows R]               every local GPU against Float64
 *      scalingcontrasts simulate [--rows R] --world S   one rank of an S-GPU world
 *      scalingcontrasts box [--rows R]                  every local GPU, peers read in place
 *      scalingcontrasts collectives [--max-mib M]       multimem barrier and reductions
 *      scalingcontrasts hosts [--rows R] --host H --hosts N --nics A,B,... --exchange DIR
 *                                                       every GPU of N hosts, one NIC per GPU
 *
 *  The oracle also takes the hosts' flags, checking each host's gradients and the world's loss.
 */
#include <cmath>   // `std::sqrt`
#include <cstring> // `std::strcmp`

#include <algorithm> // `std::sort`
#include <chrono>    // `std::chrono::steady_clock`
#include <span>      // `std::span`
#include <string>    // `std::stoul`, `std::string`
#include <thread>    // `std::this_thread::sleep_for`
#include <vector>    // `std::vector`

#include <scalingcontrasts.cuh>

using namespace ashvardanian::scalingcontrasts;

constexpr f32_t beta = 10.0f, bias = -10.0f;   // SigLIP's initial logit scale and bias
constexpr std::size_t step_calls_k = 3;        // loss steps per timed window
constexpr std::size_t collective_calls_k = 20; // collectives per timed window, nccl-tests' default
// the loss's three summands, padded to one multimem vector
constexpr std::size_t summand_scalars_k = 4;

struct settings_t {
    std::size_t rows = 1024, hidden = 1024, world = 0, windows = 10, max_mib = 128, repeats = 1000;
    std::size_t host = 0, hosts = 1;
    std::uint32_t prefixes = 5;
    std::string nics, exchange;
};

/** The splitmix64 finalizer, a keyed hash for inputs every run reproduces. */
[[nodiscard]] static __device__ std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** Fills one row of @p tower with keyed BF16 values in @b [-scale/2,scale/2). */
static __global__ void keyed_tower_kernel(nk::tensor_span<bf16_t, 2> tower, std::uint64_t salt, f32_t scale) {
    std::size_t const row = blockIdx.x, rows = tower.extent(0), hidden = tower.extent(1);
    for (std::size_t channel = threadIdx.x; channel < hidden; channel += blockDim.x) {
        std::uint64_t const bits = splitmix64((salt * rows + row) * hidden + channel);
        tower[row, channel] = __float2bfloat16((static_cast<f32_t>(bits >> 40) * 0x1.0p-24f - 0.5f) * scale);
    }
}

/** Keyed integers below 2¹², so any order of summing up to 2¹² members is exact in Float32. */
[[nodiscard]] static __device__ f32_t keyed_integer(std::size_t scalar, std::size_t member,
                                                    std::size_t members) noexcept {
    return static_cast<f32_t>(splitmix64(scalar * members + member) >> 52);
}

/** Fills @p payload with @ref keyed_integer values, or with keyed fractions in @b [-1/2,1/2) whose
 *  sums do round. */
static __global__ void keyed_payload_kernel(nk::tensor_span<f32_t, 1> payload, std::size_t member, std::size_t members,
                                            bool integers) {
    std::size_t const scalar = blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (scalar >= payload.extent(0)) return;
    payload[scalar] = integers ? keyed_integer(scalar, member, members)
                               : static_cast<f32_t>(splitmix64(scalar * members + member) >> 40) * 0x1.0p-24f - 0.5f;
}

/** Counts into @p wrong the scalars of @b [begin,end) that differ from the sum of every member's
 *  keyed integers. */
static __global__ void count_wrong_sums_kernel(nk::tensor_view<f32_t, 1> payload, std::size_t begin, std::size_t end,
                                               std::size_t members, unsigned long long *wrong) {
    std::size_t const scalar = begin + blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (scalar >= end) return;
    f32_t expected = 0.0f;
    for (std::size_t member = 0; member != members; ++member) expected += keyed_integer(scalar, member, members);
    if (payload[scalar] != expected)
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>(*wrong).fetch_add(
            1ull, cuda::std::memory_order_relaxed);
}

/** Counts into @p differing the scalars of @p actual whose bits differ from @p reference . */
static __global__ void count_differences_kernel(nk::tensor_view<f32_t, 1> actual, nk::tensor_view<f32_t, 1> reference,
                                                unsigned long long *differing) {
    std::size_t const scalar = blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (scalar >= actual.extent(0)) return;
    if (__float_as_uint(actual[scalar]) != __float_as_uint(reference[scalar]))
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>(*differing)
            .fetch_add(1ull, cuda::std::memory_order_relaxed);
}

[[nodiscard]] static std::uint32_t blocks_for(std::size_t scalars) noexcept {
    return static_cast<std::uint32_t>(divide_round_up(scalars, block_threads_k));
}

/** The minimum and median milliseconds per call of a timed run. */
struct timing_t {
    f64_t min_ms = 0, median_ms = 0;
};

/** Times @p step over windows of @p calls calls, each drained at its end. */
template <typename step_type_, typename drain_type_>
[[nodiscard]] static timing_t time_steps(step_type_ &&step, drain_type_ &&drain, std::size_t windows,
                                         std::size_t calls) noexcept {
    auto per_step = take(nk::vector<f64_t>::uninitialized(windows), "timings");
    for (std::size_t window = 0; window != windows; ++window) {
        auto const start = std::chrono::steady_clock::now();
        for (std::size_t call = 0; call != calls; ++call) step();
        drain();
        per_step[window] = std::chrono::duration<f64_t, std::milli>(std::chrono::steady_clock::now() - start).count() /
                           f64_t(calls);
    }
    std::span<f64_t> const sorted = take(per_step.span().values(), "timings");
    std::sort(sorted.begin(), sorted.end());
    return {sorted.front(), sorted[windows / 2]};
}

/** The world's Float64 loss and every vertex's gradient: @c Σ(k,i,j)w·softplus(-t·(β·cos_k+b)). */
[[nodiscard]] static f64_t oracle_loss(nk::tensor_view<f64_t, 2> world, matryoshka_t ladder,
                                       nk::tensor_span<f64_t, 2> gradient) noexcept {
    std::size_t const vertices = world.extent(0), hidden = world.extent(1);
    for (std::size_t vertex = 0; vertex != vertices; ++vertex)
        for (std::size_t channel = 0; channel != hidden; ++channel) gradient[vertex, channel] = 0.0;
    f64_t const weight = 1.0 / static_cast<f64_t>(vertices);
    f64_t loss = 0;
    auto norms = take(nk::vector<f64_t>::uninitialized(vertices), "norms");
    for (std::uint32_t band = 0; band != ladder.count; ++band) {
        std::size_t const width = ladder.ends[band];
        for (std::size_t i = 0; i != vertices; ++i) {
            f64_t sum = 0;
            for (std::size_t h = 0; h != width; ++h) sum += world[i, h] * world[i, h];
            norms[i] = std::sqrt(sum);
        }
        for (std::size_t i = 0; i != vertices; ++i)
            for (std::size_t j = 0; j != vertices; ++j) {
                f64_t dot = 0;
                for (std::size_t h = 0; h != width; ++h) dot += world[i, h] * world[j, h];
                f64_t const cosine = dot / (norms[i] * norms[j]), target = i == j ? 1.0 : -1.0;
                f64_t const signed_logit = target * (beta * cosine + bias);
                loss += weight * (std::fmax(-signed_logit, 0.0) + std::log1p(std::exp(-std::fabs(signed_logit))));
                if (i == j) continue;
                f64_t const cell_grad = weight * target * (1.0 / (1.0 + std::exp(-signed_logit)) - 1.0) * beta;
                for (std::size_t h = 0; h != width; ++h) {
                    gradient[i, h] += cell_grad * (world[j, h] / (norms[i] * norms[j]) -
                                                   cosine * world[i, h] / (norms[i] * norms[i]));
                    gradient[j, h] += cell_grad * (world[i, h] / (norms[i] * norms[j]) -
                                                   cosine * world[j, h] / (norms[j] * norms[j]));
                }
            }
    }
    return loss;
}

/** One local GPU: its stream, its tower and gradient, and its loss. */
struct member_t {
    nk::device_t device;
    nk::stream_t stream;
    nk::tensor<bf16_t, nk::allocator<bf16_t>, 2> tower, gradient;
    streamed_loss_t loss;

    /** The towers landed from every remote host, @b [remote_hosts,2,rows,hidden] by step parity. */
    nk::tensor<bf16_t, nk::allocator<bf16_t>, 4> landings;

    /** One rail per remote host, in host order. */
    std::vector<rdma_rail_t> rails;

    void drain() const noexcept {
        require(cudaSetDevice(int(device.ordinal())), "device switch");
        require(cudaStreamSynchronize(cuda_stream(stream)), "step");
    }
};

[[nodiscard]] static member_t make_member(std::size_t ordinal, settings_t const &settings, std::uint64_t salt,
                                          f32_t scale) noexcept {
    nk::device_t const device = take(nk::device_t::make(nk::device_kind_t::cuda_k, ordinal), "device");
    require(cudaSetDevice(int(ordinal)), "device switch");
    nk::stream_t stream = take(nk::stream_t::make(device), "stream");
    nk_capability_t const capabilities = take(device.capabilities_enabled(), "capabilities");
    auto tower = make_tensor<bf16_t, 2>(capabilities, stream, {settings.rows, settings.hidden});
    launch(&keyed_tower_kernel, settings.rows, block_threads_k, stream, tower.span(), salt, scale);
    auto gradient = make_tensor<bf16_t, 2>(capabilities, stream, {settings.rows, settings.hidden});
    streamed_loss_t loss = streamed_loss_t::make(
        device, stream, settings.rows, settings.hidden,
        matryoshka_t::halvings(std::uint32_t(settings.hidden), settings.prefixes));
    return member_t {device, std::move(stream), std::move(tower), std::move(gradient), std::move(loss), {}, {}};
}

/** Every local GPU's step: a multimem barrier once every tower is complete, the peers' towers read
 *  in place, each tile streamed and folded, then one multimem all-reduce of the summands, whose
 *  barriers end the peer reads. Every member's barrier is enqueued before any work, as a kernel's
 *  first, lazily loaded launch waits for its device to idle, which a barrier spinning on a member
 *  not yet enqueued never does. */
static void step_box(std::vector<member_t> &members, multimem_domain_t &domain, settings_t const &settings) noexcept {
    std::size_t const gpus = members.size();
    for (std::size_t rank = 0; rank != gpus; ++rank)
        require(cudaSetDevice(int(rank)), "device switch"), domain.barrier(rank, members[rank].stream);
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        member_t &member = members[rank];
        require(cudaSetDevice(int(rank)), "device switch");
        member.loss.begin(member.tower.view(), member.stream);
        for (std::size_t peer = 0; peer != gpus; ++peer)
            member.loss.tile(member.tower.span(), members[peer].tower.view(), rank * settings.rows,
                             peer * settings.rows, beta, bias, f32_t(gpus * settings.rows), member.stream);
        member.loss.finish(member.tower.view(), member.gradient.span(), domain.payload(rank), member.stream);
        domain.all_reduce(rank, summand_scalars_k, member.stream);
    }
}

/** Barrier latency, then reduce-scatter and all-reduce bus bandwidths from 1 MiB to @c --max-mib as
 *  nccl-tests defines them, then exact sums of keyed integers, and bitwise repeats of an all-reduce
 *  of keyed fractions. */
static void benchmark_collectives(std::vector<member_t> &members, settings_t const &settings) noexcept {
    std::size_t const gpus = members.size(), max_scalars = settings.max_mib << 18;
    multimem_domain_t domain = multimem_domain_t::make(gpus, max_scalars);
    auto const drain = [&] {
        for (member_t const &member : members) member.drain();
    };
    auto const barriers = [&] {
        for (std::size_t rank = 0; rank != gpus; ++rank)
            require(cudaSetDevice(int(rank)), "device switch"), domain.barrier(rank, members[rank].stream);
    };
    timing_t const barrier = time_steps(barriers, drain, settings.windows, collective_calls_k);
    std::printf("barrier,%zu,%zu,%.2f,%.2f\n", gpus, domain.stride, barrier.min_ms * 1e3, barrier.median_ms * 1e3);

    f64_t const ring = f64_t(gpus - 1) / f64_t(gpus);
    for (std::size_t scalars = 1 << 18; scalars <= max_scalars; scalars *= 2) {
        std::size_t const shard = scalars / gpus / 4 * 4, bytes = shard * gpus * sizeof(f32_t);
        auto const scatters = [&] {
            for (std::size_t rank = 0; rank != gpus; ++rank)
                require(cudaSetDevice(int(rank)), "device switch"),
                    domain.reduce_scatter(rank, shard, members[rank].stream);
        };
        auto const reductions = [&] {
            for (std::size_t rank = 0; rank != gpus; ++rank)
                require(cudaSetDevice(int(rank)), "device switch"),
                    domain.all_reduce(rank, shard * gpus, members[rank].stream);
        };
        timing_t const scatter = time_steps(scatters, drain, settings.windows, collective_calls_k);
        timing_t const reduce = time_steps(reductions, drain, settings.windows, collective_calls_k);
        f64_t const scatter_bandwidth = f64_t(bytes) / (scatter.min_ms * 1e6),
                    reduce_bandwidth = f64_t(bytes) / (reduce.min_ms * 1e6);
        std::printf("reduce_scatter,%zu,%zu,%.2f,%.2f,%.1f,%.1f\n", gpus, bytes, scatter.min_ms * 1e3,
                    scatter.median_ms * 1e3, scatter_bandwidth, scatter_bandwidth * ring);
        std::printf("all_reduce,%zu,%zu,%.2f,%.2f,%.1f,%.1f\n", gpus, bytes, reduce.min_ms * 1e3,
                    reduce.median_ms * 1e3, reduce_bandwidth, reduce_bandwidth * 2 * ring);
    }

    // Counters on the first GPU, which every member adds to over NVLink: wrong scatters, wrong
    // sums, differing repeats.
    member_t const &first = members.front();
    auto counters = make_tensor<unsigned long long, 1>(first.loss.capabilities, first.stream, {3});
    auto reference = make_tensor<f32_t, 1>(first.loss.capabilities, first.stream, {max_scalars});
    require(cudaSetDevice(0), "device switch");
    require(
        cudaMemsetAsync(counters.data(), 0, counters.numel() * sizeof(unsigned long long), cuda_stream(first.stream)),
        "zeroing counters");
    drain();
    // Each phase is enqueued on every member before the next, so no first launch waits behind a
    // spinning barrier.
    std::size_t const shard = max_scalars / gpus / 4 * 4;
    auto const fill = [&](bool integers) {
        for (std::size_t rank = 0; rank != gpus; ++rank)
            require(cudaSetDevice(int(rank)), "device switch"),
                launch(&keyed_payload_kernel, blocks_for(max_scalars), block_threads_k, members[rank].stream,
                       domain.payload(rank), rank, gpus, integers);
    };
    fill(true);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        require(cudaSetDevice(int(rank)), "device switch"), domain.reduce_scatter(rank, shard, members[rank].stream);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        require(cudaSetDevice(int(rank)), "device switch"),
            launch(&count_wrong_sums_kernel, blocks_for(shard), block_threads_k, members[rank].stream,
                   domain.payload(rank), rank * shard, rank * shard + shard, gpus, counters.data());
    fill(true);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        require(cudaSetDevice(int(rank)), "device switch"), domain.all_reduce(rank, max_scalars, members[rank].stream);
    for (std::size_t rank = 0; rank != gpus; ++rank)
        require(cudaSetDevice(int(rank)), "device switch"),
            launch(&count_wrong_sums_kernel, blocks_for(max_scalars), block_threads_k, members[rank].stream,
                   domain.payload(rank), std::size_t(0), max_scalars, gpus, counters.data() + 1);
    for (std::size_t repeat = 0; repeat != settings.repeats; ++repeat) {
        fill(false);
        for (std::size_t rank = 0; rank != gpus; ++rank)
            require(cudaSetDevice(int(rank)), "device switch"),
                domain.all_reduce(rank, max_scalars, members[rank].stream);
        require(cudaSetDevice(0), "device switch");
        if (repeat == 0)
            require(cudaMemcpyAsync(reference.data(), domain.payload(0).data(), max_scalars * sizeof(f32_t),
                                    cudaMemcpyDeviceToDevice, cuda_stream(first.stream)),
                    "reference copy");
        else
            launch(&count_differences_kernel, blocks_for(max_scalars), block_threads_k, first.stream, domain.payload(0),
                   reference.view(), counters.data() + 2);
    }
    auto counted = make_tensor<unsigned long long, 1>(first.loss.capabilities, first.stream, {3},
                                                      nk_allocator_init_pinned_best);
    require(cudaSetDevice(0), "device switch");
    require(cudaMemcpyAsync(counted.data(), counters.data(), counters.numel() * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, cuda_stream(first.stream)),
            "counter download");
    drain();
    std::printf("validation,%zu,%zu,%llu,%llu,%zu,%llu\n", gpus, max_scalars * sizeof(f32_t), counted[0], counted[1],
                settings.repeats, counted[2]);
}

/** The host behind the @p remote -th rail of a member, skipping @p host itself. */
[[nodiscard]] static std::size_t remote_host(std::size_t remote, std::size_t host) noexcept {
    return remote < host ? remote : remote + 1;
}

/** Every local GPU's step across hosts: towers posted over the rails, local tiles scored while they
 *  fly, landed towers read in place over NVLink, then summands all-reduced over multimem and added
 *  across hosts in host order. A host posts only after draining its last step, so with two landings
 *  no write meets a reader. */
static void step_hosts(std::vector<member_t> &members, multimem_domain_t &domain, nk::tensor_span<f32_t, 2> scalars,
                       nk::tensor_span<f32_t, 1> totals, settings_t const &settings, std::size_t step) noexcept {
    std::size_t const gpus = members.size(), parity = step % 2, live = settings.hosts * gpus * settings.rows;
    for (member_t const &member : members) member.drain();
    for (member_t &member : members)
        for (rdma_rail_t &rail : member.rails) rail.post_tower(parity);
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        member_t &member = members[rank];
        require(cudaSetDevice(int(rank)), "device switch");
        member.loss.begin(member.tower.view(), member.stream);
        for (std::size_t peer = 0; peer != gpus; ++peer)
            member.loss.tile(member.tower.span(), members[peer].tower.view(),
                             (settings.host * gpus + rank) * settings.rows,
                             (settings.host * gpus + peer) * settings.rows, beta, bias, f32_t(live), member.stream);
    }
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        for (rdma_rail_t &rail : members[rank].rails) rail.await_landed();
        require(cudaSetDevice(int(rank)), "device switch");
        require(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                   cudaFlushGPUDirectRDMAWritesToAllDevices),
                "RDMA write flush");
    }
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        member_t &member = members[rank];
        require(cudaSetDevice(int(rank)), "device switch");
        for (std::size_t remote = 0; remote + 1 != settings.hosts; ++remote)
            for (std::size_t peer = 0; peer != gpus; ++peer)
                member.loss.tile(member.tower.span(), members[peer].landings[remote, parity, nk::slice],
                                 (settings.host * gpus + rank) * settings.rows,
                                 (remote_host(remote, settings.host) * gpus + peer) * settings.rows, beta, bias,
                                 f32_t(live), member.stream);
        member.loss.finish(member.tower.view(), member.gradient.span(), domain.payload(rank), member.stream);
        domain.all_reduce(rank, summand_scalars_k, member.stream);
    }
    require(cudaSetDevice(0), "device switch");
    require(cudaMemcpyAsync(scalars[settings.host, nk::slice].data(), domain.payload(0).data(),
                            summand_scalars_k * sizeof(f32_t), cudaMemcpyDeviceToHost, cuda_stream(members[0].stream)),
            "summands download");
    members[0].drain();
    for (rdma_rail_t &rail : members[0].rails) rail.post_scalars(settings.host);
    for (rdma_rail_t &rail : members[0].rails) rail.await_landed();
    for (std::size_t scalar = 0; scalar != summand_scalars_k; ++scalar) {
        f32_t total = 0.0f;
        for (std::size_t host = 0; host != settings.hosts; ++host) total += scalars[host, scalar];
        totals[scalar] = total;
    }
    for (std::size_t rank = 0; rank != gpus; ++rank)
        require(cudaSetDevice(int(rank)), "device switch"),
            require(cudaMemcpyAsync(domain.payload(rank).data(), totals.data(), summand_scalars_k * sizeof(f32_t),
                                    cudaMemcpyHostToDevice, cuda_stream(members[rank].stream)),
                    "totals upload");
}

/** Publishes @p local as the endpoint of the rail from @p from to @p to on @p gpu , then waits
 *  for the other side's, each written whole and renamed, so none is read half-written. */
[[nodiscard]] static rdma_endpoint_t exchange_endpoint(std::string const &directory, rdma_endpoint_t const &local,
                                                       std::size_t from, std::size_t to, std::size_t gpu) noexcept {
    std::string const published = directory + "/endpoint-" + std::to_string(from) + "-" + std::to_string(to) + "-" +
                                  std::to_string(gpu);
    std::string const partial = published + ".partial";
    std::FILE *file = std::fopen(partial.c_str(), "wb");
    require(file ? 0 : errno, partial.c_str());
    require(std::fwrite(&local, sizeof(local), 1, file) == 1 ? 0 : EIO, partial.c_str());
    require(std::fclose(file) == 0 && std::rename(partial.c_str(), published.c_str()) == 0 ? 0 : errno,
            published.c_str());
    std::string const awaited = directory + "/endpoint-" + std::to_string(to) + "-" + std::to_string(from) + "-" +
                                std::to_string(gpu);
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
    rdma_endpoint_t remote;
    for (;;) {
        if (std::FILE *other = std::fopen(awaited.c_str(), "rb")) {
            std::size_t const read = std::fread(&remote, sizeof(remote), 1, other);
            std::fclose(other);
            if (read == 1) return remote;
        }
        require(std::chrono::steady_clock::now() < deadline ? 0 : ETIMEDOUT, awaited.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

/** The @p gpu -th name of the comma-separated @p nics . */
[[nodiscard]] static std::string nic_of(std::string const &nics, std::size_t gpu) noexcept {
    std::size_t begin = 0;
    for (std::size_t skipped = 0; skipped != gpu; ++skipped) begin = nics.find(',', begin) + 1;
    return nics.substr(begin, nics.find(',', begin) - begin);
}

int main(int argc, char **argv) {
    if (argc < 2)
        return std::fprintf(stderr, "usage: %s oracle|simulate|box|hosts|collectives [--rows R] [--world S]\n",
                            argv[0]),
               2;
    char const *mode = argv[1];
    settings_t settings;
    for (int index = 2; index + 1 < argc; index += 2) {
        if (!std::strcmp(argv[index], "--nics")) {
            settings.nics = argv[index + 1];
            continue;
        }
        if (!std::strcmp(argv[index], "--exchange")) {
            settings.exchange = argv[index + 1];
            continue;
        }
        std::size_t const value = std::stoul(argv[index + 1]);
        if (!std::strcmp(argv[index], "--rows")) settings.rows = value;
        else if (!std::strcmp(argv[index], "--host")) settings.host = value;
        else if (!std::strcmp(argv[index], "--hosts")) settings.hosts = value;
        else if (!std::strcmp(argv[index], "--world")) settings.world = value;
        else if (!std::strcmp(argv[index], "--prefixes")) settings.prefixes = std::uint32_t(value);
        else if (!std::strcmp(argv[index], "--windows")) settings.windows = value;
        else if (!std::strcmp(argv[index], "--max-mib")) settings.max_mib = value;
        else if (!std::strcmp(argv[index], "--repeats")) settings.repeats = value;
        else return std::fprintf(stderr, "unknown flag %s\n", argv[index]), 2;
    }
    bool const oracle = !std::strcmp(mode, "oracle");
    // unit-scale inputs keep the oracle's narrowest prefixes off the floor
    f32_t const scale = oracle ? 1.0f : 0.1f;

    if (!std::strcmp(mode, "simulate")) {
        member_t member = make_member(0, settings, 0, scale);
        auto summands = make_tensor<f32_t, 1>(member.loss.capabilities, member.stream, {summand_scalars_k});
        auto const step = [&] {
            member.loss.begin(member.tower.view(), member.stream);
            for (std::size_t subgraph = 0; subgraph != settings.world; ++subgraph)
                member.loss.tile(member.tower.span(), member.tower.view(), 0, subgraph * settings.rows, beta, bias,
                                 f32_t(settings.world * settings.rows), member.stream);
            member.loss.finish(member.tower.view(), member.gradient.span(), summands.span(), member.stream);
        };
        step(), member.drain();
        timing_t const timing = time_steps(step, [&] { member.drain(); }, settings.windows, step_calls_k);
        std::printf("simulate,%zu,%zu,%u,%zu,%zu,%.3f,%.3f\n", settings.rows, settings.hidden, settings.prefixes,
                    settings.world, member.loss.bytes(), timing.min_ms, timing.median_ms);
        return 0;
    }

    std::size_t const gpus = take(nk::device_t::count(nk::device_kind_t::cuda_k), "device count");
    bool const across_hosts = settings.hosts > 1;
    if (across_hosts && (settings.nics.empty() || settings.exchange.empty() || settings.host >= settings.hosts))
        return std::fprintf(stderr, "several hosts need --host, --nics and --exchange\n"), 2;
    // Every rail registers the host's scalars, so they outlive the members.
    nk_capability_t const capabilities = take(
        take(nk::device_t::make(nk::device_kind_t::cuda_k, 0), "device").capabilities_enabled(), "capabilities");
    auto scalars = make_tensor<f32_t, 2>(capabilities, nk::stream_t {}, {settings.hosts, summand_scalars_k},
                                         nk_allocator_init_pinned_best);
    auto totals = make_tensor<f32_t, 1>(capabilities, nk::stream_t {}, {summand_scalars_k},
                                        nk_allocator_init_pinned_best);
    std::vector<member_t> members;
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        members.push_back(make_member(rank, settings, settings.host * gpus + rank, scale));
        for (std::size_t peer = 0; peer != gpus; ++peer)
            if (peer != rank) require(cudaDeviceEnablePeerAccess(int(peer), 0), "peer access");
    }
    auto const drain = [&] {
        for (member_t const &member : members) member.drain();
    };
    if (!std::strcmp(mode, "collectives")) return benchmark_collectives(members, settings), 0;
    if (across_hosts) {
        for (std::size_t rank = 0; rank != gpus; ++rank) {
            member_t &member = members[rank];
            require(cudaSetDevice(int(rank)), "device switch");
            member.landings = make_tensor<bf16_t, 4>(member.loss.capabilities, member.stream,
                                                     {settings.hosts - 1, 2, settings.rows, settings.hidden});
            std::string const nic = nic_of(settings.nics, rank);
            for (std::size_t remote = 0; remote + 1 != settings.hosts; ++remote)
                member.rails.push_back(rdma_rail_t::make(nic.c_str(), member.tower.view(),
                                                         member.landings[remote, nk::slice], scalars.span(),
                                                         std::uint32_t(settings.host * gpus + rank)));
        }
        drain();
        for (std::size_t rank = 0; rank != gpus; ++rank)
            for (std::size_t remote = 0; remote + 1 != settings.hosts; ++remote) {
                rdma_rail_t &rail = members[rank].rails[remote];
                rail.connect(exchange_endpoint(settings.exchange, rail.local, settings.host,
                                               remote_host(remote, settings.host), rank));
            }
    }
    multimem_domain_t domain = multimem_domain_t::make(gpus, summand_scalars_k);
    std::size_t steps = 0;
    auto const step = [&] {
        if (across_hosts) step_hosts(members, domain, scalars.span(), totals.span(), settings, steps++);
        else step_box(members, domain, settings);
    };
    step(), drain();

    if (oracle) {
        // Every world rank's tower, local or landed, and every local gradient, each downloaded on
        // its member's stream.
        std::size_t const ranks = settings.hosts * gpus, parity = (steps - 1) % 2;
        nk::stream_t const &staging = members[0].stream;
        auto towers = make_tensor<bf16_t, 3>(capabilities, staging, {ranks, settings.rows, settings.hidden},
                                             nk_allocator_init_pinned_best);
        auto gradients = make_tensor<bf16_t, 3>(capabilities, staging, {gpus, settings.rows, settings.hidden},
                                                nk_allocator_init_pinned_best);
        auto outs = make_tensor<f32_t, 2>(capabilities, staging, {gpus, 3}, nk_allocator_init_pinned_best);
        std::size_t const tower_bytes = settings.rows * settings.hidden * sizeof(bf16_t);
        for (std::size_t rank = 0; rank != gpus; ++rank) {
            member_t const &member = members[rank];
            cudaStream_t const cuda = cuda_stream(member.stream);
            require(cudaSetDevice(int(rank)), "device switch");
            require(cudaMemcpyAsync(towers[settings.host * gpus + rank, nk::slice].data(), member.tower.data(),
                                    tower_bytes, cudaMemcpyDeviceToHost, cuda),
                    "download");
            for (std::size_t remote = 0; remote + 1 != settings.hosts; ++remote)
                require(cudaMemcpyAsync(towers[remote_host(remote, settings.host) * gpus + rank, nk::slice].data(),
                                        member.landings[remote, parity, nk::slice].data(), tower_bytes,
                                        cudaMemcpyDeviceToHost, cuda),
                        "download");
            require(cudaMemcpyAsync(gradients[rank, nk::slice].data(), member.gradient.data(), tower_bytes,
                                    cudaMemcpyDeviceToHost, cuda),
                    "download");
            require(cudaMemcpyAsync(outs[rank, nk::slice].data(), domain.payload(rank).data(), 3 * sizeof(f32_t),
                                    cudaMemcpyDeviceToHost, cuda),
                    "download");
        }
        drain();
        nk::aligned_allocator<f64_t> const host;
        auto world = take(nk::tensor<f64_t, nk::aligned_allocator<f64_t>, 3>::uninitialized(
                              {ranks, settings.rows, settings.hidden}, host),
                          "world");
        for (std::size_t rank = 0; rank != ranks; ++rank)
            for (std::size_t row = 0; row != settings.rows; ++row)
                for (std::size_t channel = 0; channel != settings.hidden; ++channel)
                    world[rank, row, channel] = __bfloat162float(towers[rank, row, channel]);
        std::size_t disagreeing = 0; // local ranks whose loss differs from the first's
        for (std::size_t rank = 0; rank != gpus; ++rank) disagreeing += outs[rank, 0] != outs[0, 0];
        matryoshka_t const ladder = matryoshka_t::halvings(std::uint32_t(settings.hidden), settings.prefixes);
        auto exact = take(nk::tensor<f64_t, nk::aligned_allocator<f64_t>, 3>::uninitialized(
                              {ranks, settings.rows, settings.hidden}, host),
                          "oracle gradients");
        f64_t const exact_loss = oracle_loss(world.view().reshape<2>({ranks * settings.rows, settings.hidden}), ladder,
                                             exact.span().reshape<2>({ranks * settings.rows, settings.hidden}));
        f64_t difference = 0, reference = 0;
        for (std::size_t rank = 0; rank != gpus; ++rank)
            for (std::size_t row = 0; row != settings.rows; ++row)
                for (std::size_t channel = 0; channel != settings.hidden; ++channel) {
                    f64_t const expected = exact[settings.host * gpus + rank, row, channel];
                    f64_t const delta = __bfloat162float(gradients[rank, row, channel]) - expected;
                    difference += delta * delta, reference += expected * expected;
                }
        std::printf("oracle,%zu,%zu,%u,%zu,%.9f,%.9f,%.3e,%zu\n", settings.rows, settings.hidden, settings.prefixes,
                    ranks, exact_loss, f64_t(outs[0, 0]), std::sqrt(difference / reference), disagreeing);
        return 0;
    }
    timing_t const timing = time_steps(step, drain, settings.windows, step_calls_k);
    std::size_t const landing_bytes = members[0].landings.numel() * sizeof(bf16_t);
    std::printf("%s,%zu,%zu,%u,%zu,%zu,%.3f,%.3f\n", across_hosts ? "hosts" : "box", settings.rows, settings.hidden,
                settings.prefixes, settings.hosts * gpus, members[0].loss.bytes() + landing_bytes, timing.min_ms,
                timing.median_ms);
    return 0;
}
