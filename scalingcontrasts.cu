/**
 *  @file scalingcontrasts.cu
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Benchmarks of the primitives in @c scalingcontrasts.cuh, each printing CSV rows.
 *
 *      scalingcontrasts oracle   [--rows R]            every local GPU, against a Float64 evaluation of the world
 *      scalingcontrasts simulate [--rows R] --world S  one rank of an S-GPU world on the first GPU
 *      scalingcontrasts box      [--rows R]            every local GPU, peers' towers read in place over NVLink
 *      scalingcontrasts collectives [--max-mib M]      every local GPU's multimem barrier, reduce-scatter, all-reduce
 */
#include <chrono>  // `std::chrono::steady_clock`
#include <cmath>   // `std::sqrt`
#include <cstring> // `std::strcmp`

#include <algorithm> // `std::sort`
#include <span>      // `std::span`
#include <string>    // `std::stoul`
#include <vector>    // `std::vector`

#include <scalingcontrasts.cuh>

using namespace ashvardanian::scalingcontrasts;

constexpr f32_t beta = 10.0f, bias = -10.0f;   // SigLIP's initial logit scale and bias
constexpr std::size_t step_calls_k = 3;        // loss steps per timed window
constexpr std::size_t collective_calls_k = 20; // collectives per timed window, nccl-tests' default
constexpr std::size_t summand_scalars_k = 4;   // the loss's three summands, padded to one multimem vector

struct settings_t {
    std::size_t rows = 1024, hidden = 1024, world = 0, windows = 10, max_mib = 128, repeats = 1000;
    std::uint32_t prefixes = 5;
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

/** Fills @p payload with @ref keyed_integer values, or with keyed fractions in @b [-1/2,1/2) whose sums do round. */
static __global__ void keyed_payload_kernel(nk::tensor_span<f32_t, 1> payload, std::size_t member, std::size_t members,
                                            bool integers) {
    std::size_t const scalar = blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (scalar >= payload.extent(0)) return;
    payload[scalar] = integers ? keyed_integer(scalar, member, members)
                               : static_cast<f32_t>(splitmix64(scalar * members + member) >> 40) * 0x1.0p-24f - 0.5f;
}

/** Counts into @p wrong the scalars of @b [begin,end) that differ from the sum of every member's keyed integers. */
static __global__ void count_wrong_sums_kernel(nk::tensor_view<f32_t, 1> payload, std::size_t begin, std::size_t end,
                                               std::size_t members, unsigned long long *wrong) {
    std::size_t const scalar = begin + blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (scalar >= end) return;
    f32_t expected = 0.0f;
    for (std::size_t member = 0; member != members; ++member) expected += keyed_integer(scalar, member, members);
    if (payload[scalar] != expected) atomicAdd(wrong, 1ull);
}

/** Counts into @p differing the scalars of @p actual whose bits differ from @p reference . */
static __global__ void count_differences_kernel(nk::tensor_view<f32_t, 1> actual, nk::tensor_view<f32_t, 1> reference,
                                                unsigned long long *differing) {
    std::size_t const scalar = blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (scalar >= actual.extent(0)) return;
    if (__float_as_uint(actual[scalar]) != __float_as_uint(reference[scalar])) atomicAdd(differing, 1ull);
}

[[nodiscard]] static std::uint32_t blocks_for(std::size_t scalars) noexcept {
    return static_cast<std::uint32_t>((scalars + block_threads_k - 1) / block_threads_k);
}

/** Minimum and median milliseconds per call, over windows of @p calls calls each drained at the end. */
template <typename step_type_, typename drain_type_>
[[nodiscard]] static std::pair<f64_t, f64_t> time_steps(step_type_ &&step, drain_type_ &&drain, std::size_t windows,
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
    return member_t {device, std::move(stream), std::move(tower), std::move(gradient), std::move(loss)};
}

/** Every local GPU's step: a multimem barrier once every tower is complete, the peers' towers read in place, each
 *  tile streamed and folded, then one multimem all-reduce of the summands, whose barriers end the peer reads.
 *  Every member's barrier is enqueued before any work, as a kernel's first, lazily loaded launch waits for its
 *  device to idle, which a barrier spinning on a member not yet enqueued never does. */
static void step_box(std::vector<member_t> &members, multimem_domain_t &domain, settings_t const &settings) noexcept {
    std::size_t const gpus = members.size();
    for (std::size_t rank = 0; rank != gpus; ++rank)
        require(cudaSetDevice(int(rank)), "device switch"), domain.barrier(rank, members[rank].stream);
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        member_t &member = members[rank];
        require(cudaSetDevice(int(rank)), "device switch");
        member.loss.begin(member.tower.view(), member.stream);
        for (std::size_t peer = 0; peer != gpus; ++peer)
            member.loss.tile(member.tower.view(), members[peer].tower.view(), rank * settings.rows,
                             peer * settings.rows, beta, bias, f32_t(gpus * settings.rows), member.stream);
        member.loss.finish(member.tower.view(), member.gradient.span(), domain.payload(rank), member.stream);
        domain.all_reduce(rank, summand_scalars_k, member.stream);
    }
}

/** Barrier latency, then reduce-scatter and all-reduce bus bandwidths from 1 MiB to @c --max-mib as nccl-tests
 *  defines them, then exact sums of keyed integers, and bitwise repeats of an all-reduce of keyed fractions. */
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
    auto const [barrier_min, barrier_median] = time_steps(barriers, drain, settings.windows, collective_calls_k);
    std::printf("barrier,%zu,%zu,%.2f,%.2f\n", gpus, domain.stride, barrier_min * 1e3, barrier_median * 1e3);

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
        auto const [scatter_min, scatter_median] = time_steps(scatters, drain, settings.windows, collective_calls_k);
        auto const [reduce_min, reduce_median] = time_steps(reductions, drain, settings.windows, collective_calls_k);
        f64_t const scatter_bandwidth = f64_t(bytes) / (scatter_min * 1e6),
                    reduce_bandwidth = f64_t(bytes) / (reduce_min * 1e6);
        std::printf("reduce_scatter,%zu,%zu,%.2f,%.2f,%.1f,%.1f\n", gpus, bytes, scatter_min * 1e3,
                    scatter_median * 1e3, scatter_bandwidth, scatter_bandwidth * ring);
        std::printf("all_reduce,%zu,%zu,%.2f,%.2f,%.1f,%.1f\n", gpus, bytes, reduce_min * 1e3, reduce_median * 1e3,
                    reduce_bandwidth, reduce_bandwidth * 2 * ring);
    }

    // Counters on the first GPU, which every member adds to over NVLink: wrong scatters, wrong sums, differing repeats.
    member_t const &first = members.front();
    auto counters = make_tensor<unsigned long long, 1>(first.loss.capabilities, first.stream, {3});
    auto reference = make_tensor<f32_t, 1>(first.loss.capabilities, first.stream, {max_scalars});
    require(cudaSetDevice(0), "device switch");
    require(
        cudaMemsetAsync(counters.data(), 0, counters.numel() * sizeof(unsigned long long), cuda_stream(first.stream)),
        "zeroing counters");
    drain();
    // Each phase is enqueued on every member before the next, so no first launch waits behind a spinning barrier.
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

int main(int argc, char **argv) {
    if (argc < 2)
        return std::fprintf(stderr, "usage: %s oracle|simulate|box|collectives [--rows R] [--world S]\n", argv[0]), 2;
    char const *mode = argv[1];
    settings_t settings;
    for (int index = 2; index + 1 < argc; index += 2) {
        std::size_t const value = std::stoul(argv[index + 1]);
        if (!std::strcmp(argv[index], "--rows")) settings.rows = value;
        else if (!std::strcmp(argv[index], "--world")) settings.world = value;
        else if (!std::strcmp(argv[index], "--prefixes")) settings.prefixes = std::uint32_t(value);
        else if (!std::strcmp(argv[index], "--windows")) settings.windows = value;
        else if (!std::strcmp(argv[index], "--max-mib")) settings.max_mib = value;
        else if (!std::strcmp(argv[index], "--repeats")) settings.repeats = value;
        else return std::fprintf(stderr, "unknown flag %s\n", argv[index]), 2;
    }
    bool const oracle = !std::strcmp(mode, "oracle");
    f32_t const scale = oracle ? 1.0f : 0.1f; // unit-scale inputs keep the oracle's narrowest prefixes off the floor

    if (!std::strcmp(mode, "simulate")) {
        member_t member = make_member(0, settings, 0, scale);
        auto summands = make_tensor<f32_t, 1>(member.loss.capabilities, member.stream, {summand_scalars_k});
        auto const step = [&] {
            member.loss.begin(member.tower.view(), member.stream);
            for (std::size_t subgraph = 0; subgraph != settings.world; ++subgraph)
                member.loss.tile(member.tower.view(), member.tower.view(), 0, subgraph * settings.rows, beta, bias,
                                 f32_t(settings.world * settings.rows), member.stream);
            member.loss.finish(member.tower.view(), member.gradient.span(), summands.span(), member.stream);
        };
        step(), member.drain();
        auto const [min_ms, median_ms] = time_steps(step, [&] { member.drain(); }, settings.windows, step_calls_k);
        std::printf("simulate,%zu,%zu,%u,%zu,%zu,%.3f,%.3f\n", settings.rows, settings.hidden, settings.prefixes,
                    settings.world, member.loss.bytes(), min_ms, median_ms);
        return 0;
    }

    std::size_t const gpus = take(nk::device_t::count(nk::device_kind_t::cuda_k), "device count");
    std::vector<member_t> members;
    for (std::size_t rank = 0; rank != gpus; ++rank) {
        members.push_back(make_member(rank, settings, rank, scale));
        for (std::size_t peer = 0; peer != gpus; ++peer)
            if (peer != rank) require(cudaDeviceEnablePeerAccess(int(peer), 0), "peer access");
    }
    auto const drain = [&] {
        for (member_t const &member : members) member.drain();
    };
    if (!std::strcmp(mode, "collectives")) return benchmark_collectives(members, settings), 0;
    multimem_domain_t domain = multimem_domain_t::make(gpus, summand_scalars_k);
    step_box(members, domain, settings), drain();

    if (oracle) {
        // Every rank downloads on its own stream into one pinned staging per quantity, drained once.
        nk_capability_t const capabilities = members[0].loss.capabilities;
        nk::stream_t const &staging = members[0].stream;
        auto towers = make_tensor<bf16_t, 3>(capabilities, staging, {gpus, settings.rows, settings.hidden},
                                             nk_allocator_init_pinned_best);
        auto gradients = make_tensor<bf16_t, 3>(capabilities, staging, {gpus, settings.rows, settings.hidden},
                                                nk_allocator_init_pinned_best);
        auto outs = make_tensor<f32_t, 2>(capabilities, staging, {gpus, 3}, nk_allocator_init_pinned_best);
        for (std::size_t rank = 0; rank != gpus; ++rank) {
            member_t const &member = members[rank];
            cudaStream_t const cuda = cuda_stream(member.stream);
            require(cudaSetDevice(int(rank)), "device switch");
            require(cudaMemcpyAsync(towers.span()[rank, nk::slice].data(), member.tower.data(),
                                    member.tower.numel() * sizeof(bf16_t), cudaMemcpyDeviceToHost, cuda),
                    "download");
            require(cudaMemcpyAsync(gradients.span()[rank, nk::slice].data(), member.gradient.data(),
                                    member.gradient.numel() * sizeof(bf16_t), cudaMemcpyDeviceToHost, cuda),
                    "download");
            require(cudaMemcpyAsync(outs.span()[rank, nk::slice].data(), domain.payload(rank).data(), 3 * sizeof(f32_t),
                                    cudaMemcpyDeviceToHost, cuda),
                    "download");
        }
        drain();
        nk::aligned_allocator<f64_t> const host;
        auto world = take(nk::tensor<f64_t, nk::aligned_allocator<f64_t>, 3>::uninitialized(
                              {gpus, settings.rows, settings.hidden}, host),
                          "world");
        auto actual = take(nk::tensor<f64_t, nk::aligned_allocator<f64_t>, 3>::uninitialized(
                               {gpus, settings.rows, settings.hidden}, host),
                           "gradients");
        f64_t const loss = outs[0, 0];
        std::size_t disagreeing = 0; // ranks whose all-reduced loss differs from the first's
        for (std::size_t rank = 0; rank != gpus; ++rank) {
            for (std::size_t row = 0; row != settings.rows; ++row)
                for (std::size_t channel = 0; channel != settings.hidden; ++channel) {
                    world[rank, row, channel] = __bfloat162float(towers[rank, row, channel]);
                    actual[rank, row, channel] = __bfloat162float(gradients[rank, row, channel]);
                }
            disagreeing += outs[rank, 0] != outs[0, 0];
        }
        matryoshka_t const ladder = matryoshka_t::halvings(std::uint32_t(settings.hidden), settings.prefixes);
        auto exact = take(nk::tensor<f64_t, nk::aligned_allocator<f64_t>, 2>::uninitialized(
                              {gpus * settings.rows, settings.hidden}, host),
                          "oracle gradients");
        f64_t const exact_loss = oracle_loss(world.view().reshape<2>({gpus * settings.rows, settings.hidden}), ladder,
                                             exact.span());
        auto const flat = actual.view().reshape<2>({gpus * settings.rows, settings.hidden});
        f64_t difference = 0, reference = 0;
        for (std::size_t vertex = 0; vertex != exact.extent(0); ++vertex)
            for (std::size_t channel = 0; channel != exact.extent(1); ++channel) {
                f64_t const delta = flat[vertex, channel] - exact[vertex, channel];
                difference += delta * delta, reference += exact[vertex, channel] * exact[vertex, channel];
            }
        std::printf("oracle,%zu,%zu,%u,%zu,%.9f,%.9f,%.3e,%zu\n", settings.rows, settings.hidden, settings.prefixes,
                    gpus, exact_loss, loss, std::sqrt(difference / reference), disagreeing);
        return 0;
    }
    auto const [min_ms, median_ms] = time_steps([&] { step_box(members, domain, settings); }, drain, settings.windows,
                                                step_calls_k);
    std::printf("box,%zu,%zu,%u,%zu,%zu,%.3f,%.3f\n", settings.rows, settings.hidden, settings.prefixes, gpus,
                members[0].loss.bytes(), min_ms, median_ms);
    return 0;
}
