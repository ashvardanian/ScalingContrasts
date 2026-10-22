/**
 *  @file scalingcontrasts.cu
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Benchmarks of the primitives in @c scalingcontrasts.cuh, each printing CSV rows.
 *
 *      scalingcontrasts oracle [--rows R]               every local GPU against Float64
 *      scalingcontrasts simulate [--rows R] --gpus S    one rank of an S-GPU world
 *      scalingcontrasts box [--rows R]                  every local GPU, peers read in place
 *      scalingcontrasts collectives [--max-mib M]       multimem barrier and reductions
 *      scalingcontrasts hosts [--rows R] --host H --hosts N --nics A,B,... --exchange DIR
 *                                                       every GPU of N hosts, one NIC per GPU
 *
 *  The oracle also takes the hosts' flags, checking each host's gradients and the world's loss.
 */
#include <cmath>   // `std::sqrt`
#include <cstring> // `std::strcmp`

#include <algorithm>   // `std::sort`
#include <chrono>      // `std::chrono::steady_clock`
#include <span>        // `std::span`
#include <iterator>    // `std::begin`, `std::end`
#include <string>      // `std::stoul`, `std::string`
#include <string_view> // `std::string_view`
#include <thread>      // `std::this_thread::sleep_for`
#include <vector>      // `std::vector`

#include <scalingcontrasts.cuh>

using namespace ashvardanian::scalingcontrasts;

constexpr f32_t beta = 10.0f, bias = -10.0f;   // SigLIP's initial logit scale and bias
constexpr std::size_t step_calls_k = 3;        // loss steps per timed window
constexpr std::size_t collective_calls_k = 20; // collectives per timed window, nccl-tests' default
// the loss's three summands, padded to one multimem vector
constexpr std::size_t summand_scalars_k = 4;
constexpr f32_t oracle_scale_k = 1.0f;    // unit-scale inputs keep the oracle's narrowest prefixes off the floor
constexpr f32_t benchmark_scale_k = 0.1f; // the scale of the embeddings the benchmarks score

struct settings_t {
    std::size_t rows = 1024, hidden = 1024, simulated_gpus = 0, windows = 10, max_mib = 128, repeats = 1000;
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

/** Fills one row of @p embeddings with keyed BF16 values in @b [-scale/2,scale/2). */
static __global__ void keyed_embeddings_kernel(nk::tensor_span<bf16_t, 2> embeddings, std::uint64_t salt, f32_t scale) {
    std::size_t const row = blockIdx.x, rows = embeddings.extent(0), hidden = embeddings.extent(1);
    for (std::size_t channel = threadIdx.x; channel < hidden; channel += blockDim.x) {
        std::uint64_t const bits = splitmix64((salt * rows + row) * hidden + channel);
        embeddings[row, channel] = __float2bfloat16((static_cast<f32_t>(bits >> 40) * 0x1.0p-24f - 0.5f) * scale);
    }
}

/** Keyed integers below 2¹², so any order of summing up to 2¹² members is exact in Float32. */
[[nodiscard]] static __device__ f32_t keyed_integer(std::size_t scalar, std::size_t member,
                                                    std::size_t members) noexcept {
    return static_cast<f32_t>(splitmix64(scalar * members + member) >> 52);
}

/** Fills @p payload with @ref keyed_integer values, whose sums are exact. */
static __global__ void keyed_integers_kernel(nk::tensor_span<f32_t, 1> payload, std::size_t member,
                                             std::size_t members) {
    std::size_t const scalar = blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (scalar < payload.extent(0)) payload[scalar] = keyed_integer(scalar, member, members);
}

/** Fills @p payload with keyed fractions in @b [-1/2,1/2), whose sums round. */
static __global__ void keyed_fractions_kernel(nk::tensor_span<f32_t, 1> payload, std::size_t member,
                                              std::size_t members) {
    std::size_t const scalar = blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (scalar < payload.extent(0))
        payload[scalar] = static_cast<f32_t>(splitmix64(scalar * members + member) >> 40) * 0x1.0p-24f - 0.5f;
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

/** The world's Float64 loss and every query's gradient: @c Σ(k,i,j)w·softplus(-t·(β·cos_k+b)). */
[[nodiscard]] static f64_t oracle_loss(nk::tensor_view<f64_t, 2> embeddings, matryoshka_t ladder,
                                       nk::tensor_span<f64_t, 2> gradient) noexcept {
    std::size_t const queries = embeddings.extent(0), hidden = embeddings.extent(1);
    for (std::size_t query = 0; query != queries; ++query)
        for (std::size_t channel = 0; channel != hidden; ++channel) gradient[query, channel] = 0.0;
    f64_t const weight = 1.0 / static_cast<f64_t>(queries);
    f64_t loss = 0;
    auto norms = take(nk::vector<f64_t>::uninitialized(queries), "norms");
    for (std::uint32_t band = 0; band != ladder.count; ++band) {
        std::size_t const width = ladder.ends[band];
        for (std::size_t i = 0; i != queries; ++i) {
            f64_t sum = 0;
            for (std::size_t h = 0; h != width; ++h) sum += embeddings[i, h] * embeddings[i, h];
            norms[i] = std::sqrt(sum);
        }
        for (std::size_t i = 0; i != queries; ++i)
            for (std::size_t j = 0; j != queries; ++j) {
                f64_t dot = 0;
                for (std::size_t h = 0; h != width; ++h) dot += embeddings[i, h] * embeddings[j, h];
                f64_t const cosine = dot / (norms[i] * norms[j]), target = i == j ? 1.0 : -1.0;
                f64_t const signed_logit = target * (beta * cosine + bias);
                loss += weight * (std::fmax(-signed_logit, 0.0) + std::log1p(std::exp(-std::fabs(signed_logit))));
                if (i == j) continue;
                f64_t const cell_grad = weight * target * (1.0 / (1.0 + std::exp(-signed_logit)) - 1.0) * beta;
                for (std::size_t h = 0; h != width; ++h) {
                    gradient[i, h] += cell_grad * (embeddings[j, h] / (norms[i] * norms[j]) -
                                                   cosine * embeddings[i, h] / (norms[i] * norms[i]));
                    gradient[j, h] += cell_grad * (embeddings[i, h] / (norms[i] * norms[j]) -
                                                   cosine * embeddings[j, h] / (norms[j] * norms[j]));
                }
            }
    }
    return loss;
}

/** One local GPU: its stream, its embeddings and their gradient, and its loss. */
struct gpu_t {
    nk::device_t device;
    nk::stream_t stream;
    nk::tensor<bf16_t, nk::allocator<bf16_t>, 2> embeddings, gradient;
    streamed_loss_t loss;

    /** The embeddings landed from every remote host, @b [remote_hosts,2,rows,hidden] by parity. */
    nk::tensor<bf16_t, nk::allocator<bf16_t>, 4> landings;

    /** One rail per remote host, in host order. */
    std::vector<rdma_rail_t> rails;

    [[nodiscard]] std::size_t local() const noexcept { return device.ordinal(); }
    void enter() const noexcept { require(cudaSetDevice(int(device.ordinal())), "device switch"); }
    void drain() const noexcept { enter(), require(cudaStreamSynchronize(cuda_stream(stream)), "step"); }
};

[[nodiscard]] static gpu_t make_gpu(std::size_t local, settings_t const &settings, std::uint64_t rank,
                                    f32_t scale) noexcept {
    nk::device_t const device = take(nk::device_t::make(nk::device_kind_t::cuda_k, local), "device");
    require(cudaSetDevice(int(local)), "device switch");
    nk::stream_t stream = take(nk::stream_t::make(device), "stream");
    nk_capability_t const capabilities = take(device.capabilities_enabled(), "capabilities");
    auto embeddings = make_tensor<bf16_t, 2>(capabilities, stream, {settings.rows, settings.hidden});
    launch(&keyed_embeddings_kernel, settings.rows, block_threads_k, stream, embeddings.span(), rank, scale);
    auto gradient = make_tensor<bf16_t, 2>(capabilities, stream, {settings.rows, settings.hidden});
    streamed_loss_t loss = streamed_loss_t::make(
        device, stream, settings.rows, settings.hidden,
        matryoshka_t::halvings(std::uint32_t(settings.hidden), settings.prefixes));
    return gpu_t {device, std::move(stream), std::move(embeddings), std::move(gradient), std::move(loss), {}, {}};
}

/** The host behind the @p remote -th rail of a GPU, skipping @p host itself. */
[[nodiscard]] static std::size_t remote_host(std::size_t remote, std::size_t host) noexcept {
    return remote < host ? remote : remote + 1;
}

/** One host's GPUs: their multimem domain, and the host-pinned summands every rail registers, which
 *  are declared first so they outlive the rails. */
struct cluster_t {
    nk::tensor<f32_t, nk::allocator<f32_t>, 2> summand_rows; // @b [hosts,4], one row per host
    nk::tensor<f32_t, nk::allocator<f32_t>, 1> totals;
    std::vector<gpu_t> gpus;
    multimem_domain_t domain;
    std::size_t steps = 0;

    void drain() const noexcept {
        for (gpu_t const &gpu : gpus) gpu.drain();
    }
};

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

/** Every local GPU with peer access to the others, keyed embeddings of @p scale , the multimem
 *  domain, and with several hosts a connected rail from every GPU to its counterpart on each. */
[[nodiscard]] static cluster_t make_cluster(settings_t const &settings, f32_t scale) noexcept {
    std::size_t const count = take(nk::device_t::count(nk::device_kind_t::cuda_k), "device count");
    nk_capability_t const capabilities = take(
        take(nk::device_t::make(nk::device_kind_t::cuda_k, 0), "device").capabilities_enabled(), "capabilities");
    cluster_t cluster;
    cluster.summand_rows = make_tensor<f32_t, 2>(capabilities, nk::stream_t {}, {settings.hosts, summand_scalars_k},
                                                 nk_allocator_init_pinned_best);
    cluster.totals = make_tensor<f32_t, 1>(capabilities, nk::stream_t {}, {summand_scalars_k},
                                           nk_allocator_init_pinned_best);
    for (std::size_t local = 0; local != count; ++local) {
        cluster.gpus.push_back(make_gpu(local, settings, settings.host * count + local, scale));
        for (std::size_t peer = 0; peer != count; ++peer)
            if (peer != local) require(cudaDeviceEnablePeerAccess(int(peer), 0), "peer access");
    }
    cluster.domain = multimem_domain_t::make(count, summand_scalars_k);
    if (settings.hosts == 1) return cluster;
    for (gpu_t &gpu : cluster.gpus) {
        gpu.enter();
        gpu.landings = make_tensor<bf16_t, 4>(gpu.loss.capabilities, gpu.stream,
                                              {settings.hosts - 1, 2, settings.rows, settings.hidden});
        std::string const nic = nic_of(settings.nics, gpu.local());
        for (std::size_t remote = 0; remote + 1 < settings.hosts; ++remote)
            gpu.rails.push_back(rdma_rail_t::make(nic.c_str(), gpu.embeddings.view(), gpu.landings[remote, nk::slice],
                                                  cluster.summand_rows.span(),
                                                  std::uint32_t(settings.host * count + gpu.local())));
    }
    cluster.drain();
    for (gpu_t &gpu : cluster.gpus)
        for (std::size_t remote = 0; remote + 1 < settings.hosts; ++remote)
            gpu.rails[remote].connect(exchange_endpoint(settings.exchange, gpu.rails[remote].local, settings.host,
                                                        remote_host(remote, settings.host), gpu.local()));
    return cluster;
}

/** Every local GPU's step: a multimem barrier once all embeddings are complete, the peers' read in
 *  place, each candidate block streamed and folded, then one multimem all-reduce of the summands,
 *  whose barriers end the peer reads. Every barrier is enqueued before any work, as a kernel's
 *  first, lazily loaded launch waits for its device to idle, which a barrier spinning on a GPU not
 *  yet enqueued never does. */
static void step_box(cluster_t &cluster, settings_t const &settings) noexcept {
    std::size_t const count = cluster.gpus.size();
    for (gpu_t const &gpu : cluster.gpus) gpu.enter(), cluster.domain.barrier(gpu.local(), gpu.stream);
    for (gpu_t &gpu : cluster.gpus) {
        gpu.enter();
        gpu.loss.begin(gpu.embeddings.view(), gpu.stream);
        for (gpu_t const &peer : cluster.gpus)
            gpu.loss.score(gpu.embeddings.span(), peer.embeddings.view(), gpu.local() * settings.rows,
                           peer.local() * settings.rows, beta, bias, f32_t(count * settings.rows), gpu.stream);
        gpu.loss.finish(gpu.embeddings.view(), gpu.gradient.span(), cluster.domain.payload(gpu.local()), gpu.stream);
        cluster.domain.all_reduce(gpu.local(), summand_scalars_k, gpu.stream);
    }
}

/** Every local GPU's step across hosts: embeddings posted over the rails, local candidates scored
 *  while they fly, landed ones read in place over NVLink, then summands all-reduced over multimem
 *  and added across hosts in host order. A host posts only after draining its last step, so with
 *  two landings no write meets a reader. */
static void step_hosts(cluster_t &cluster, settings_t const &settings) noexcept {
    std::size_t const count = cluster.gpus.size(), parity = cluster.steps++ % 2;
    std::size_t const candidates = settings.hosts * count * settings.rows;
    auto const rank_of = [&](std::size_t host, std::size_t local) { return host * count + local; };
    cluster.drain();
    for (gpu_t &gpu : cluster.gpus)
        for (rdma_rail_t &rail : gpu.rails) rail.post_embeddings(parity);
    for (gpu_t &gpu : cluster.gpus) {
        gpu.enter();
        gpu.loss.begin(gpu.embeddings.view(), gpu.stream);
        for (gpu_t const &peer : cluster.gpus)
            gpu.loss.score(
                gpu.embeddings.span(), peer.embeddings.view(), rank_of(settings.host, gpu.local()) * settings.rows,
                rank_of(settings.host, peer.local()) * settings.rows, beta, bias, f32_t(candidates), gpu.stream);
    }
    for (gpu_t &gpu : cluster.gpus) {
        for (rdma_rail_t &rail : gpu.rails) rail.await_landed();
        gpu.enter();
        require(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                   cudaFlushGPUDirectRDMAWritesToAllDevices),
                "RDMA write flush");
    }
    for (gpu_t &gpu : cluster.gpus) {
        gpu.enter();
        for (std::size_t remote = 0; remote + 1 != settings.hosts; ++remote)
            for (gpu_t const &peer : cluster.gpus)
                gpu.loss.score(gpu.embeddings.span(), peer.landings[remote, parity, nk::slice],
                               rank_of(settings.host, gpu.local()) * settings.rows,
                               rank_of(remote_host(remote, settings.host), peer.local()) * settings.rows, beta, bias,
                               f32_t(candidates), gpu.stream);
        gpu.loss.finish(gpu.embeddings.view(), gpu.gradient.span(), cluster.domain.payload(gpu.local()), gpu.stream);
        cluster.domain.all_reduce(gpu.local(), summand_scalars_k, gpu.stream);
    }
    gpu_t const &first = cluster.gpus.front();
    first.enter();
    require(cudaMemcpyAsync(cluster.summand_rows[settings.host, nk::slice].data(), cluster.domain.payload(0).data(),
                            summand_scalars_k * sizeof(f32_t), cudaMemcpyDeviceToHost, cuda_stream(first.stream)),
            "summands download");
    first.drain();
    for (rdma_rail_t &rail : cluster.gpus.front().rails) rail.post_summands(settings.host);
    for (rdma_rail_t &rail : cluster.gpus.front().rails) rail.await_landed();
    for (std::size_t summand = 0; summand != summand_scalars_k; ++summand) {
        f32_t total = 0.0f;
        for (std::size_t host = 0; host != settings.hosts; ++host) total += cluster.summand_rows[host, summand];
        cluster.totals[summand] = total;
    }
    for (gpu_t const &gpu : cluster.gpus)
        gpu.enter(),
            require(cudaMemcpyAsync(cluster.domain.payload(gpu.local()).data(), cluster.totals.data(),
                                    summand_scalars_k * sizeof(f32_t), cudaMemcpyHostToDevice, cuda_stream(gpu.stream)),
                    "totals upload");
}

using step_t = void (*)(cluster_t &, settings_t const &);

/** The step of one host, or of several over their rails. */
[[nodiscard]] static step_t step_of(settings_t const &settings) noexcept {
    return settings.hosts > 1 ? &step_hosts : &step_box;
}

/** One rank of a @c --gpus world on the first GPU, its own embeddings standing in for every
 *  candidate block. */
static int run_simulate(settings_t const &settings) noexcept {
    gpu_t gpu = make_gpu(0, settings, 0, benchmark_scale_k);
    auto summands = make_tensor<f32_t, 1>(gpu.loss.capabilities, gpu.stream, {summand_scalars_k});
    auto const step = [&] {
        gpu.loss.begin(gpu.embeddings.view(), gpu.stream);
        for (std::size_t block = 0; block != settings.simulated_gpus; ++block)
            gpu.loss.score(gpu.embeddings.span(), gpu.embeddings.view(), 0, block * settings.rows, beta, bias,
                           f32_t(settings.simulated_gpus * settings.rows), gpu.stream);
        gpu.loss.finish(gpu.embeddings.view(), gpu.gradient.span(), summands.span(), gpu.stream);
    };
    step(), gpu.drain();
    timing_t const timing = time_steps(step, [&] { gpu.drain(); }, settings.windows, step_calls_k);
    std::printf("simulate,%zu,%zu,%u,%zu,%zu,%.3f,%.3f\n", settings.rows, settings.hidden, settings.prefixes,
                settings.simulated_gpus, gpu.loss.bytes(), timing.min_ms, timing.median_ms);
    return 0;
}

/** Every local GPU's, or every host's, step time and memory. */
static int run_steps(settings_t const &settings) noexcept {
    cluster_t cluster = make_cluster(settings, benchmark_scale_k);
    auto const step = [&, advance = step_of(settings)] { advance(cluster, settings); };
    step(), cluster.drain();
    timing_t const timing = time_steps(step, [&] { cluster.drain(); }, settings.windows, step_calls_k);
    gpu_t const &first = cluster.gpus.front();
    std::printf("%s,%zu,%zu,%u,%zu,%zu,%.3f,%.3f\n", settings.hosts > 1 ? "hosts" : "box", settings.rows,
                settings.hidden, settings.prefixes, settings.hosts * cluster.gpus.size(),
                first.loss.bytes() + first.landings.numel() * sizeof(bf16_t), timing.min_ms, timing.median_ms);
    return 0;
}

/** One step against a Float64 evaluation of the whole world: every rank's embeddings, local or
 *  landed, and every local gradient, each downloaded on its GPU's stream. */
static int run_oracle(settings_t const &settings) noexcept {
    cluster_t cluster = make_cluster(settings, oracle_scale_k);
    step_of(settings)(cluster, settings), cluster.drain();
    // The last step's parity, which only the landings of several hosts depend on.
    std::size_t const count = cluster.gpus.size(), ranks = settings.hosts * count, parity = (cluster.steps + 1) % 2;
    gpu_t const &first = cluster.gpus.front();
    auto embeddings = make_tensor<bf16_t, 3>(first.loss.capabilities, first.stream,
                                             {ranks, settings.rows, settings.hidden}, nk_allocator_init_pinned_best);
    auto gradients = make_tensor<bf16_t, 3>(first.loss.capabilities, first.stream,
                                            {count, settings.rows, settings.hidden}, nk_allocator_init_pinned_best);
    auto summands = make_tensor<f32_t, 2>(first.loss.capabilities, first.stream, {count, 3},
                                          nk_allocator_init_pinned_best);
    std::size_t const bytes = settings.rows * settings.hidden * sizeof(bf16_t);
    for (gpu_t const &gpu : cluster.gpus) {
        cudaStream_t const cuda = cuda_stream(gpu.stream);
        gpu.enter();
        require(cudaMemcpyAsync(embeddings[settings.host * count + gpu.local(), nk::slice].data(),
                                gpu.embeddings.data(), bytes, cudaMemcpyDeviceToHost, cuda),
                "download");
        for (std::size_t remote = 0; remote + 1 < settings.hosts; ++remote)
            require(
                cudaMemcpyAsync(embeddings[remote_host(remote, settings.host) * count + gpu.local(), nk::slice].data(),
                                gpu.landings[remote, parity, nk::slice].data(), bytes, cudaMemcpyDeviceToHost, cuda),
                "download");
        require(cudaMemcpyAsync(gradients[gpu.local(), nk::slice].data(), gpu.gradient.data(), bytes,
                                cudaMemcpyDeviceToHost, cuda),
                "download");
        require(cudaMemcpyAsync(summands[gpu.local(), nk::slice].data(), cluster.domain.payload(gpu.local()).data(),
                                3 * sizeof(f32_t), cudaMemcpyDeviceToHost, cuda),
                "download");
    }
    cluster.drain();
    nk::aligned_allocator<f64_t> const host;
    auto world = take(nk::tensor<f64_t, nk::aligned_allocator<f64_t>, 3>::uninitialized(
                          {ranks, settings.rows, settings.hidden}, host),
                      "world");
    for (std::size_t rank = 0; rank != ranks; ++rank)
        for (std::size_t row = 0; row != settings.rows; ++row)
            for (std::size_t channel = 0; channel != settings.hidden; ++channel)
                world[rank, row, channel] = __bfloat162float(embeddings[rank, row, channel]);
    std::size_t disagreeing = 0; // local GPUs whose loss differs from the first's
    for (std::size_t local = 0; local != count; ++local) disagreeing += summands[local, 0] != summands[0, 0];
    matryoshka_t const ladder = matryoshka_t::halvings(std::uint32_t(settings.hidden), settings.prefixes);
    auto exact = take(nk::tensor<f64_t, nk::aligned_allocator<f64_t>, 3>::uninitialized(
                          {ranks, settings.rows, settings.hidden}, host),
                      "oracle gradients");
    f64_t const exact_loss = oracle_loss(world.view().reshape<2>({ranks * settings.rows, settings.hidden}), ladder,
                                         exact.span().reshape<2>({ranks * settings.rows, settings.hidden}));
    f64_t difference = 0, reference = 0;
    for (std::size_t local = 0; local != count; ++local)
        for (std::size_t row = 0; row != settings.rows; ++row)
            for (std::size_t channel = 0; channel != settings.hidden; ++channel) {
                f64_t const expected = exact[settings.host * count + local, row, channel];
                f64_t const delta = __bfloat162float(gradients[local, row, channel]) - expected;
                difference += delta * delta, reference += expected * expected;
            }
    std::printf("oracle,%zu,%zu,%u,%zu,%.9f,%.9f,%.3e,%zu\n", settings.rows, settings.hidden, settings.prefixes, ranks,
                exact_loss, f64_t(summands[0, 0]), std::sqrt(difference / reference), disagreeing);
    return 0;
}

/** Barrier latency, then reduce-scatter and all-reduce bus bandwidths from 1 MiB to @c --max-mib as
 *  nccl-tests defines them, then exact sums of keyed integers, and bitwise repeats of an all-reduce
 *  of keyed fractions. Each phase is enqueued on every GPU before the next, so no first launch
 *  waits behind a spinning barrier. */
static int run_collectives(settings_t const &settings) noexcept {
    cluster_t cluster = make_cluster(settings, benchmark_scale_k);
    std::vector<gpu_t> const &gpus = cluster.gpus;
    std::size_t const count = gpus.size(), max_scalars = settings.max_mib << 18;
    multimem_domain_t domain = multimem_domain_t::make(count, max_scalars);
    auto const drain = [&] { cluster.drain(); };
    auto const barriers = [&] {
        for (gpu_t const &gpu : gpus) gpu.enter(), domain.barrier(gpu.local(), gpu.stream);
    };
    timing_t const barrier = time_steps(barriers, drain, settings.windows, collective_calls_k);
    std::printf("barrier,%zu,%zu,%.2f,%.2f\n", count, domain.stride, barrier.min_ms * 1e3, barrier.median_ms * 1e3);

    f64_t const ring = f64_t(count - 1) / f64_t(count);
    for (std::size_t scalars = 1 << 18; scalars <= max_scalars; scalars *= 2) {
        std::size_t const shard = scalars / count / 4 * 4, bytes = shard * count * sizeof(f32_t);
        auto const scatters = [&] {
            for (gpu_t const &gpu : gpus) gpu.enter(), domain.reduce_scatter(gpu.local(), shard, gpu.stream);
        };
        auto const reductions = [&] {
            for (gpu_t const &gpu : gpus) gpu.enter(), domain.all_reduce(gpu.local(), shard * count, gpu.stream);
        };
        timing_t const scatter = time_steps(scatters, drain, settings.windows, collective_calls_k);
        timing_t const reduce = time_steps(reductions, drain, settings.windows, collective_calls_k);
        f64_t const scatter_bandwidth = f64_t(bytes) / (scatter.min_ms * 1e6),
                    reduce_bandwidth = f64_t(bytes) / (reduce.min_ms * 1e6);
        std::printf("reduce_scatter,%zu,%zu,%.2f,%.2f,%.1f,%.1f\n", count, bytes, scatter.min_ms * 1e3,
                    scatter.median_ms * 1e3, scatter_bandwidth, scatter_bandwidth * ring);
        std::printf("all_reduce,%zu,%zu,%.2f,%.2f,%.1f,%.1f\n", count, bytes, reduce.min_ms * 1e3,
                    reduce.median_ms * 1e3, reduce_bandwidth, reduce_bandwidth * 2 * ring);
    }

    // Counters on the first GPU, which every GPU adds to over NVLink: wrong scatters, wrong sums,
    // differing repeats.
    gpu_t const &first = gpus.front();
    auto counters = make_tensor<unsigned long long, 1>(first.loss.capabilities, first.stream, {3});
    auto reference = make_tensor<f32_t, 1>(first.loss.capabilities, first.stream, {max_scalars});
    first.enter();
    require(
        cudaMemsetAsync(counters.data(), 0, counters.numel() * sizeof(unsigned long long), cuda_stream(first.stream)),
        "zeroing counters");
    drain();
    std::size_t const shard = max_scalars / count / 4 * 4;
    auto const fill = [&](void (*kernel)(nk::tensor_span<f32_t, 1>, std::size_t, std::size_t)) {
        for (gpu_t const &gpu : gpus)
            gpu.enter(), launch(kernel, blocks_for(max_scalars), block_threads_k, gpu.stream,
                                domain.payload(gpu.local()), gpu.local(), count);
    };
    auto const all_reduce = [&] {
        for (gpu_t const &gpu : gpus) gpu.enter(), domain.all_reduce(gpu.local(), max_scalars, gpu.stream);
    };
    fill(&keyed_integers_kernel);
    for (gpu_t const &gpu : gpus) gpu.enter(), domain.reduce_scatter(gpu.local(), shard, gpu.stream);
    for (gpu_t const &gpu : gpus)
        gpu.enter(), launch(&count_wrong_sums_kernel, blocks_for(shard), block_threads_k, gpu.stream,
                            domain.payload(gpu.local()), gpu.local() * shard, gpu.local() * shard + shard, count,
                            counters.data());
    fill(&keyed_integers_kernel);
    all_reduce();
    for (gpu_t const &gpu : gpus)
        gpu.enter(), launch(&count_wrong_sums_kernel, blocks_for(max_scalars), block_threads_k, gpu.stream,
                            domain.payload(gpu.local()), std::size_t(0), max_scalars, count, counters.data() + 1);
    fill(&keyed_fractions_kernel);
    all_reduce();
    first.enter();
    require(cudaMemcpyAsync(reference.data(), domain.payload(0).data(), max_scalars * sizeof(f32_t),
                            cudaMemcpyDeviceToDevice, cuda_stream(first.stream)),
            "reference copy");
    for (std::size_t repeat = 1; repeat < settings.repeats; ++repeat) {
        fill(&keyed_fractions_kernel);
        all_reduce();
        first.enter();
        launch(&count_differences_kernel, blocks_for(max_scalars), block_threads_k, first.stream, domain.payload(0),
               reference.view(), counters.data() + 2);
    }
    auto counted = make_tensor<unsigned long long, 1>(first.loss.capabilities, first.stream, {3},
                                                      nk_allocator_init_pinned_best);
    first.enter();
    require(cudaMemcpyAsync(counted.data(), counters.data(), counters.numel() * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, cuda_stream(first.stream)),
            "counter download");
    drain();
    std::printf("validation,%zu,%zu,%llu,%llu,%zu,%llu\n", count, max_scalars * sizeof(f32_t), counted[0], counted[1],
                settings.repeats, counted[2]);
    return 0;
}

/** A mode the binary runs, by the name its first argument gives. */
struct run_mode_t {
    char const *name;
    int (*run)(settings_t const &);
};

int main(int argc, char **argv) {
    static constexpr run_mode_t modes[] = {{"oracle", &run_oracle},
                                           {"simulate", &run_simulate},
                                           {"box", &run_steps},
                                           {"hosts", &run_steps},
                                           {"collectives", &run_collectives}};
    run_mode_t const *mode = argc > 1 ? std::find_if(
                                            std::begin(modes), std::end(modes),
                                            [&](run_mode_t const &known) { return !std::strcmp(known.name, argv[1]); })
                                      : std::end(modes);
    if (mode == std::end(modes))
        return std::fprintf(stderr, "usage: %s oracle|simulate|box|hosts|collectives [--rows R] [--gpus S]\n", argv[0]),
               2;
    settings_t settings;
    for (int index = 2; index + 1 < argc; index += 2) {
        std::string_view const flag = argv[index];
        char const *const value = argv[index + 1];
        if (flag == "--nics") settings.nics = value;
        else if (flag == "--exchange") settings.exchange = value;
        else if (flag == "--rows") settings.rows = std::stoul(value);
        else if (flag == "--host") settings.host = std::stoul(value);
        else if (flag == "--hosts") settings.hosts = std::stoul(value);
        else if (flag == "--gpus") settings.simulated_gpus = std::stoul(value);
        else if (flag == "--prefixes") settings.prefixes = std::uint32_t(std::stoul(value));
        else if (flag == "--windows") settings.windows = std::stoul(value);
        else if (flag == "--max-mib") settings.max_mib = std::stoul(value);
        else if (flag == "--repeats") settings.repeats = std::stoul(value);
        else return std::fprintf(stderr, "unknown flag %s\n", argv[index]), 2;
    }
    if (settings.hosts > 1 && (settings.nics.empty() || settings.exchange.empty() || settings.host >= settings.hosts))
        return std::fprintf(stderr, "several hosts need --host, --nics and --exchange\n"), 2;
    return mode->run(settings);
}
