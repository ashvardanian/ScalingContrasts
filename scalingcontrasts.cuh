/**
 *  @file scalingcontrasts.cuh
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Direct NVLink scale-up primitives for training, and a contrastive loss built on them.
 *
 *  Retrieval and perception models learn their embeddings contrastively, scoring every query of a
 *  batch against every candidate, so the batch, and with it the number of negatives each pair
 *  learns from, is bounded by how cheaply GPUs see each other's embeddings. This header holds the
 *  pieces that keep that cheap: multimem collectives, RDMA rails, and a streamed, row-owned sigmoid
 *  loss whose memory does not grow with the world and whose gradient exchanges nothing.
 */
#pragma once

#include <cerrno>  // `errno`
#include <cmath>   // `std::fmin`, `std::fmax`, `std::fabs`, `std::sqrt`, `std::fma`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fprintf`
#include <cstdlib> // `std::exit`
#include <cstring> // `std::strerror`, `std::strcmp`

#include <arpa/inet.h> // `htonl`
#include <unistd.h>    // `close`, `sysconf`

#include <algorithm> // `std::max`, `std::clamp`
#include <array>     // `std::array`
#include <span>      // `std::span`
#include <tuple>     // `std::tuple`, `std::apply`
#include <utility>   // `std::exchange`, `std::swap`

#include <cuda.h>             // `cuMulticastCreate`, `cuMulticastBindMem`, `cuMemMap`, `cuMemSetAccess`
#include <cuda/atomic>        // `cuda::atomic_ref`
#include <cuda_bf16.h>        // `__nv_bfloat16`
#include <cuda_runtime.h>     // `cudaStream_t`, `cudaMemsetAsync`
#include <infiniband/verbs.h> // `ibv_reg_dmabuf_mr`, `ibv_post_send`, `ibv_poll_cq`

#include <numkong/numkong.hpp> // `nk::tensor`, `nk::allocator`, `nk::stream_t`, `nk::device_t`

namespace ashvardanian::scalingcontrasts {

namespace nk = ashvardanian::numkong;

using bf16_t = __nv_bfloat16;
using byte_t = std::byte;
using f32_t = float;
using f64_t = double;

inline constexpr std::uint32_t block_threads_k = 256;
inline constexpr f32_t cosine_floor_k = 1.0e-2f; // norm products at or below this read as zero cosines
inline constexpr f32_t inverse_cosine_floor_k = 1.0f / cosine_floor_k;

/** The smallest count of @p divisor -sized pieces that covers @p value . */
[[nodiscard]] constexpr std::size_t divide_round_up(std::size_t value, std::size_t divisor) noexcept {
    return (value + divisor - 1) / divisor;
}

/** Ends the process with @p what when @p status failed: a benchmark has no recovery path. */
inline void require(nk::status_t status, char const *what) noexcept {
    if (nk::succeeded(status)) return;
    std::fprintf(stderr, "%s failed: %s\n", what, nk::status_name(status));
    std::exit(1);
}

inline void require(cudaError_t error, char const *what) noexcept {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(error));
    std::exit(1);
}

[[nodiscard]] inline cudaStream_t cuda_stream(nk::stream_t const &stream) noexcept {
    return reinterpret_cast<cudaStream_t>(stream.get());
}

/** Launches @p kernel on @p stream with @p shared_bytes of dynamic shared memory and @p arguments
 *  copied by value, and checks that it launched.
 *  No triple chevrons: https://ashvardanian.com/posts/less-wrong-cuda-hello-world/ */
template <typename... parameter_types_, typename... argument_types_>
void launch(void (*kernel)(parameter_types_...), dim3 grid, dim3 block, std::size_t shared_bytes,
            nk::stream_t const &stream, argument_types_ &&...arguments) noexcept {
    std::tuple<parameter_types_...> values {std::forward<argument_types_>(arguments)...};
    void *addresses[sizeof...(parameter_types_)];
    std::apply(
        [&](auto &...value) {
            std::size_t index = 0;
            ((addresses[index++] = &value), ...);
        },
        values);
    require(cudaLaunchKernel(reinterpret_cast<void const *>(kernel), grid, block, addresses, shared_bytes,
                             cuda_stream(stream)),
            "kernel launch");
}

/** Launches @p kernel on @p stream with no dynamic shared memory. */
template <typename... parameter_types_, typename... argument_types_>
void launch(void (*kernel)(parameter_types_...), dim3 grid, dim3 block, nk::stream_t const &stream,
            argument_types_ &&...arguments) noexcept {
    launch(kernel, grid, block, 0, stream, std::forward<argument_types_>(arguments)...);
}

/** The value of a NumKong factory, or the end of the process. */
template <typename value_type_>
[[nodiscard]] value_type_ take(nk::expected<value_type_> &&made, char const *what) noexcept {
    require(made.status, what);
    return std::move(made.value);
}

/** A tensor of @p extents from NumKong's @p initialize policy bound to @p stream: allocated on the
 *  stream's device and freed in its order, as device, pinned or unified memory. */
template <typename value_type_, std::size_t rank_>
[[nodiscard]] nk::tensor<value_type_, nk::allocator<value_type_>, rank_> make_tensor(
    nk_capability_t capabilities, nk::stream_t const &stream, std::initializer_list<std::size_t> extents,
    nk_status_t (*initialize)(nk_allocator_t *, nk_capability_t) = nk_allocator_init_device_best) noexcept {
    return take(
        nk::tensor<value_type_, nk::allocator<value_type_>, rank_>::uninitialized(
            extents, take(nk::allocator<value_type_>::make(capabilities, stream.get(), initialize), "allocator")),
        "allocation");
}

/** The Matryoshka ladder, ascending prefix widths; band @c b covers the channels
 *  @b [begin(b),ends[b]). */
struct matryoshka_t {
    std::uint32_t count = 0;
    std::uint32_t ends[8] = {};

    [[nodiscard]] __host__ __device__ std::uint32_t begin(std::uint32_t band) const noexcept {
        return band == 0 ? 0 : ends[band - 1];
    }
    [[nodiscard]] __host__ __device__ std::uint32_t width(std::uint32_t band) const noexcept {
        return ends[band] - begin(band);
    }

    /** @p count halvings of @p hidden , narrowest first. */
    [[nodiscard]] static matryoshka_t halvings(std::uint32_t hidden, std::uint32_t count) noexcept {
        matryoshka_t ladder;
        ladder.count = count;
        for (std::uint32_t prefix = 0; prefix != count; ++prefix) ladder.ends[prefix] = hidden >> (count - 1 - prefix);
        return ladder;
    }
};

/** One member's counters in the multimem domain: the multicast arrival flag, and its own replica's flag,
 *  round and count of blocks done. */
struct multimem_counters_t {
    std::uint32_t *multicast_flag = nullptr, *own_flag = nullptr, *own_round = nullptr, *own_finished = nullptr;
    std::uint32_t members = 0;
};

/** Adds one arrival for this member to every replica's flag. */
__device__ inline void multimem_arrive_(multimem_counters_t const &counters) noexcept {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(counters.multicast_flag), "r"(1u)
                 : "memory");
}

/** Waits until this member's own flag reaches @p target , wrap-safe past 2³² arrivals. */
__device__ inline void multimem_await_(multimem_counters_t const &counters, std::uint32_t target) noexcept {
    std::uint32_t arrived = 0;
    do asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(arrived) : "l"(counters.own_flag) : "memory");
    while (static_cast<std::int32_t>(arrived - target) < 0);
}

/** Enters a collective round: block 0 arrives for the member, every block waits for all members, and all of
 *  them read the round before the last block to leave advances it. */
[[nodiscard]] __device__ inline std::uint32_t multimem_enter_(multimem_counters_t const &counters) noexcept {
    std::uint32_t const round = *counters.own_round;
    if (threadIdx.x == 0 && blockIdx.x == 0) multimem_arrive_(counters);
    if (threadIdx.x == 0) multimem_await_(counters, round + counters.members);
    __syncthreads();
    return round;
}

/** Leaves a collective round: the last block done arrives again, waits for every member's last block, and
 *  advances the round, so the kernel ends only once no member still reads any payload. */
__device__ inline void multimem_leave_(multimem_counters_t const &counters, std::uint32_t round) noexcept {
    __syncthreads();
    if (threadIdx.x != 0) return;
    // System scope, as the blocks' multicast stores land on peers before the last block's release.
    cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> finished(*counters.own_finished);
    if (finished.fetch_add(1u, cuda::std::memory_order_acq_rel) != gridDim.x - 1) return;
    finished.store(0u, cuda::std::memory_order_relaxed);
    multimem_arrive_(counters);
    multimem_await_(counters, round + 2 * counters.members);
    *counters.own_round = round + 2 * counters.members;
}

/** A barrier on its own: one arrival, then the round's last. */
static __global__ void multimem_barrier_cuda_(multimem_counters_t counters) {
    std::uint32_t const round = *counters.own_round;
    multimem_arrive_(counters);
    multimem_await_(counters, round + counters.members);
    *counters.own_round = round + counters.members;
}

/** @c own[i]=Σ(members)payload[i] over @b [begin,end), one switch-reduced load per four scalars,
 *  between the round's entry and exit. */
static __global__ void multimem_reduce_scatter_cuda_(multimem_counters_t counters, nk::tensor_view<f32_t, 1> multicast,
                                                     nk::tensor_span<f32_t, 1> own, std::size_t begin,
                                                     std::size_t end) {
    std::uint32_t const round = multimem_enter_(counters);
    for (std::size_t scalar = begin + 4 * (blockIdx.x * std::size_t {blockDim.x} + threadIdx.x); scalar < end;
         scalar += 4 * std::size_t {gridDim.x} * blockDim.x) {
        f32_t x, y, z, w;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 " //
                     "{%0, %1, %2, %3}, [%4];"                           //
                     : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                     : "l"(&multicast[scalar])
                     : "memory");
        asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(&own[scalar]), "f"(x), "f"(y), "f"(z), "f"(w)
                     : "memory");
    }
    multimem_leave_(counters, round);
}

/** @c payload[i]=Σ(members)payload[i] over @b [begin,end) on every replica: one switch-reduced
 *  load and one multicast store per four scalars, between the round's entry and exit. */
static __global__ void multimem_all_reduce_cuda_(multimem_counters_t counters, nk::tensor_span<f32_t, 1> multicast,
                                                 std::size_t begin, std::size_t end) {
    std::uint32_t const round = multimem_enter_(counters);
    for (std::size_t scalar = begin + 4 * (blockIdx.x * std::size_t {blockDim.x} + threadIdx.x); scalar < end;
         scalar += 4 * std::size_t {gridDim.x} * blockDim.x) {
        f32_t x, y, z, w;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 " //
                     "{%0, %1, %2, %3}, [%4];"                           //
                     : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                     : "l"(&multicast[scalar])
                     : "memory");
        asm volatile("multimem.st.relaxed.sys.global.v4.f32 " //
                     "[%0], {%1, %2, %3, %4};"                //
                     ::"l"(&multicast[scalar]),
                     "f"(x), "f"(y), "f"(z), "f"(w)
                     : "memory");
    }
    multimem_leave_(counters, round);
}

inline void require(CUresult result, char const *what) noexcept {
    if (result == CUDA_SUCCESS) return;
    char const *name = nullptr;
    [[maybe_unused]] CUresult const named = cuGetErrorName(result, &name);
    std::fprintf(stderr, "%s failed: %s\n", what, name ? name : "an unnamed driver error");
    std::exit(1);
}

/**
 *  @brief One host's GPUs as an NVLink multicast domain: a replica per GPU, all bound into one
 *      multicast object that any of them addresses to reduce in the switch or to store into every
 *      replica at once.
 *
 *  A replica holds @ref payload_scalars scalars, then its arrival flag, its barrier round and its
 *  count of blocks done with a collective. Only
 *  its own GPU reads the round, so barriers advance with no host writes. Every call enqueues on
 *  @p member 's stream, with that member's device current.
 */
struct multimem_domain_t {
    std::size_t members = 0, payload_scalars = 0, stride = 0;
    int multiprocessors = 0;
    CUmemGenericAllocationHandle multicast = 0;
    CUdeviceptr multicast_base = 0, replicas_base = 0;

    multimem_domain_t() noexcept = default;
    multimem_domain_t(multimem_domain_t const &) = delete;
    multimem_domain_t(multimem_domain_t &&other) noexcept
        : members(other.members), payload_scalars(other.payload_scalars), stride(other.stride),
          multiprocessors(other.multiprocessors), multicast(std::exchange(other.multicast, 0)),
          multicast_base(other.multicast_base), replicas_base(other.replicas_base) {}
    multimem_domain_t &operator=(multimem_domain_t &&other) noexcept {
        std::swap(members, other.members), std::swap(payload_scalars, other.payload_scalars);
        std::swap(stride, other.stride), std::swap(multiprocessors, other.multiprocessors);
        std::swap(multicast, other.multicast), std::swap(multicast_base, other.multicast_base);
        std::swap(replicas_base, other.replicas_base);
        return *this;
    }

    ~multimem_domain_t() noexcept {
        if (!multicast) return;
        require(cuMemUnmap(multicast_base, stride), "multicast unmapping");
        require(cuMemAddressFree(multicast_base, stride), "multicast reservation release");
        for (std::size_t member = 0; member != members; ++member)
            require(cuMulticastUnbind(multicast, device_(member), 0, stride), "multicast unbinding");
        require(cuMemUnmap(replicas_base, members * stride), "replica unmapping");
        require(cuMemAddressFree(replicas_base, members * stride), "replica reservation release");
        require(cuMemRelease(multicast), "multicast release");
    }

    /** Binds a zeroed replica of @p payload_scalars, a multiple of four, on each of the first
     *  @p members GPUs. */
    [[nodiscard]] static multimem_domain_t make(std::size_t members, std::size_t payload_scalars) noexcept {
        multimem_domain_t domain;
        domain.members = members, domain.payload_scalars = payload_scalars;
        CUmemAllocationProp allocation {};
        allocation.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        allocation.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        allocation.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        CUmulticastObjectProp properties {};
        properties.numDevices = static_cast<unsigned>(members);
        properties.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        // the payload, then the flag, the round and the count of blocks done
        properties.size = (payload_scalars + 4) * sizeof(f32_t);
        std::size_t allocation_granularity = 0, multicast_granularity = 0;
        require(cuMemGetAllocationGranularity(&allocation_granularity, &allocation, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                "allocation granularity");
        require(cuMulticastGetGranularity(&multicast_granularity, &properties, CU_MULTICAST_GRANULARITY_MINIMUM),
                "multicast granularity");
        std::size_t const granularity = std::max(allocation_granularity, multicast_granularity);
        domain.stride = divide_round_up(properties.size, granularity) * granularity;
        properties.size = domain.stride;
        require(cuMulticastCreate(&domain.multicast, &properties), "multicast creation");
        for (std::size_t member = 0; member != members; ++member)
            require(cuMulticastAddDevice(domain.multicast, device_(member)), "multicast device addition");
        require(cuMemAddressReserve(&domain.replicas_base, members * domain.stride, 0, 0, 0), "replica reservation");
        for (std::size_t member = 0; member != members; ++member) {
            CUdeviceptr const replica = domain.replicas_base + member * domain.stride;
            CUmemGenericAllocationHandle physical = 0;
            allocation.location.id = int(member);
            CUmemAccessDesc const access = access_(member);
            require(cuMemCreate(&physical, domain.stride, &allocation, 0), "replica allocation");
            require(cuMulticastBindMem(domain.multicast, 0, physical, 0, domain.stride, 0), "multicast binding");
            require(cuMemMap(replica, domain.stride, 0, physical, 0), "replica mapping");
            // the mapping keeps the memory
            require(cuMemRelease(physical), "replica handle release");
            require(cuMemSetAccess(replica, domain.stride, &access, 1), "replica access");
            // Blocking, so no member's first arrival can land before another's zeroing.
            require(cudaSetDevice(int(member)), "device switch");
            require(cuMemsetD8(replica, 0, domain.stride), "replica zeroing");
            require(cudaDeviceSynchronize(), "replica zeroing");
        }
        require(cuMemAddressReserve(&domain.multicast_base, domain.stride, 0, 0, 0), "multicast reservation");
        require(cuMemMap(domain.multicast_base, domain.stride, 0, domain.multicast, 0), "multicast mapping");
        for (std::size_t member = 0; member != members; ++member) {
            CUmemAccessDesc const access = access_(member);
            require(cuMemSetAccess(domain.multicast_base, domain.stride, &access, 1), "multicast access");
        }
        require(cuDeviceGetAttribute(&domain.multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_(0)),
                "multiprocessor count");
        // A kernel's first launch loads it lazily, waiting for an idle device, so load the
        // collectives while every device is idle rather than behind a member that spins.
        for (std::size_t member = 0; member != members; ++member) {
            cudaFuncAttributes attributes {};
            require(cudaSetDevice(int(member)), "device switch");
            require(cudaFuncGetAttributes(&attributes, &multimem_barrier_cuda_), "kernel loading");
            require(cudaFuncGetAttributes(&attributes, &multimem_reduce_scatter_cuda_), "kernel loading");
            require(cudaFuncGetAttributes(&attributes, &multimem_all_reduce_cuda_), "kernel loading");
        }
        return domain;
    }

    /** The scalars @p member contributes and receives, in its own replica. */
    [[nodiscard]] nk::tensor_span<f32_t, 1> payload(std::size_t member) const noexcept {
        return {pointer_<f32_t>(replicas_base + member * stride), payload_scalars};
    }

    /** Returns on @p member 's stream once every member has entered the same round. */
    void barrier(std::size_t member, nk::stream_t const &stream) noexcept {
        launch(&multimem_barrier_cuda_, 1, 1, stream, counters_(member));
    }

    /** In place: @p member 's slice of @p shard_scalars , a multiple of four, at
     *  @c member·shard_scalars of its payload becomes the sum of every payload's same slice. */
    void reduce_scatter(std::size_t member, std::size_t shard_scalars, nk::stream_t const &stream) noexcept {
        std::size_t const begin = member * shard_scalars, end = begin + shard_scalars;
        launch(&multimem_reduce_scatter_cuda_, blocks_(shard_scalars), block_threads_k, stream, counters_(member),
               multicast_payload_(), payload(member), begin, end);
    }

    /** In place: the first @p scalars , a multiple of four, of every payload become their sum, each
     *  member reducing and broadcasting its own share. */
    void all_reduce(std::size_t member, std::size_t scalars, nk::stream_t const &stream) noexcept {
        std::size_t const vectors = scalars / 4, share = divide_round_up(vectors, members);
        std::size_t const begin = std::min(member * share, vectors) * 4,
                          end = std::min(member * share + share, vectors) * 4;
        launch(&multimem_all_reduce_cuda_, blocks_(end - begin), block_threads_k, stream, counters_(member),
               multicast_payload_(), begin, end);
    }

  private:
    template <typename value_type_>
    [[nodiscard]] static value_type_ *pointer_(CUdeviceptr address) noexcept {
        return reinterpret_cast<value_type_ *>(address);
    }

    [[nodiscard]] multimem_counters_t counters_(std::size_t member) const noexcept {
        CUdeviceptr const own = replicas_base + member * stride + payload_scalars * sizeof(f32_t);
        multimem_counters_t counters;
        counters.multicast_flag = pointer_<std::uint32_t>(multicast_base + payload_scalars * sizeof(f32_t));
        counters.own_flag = pointer_<std::uint32_t>(own);
        counters.own_round = pointer_<std::uint32_t>(own + sizeof(std::uint32_t));
        counters.own_finished = pointer_<std::uint32_t>(own + 2 * sizeof(std::uint32_t));
        counters.members = static_cast<std::uint32_t>(members);
        return counters;
    }
    [[nodiscard]] nk::tensor_span<f32_t, 1> multicast_payload_() const noexcept {
        return {pointer_<f32_t>(multicast_base), payload_scalars};
    }
    [[nodiscard]] std::uint32_t blocks_(std::size_t scalars) const noexcept {
        std::size_t const wanted = divide_round_up(scalars / 4, block_threads_k);
        return static_cast<std::uint32_t>(std::clamp<std::size_t>(wanted, 1, std::size_t(multiprocessors)));
    }
    [[nodiscard]] static CUdevice device_(std::size_t ordinal) noexcept {
        CUdevice device = 0;
        require(cuDeviceGet(&device, int(ordinal)), "device handle");
        return device;
    }
    [[nodiscard]] static CUmemAccessDesc access_(std::size_t ordinal) noexcept {
        CUmemAccessDesc access {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = int(ordinal);
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        return access;
    }
};

/** Ends the process when a verbs call returned @p result , an @c errno value or zero. */
inline void require(int result, char const *what) noexcept {
    if (result == 0) return;
    std::fprintf(stderr, "%s failed: %s\n", what, std::strerror(result));
    std::exit(1);
}

/** What a rail's counterpart needs to reach it: its queue pair, its address, and the landings it
 *  may write. */
struct rdma_endpoint_t {
    ibv_gid gid {};
    std::uint64_t landing_address = 0, summands_address = 0;
    std::uint32_t queue_pair = 0, packet_sequence = 0, landing_key = 0, summands_key = 0;
};

/**
 *  @brief One RoCE rail between a GPU and its counterpart on another host: an RC queue pair on the
 *      GPU's own NIC, writing GPU memory straight into GPU memory, registered as dma-buf.
 *
 *  - @b Embeddings: the GPU's embeddings land in the counterpart's @c landing[parity],
 *    double-buffered by step parity, and the counterpart's other GPUs read them over NVLink.
 *  - @b Summands: a row of host-pinned @c summands, at the sender's host index, lands in the same
 *    row on the other side, so every host adds the rows in one order.
 *
 *  Every write carries an immediate, so its landing completes one receive that @ref await_landed
 *  reaps.
 */
struct rdma_rail_t {
    ibv_context *context = nullptr;
    ibv_pd *protection = nullptr;
    ibv_cq *completions = nullptr;
    ibv_qp *queue_pair = nullptr;
    ibv_mr *embeddings = nullptr, *landing = nullptr, *summands = nullptr;
    rdma_endpoint_t local, remote;
    ibv_mtu mtu = IBV_MTU_1024;
    int gid_index = 0;
    std::uint64_t embeddings_address = 0;
    std::size_t embeddings_bytes = 0, summands_row_bytes = 0, landed = 0;

    rdma_rail_t() noexcept = default;
    rdma_rail_t(rdma_rail_t const &) = delete;
    rdma_rail_t(rdma_rail_t &&other) noexcept
        : context(std::exchange(other.context, nullptr)), protection(std::exchange(other.protection, nullptr)),
          completions(std::exchange(other.completions, nullptr)), queue_pair(std::exchange(other.queue_pair, nullptr)),
          embeddings(std::exchange(other.embeddings, nullptr)), landing(std::exchange(other.landing, nullptr)),
          summands(std::exchange(other.summands, nullptr)), local(other.local), remote(other.remote), mtu(other.mtu),
          gid_index(other.gid_index), embeddings_address(other.embeddings_address),
          embeddings_bytes(other.embeddings_bytes), summands_row_bytes(other.summands_row_bytes), landed(other.landed) {
    }

    ~rdma_rail_t() noexcept {
        if (!context) return;
        require(ibv_destroy_qp(queue_pair), "queue pair release");
        for (ibv_mr *registration : {embeddings, landing, summands})
            require(ibv_dereg_mr(registration), "deregistration");
        require(ibv_destroy_cq(completions), "completion queue release");
        require(ibv_dealloc_pd(protection), "protection domain release");
        require(ibv_close_device(context), "device release");
    }

    /** Opens @p nic , registers @p embeddings , @p landing @b [2,rows,hidden] and the host-pinned
     *  @p summands @b [hosts,4], and readies a queue pair whose @ref local endpoint the
     *  counterpart must receive. */
    [[nodiscard]] static rdma_rail_t make(char const *nic, nk::tensor_view<bf16_t, 2> embeddings,
                                          nk::tensor_span<bf16_t, 3> landing, nk::tensor_span<f32_t, 2> summands,
                                          std::uint32_t packet_sequence) noexcept {
        constexpr std::uint8_t port = 1;
        rdma_rail_t rail;
        int devices = 0;
        ibv_device **list = ibv_get_device_list(&devices);
        for (int index = 0; index != devices && !rail.context; ++index)
            if (!std::strcmp(ibv_get_device_name(list[index]), nic)) rail.context = ibv_open_device(list[index]);
        ibv_free_device_list(list);
        require(rail.context ? 0 : ENODEV, nic);
        ibv_port_attr port_attributes {};
        require(ibv_query_port(rail.context, port, &port_attributes), "port query");
        rail.mtu = port_attributes.active_mtu;
        rail.local.gid = roce_v2_gid_(rail.context, port, port_attributes.gid_tbl_len, rail.gid_index);
        require((rail.protection = ibv_alloc_pd(rail.context)) ? 0 : errno, "protection domain");
        require((rail.completions = ibv_create_cq(rail.context, 64, nullptr, nullptr, 0)) ? 0 : errno,
                "completion queue");
        ibv_qp_init_attr initial {};
        initial.send_cq = initial.recv_cq = rail.completions;
        initial.qp_type = IBV_QPT_RC;
        initial.cap.max_send_wr = initial.cap.max_recv_wr = 16;
        initial.cap.max_send_sge = 1;
        require((rail.queue_pair = ibv_create_qp(rail.protection, &initial)) ? 0 : errno, "queue pair");

        rail.embeddings_address = reinterpret_cast<std::uint64_t>(embeddings.data());
        rail.embeddings_bytes = embeddings.numel() * sizeof(bf16_t);
        rail.summands_row_bytes = summands.extent(1) * sizeof(f32_t);
        rail.embeddings = register_device_(rail.protection, embeddings.data(), rail.embeddings_bytes,
                                           IBV_ACCESS_LOCAL_WRITE);
        rail.landing = register_device_(rail.protection, landing.data(), landing.numel() * sizeof(bf16_t),
                                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        require((rail.summands = ibv_reg_mr(rail.protection, summands.data(), summands.numel() * sizeof(f32_t),
                                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))
                    ? 0
                    : errno,
                "summands registration");
        rail.local.queue_pair = rail.queue_pair->qp_num;
        rail.local.packet_sequence = packet_sequence & 0xFFFFFFu;
        rail.local.landing_address = reinterpret_cast<std::uint64_t>(landing.data());
        rail.local.landing_key = rail.landing->rkey;
        rail.local.summands_address = reinterpret_cast<std::uint64_t>(summands.data());
        rail.local.summands_key = rail.summands->rkey;

        ibv_qp_attr attributes {};
        attributes.qp_state = IBV_QPS_INIT;
        attributes.port_num = port;
        attributes.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        require(ibv_modify_qp(rail.queue_pair, &attributes,
                              IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
                "queue pair initialization");
        for (std::uint32_t receive = 0; receive != initial.cap.max_recv_wr; ++receive) rail.post_receive_();
        return rail;
    }

    /** Joins the queue pair to the @p counterpart endpoint, after which both sides may write. */
    void connect(rdma_endpoint_t const &counterpart) noexcept {
        constexpr std::uint8_t port = 1;
        remote = counterpart;
        ibv_qp_attr attributes {};
        attributes.qp_state = IBV_QPS_RTR;
        attributes.path_mtu = mtu;
        attributes.dest_qp_num = remote.queue_pair;
        attributes.rq_psn = remote.packet_sequence;
        attributes.max_dest_rd_atomic = 1;
        attributes.min_rnr_timer = 12;
        attributes.ah_attr.is_global = 1;
        attributes.ah_attr.grh.dgid = remote.gid;
        attributes.ah_attr.grh.sgid_index = std::uint8_t(gid_index);
        attributes.ah_attr.grh.hop_limit = 64;
        attributes.ah_attr.port_num = port;
        require(ibv_modify_qp(queue_pair, &attributes,
                              IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                  IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
                "queue pair readiness to receive");
        attributes = {};
        attributes.qp_state = IBV_QPS_RTS;
        attributes.timeout = 14;
        attributes.retry_cnt = 7;
        attributes.rnr_retry = 7;
        attributes.sq_psn = local.packet_sequence;
        attributes.max_rd_atomic = 1;
        require(ibv_modify_qp(queue_pair, &attributes,
                              IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                  IBV_QP_MAX_QP_RD_ATOMIC),
                "queue pair readiness to send");
    }

    /** Writes the whole of this GPU's embeddings into the counterpart's @c landing[parity]. */
    void post_embeddings(std::size_t parity) noexcept {
        post_write_(embeddings_address, embeddings_bytes, embeddings->lkey,
                    remote.landing_address + parity * embeddings_bytes, remote.landing_key);
    }

    /** Writes @c summands[host] into the counterpart's @c summands[host]. */
    void post_summands(std::size_t host) noexcept {
        std::uint64_t const offset = host * summands_row_bytes;
        post_write_(reinterpret_cast<std::uint64_t>(summands->addr) + offset, summands_row_bytes, summands->lkey,
                    remote.summands_address + offset, remote.summands_key);
    }

    /** Returns once the counterpart's next write has landed here, in the order the counterpart
     *  posted them. */
    void await_landed() noexcept {
        while (!landed) {
            ibv_wc entries[16];
            int const polled = ibv_poll_cq(completions, 16, entries);
            require(polled < 0 ? -polled : 0, "completion polling");
            for (int index = 0; index != polled; ++index) {
                require(entries[index].status == IBV_WC_SUCCESS ? 0 : EIO, ibv_wc_status_str(entries[index].status));
                if (entries[index].opcode == IBV_WC_RECV_RDMA_WITH_IMM) ++landed, post_receive_();
            }
        }
        --landed;
    }

  private:

    /** The port's routable RoCE GID with an IPv4-mapped address, the kind routed rails use. */
    [[nodiscard]] static ibv_gid roce_v2_gid_(ibv_context *context, std::uint8_t port, int entries,
                                              int &found) noexcept {
        for (int index = 0; index != entries; ++index) {
            ibv_gid_entry entry {};
            if (ibv_query_gid_ex(context, port, std::uint32_t(index), &entry, 0) != 0) continue;
            std::uint8_t const *raw = entry.gid.raw;
            bool const ipv4 = std::all_of(raw, raw + 10, [](std::uint8_t byte) { return byte == 0; }) &&
                              raw[10] == 0xFF && raw[11] == 0xFF;
            if (entry.gid_type != IBV_GID_TYPE_ROCE_V2 || !ipv4) continue;
            found = index;
            return entry.gid;
        }
        require(ENOENT, "RoCE v2 IPv4 GID lookup");
        return {};
    }

    /** Registers device memory as a dma-buf, so no peer-memory kernel module is needed. */
    [[nodiscard]] static ibv_mr *register_device_(ibv_pd *protection, void const *pointer, std::size_t bytes,
                                                  int access) noexcept {
        // The export takes whole host pages, so the buffer goes out with the pages around it.
        std::size_t const page = std::size_t(sysconf(_SC_PAGESIZE));
        CUdeviceptr const first = reinterpret_cast<CUdeviceptr>(pointer);
        CUdeviceptr const base = first / page * page;
        std::size_t const span = divide_round_up(first + bytes - base, page) * page;
        int descriptor = -1;
        require(cuMemGetHandleForAddressRange(&descriptor, base, span, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
                "dma-buf export");
        ibv_mr *registration = ibv_reg_dmabuf_mr(protection, reinterpret_cast<CUdeviceptr>(pointer) - base, bytes,
                                                 reinterpret_cast<std::uint64_t>(pointer), descriptor, access);
        int const failure = registration ? 0 : errno;
        // the registration holds the buffer
        require(close(descriptor) == 0 ? 0 : errno, "dma-buf release");
        require(failure, "dma-buf registration");
        return registration;
    }

    void post_receive_() noexcept {
        ibv_recv_wr request {}, *refused = nullptr;
        require(ibv_post_recv(queue_pair, &request, &refused), "receive posting");
    }

    void post_write_(std::uint64_t source, std::size_t bytes, std::uint32_t source_key, std::uint64_t target,
                     std::uint32_t target_key) noexcept {
        ibv_sge piece {source, std::uint32_t(bytes), source_key};
        ibv_send_wr request {}, *refused = nullptr;
        request.sg_list = &piece, request.num_sge = 1;
        request.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        request.send_flags = IBV_SEND_SIGNALED;
        request.imm_data = htonl(std::uint32_t(bytes));
        request.wr.rdma.remote_addr = target, request.wr.rdma.rkey = target_key;
        require(ibv_post_send(queue_pair, &request, &refused), "write posting");
    }
};

/** The block-wide sum, every thread receiving it, in a fixed tree order. */
[[nodiscard]] __device__ inline f32_t block_sum(f32_t value) noexcept {
    __shared__ f32_t warp_sums[block_threads_k / 32];
    for (std::uint32_t offset = 16; offset != 0; offset /= 2) value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    __syncthreads();
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = value;
    __syncthreads();
    value = threadIdx.x % 32 < blockDim.x / 32 ? warp_sums[threadIdx.x % 32] : 0.0f;
    for (std::uint32_t offset = 16; offset != 0; offset /= 2) value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    return value;
}

static __global__ void transpose_cuda_(nk::tensor_view<bf16_t, 2> source, nk::tensor_span<bf16_t, 2> destination) {
    __shared__ bf16_t tile[32][33];
    std::size_t const candidate_base = blockIdx.x * 32, query_base = blockIdx.y * 32;
    for (std::uint32_t step = threadIdx.y; step < 32; step += blockDim.y)
        if (query_base + step < source.extent(0) && candidate_base + threadIdx.x < source.extent(1))
            tile[step][threadIdx.x] = source[query_base + step, candidate_base + threadIdx.x];
    __syncthreads();
    for (std::uint32_t step = threadIdx.y; step < 32; step += blockDim.y)
        if (candidate_base + step < destination.extent(0) && query_base + threadIdx.x < destination.extent(1))
            destination[candidate_base + step, query_base + threadIdx.x] = tile[threadIdx.x][step];
}

static __global__ void prefix_inverse_norms_cuda_(nk::tensor_view<bf16_t, 2> embeddings,
                                                  nk::tensor_span<f32_t, 2> norms, matryoshka_t ladder) {
    std::size_t const row = blockIdx.x;
    f32_t running = 0.0f;
    for (std::uint32_t band = 0; band != ladder.count; ++band) {
        f32_t partial = 0.0f;
        for (std::uint32_t channel = ladder.begin(band) + threadIdx.x; channel < ladder.ends[band];
             channel += blockDim.x) {
            f32_t const value = __bfloat162float(embeddings[row, channel]);
            partial += value * value;
        }
        running += block_sum(partial);
        if (threadIdx.x == 0) norms[band, row] = running > 0.0f ? 1.0f / std::sqrt(running) : 0.0f;
    }
}

inline constexpr std::uint32_t forward_side_k = 128; // tile rows and columns, the MMA's M and N
inline constexpr std::uint32_t forward_depth_k = 64; // channels per stage, one 128-byte swizzle row
inline constexpr std::uint32_t forward_ring_k = 4;   // stages in flight, one block per SM
inline constexpr std::uint32_t forward_chunk_k = 32; // columns one `tcgen05.ld` moves per lane
inline constexpr std::uint32_t forward_stage_bytes_k = forward_side_k * forward_depth_k * sizeof(bf16_t);
inline constexpr std::uint32_t forward_shared_bytes_k = 2 * forward_ring_k * forward_stage_bytes_k +
                                                        1024;     // + alignment
inline constexpr std::uint32_t forward_epilogue_warps_k = 8;      // two per TMEM lane quarter
inline constexpr std::uint32_t forward_first_epilogue_warp_k = 2; // after the TMA and MMA warps
inline constexpr std::uint32_t forward_threads_k = (forward_first_epilogue_warp_k + forward_epilogue_warps_k) * 32;
inline constexpr std::uint32_t forward_halves_chunks_k = forward_side_k / forward_chunk_k / 2; // chunks per column half
inline constexpr std::uint32_t forward_tmem_columns_k = 4 * forward_side_k; // raw dots, suffix and two segments

/** The instruction descriptor of a @b [128,128] BF16 MMA into Float32, both operands K-major. */
inline constexpr std::uint32_t forward_instruction_k = (1u << 4)                        // Float32 accumulator
                                                       | (1u << 7) | (1u << 10)         // BF16 A and B
                                                       | ((forward_side_k >> 3) << 17)  // N
                                                       | ((forward_side_k >> 4) << 24); // M
inline constexpr std::uint32_t forward_negated_instruction_k = forward_instruction_k | (1u << 13); // A negated

/** What every forward tile of one candidate block shares. */
struct forward_tile_t {
    nk::tensor_view<f32_t, 2> query_inverse_norms, candidate_inverse_norms;
    nk::tensor_span<bf16_t, 3> similarity_gradients;
    nk::tensor_span<f32_t, 3> row_partials;
    std::size_t query_base, candidate_base;
    f32_t beta, bias, weight;
};

[[nodiscard]] __device__ inline std::uint32_t shared_address_(void const *pointer) noexcept {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ inline void mbarrier_init_(std::uint64_t &barrier, std::uint32_t arrivals) noexcept {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(shared_address_(&barrier)), "r"(arrivals) : "memory");
}

__device__ inline void mbarrier_arrive_(std::uint64_t &barrier) noexcept {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(shared_address_(&barrier)) : "memory");
}

__device__ inline void mbarrier_wait_(std::uint64_t &barrier, std::uint32_t parity) noexcept {
    std::uint32_t done = 0;
    do
        asm volatile("{\n"                                                            //
                     ".reg .pred complete;\n"                                         //
                     "mbarrier.try_wait.parity.shared::cta.b64 complete, [%1], %2;\n" //
                     "selp.u32 %0, 1, 0, complete;\n"                                 //
                     "}\n"                                                            //
                     : "=r"(done)
                     : "r"(shared_address_(&barrier)), "r"(parity)
                     : "memory");
    while (!done);
}

/** Arms @p barrier to complete once @p bytes of TMA copies land. */
__device__ inline void mbarrier_expect_(std::uint64_t &barrier, std::uint32_t bytes) noexcept {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(shared_address_(&barrier)), "r"(bytes)
                 : "memory");
}

/** Copies the @b [128,64] box at @p row and channel @p depth of @p map into @p destination ,
 *  128-byte swizzled. */
__device__ inline void tma_load_(void *destination, CUtensorMap const &map, std::uint32_t depth, std::uint32_t row,
                                 std::uint64_t &barrier) noexcept {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes " //
                 "[%0], [%1, {%2, %3}], [%4];"                                                       //
                 ::"r"(shared_address_(destination)),
                 "l"(&map), "r"(depth), "r"(row), "r"(shared_address_(&barrier))
                 : "memory");
}

/** Copies the @b [128,64] box at @p row and channel @p depth of matrix @p matrix of @p map into
 *  @p destination , 128-byte swizzled. */
__device__ inline void tma_load_(void *destination, CUtensorMap const &map, std::uint32_t depth, std::uint32_t row,
                                 std::uint32_t matrix, std::uint64_t &barrier) noexcept {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes " //
                 "[%0], [%1, {%2, %3, %4}], [%5];"                                                   //
                 ::"r"(shared_address_(destination)),
                 "l"(&map), "r"(depth), "r"(row), "r"(matrix), "r"(shared_address_(&barrier))
                 : "memory");
}

/** The shared-memory descriptor of a K-major, 128-byte swizzled @b [128,64] stage, @p step k16
 *  steps in. */
[[nodiscard]] __device__ inline std::uint64_t stage_descriptor_(void const *stage, std::uint32_t step) noexcept {
    std::uint64_t const start = (shared_address_(stage) + step * 16 * sizeof(bf16_t)) >> 4;
    return start                    // start address
           | (1ull << 16)           // leading offset, unused when swizzled
           | ((1024ull >> 4) << 32) // stride between 8-row swizzle atoms
           | (1ull << 46)           // the Blackwell descriptor version
           | (2ull << 61);          // 128-byte swizzle
}

__device__ inline void tcgen05_mma_(std::uint32_t accumulator, std::uint64_t a, std::uint64_t b,
                                    std::uint32_t instruction, std::uint32_t accumulate) noexcept {
    asm volatile("{\n"                                                                //
                 ".reg .pred accumulate;\n"                                           //
                 "setp.ne.b32 accumulate, %4, 0;\n"                                   //
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accumulate;\n" //
                 "}\n"                                                                //
                 ::"r"(accumulator),
                 "l"(a), "l"(b), "r"(instruction), "r"(accumulate)
                 : "memory");
}

/** Arrives on @p barrier once every MMA this thread issued so far has completed. */
__device__ inline void tcgen05_commit_(std::uint64_t &barrier) noexcept {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(shared_address_(&barrier))
        : "memory");
}

/** Loads 32 consecutive TMEM columns of this lane's row. */
__device__ inline void tmem_load_(std::uint32_t address, f32_t (&values)[forward_chunk_k]) noexcept {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "                                           //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "           //
                 "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, " //
                 "[%32];"                                                                            //
                 : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
                   "=f"(values[6]), "=f"(values[7]), "=f"(values[8]), "=f"(values[9]), "=f"(values[10]),
                   "=f"(values[11]), "=f"(values[12]), "=f"(values[13]), "=f"(values[14]), "=f"(values[15]),
                   "=f"(values[16]), "=f"(values[17]), "=f"(values[18]), "=f"(values[19]), "=f"(values[20]),
                   "=f"(values[21]), "=f"(values[22]), "=f"(values[23]), "=f"(values[24]), "=f"(values[25]),
                   "=f"(values[26]), "=f"(values[27]), "=f"(values[28]), "=f"(values[29]), "=f"(values[30]),
                   "=f"(values[31])
                 : "r"(address)
                 : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

/** Stores 32 consecutive TMEM columns of this lane's row. */
__device__ inline void tmem_store_(std::uint32_t address, f32_t const (&values)[forward_chunk_k]) noexcept {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "                                    //
                 "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "         //
                 "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};" //
                 ::"r"(address),
                 "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]), "f"(values[4]), "f"(values[5]),
                 "f"(values[6]), "f"(values[7]), "f"(values[8]), "f"(values[9]), "f"(values[10]), "f"(values[11]),
                 "f"(values[12]), "f"(values[13]), "f"(values[14]), "f"(values[15]), "f"(values[16]), "f"(values[17]),
                 "f"(values[18]), "f"(values[19]), "f"(values[20]), "f"(values[21]), "f"(values[22]), "f"(values[23]),
                 "f"(values[24]), "f"(values[25]), "f"(values[26]), "f"(values[27]), "f"(values[28]), "f"(values[29]),
                 "f"(values[30]), "f"(values[31])
                 : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

/** @c log(1+x) for @p x in @b [0,1], as @p x times a degree-8 Chebyshev least-squares fit of
 *  @c log1p(x)/x on that interval, within 3.2e-7 of it relative in Float32. */
[[nodiscard]] __device__ inline f32_t log1p_unit_(f32_t x) noexcept {
    f32_t fit = 5.185916197e-03f;
    for (f32_t const coefficient : {-2.920989730e-02f, 7.753973263e-02f, -1.358388136e-01f, 1.905592322e-01f,
                                    -2.482563861e-01f, 3.331600901e-01f, -4.999925577e-01f, 9.999999208e-01f})
        fit = std::fma(fit, x, coefficient);
    return fit * x;
}

/** This lane's row summands of a forward tile. */
struct forward_row_t {
    f32_t loss = 0, bias_grad = 0, log_beta_grad = 0;
};

/** Scores @p band for this lane's row over half @p half of the tile's columns: folds the segment in
 *  TMEM at @p segment into the raw dots at @p raw_dots , adds the band's gradient to the suffix at
 *  @p suffix , and writes that suffix as the band's BF16 gradient, with the row's coefficient
 *  partial. */
__device__ inline void score_forward_row_(forward_tile_t const &tile, std::uint32_t raw_dots, std::uint32_t suffix,
                                          std::uint32_t segment, std::uint32_t band, std::uint32_t half,
                                          forward_row_t &summands) noexcept {
    std::uint32_t const local_row = threadIdx.x % 128;
    std::size_t const row = std::size_t {blockIdx.y} * forward_side_k + local_row;
    f32_t const row_inverse_norm = tile.query_inverse_norms[band, row];
    f32_t coefficient = 0;
    for (std::uint32_t chunk = half * forward_halves_chunks_k; chunk != (half + 1) * forward_halves_chunks_k; ++chunk) {
        f32_t similarities[forward_chunk_k], swept[forward_chunk_k], running[forward_chunk_k];
        tmem_load_(raw_dots + chunk * forward_chunk_k, similarities);
        tmem_load_(segment + chunk * forward_chunk_k, swept);
#pragma unroll
        for (std::uint32_t cell = 0; cell != forward_chunk_k; ++cell) similarities[cell] += swept[cell];
        tmem_store_(raw_dots + chunk * forward_chunk_k, similarities);
        tmem_load_(suffix + chunk * forward_chunk_k, running);
        std::size_t const first_column = std::size_t {blockIdx.x} * forward_side_k + chunk * forward_chunk_k;
        std::uint32_t packed[forward_chunk_k / 2];
#pragma unroll
        for (std::uint32_t cell = 0; cell != forward_chunk_k; ++cell) {
            std::size_t const column = first_column + cell;
            bool const own_copy = tile.query_base + row == tile.candidate_base + column;
            // The inverse of the norms' product, clamped where they read as a zero cosine.
            f32_t const inverse_denominator = std::fmin(row_inverse_norm * tile.candidate_inverse_norms[band, column],
                                                        inverse_cosine_floor_k);
            f32_t const cosine = inverse_denominator < inverse_cosine_floor_k ? similarities[cell] * inverse_denominator
                                                                              : 0.0f;
            // A query's cosine with its own copy is one, whatever its raw dot rounds to.
            f32_t const logit = tile.beta * (own_copy ? 1.0f : cosine);
            f32_t const target = own_copy ? 1.0f : -1.0f;
            f32_t const signed_logit = target * (logit + tile.bias);
            f32_t const decay = __expf(-std::fabs(signed_logit));
            f32_t const complement = __fdividef(signed_logit >= 0.0f ? decay : 1.0f, 1.0f + decay); // 1 - sigmoid
            f32_t const cell_grad = -tile.weight * target * complement;
            summands.loss += tile.weight * (std::fmax(-signed_logit, 0.0f) + log1p_unit_(decay));
            summands.bias_grad += cell_grad;
            summands.log_beta_grad += cell_grad * logit;
            // The mirror cell (j, i) moves v_i the same way, and the own copy moves nothing.
            f32_t const raw_grad = own_copy || inverse_denominator >= inverse_cosine_floor_k
                                       ? 0.0f
                                       : 2.0f * tile.beta * cell_grad * inverse_denominator;
            coefficient += raw_grad * similarities[cell];
            running[cell] += raw_grad;
            if (cell % 2)
                packed[cell / 2] = __bfloat16_as_ushort(__float2bfloat16(running[cell - 1])) |
                                   std::uint32_t {__bfloat16_as_ushort(__float2bfloat16(running[cell]))} << 16;
        }
        tmem_store_(suffix + chunk * forward_chunk_k, running);
        for (std::uint32_t quarter = 0; quarter != 4; ++quarter)
            asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                             &tile.similarity_gradients[band, row, first_column + quarter * 8]),
                         "r"(packed[quarter * 4]), "r"(packed[quarter * 4 + 1]), "r"(packed[quarter * 4 + 2]),
                         "r"(packed[quarter * 4 + 3])
                         : "memory");
    }
    tile.row_partials[3 + band, blockIdx.x * 2 + half, row] = coefficient;
}

/** The stages one segment of a forward tile sweeps, and the MMA that sweeps them. */
struct forward_segment_t {
    std::uint32_t begin = 0, end = 0, instruction = 0;
};

/** Segment @p segment of the sweep: first every channel, then each band's own channels taken back
 *  off, widest band first, so after folding segment @c s the raw dots are band @c K-1-s 's. */
[[nodiscard]] __device__ inline forward_segment_t forward_segment_(matryoshka_t const &ladder,
                                                                   std::uint32_t segment) noexcept {
    std::uint32_t const widest = ladder.count - 1;
    if (segment == 0) return {0, ladder.ends[widest] / forward_depth_k, forward_instruction_k};
    return {ladder.ends[widest - segment] / forward_depth_k, ladder.ends[widest - segment + 1] / forward_depth_k,
            forward_negated_instruction_k};
}

/**
 *  @brief One @b [128,128] tile of queries against candidates, scored at every Matryoshka prefix
 *      with @c tcgen05 MMAs fed by TMA, with no Float32 @b [K,R,R] stash.
 *
 *  A forward sweep over the depth accumulates every cell's full raw dot in TMEM. A reverse sweep
 *  with negated rows then takes the suffix back off one segment between band edges at a time, each
 *  into the other of two TMEM segment buffers, so the MMA warp sweeps the next segment while eight
 *  epilogue warps fold the last one into the raw dots and score its band. Each band's
 *  suffix-summed gradient stays in TMEM beside the dots. Row summands land in the tile's
 *  @b [3+K,R/64,R] partials, which @ref fold_partials_cuda_ adds in order.
 */
static __global__ void __launch_bounds__(forward_threads_k, 1)
    forward_tile_cuda_(CUtensorMap const __grid_constant__ rows_map, CUtensorMap const __grid_constant__ columns_map,
                       forward_tile_t const tile, matryoshka_t ladder) {
    extern __shared__ byte_t dynamic_shared[];
    __shared__ alignas(8) std::uint64_t filled[forward_ring_k], emptied[forward_ring_k], swept[2], drained[2];
    __shared__ std::uint32_t tmem_base;
    byte_t *const ring = dynamic_shared + (1024 - shared_address_(dynamic_shared) % 1024) % 1024;
    std::uint32_t const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    auto const rows_stage = [&](std::uint32_t slot) { return ring + slot * forward_stage_bytes_k; };
    auto const columns_stage = [&](std::uint32_t slot) {
        return ring + (forward_ring_k + slot) * forward_stage_bytes_k;
    };

    if (threadIdx.x == 0) {
        for (std::uint32_t slot = 0; slot != forward_ring_k; ++slot)
            mbarrier_init_(filled[slot], 1), mbarrier_init_(emptied[slot], 1);
        for (std::uint32_t buffer = 0; buffer != 2; ++buffer)
            mbarrier_init_(swept[buffer], 1), mbarrier_init_(drained[buffer], forward_epilogue_warps_k);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == forward_first_epilogue_warp_k) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(shared_address_(&tmem_base)),
            "r"(forward_tmem_columns_k)
            : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    // The raw dots, the gradient suffix and the two segment buffers take 128 TMEM columns each.
    std::uint32_t const raw_dots = tmem_base, suffix = raw_dots + forward_side_k,
                        segments = raw_dots + 2 * forward_side_k;
    if (warp == 0) {
        if (lane != 0) return;
        std::uint32_t position = 0;
        for (std::uint32_t segment = 0; segment != ladder.count; ++segment) {
            forward_segment_t const sweep = forward_segment_(ladder, segment);
            for (std::uint32_t stage = sweep.end; stage-- != sweep.begin; ++position) {
                std::uint32_t const slot = position % forward_ring_k;
                if (position >= forward_ring_k) mbarrier_wait_(emptied[slot], (position / forward_ring_k - 1) % 2);
                mbarrier_expect_(filled[slot], 2 * forward_stage_bytes_k);
                tma_load_(rows_stage(slot), rows_map, stage * forward_depth_k, blockIdx.y * forward_side_k,
                          filled[slot]);
                tma_load_(columns_stage(slot), columns_map, stage * forward_depth_k, blockIdx.x * forward_side_k,
                          filled[slot]);
            }
        }
        return;
    }
    if (warp == 1) {
        if (lane != 0) return;
        std::uint32_t position = 0;
        for (std::uint32_t segment = 0; segment != ladder.count; ++segment) {
            // A segment buffer refills only once every epilogue warp folded what it held.
            if (segment >= 2) mbarrier_wait_(drained[segment % 2], (segment / 2 - 1) % 2);
            forward_segment_t const sweep = forward_segment_(ladder, segment);
            std::uint32_t const accumulator = (segments + segment % 2 * forward_side_k) & 0xFFFFu;
            for (std::uint32_t stage = sweep.end; stage-- != sweep.begin; ++position) {
                std::uint32_t const slot = position % forward_ring_k;
                mbarrier_wait_(filled[slot], position / forward_ring_k % 2);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                for (std::uint32_t step = 0; step != forward_depth_k / 16; ++step)
                    tcgen05_mma_(accumulator, stage_descriptor_(rows_stage(slot), step),
                                 stage_descriptor_(columns_stage(slot), step), sweep.instruction,
                                 stage + 1 != sweep.end || step != 0);
                tcgen05_commit_(emptied[slot]);
            }
            tcgen05_commit_(swept[segment % 2]);
        }
        return;
    }

    // Two epilogue warps share each 32-lane TMEM quarter, one per half of the tile's columns.
    std::uint32_t const quarter = warp % 4, half = (warp - forward_first_epilogue_warp_k) / 4;
    std::uint32_t const lanes = quarter * 32 << 16;
    f32_t const zeros[forward_chunk_k] = {};
    for (std::uint32_t chunk = half * forward_halves_chunks_k; chunk != (half + 1) * forward_halves_chunks_k; ++chunk)
        tmem_store_(raw_dots + lanes + chunk * forward_chunk_k, zeros),
            tmem_store_(suffix + lanes + chunk * forward_chunk_k, zeros);
    forward_row_t summands;
    for (std::uint32_t segment = 0; segment != ladder.count; ++segment) {
        mbarrier_wait_(swept[segment % 2], segment / 2 % 2);
        __syncwarp();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        score_forward_row_(tile, raw_dots + lanes, suffix + lanes, segments + segment % 2 * forward_side_k + lanes,
                           ladder.count - 1 - segment, half, summands);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncwarp();
        if (lane == 0) mbarrier_arrive_(drained[segment % 2]);
    }

    std::size_t const row = std::size_t {blockIdx.y} * forward_side_k + quarter * 32 + lane;
    tile.row_partials[0, blockIdx.x * 2 + half, row] = summands.loss;
    tile.row_partials[1, blockIdx.x * 2 + half, row] = summands.bias_grad;
    tile.row_partials[2, blockIdx.x * 2 + half, row] = summands.log_beta_grad;
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("bar.sync 1, %0;" ::"n"(forward_epilogue_warps_k * 32) : "memory");
    if (warp == forward_first_epilogue_warp_k)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_base),
                     "r"(forward_tmem_columns_k)
                     : "memory");
}

inline constexpr std::uint32_t backward_channels_k = 64; // gradient channels per tile, each in one band
inline constexpr std::uint32_t backward_ring_k = 4;
inline constexpr std::uint32_t backward_threads_k = 160; // four epilogue warps, the first issuing MMAs, and a TMA warp
inline constexpr std::uint32_t backward_rows_bytes_k = forward_side_k * forward_depth_k * sizeof(bf16_t);
inline constexpr std::uint32_t backward_columns_bytes_k = backward_channels_k * forward_depth_k * sizeof(bf16_t);
inline constexpr std::uint32_t backward_shared_bytes_k =
    backward_ring_k * (backward_rows_bytes_k + backward_columns_bytes_k) + 1024;             // + alignment
inline constexpr std::uint32_t backward_instruction_k = (1u << 4)                            // Float32 accumulator
                                                        | (1u << 7) | (1u << 10)             // BF16 A and B
                                                        | ((backward_channels_k >> 3) << 17) // N
                                                        | ((forward_side_k >> 4) << 24);     // M

/**
 *  @brief Adds one @b [128,64] tile of the candidates' pull on this rank's queries into
 *      @p query_gradient : the band's BF16 similarity gradients times the candidates' channels in
 *      that band, over every candidate.
 *
 *  A TMA warp streams the gradients' rows and the transposed candidates' channels through a ring of
 *  @c backward_ring_k stages, one thread issues @c tcgen05 MMAs into tensor memory, and once they
 *  finish each of four warps adds its 32 rows into the Float32 query gradient.
 */
static __global__ void __launch_bounds__(backward_threads_k)
    backward_tile_cuda_(CUtensorMap const __grid_constant__ gradient_map,
                        CUtensorMap const __grid_constant__ candidates_map, std::uint32_t stages,
                        nk::tensor_span<f32_t, 2> query_gradient, matryoshka_t ladder) {
    extern __shared__ byte_t dynamic_shared[];
    __shared__ alignas(8) std::uint64_t filled[backward_ring_k], emptied[backward_ring_k], finished;
    __shared__ std::uint32_t tmem_base;
    byte_t *const ring = dynamic_shared + (1024 - shared_address_(dynamic_shared) % 1024) % 1024;
    std::uint32_t const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    std::uint32_t const channel = blockIdx.x * backward_channels_k, row = blockIdx.y * forward_side_k;
    std::uint32_t band = 0;
    while (ladder.ends[band] <= channel) ++band;
    auto const rows_stage = [&](std::uint32_t slot) { return ring + slot * backward_rows_bytes_k; };
    auto const columns_stage = [&](std::uint32_t slot) {
        return ring + backward_ring_k * backward_rows_bytes_k + slot * backward_columns_bytes_k;
    };

    if (threadIdx.x == 0) {
        for (std::uint32_t slot = 0; slot != backward_ring_k; ++slot)
            mbarrier_init_(filled[slot], 1), mbarrier_init_(emptied[slot], 1);
        mbarrier_init_(finished, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(shared_address_(&tmem_base)),
            "r"(backward_channels_k)
            : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (warp == 4) {
        if (lane != 0) return;
        for (std::uint32_t position = 0; position != stages; ++position) {
            std::uint32_t const slot = position % backward_ring_k;
            if (position >= backward_ring_k) mbarrier_wait_(emptied[slot], (position / backward_ring_k - 1) % 2);
            mbarrier_expect_(filled[slot], backward_rows_bytes_k + backward_columns_bytes_k);
            tma_load_(rows_stage(slot), gradient_map, position * forward_depth_k, row, band, filled[slot]);
            tma_load_(columns_stage(slot), candidates_map, position * forward_depth_k, channel, filled[slot]);
        }
        return;
    }

    if (threadIdx.x == 0) {
        for (std::uint32_t position = 0; position != stages; ++position) {
            std::uint32_t const slot = position % backward_ring_k;
            mbarrier_wait_(filled[slot], position / backward_ring_k % 2);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (std::uint32_t step = 0; step != forward_depth_k / 16; ++step)
                tcgen05_mma_(tmem_base & 0xFFFFu, stage_descriptor_(rows_stage(slot), step),
                             stage_descriptor_(columns_stage(slot), step), backward_instruction_k, position | step);
            tcgen05_commit_(emptied[slot]);
        }
        tcgen05_commit_(finished);
    }
    mbarrier_wait_(finished, 0);
    __syncwarp();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    for (std::uint32_t chunk = 0; chunk != backward_channels_k / forward_chunk_k; ++chunk) {
        f32_t gradient_tile[forward_chunk_k];
        tmem_load_(tmem_base + (warp * 32 << 16) + chunk * forward_chunk_k, gradient_tile);
#pragma unroll
        for (std::uint32_t cell = 0; cell != forward_chunk_k; ++cell)
            query_gradient[row + threadIdx.x, channel + chunk * forward_chunk_k + cell] += gradient_tile[cell];
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"(backward_channels_k)
                     : "memory");
}

/** Adds each row's column-tile @p row_partials , in tile order, to its @p summand_rows and
 *  @p self_coefficients . */
static __global__ void fold_partials_cuda_(nk::tensor_view<f32_t, 3> row_partials,
                                           nk::tensor_span<f32_t, 2> summand_rows,
                                           nk::tensor_span<f32_t, 2> self_coefficients) {
    std::size_t const row = blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (row >= row_partials.extent(2)) return;
    for (std::size_t quantity = 0; quantity != row_partials.extent(0); ++quantity) {
        f32_t total = 0.0f;
        for (std::size_t tile = 0; tile != row_partials.extent(1); ++tile) total += row_partials[quantity, tile, row];
        if (quantity < 3) summand_rows[quantity, row] += total;
        else self_coefficients[quantity - 3, row] += total;
    }
}

/** The cosine's self term folded into the query gradient, then its BF16 copy. */
static __global__ void finalize_cuda_(nk::tensor_view<bf16_t, 2> queries, nk::tensor_view<f32_t, 2> inverse_norms,
                                      nk::tensor_view<f32_t, 2> self_coefficients,
                                      nk::tensor_view<f32_t, 2> query_gradient, nk::tensor_span<bf16_t, 2> gradient_out,
                                      matryoshka_t ladder) {
    std::size_t const row = blockIdx.x;
    for (std::uint32_t channel = threadIdx.x; channel < queries.extent(1); channel += blockDim.x) {
        f32_t fold = 0.0f;
        for (std::uint32_t band = 0; band != ladder.count; ++band)
            if (channel < ladder.ends[band])
                fold += self_coefficients[band, row] * inverse_norms[band, row] * inverse_norms[band, row];
        gradient_out[row, channel] = __float2bfloat16(query_gradient[row, channel] -
                                                      fold * __bfloat162float(queries[row, channel]));
    }
}

static __global__ void sum_summands_cuda_(nk::tensor_view<f32_t, 2> summand_rows, nk::tensor_span<f32_t, 1> summands) {
    for (std::uint32_t which = 0; which != 3; ++which) {
        f32_t partial = 0.0f;
        for (std::size_t row = threadIdx.x; row < summand_rows.extent(1); row += blockDim.x)
            partial += summand_rows[which, row];
        f32_t const total = block_sum(partial);
        if (threadIdx.x == 0) summands[which] = total;
    }
}

/**
 *  @brief A rank's Matryoshka sigmoid contrastive loss, streamed one candidate block at a time and
 *      row-owned.
 *
 *  Every query is a row against all @c S·R candidates of the world, at every prefix of the ladder,
 *  scored by SigLIP's sigmoid BCE with weight @c 1/(S·R): its own copy is the positive, every other
 *  candidate a negative.
 *
 *  - @b Streamed: each candidate block's @b [K,R,R] tile is scored, lossed and folded into the
 *    rank's query gradient before the next is touched, so the scratch is one tile whatever the
 *    world.
 *  - @b Row-owned: cell @c (i,j) and its mirror @c (j,i) share the cosine, target and weight, so
 *    the owner of @c i differentiates both. No rank computes another's gradient, and the gradient
 *    exchanges nothing.
 *
 *  Call @ref begin , then @ref score once per candidate block in any order, then @ref finish ,
 *  which writes the rank's loss, bias-gradient and log-β-gradient summands, which add up across
 *  ranks.
 */
struct streamed_loss_t {
    matryoshka_t ladder;
    std::size_t rows = 0, hidden = 0;
    nk_capability_t capabilities = 0;
    nk::tensor<f32_t, nk::allocator<f32_t>, 2> query_inverse_norms, candidate_inverse_norms, query_gradient,
        summand_rows, self_coefficients;
    nk::tensor<f32_t, nk::allocator<f32_t>, 3> row_partials;
    nk::tensor<bf16_t, nk::allocator<bf16_t>, 3> similarity_gradients;
    nk::tensor<bf16_t, nk::allocator<bf16_t>, 2> transposed_candidates, staged_candidates;

    [[nodiscard]] static streamed_loss_t make(nk::device_t device, nk::stream_t const &stream, std::size_t rows,
                                              std::size_t hidden, matryoshka_t ladder) noexcept {
        streamed_loss_t loss;
        loss.ladder = ladder, loss.rows = rows, loss.hidden = hidden;
        loss.capabilities = take(device.capabilities_enabled(), "capability query");
        std::size_t const k = ladder.count;
        bool const edges_fit = std::all_of(ladder.ends, ladder.ends + k,
                                           [](std::uint32_t end) { return end % backward_channels_k == 0; });
        require(rows % forward_side_k == 0 && hidden % forward_depth_k == 0 && edges_fit && ladder.ends[k - 1] == hidden
                    ? 0
                    : EINVAL,
                "tile shape, which takes rows in multiples of 128, and channels and band edges in 64");
        require(cudaFuncSetAttribute(&forward_tile_cuda_, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     int(forward_shared_bytes_k)),
                "forward tile shared memory");
        require(cudaFuncSetAttribute(&backward_tile_cuda_, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     int(backward_shared_bytes_k)),
                "backward tile shared memory");
        loss.query_inverse_norms = make_tensor<f32_t, 2>(loss.capabilities, stream, {k, rows});
        loss.candidate_inverse_norms = make_tensor<f32_t, 2>(loss.capabilities, stream, {k, rows});
        loss.query_gradient = make_tensor<f32_t, 2>(loss.capabilities, stream, {rows, hidden});
        loss.summand_rows = make_tensor<f32_t, 2>(loss.capabilities, stream, {3, rows});
        loss.self_coefficients = make_tensor<f32_t, 2>(loss.capabilities, stream, {k, rows});
        loss.row_partials = make_tensor<f32_t, 3>(loss.capabilities, stream, {3 + k, 2 * rows / forward_side_k, rows});
        loss.similarity_gradients = make_tensor<bf16_t, 3>(loss.capabilities, stream, {k, rows, rows});
        loss.transposed_candidates = make_tensor<bf16_t, 2>(loss.capabilities, stream, {hidden, rows});
        loss.staged_candidates = make_tensor<bf16_t, 2>(loss.capabilities, stream, {rows, hidden});
        return loss;
    }

    /** Device bytes the loss holds, independent of the world's size. */
    [[nodiscard]] std::size_t bytes() const noexcept {
        return (query_inverse_norms.numel() + candidate_inverse_norms.numel() + query_gradient.numel() +
                summand_rows.numel() + self_coefficients.numel() + row_partials.numel()) *
                   sizeof(f32_t) +
               (similarity_gradients.numel() + transposed_candidates.numel() + staged_candidates.numel()) *
                   sizeof(bf16_t);
    }

    void begin(nk::tensor_view<bf16_t, 2> queries, nk::stream_t const &stream) noexcept {
        cudaStream_t const cuda = cuda_stream(stream);
        require(cudaMemsetAsync(query_gradient.data(), 0, query_gradient.numel() * sizeof(f32_t), cuda),
                "zeroing the query gradient");
        require(cudaMemsetAsync(summand_rows.data(), 0, summand_rows.numel() * sizeof(f32_t), cuda),
                "zeroing summands");
        require(cudaMemsetAsync(self_coefficients.data(), 0, self_coefficients.numel() * sizeof(f32_t), cuda),
                "zeroing coefficients");
        launch(&prefix_inverse_norms_cuda_, rows, block_threads_k, stream, queries, query_inverse_norms.span(), ladder);
    }

    /** Scores this rank's @p queries against @p candidates , whose first is @p candidate_base on
     *  the global axis; @p queries is a span only so TMA can map it, and is never written. */
    void score(nk::tensor_span<bf16_t, 2> queries, nk::tensor_view<bf16_t, 2> candidates, std::size_t query_base,
               std::size_t candidate_base, f32_t beta, f32_t bias, f32_t live, nk::stream_t const &stream) noexcept {
        // TMA reads local memory, so a peer's candidates cross NVLink once, into the staging copy.
        require(cudaMemcpyAsync(staged_candidates.data(), candidates.data(), staged_candidates.numel() * sizeof(bf16_t),
                                cudaMemcpyDefault, cuda_stream(stream)),
                "candidate staging");
        launch(&prefix_inverse_norms_cuda_, rows, block_threads_k, stream, staged_candidates.view(),
               candidate_inverse_norms.span(), ladder);
        dim3 const transpose_grid {static_cast<std::uint32_t>(hidden / 32), static_cast<std::uint32_t>(rows / 32)};
        launch(&transpose_cuda_, transpose_grid, dim3 {32, 8}, stream, staged_candidates.view(),
               transposed_candidates.span());
        std::uint32_t const tiles = static_cast<std::uint32_t>(rows / forward_side_k);
        forward_tile_t const tile {query_inverse_norms.view(),
                                   candidate_inverse_norms.view(),
                                   similarity_gradients.span(),
                                   row_partials.span(),
                                   query_base,
                                   candidate_base,
                                   beta,
                                   bias,
                                   1.0f / live};
        launch(&forward_tile_cuda_, dim3 {tiles, tiles}, forward_threads_k, forward_shared_bytes_k, stream,
               tensor_map_<2>(queries.data(), {hidden, rows}, {forward_depth_k, forward_side_k}),
               tensor_map_<2>(staged_candidates.data(), {hidden, rows}, {forward_depth_k, forward_side_k}), tile,
               ladder);
        launch(&fold_partials_cuda_, divide_round_up(rows, block_threads_k), block_threads_k, stream,
               row_partials.view(), summand_rows.span(), self_coefficients.span());
        launch(&backward_tile_cuda_, dim3 {static_cast<std::uint32_t>(hidden / backward_channels_k), tiles},
               backward_threads_k, backward_shared_bytes_k, stream,
               tensor_map_<3>(similarity_gradients.data(), {rows, rows, ladder.count},
                              {forward_depth_k, forward_side_k, 1}),
               tensor_map_<2>(transposed_candidates.data(), {rows, hidden}, {forward_depth_k, backward_channels_k}),
               static_cast<std::uint32_t>(rows / forward_depth_k), query_gradient.span(), ladder);
    }

    /** Writes the BF16 @p gradient_out and the three @p summands , a multimem payload or any device
     *  span. */
    void finish(nk::tensor_view<bf16_t, 2> queries, nk::tensor_span<bf16_t, 2> gradient_out,
                nk::tensor_span<f32_t, 1> summands, nk::stream_t const &stream) noexcept {
        launch(&finalize_cuda_, rows, block_threads_k, stream, queries, query_inverse_norms.view(),
               self_coefficients.view(), query_gradient.view(), gradient_out, ladder);
        launch(&sum_summands_cuda_, 1, block_threads_k, stream, summand_rows.view(), summands);
    }

  private:

    /** The TMA map of a row-major BF16 tensor at @p data with @p extents , innermost first, read in
     *  @p box boxes, 128-byte swizzled. */
    template <std::size_t rank_>
    [[nodiscard]] static CUtensorMap tensor_map_(bf16_t *data, std::array<std::uint64_t, rank_> extents,
                                                 std::array<std::uint32_t, rank_> box) noexcept {
        CUtensorMap map;
        std::uint64_t strides[rank_ - 1];
        std::uint32_t steps[rank_];
        for (std::size_t dimension = 0; dimension != rank_; ++dimension) steps[dimension] = 1;
        strides[0] = extents[0] * sizeof(bf16_t);
        for (std::size_t dimension = 1; dimension + 1 < rank_; ++dimension)
            strides[dimension] = strides[dimension - 1] * extents[dimension];
        require(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank_, data, extents.data(), strides,
                                       box.data(), steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                "tensor map");
        return map;
    }
};

} // namespace ashvardanian::scalingcontrasts
