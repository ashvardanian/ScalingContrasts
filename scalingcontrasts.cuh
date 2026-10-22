/**
 *  @file scalingcontrasts.cuh
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief Direct NVLink scale-up primitives for training, and a contrastive loss built on them.
 *
 *  A contrastive loss scores every vertex of a batch against every other one, so its batch, and
 *  with it the number of negatives each pair learns from, is bounded by how cheaply GPUs can see
 *  each other's vertices. This header holds the pieces that keep that cheap: a streamed, row-owned
 *  sigmoid loss whose memory does not grow with the world and whose backward exchanges nothing.
 */
#pragma once

#include <cerrno>  // `errno`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fprintf`
#include <cstdlib> // `std::exit`
#include <cstring> // `std::strerror`, `std::strcmp`

#include <arpa/inet.h> // `htonl`
#include <unistd.h>    // `close`, `sysconf`

#include <algorithm> // `std::max`, `std::clamp`
#include <span>      // `std::span`
#include <tuple>     // `std::tuple`, `std::apply`
#include <utility>   // `std::exchange`

#include <cuda.h>             // `cuMulticastCreate`, `cuMulticastBindMem`, `cuMemMap`, `cuMemSetAccess`
#include <cuda/atomic>        // `cuda::atomic_ref`
#include <cuda_bf16.h>        // `__nv_bfloat16`
#include <cuda_runtime.h>     // `cudaStream_t`, `cudaMemsetAsync`
#include <infiniband/verbs.h> // `ibv_reg_dmabuf_mr`, `ibv_post_send`, `ibv_poll_cq`

#include <numkong/numkong.hpp> // `nk::tensor`, `nk::allocator`, `nk::dots_pack`, `nk::dots_packed`

namespace ashvardanian::scalingcontrasts {

namespace nk = ashvardanian::numkong;

using bf16_t = __nv_bfloat16;
using byte_t = std::byte;
using f32_t = float;
using f64_t = double;

inline constexpr std::uint32_t block_threads_k = 256;
inline constexpr f32_t cosine_floor_k = 1.0e-2f; // smaller norm products read as zero cosines

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
static __global__ void multimem_barrier_kernel(multimem_counters_t counters) {
    std::uint32_t const round = *counters.own_round;
    multimem_arrive_(counters);
    multimem_await_(counters, round + counters.members);
    *counters.own_round = round + counters.members;
}

/** @c own[i]=Σ(members)payload[i] over @b [begin,end), one switch-reduced load per four scalars,
 *  between the round's entry and exit. */
static __global__ void multimem_reduce_scatter_kernel(multimem_counters_t counters, nk::tensor_view<f32_t, 1> multicast,
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
static __global__ void multimem_all_reduce_kernel(multimem_counters_t counters, nk::tensor_span<f32_t, 1> multicast,
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
            require(cudaFuncGetAttributes(&attributes, &multimem_barrier_kernel), "kernel loading");
            require(cudaFuncGetAttributes(&attributes, &multimem_reduce_scatter_kernel), "kernel loading");
            require(cudaFuncGetAttributes(&attributes, &multimem_all_reduce_kernel), "kernel loading");
        }
        return domain;
    }

    /** The scalars @p member contributes and receives, in its own replica. */
    [[nodiscard]] nk::tensor_span<f32_t, 1> payload(std::size_t member) const noexcept {
        return {pointer_<f32_t>(replicas_base + member * stride), payload_scalars};
    }

    /** Returns on @p member 's stream once every member has entered the same round. */
    void barrier(std::size_t member, nk::stream_t const &stream) noexcept {
        launch(&multimem_barrier_kernel, 1, 1, stream, counters_(member));
    }

    /** In place: @p member 's slice of @p shard_scalars , a multiple of four, at
     *  @c member·shard_scalars of its payload becomes the sum of every payload's same slice. */
    void reduce_scatter(std::size_t member, std::size_t shard_scalars, nk::stream_t const &stream) noexcept {
        std::size_t const begin = member * shard_scalars, end = begin + shard_scalars;
        launch(&multimem_reduce_scatter_kernel, blocks_(shard_scalars), block_threads_k, stream, counters_(member),
               multicast_payload_(), payload(member), begin, end);
    }

    /** In place: the first @p scalars , a multiple of four, of every payload become their sum, each
     *  member reducing and broadcasting its own share. */
    void all_reduce(std::size_t member, std::size_t scalars, nk::stream_t const &stream) noexcept {
        std::size_t const vectors = scalars / 4, share = divide_round_up(vectors, members);
        std::size_t const begin = std::min(member * share, vectors) * 4,
                          end = std::min(member * share + share, vectors) * 4;
        launch(&multimem_all_reduce_kernel, blocks_(end - begin), block_threads_k, stream, counters_(member),
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
    std::uint64_t landing_address = 0, scalars_address = 0;
    std::uint32_t queue_pair = 0, packet_sequence = 0, landing_key = 0, scalars_key = 0;
};

/**
 *  @brief One RoCE rail between a GPU and its counterpart on another host: an RC queue pair on the
 *      GPU's own NIC, writing GPU memory straight into GPU memory, registered as dma-buf.
 *
 *  - @b Towers: the GPU's tower lands in the counterpart's @c landing[parity], double-buffered by
 *    step parity, and the counterpart's other GPUs read it in place over NVLink.
 *  - @b Scalars: a row of host-pinned @c scalars, at the sender's host index, lands in the same row
 *    on the other side, so every host adds the rows in one order.
 *
 *  Every write carries an immediate, so its landing completes one receive that @ref await_landed
 *  reaps.
 */
struct rdma_rail_t {
    ibv_context *context = nullptr;
    ibv_pd *protection = nullptr;
    ibv_cq *completions = nullptr;
    ibv_qp *queue_pair = nullptr;
    ibv_mr *tower = nullptr, *landing = nullptr, *scalars = nullptr;
    rdma_endpoint_t local, remote;
    ibv_mtu mtu = IBV_MTU_1024;
    int gid_index = 0;
    std::uint64_t tower_address = 0;
    std::size_t tower_bytes = 0, scalars_row_bytes = 0, landed = 0;

    rdma_rail_t() noexcept = default;
    rdma_rail_t(rdma_rail_t const &) = delete;
    rdma_rail_t(rdma_rail_t &&other) noexcept
        : context(std::exchange(other.context, nullptr)), protection(std::exchange(other.protection, nullptr)),
          completions(std::exchange(other.completions, nullptr)), queue_pair(std::exchange(other.queue_pair, nullptr)),
          tower(std::exchange(other.tower, nullptr)), landing(std::exchange(other.landing, nullptr)),
          scalars(std::exchange(other.scalars, nullptr)), local(other.local), remote(other.remote), mtu(other.mtu),
          gid_index(other.gid_index), tower_address(other.tower_address), tower_bytes(other.tower_bytes),
          scalars_row_bytes(other.scalars_row_bytes), landed(other.landed) {}

    ~rdma_rail_t() noexcept {
        if (!context) return;
        require(ibv_destroy_qp(queue_pair), "queue pair release");
        for (ibv_mr *registration : {tower, landing, scalars}) require(ibv_dereg_mr(registration), "deregistration");
        require(ibv_destroy_cq(completions), "completion queue release");
        require(ibv_dealloc_pd(protection), "protection domain release");
        require(ibv_close_device(context), "device release");
    }

    /** Opens @p nic , registers @p tower , @p landing @b [2,rows,hidden] and the host-pinned
     *  @p scalars @b [hosts,4], and readies a queue pair whose @ref local endpoint the
     *  counterpart must receive. */
    [[nodiscard]] static rdma_rail_t make(char const *nic, nk::tensor_view<bf16_t, 2> tower,
                                          nk::tensor_span<bf16_t, 3> landing, nk::tensor_span<f32_t, 2> scalars,
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

        rail.tower_address = reinterpret_cast<std::uint64_t>(tower.data());
        rail.tower_bytes = tower.numel() * sizeof(bf16_t);
        rail.scalars_row_bytes = scalars.extent(1) * sizeof(f32_t);
        rail.tower = register_device_(rail.protection, tower.data(), rail.tower_bytes, IBV_ACCESS_LOCAL_WRITE);
        rail.landing = register_device_(rail.protection, landing.data(), landing.numel() * sizeof(bf16_t),
                                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        require((rail.scalars = ibv_reg_mr(rail.protection, scalars.data(), scalars.numel() * sizeof(f32_t),
                                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))
                    ? 0
                    : errno,
                "scalars registration");
        rail.local.queue_pair = rail.queue_pair->qp_num;
        rail.local.packet_sequence = packet_sequence & 0xFFFFFFu;
        rail.local.landing_address = reinterpret_cast<std::uint64_t>(landing.data());
        rail.local.landing_key = rail.landing->rkey;
        rail.local.scalars_address = reinterpret_cast<std::uint64_t>(scalars.data());
        rail.local.scalars_key = rail.scalars->rkey;

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

    /** Writes the whole tower into the counterpart's @c landing[parity]. */
    void post_tower(std::size_t parity) noexcept {
        post_write_(tower_address, tower_bytes, tower->lkey, remote.landing_address + parity * tower_bytes,
                    remote.landing_key);
    }

    /** Writes @c scalars[host] into the counterpart's @c scalars[host]. */
    void post_scalars(std::size_t host) noexcept {
        std::uint64_t const offset = host * scalars_row_bytes;
        post_write_(reinterpret_cast<std::uint64_t>(scalars->addr) + offset, scalars_row_bytes, scalars->lkey,
                    remote.scalars_address + offset, remote.scalars_key);
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
    __shared__ f32_t partials[block_threads_k / 32];
    for (std::uint32_t offset = 16; offset != 0; offset /= 2) value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    __syncthreads();
    if (threadIdx.x % 32 == 0) partials[threadIdx.x / 32] = value;
    __syncthreads();
    value = threadIdx.x % 32 < blockDim.x / 32 ? partials[threadIdx.x % 32] : 0.0f;
    for (std::uint32_t offset = 16; offset != 0; offset /= 2) value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    return value;
}

static __global__ void transpose_kernel(nk::tensor_view<bf16_t, 2> source, nk::tensor_span<bf16_t, 2> destination) {
    __shared__ bf16_t tile[32][33];
    std::size_t const column_base = blockIdx.x * 32, row_base = blockIdx.y * 32;
    for (std::uint32_t step = threadIdx.y; step < 32; step += blockDim.y)
        if (row_base + step < source.extent(0) && column_base + threadIdx.x < source.extent(1))
            tile[step][threadIdx.x] = source[row_base + step, column_base + threadIdx.x];
    __syncthreads();
    for (std::uint32_t step = threadIdx.y; step < 32; step += blockDim.y)
        if (column_base + step < destination.extent(0) && row_base + threadIdx.x < destination.extent(1))
            destination[column_base + step, row_base + threadIdx.x] = tile[threadIdx.x][step];
}

static __global__ void prefix_norms_kernel(nk::tensor_view<bf16_t, 2> vertices, nk::tensor_span<f32_t, 2> norms,
                                           matryoshka_t ladder) {
    std::size_t const row = blockIdx.x;
    f32_t running = 0.0f;
    for (std::uint32_t band = 0; band != ladder.count; ++band) {
        f32_t partial = 0.0f;
        for (std::uint32_t channel = ladder.begin(band) + threadIdx.x; channel < ladder.ends[band];
             channel += blockDim.x) {
            f32_t const value = __bfloat162float(vertices[row, channel]);
            partial += value * value;
        }
        running += block_sum(partial);
        if (threadIdx.x == 0) norms[band, row] = sqrtf(running);
    }
}

inline constexpr std::uint32_t fused_side_k = 128; // tile rows and columns, the MMA's M and N
inline constexpr std::uint32_t fused_depth_k = 64; // channels per stage, one 128-byte swizzle row
inline constexpr std::uint32_t fused_ring_k = 3;   // stages in flight, two blocks per SM
inline constexpr std::uint32_t fused_chunk_k = 32; // columns one `tcgen05.ld` moves per lane
inline constexpr std::uint32_t fused_stage_bytes_k = fused_side_k * fused_depth_k * sizeof(bf16_t);
inline constexpr std::uint32_t fused_shared_bytes_k = 2 * fused_ring_k * fused_stage_bytes_k + 1024; // + alignment
inline constexpr std::uint32_t fused_threads_k = 160; // four epilogue warps, the first issuing MMAs, and a TMA warp

/** The instruction descriptor of a @b [128,128] BF16 MMA into Float32, both operands K-major. */
inline constexpr std::uint32_t fused_instruction_k = (1u << 4)                                 // Float32 accumulator
                                                     | (1u << 7) | (1u << 10)                  // BF16 A and B
                                                     | ((fused_side_k >> 3) << 17)             // N
                                                     | ((fused_side_k >> 4) << 24);            // M
inline constexpr std::uint32_t fused_negated_instruction_k = fused_instruction_k | (1u << 13); // A negated

/** What every fused tile of one subgraph shares. */
struct fused_tile_t {
    nk::tensor_view<f32_t, 2> row_norms, column_norms;
    nk::tensor_span<bf16_t, 3> gradient;
    nk::tensor_span<f32_t, 3> partials;
    std::size_t row_base, column_base;
    f32_t beta, bias, weight;
};

[[nodiscard]] __device__ inline std::uint32_t shared_address_(void const *pointer) noexcept {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ inline void mbarrier_init_(std::uint64_t &barrier, std::uint32_t arrivals) noexcept {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(shared_address_(&barrier)), "r"(arrivals) : "memory");
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
__device__ inline void tmem_load_(std::uint32_t address, f32_t (&values)[fused_chunk_k]) noexcept {
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
__device__ inline void tmem_store_(std::uint32_t address, f32_t const (&values)[fused_chunk_k]) noexcept {
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

/** This lane's row summands of a fused tile. */
struct fused_row_t {
    f32_t loss = 0, bias_grad = 0, log_beta_grad = 0;
};

/** Scores @p band for this lane's row from the raw dots in TMEM at @p dots , adds its gradient to
 *  the suffix in TMEM at @p suffix , and writes that suffix as the band's BF16 gradient row, with
 *  the row's coefficient. */
__device__ inline void score_fused_row_(fused_tile_t const &tile, std::uint32_t dots, std::uint32_t suffix,
                                        std::uint32_t band, bool first, fused_row_t &summands) noexcept {
    std::uint32_t const local_row = threadIdx.x;
    std::size_t const row = std::size_t {blockIdx.y} * fused_side_k + local_row;
    f32_t const row_norm = tile.row_norms[band, row];
    f32_t coefficient = 0;
    for (std::uint32_t chunk = 0; chunk != fused_side_k / fused_chunk_k; ++chunk) {
        f32_t raw[fused_chunk_k], running[fused_chunk_k] = {};
        tmem_load_(dots + chunk * fused_chunk_k, raw);
        if (!first) tmem_load_(suffix + chunk * fused_chunk_k, running);
        std::size_t const first_column = std::size_t {blockIdx.x} * fused_side_k + chunk * fused_chunk_k;
        std::uint32_t packed[fused_chunk_k / 2];
#pragma unroll
        for (std::uint32_t cell = 0; cell != fused_chunk_k; ++cell) {
            std::size_t const column = first_column + cell;
            bool const own_copy = tile.row_base + row == tile.column_base + column;
            f32_t const denominator = row_norm * tile.column_norms[band, column];
            f32_t const cosine = denominator > cosine_floor_k ? __fdividef(raw[cell], denominator) : 0.0f;
            // A vertex's cosine with its own copy is one, whatever its raw dot rounds to.
            f32_t const logit = tile.beta * (own_copy ? 1.0f : cosine);
            f32_t const target = own_copy ? 1.0f : -1.0f;
            f32_t const signed_logit = target * (logit + tile.bias);
            f32_t const decay = __expf(-fabsf(signed_logit));
            f32_t const complement = __fdividef(signed_logit >= 0.0f ? decay : 1.0f, 1.0f + decay); // 1 - sigmoid
            f32_t const cell_grad = -tile.weight * target * complement;
            summands.loss += tile.weight * (fmaxf(-signed_logit, 0.0f) + log1pf(decay));
            summands.bias_grad += cell_grad;
            summands.log_beta_grad += cell_grad * logit;
            // The mirror cell (j, i) moves v_i the same way, and the own copy moves nothing.
            f32_t const raw_grad = own_copy
                                       ? 0.0f
                                       : __fdividef(2.0f * tile.beta * cell_grad, fmaxf(denominator, cosine_floor_k));
            coefficient += raw_grad * raw[cell];
            running[cell] += raw_grad;
            if (cell % 2)
                packed[cell / 2] = __bfloat16_as_ushort(__float2bfloat16(running[cell - 1])) |
                                   std::uint32_t {__bfloat16_as_ushort(__float2bfloat16(running[cell]))} << 16;
        }
        tmem_store_(suffix + chunk * fused_chunk_k, running);
        for (std::uint32_t quarter = 0; quarter != 4; ++quarter)
            asm volatile(
                "st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(&tile.gradient[band, row, first_column + quarter * 8]),
                "r"(packed[quarter * 4]), "r"(packed[quarter * 4 + 1]), "r"(packed[quarter * 4 + 2]),
                "r"(packed[quarter * 4 + 3])
                : "memory");
    }
    tile.partials[3 + band, row, blockIdx.x] = coefficient;
}

/**
 *  @brief One @b [128,128] tile of a subgraph, scored at every Matryoshka prefix with @c tcgen05
 *      MMAs fed by TMA, with no Float32 @b [K,R,R] stash.
 *
 *  A forward sweep over the depth accumulates every cell's full raw dot in TMEM. A reverse sweep
 *  with negated rows then takes the suffix back off, meeting the band edges widest first, and at
 *  each edge the four epilogue warps score their rows, keeping each band's suffix-summed gradient
 *  in TMEM beside the dots. Row summands land in the tile's @b [3+K,R,R/128] partials, which
 *  @ref fold_partials_kernel adds in order. The fifth warp only streams stages through a ring of
 *  @c fused_ring_k slots.
 */
static __global__ void __launch_bounds__(fused_threads_k)
    matryoshka_tile_kernel(CUtensorMap const __grid_constant__ rows_map,
                           CUtensorMap const __grid_constant__ columns_map, std::uint32_t stages,
                           fused_tile_t const tile, matryoshka_t ladder) {
    extern __shared__ byte_t dynamic_shared[];
    __shared__ alignas(8) std::uint64_t filled[fused_ring_k], emptied[fused_ring_k], scored;
    __shared__ std::uint32_t tmem_base;
    byte_t *const ring = dynamic_shared + (1024 - shared_address_(dynamic_shared) % 1024) % 1024;
    std::uint32_t const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    std::uint32_t const reversed = ladder.count > 1 ? stages - ladder.ends[0] / fused_depth_k : 0;
    std::uint32_t const positions = stages + reversed;
    // Forward positions take stages in order, reverse ones from the last to the narrowest band's.
    auto const depth_of = [&](std::uint32_t position) {
        return (position < stages ? position : 2 * stages - 1 - position) * fused_depth_k;
    };
    auto const rows_stage = [&](std::uint32_t slot) { return ring + slot * fused_stage_bytes_k; };
    auto const columns_stage = [&](std::uint32_t slot) { return ring + (fused_ring_k + slot) * fused_stage_bytes_k; };

    if (threadIdx.x == 0) {
        for (std::uint32_t slot = 0; slot != fused_ring_k; ++slot)
            mbarrier_init_(filled[slot], 1), mbarrier_init_(emptied[slot], 1);
        mbarrier_init_(scored, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(shared_address_(&tmem_base)),
            "r"(2 * fused_side_k)
            : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (warp == 4) {
        if (lane != 0) return;
        for (std::uint32_t position = 0; position != positions; ++position) {
            std::uint32_t const slot = position % fused_ring_k;
            if (position >= fused_ring_k) mbarrier_wait_(emptied[slot], (position / fused_ring_k - 1) % 2);
            mbarrier_expect_(filled[slot], 2 * fused_stage_bytes_k);
            tma_load_(rows_stage(slot), rows_map, depth_of(position), blockIdx.y * fused_side_k, filled[slot]);
            tma_load_(columns_stage(slot), columns_map, depth_of(position), blockIdx.x * fused_side_k, filled[slot]);
        }
        return;
    }

    // Each epilogue warp owns 32 TMEM lanes, one per row; dots and suffix take 128 columns each.
    std::uint32_t const dots = tmem_base + (warp * 32 << 16), suffix = dots + fused_side_k;
    bool const issuer = threadIdx.x == 0;
    std::uint32_t band = ladder.count - 1, epilogues = 0;
    fused_row_t summands;
    auto const score = [&](std::uint32_t scored_band) {
        if (issuer) tcgen05_commit_(scored);
        mbarrier_wait_(scored, epilogues++ % 2);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        score_fused_row_(tile, dots, suffix, scored_band, scored_band + 1 == ladder.count, summands);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        asm volatile("bar.sync 1, 128;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    };
    for (std::uint32_t position = 0; position != positions; ++position) {
        std::uint32_t const slot = position % fused_ring_k, depth = depth_of(position);
        bool const reverse = position >= stages;
        if (issuer) {
            mbarrier_wait_(filled[slot], position / fused_ring_k % 2);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        }
        for (std::uint32_t order = 0; order != fused_depth_k / 16; ++order) {
            std::uint32_t const step = reverse ? fused_depth_k / 16 - 1 - order : order;
            if (issuer)
                tcgen05_mma_(dots & 0xFFFFu, stage_descriptor_(rows_stage(slot), step),
                             stage_descriptor_(columns_stage(slot), step),
                             reverse ? fused_negated_instruction_k : fused_instruction_k, position | step);
            if (reverse && band && depth + step * 16 == ladder.ends[band - 1]) score(--band);
        }
        if (issuer) tcgen05_commit_(emptied[slot]);
        if (position + 1 == stages) score(band);
    }

    std::size_t const row = std::size_t {blockIdx.y} * fused_side_k + threadIdx.x;
    tile.partials[0, row, blockIdx.x] = summands.loss;
    tile.partials[1, row, blockIdx.x] = summands.bias_grad;
    tile.partials[2, row, blockIdx.x] = summands.log_beta_grad;
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"(2 * fused_side_k)
                     : "memory");
}

/** Adds each row's column-tile @p partials , in tile order, to its @p scalars and
 *  @p self_coefficients . */
static __global__ void fold_partials_kernel(nk::tensor_view<f32_t, 3> partials, nk::tensor_span<f32_t, 2> scalars,
                                            nk::tensor_span<f32_t, 2> self_coefficients) {
    std::size_t const row = blockIdx.x * std::size_t {blockDim.x} + threadIdx.x;
    if (row >= partials.extent(1)) return;
    for (std::size_t quantity = 0; quantity != partials.extent(0); ++quantity) {
        f32_t total = 0.0f;
        for (std::size_t tile = 0; tile != partials.extent(2); ++tile) total += partials[quantity, row, tile];
        if (quantity < 3) scalars[quantity, row] += total;
        else self_coefficients[quantity - 3, row] += total;
    }
}

static __global__ void add_kernel(nk::tensor_view<f32_t, 2> addend, nk::tensor_span<f32_t, 2> accumulator) {
    std::size_t const columns = accumulator.extent(1), total = accumulator.extent(0) * columns;
    for (std::size_t cell = blockIdx.x * std::size_t {blockDim.x} + threadIdx.x; cell < total;
         cell += std::size_t {gridDim.x} * blockDim.x)
        accumulator[cell / columns, cell % columns] += addend[cell / columns, cell % columns];
}

/** The cosine's self term folded into the forces, then the BF16 gradient. */
static __global__ void finalize_kernel(nk::tensor_view<bf16_t, 2> vertices, nk::tensor_view<f32_t, 2> norms,
                                       nk::tensor_view<f32_t, 2> self_coefficients, nk::tensor_view<f32_t, 2> forces,
                                       nk::tensor_span<bf16_t, 2> gradient, matryoshka_t ladder) {
    std::size_t const row = blockIdx.x;
    for (std::uint32_t channel = threadIdx.x; channel < vertices.extent(1); channel += blockDim.x) {
        f32_t fold = 0.0f;
        for (std::uint32_t band = 0; band != ladder.count; ++band)
            if (channel < ladder.ends[band])
                fold += self_coefficients[band, row] / fmaxf(norms[band, row] * norms[band, row], cosine_floor_k);
        gradient[row, channel] = __float2bfloat16(forces[row, channel] -
                                                  fold * __bfloat162float(vertices[row, channel]));
    }
}

static __global__ void scalars_kernel(nk::tensor_view<f32_t, 2> scalars, nk::tensor_span<f32_t, 1> out) {
    for (std::uint32_t which = 0; which != 3; ++which) {
        f32_t partial = 0.0f;
        for (std::size_t row = threadIdx.x; row < scalars.extent(1); row += blockDim.x) partial += scalars[which, row];
        f32_t const total = block_sum(partial);
        if (threadIdx.x == 0) out[which] = total;
    }
}

/**
 *  @brief A rank's Matryoshka sigmoid contrastive loss, streamed one subgraph at a time and
 *      row-owned.
 *
 *  Every vertex is a row against all @c S·R world vertices, at every prefix of the ladder, scored
 *  by SigLIP's sigmoid BCE with weight @c 1/(S·R): its own copy is the positive, every other vertex
 *  a negative.
 *
 *  - @b Streamed: each subgraph's @b [K,R,R] tile is scored, lossed and folded into the rank's
 *    forces before the next is touched, so the scratch is one tile whatever the world.
 *  - @b Row-owned: cell @c (i,j) and its mirror @c (j,i) share the cosine, target and weight, so
 *    the owner of @c i differentiates both. No rank computes another's forces, and the backward
 *    exchanges nothing.
 *
 *  Call @ref begin , then @ref tile once per subgraph in any order, then @ref finish , which writes
 *  the rank's loss, bias-gradient and log-β-gradient summands, which add up across ranks.
 */
struct streamed_loss_t {
    matryoshka_t ladder;
    std::size_t rows = 0, hidden = 0;
    nk_capability_t capabilities = 0;
    nk::tensor<f32_t, nk::allocator<f32_t>, 2> row_norms, column_norms, products, forces, scalars, self_coefficients;
    nk::tensor<f32_t, nk::allocator<f32_t>, 3> partials;
    nk::tensor<bf16_t, nk::allocator<bf16_t>, 3> gradient;
    nk::tensor<bf16_t, nk::allocator<bf16_t>, 2> transposed, staged;
    nk::tensor<byte_t, nk::allocator<byte_t>, 1> packed, workspace;

    [[nodiscard]] static streamed_loss_t make(nk::device_t device, nk::stream_t const &stream, std::size_t rows,
                                              std::size_t hidden, matryoshka_t ladder) noexcept {
        streamed_loss_t loss;
        loss.ladder = ladder, loss.rows = rows, loss.hidden = hidden;
        loss.capabilities = take(device.capabilities_enabled(), "capability query");
        std::size_t const k = ladder.count;
        bool const edges_fit = std::all_of(ladder.ends, ladder.ends + k,
                                           [](std::uint32_t end) { return end % 16 == 0; });
        require(rows % fused_side_k == 0 && hidden % fused_depth_k == 0 && edges_fit && ladder.ends[k - 1] == hidden
                    ? 0
                    : EINVAL,
                "fused tile shape, which takes rows in multiples of 128, channels in 64 and band edges in 16");
        require(cudaFuncSetAttribute(&matryoshka_tile_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     int(fused_shared_bytes_k)),
                "fused tile shared memory");
        // a zero-byte tensor would read as a failed allocation
        std::size_t pack_bytes = 1, workspace_bytes = 1;
        for (std::uint32_t band = 0; band != k; ++band) {
            std::size_t const width = ladder.width(band);
            pack_bytes = std::max(pack_bytes,
                                  take(nk::dots_pack_size<nk::bf16_t>(width, rows, loss.capabilities), "pack size"));
            workspace_bytes = std::max(
                workspace_bytes, take(nk::dots_packed_workspace_size<nk::bf16_t>(rows, width, rows, loss.capabilities),
                                      "workspace size"));
        }
        loss.row_norms = make_tensor<f32_t, 2>(loss.capabilities, stream, {k, rows});
        loss.column_norms = make_tensor<f32_t, 2>(loss.capabilities, stream, {k, rows});
        loss.products = make_tensor<f32_t, 2>(loss.capabilities, stream, {rows, hidden});
        loss.forces = make_tensor<f32_t, 2>(loss.capabilities, stream, {rows, hidden});
        loss.scalars = make_tensor<f32_t, 2>(loss.capabilities, stream, {3, rows});
        loss.self_coefficients = make_tensor<f32_t, 2>(loss.capabilities, stream, {k, rows});
        loss.partials = make_tensor<f32_t, 3>(loss.capabilities, stream, {3 + k, rows, rows / fused_side_k});
        loss.gradient = make_tensor<bf16_t, 3>(loss.capabilities, stream, {k, rows, rows});
        loss.transposed = make_tensor<bf16_t, 2>(loss.capabilities, stream, {hidden, rows});
        loss.staged = make_tensor<bf16_t, 2>(loss.capabilities, stream, {rows, hidden});
        loss.packed = make_tensor<byte_t, 1>(loss.capabilities, stream, {pack_bytes});
        loss.workspace = make_tensor<byte_t, 1>(loss.capabilities, stream, {workspace_bytes});
        return loss;
    }

    /** Device bytes the loss holds, independent of the world's size. */
    [[nodiscard]] std::size_t bytes() const noexcept {
        return (row_norms.numel() + column_norms.numel() + products.numel() + forces.numel() + scalars.numel() +
                self_coefficients.numel() + partials.numel()) *
                   sizeof(f32_t) +
               (gradient.numel() + transposed.numel() + staged.numel()) * sizeof(bf16_t) + packed.numel() +
               workspace.numel();
    }

    void begin(nk::tensor_view<bf16_t, 2> vertices, nk::stream_t const &stream) noexcept {
        cudaStream_t const cuda = cuda_stream(stream);
        require(cudaMemsetAsync(forces.data(), 0, forces.numel() * sizeof(f32_t), cuda), "zeroing forces");
        require(cudaMemsetAsync(scalars.data(), 0, scalars.numel() * sizeof(f32_t), cuda), "zeroing scalars");
        require(cudaMemsetAsync(self_coefficients.data(), 0, self_coefficients.numel() * sizeof(f32_t), cuda),
                "zeroing coefficients");
        launch(&prefix_norms_kernel, rows, block_threads_k, stream, vertices, row_norms.span(), ladder);
    }

    /** Score this rank's rows against @p subgraph , whose first vertex is @p column_base on the
     *  global axis; @p vertices is a span only so TMA can map it, and is never written. */
    void tile(nk::tensor_span<bf16_t, 2> vertices, nk::tensor_view<bf16_t, 2> subgraph, std::size_t row_base,
              std::size_t column_base, f32_t beta, f32_t bias, f32_t live, nk::stream_t const &stream) noexcept {
        // TMA reads local memory, so a peer's tower crosses NVLink once, into the staging copy.
        require(cudaMemcpyAsync(staged.data(), subgraph.data(), staged.numel() * sizeof(bf16_t), cudaMemcpyDefault,
                                cuda_stream(stream)),
                "subgraph staging");
        launch(&prefix_norms_kernel, rows, block_threads_k, stream, staged.view(), column_norms.span(), ladder);
        dim3 const transpose_grid {static_cast<std::uint32_t>(hidden / 32), static_cast<std::uint32_t>(rows / 32)};
        launch(&transpose_kernel, transpose_grid, dim3 {32, 8}, stream, staged.view(), transposed.span());
        std::uint32_t const tiles = static_cast<std::uint32_t>(rows / fused_side_k);
        fused_tile_t const tile {
            row_norms.view(), column_norms.view(), gradient.span(), partials.span(), row_base, column_base, beta, bias,
            1.0f / live};
        launch(&matryoshka_tile_kernel, dim3 {tiles, tiles}, fused_threads_k, fused_shared_bytes_k, stream,
               tensor_map_(vertices), tensor_map_(staged.span()), static_cast<std::uint32_t>(hidden / fused_depth_k),
               tile, ladder);
        launch(&fold_partials_kernel, divide_round_up(rows, block_threads_k), block_threads_k, stream, partials.view(),
               scalars.span(), self_coefficients.span());
        for (std::uint32_t band = 0; band != ladder.count; ++band)
            multiply_(gradient.data() + band * rows * rows, rows, transposed.data() + ladder.begin(band) * rows, rows,
                      products.data() + ladder.begin(band), hidden, rows, ladder.width(band), rows, stream);
        launch(&add_kernel, divide_round_up(rows * hidden, block_threads_k), block_threads_k, stream, products.view(),
               forces.span());
    }

    /** Writes the BF16 @p gradient_out and the three @p summands , a multimem payload or any device
     *  span. */
    void finish(nk::tensor_view<bf16_t, 2> vertices, nk::tensor_span<bf16_t, 2> gradient_out,
                nk::tensor_span<f32_t, 1> summands, nk::stream_t const &stream) noexcept {
        launch(&finalize_kernel, rows, block_threads_k, stream, vertices, row_norms.view(), self_coefficients.view(),
               forces.view(), gradient_out, ladder);
        launch(&scalars_kernel, 1, block_threads_k, stream, scalars.view(), summands);
    }

  private:

    /** The TMA map of a K-major @b [rows,hidden] tower in @b [128,64] boxes, 128-byte swizzled. */
    [[nodiscard]] CUtensorMap tensor_map_(nk::tensor_span<bf16_t, 2> tower) const noexcept {
        CUtensorMap map;
        std::uint64_t const extents[2] = {hidden, rows}, strides[1] = {hidden * sizeof(bf16_t)};
        std::uint32_t const box[2] = {fused_depth_k, fused_side_k}, steps[2] = {1, 1};
        require(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, tower.data(), extents, strides, box,
                                       steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                "tensor map");
        return map;
    }

    /** @c c=a·bᵀ for @p a @b [m,depth], @p b @b [n,depth] and @p c @b [m,n], each with a row stride
     *  in scalars. */
    void multiply_(bf16_t const *a, std::size_t a_stride, bf16_t const *b, std::size_t b_stride, f32_t *c,
                   std::size_t c_stride, std::size_t m, std::size_t n, std::size_t depth,
                   nk::stream_t const &stream) noexcept {
        require(nk::dots_pack<nk::bf16_t>(reinterpret_cast<nk::bf16_t const *>(b), n, depth, b_stride * sizeof(bf16_t),
                                          packed.data(), capabilities, stream.get()),
                "pack");
        require(nk::dots_packed<nk::bf16_t>(
                    reinterpret_cast<nk::bf16_t const *>(a), packed.data(), reinterpret_cast<nk::f32_t *>(c), m, n,
                    depth, a_stride * sizeof(bf16_t), c_stride * sizeof(f32_t),
                    std::span<byte_t> {workspace.data(), workspace.numel()}, capabilities, stream.get()),
                "product");
    }
};

} // namespace ashvardanian::scalingcontrasts
