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
#include <unistd.h>    // `close`

#include <algorithm> // `std::max`, `std::clamp`
#include <span>      // `std::span`
#include <tuple>     // `std::tuple`, `std::apply`
#include <utility>   // `std::exchange`

#include <cuda.h>             // `cuMulticastCreate`, `cuMulticastBindMem`, `cuMemMap`, `cuMemSetAccess`
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

/** Launches @p kernel on @p stream with @p arguments copied by value, and checks that it launched.
 *  No triple chevrons: https://ashvardanian.com/posts/less-wrong-cuda-hello-world/ */
template <typename... parameter_types_, typename... argument_types_>
void launch(void (*kernel)(parameter_types_...), dim3 grid, dim3 block, nk::stream_t const &stream,
            argument_types_ &&...arguments) noexcept {
    std::tuple<parameter_types_...> values {std::forward<argument_types_>(arguments)...};
    void *addresses[sizeof...(parameter_types_)];
    std::apply(
        [&](auto &...value) {
            std::size_t index = 0;
            ((addresses[index++] = &value), ...);
        },
        values);
    require(cudaLaunchKernel(reinterpret_cast<void const *>(kernel), grid, block, addresses, 0, cuda_stream(stream)),
            "kernel launch");
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

/** Adds this member's arrival to every replica's flag, waits on its own flag for the round's last
 *  arrival, and advances its own round, so the counters never reset. */
static __global__ void multimem_barrier_kernel(std::uint32_t *multicast_flag, std::uint32_t *own_flag,
                                               std::uint32_t *own_round, std::uint32_t members) {
    std::uint32_t const target = *own_round + members;
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(multicast_flag), "r"(1u) : "memory");
    std::uint32_t arrived = 0;
    do asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(arrived) : "l"(own_flag) : "memory");
    while (static_cast<std::int32_t>(arrived - target) < 0); // wrap-safe past 2³² arrivals
    *own_round = target;
}

/** @c own[i]=Σ(members)payload[i] over @b [begin,end), one switch-reduced load per four scalars. */
static __global__ void multimem_reduce_scatter_kernel(nk::tensor_view<f32_t, 1> multicast,
                                                      nk::tensor_span<f32_t, 1> own, std::size_t begin,
                                                      std::size_t end) {
    for (std::size_t scalar = begin + 4 * (blockIdx.x * std::size_t {blockDim.x} + threadIdx.x); scalar < end;
         scalar += 4 * std::size_t {gridDim.x} * blockDim.x) {
        f32_t x, y, z, w;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                     : "l"(&multicast[scalar])
                     : "memory");
        asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(&own[scalar]), "f"(x), "f"(y), "f"(z), "f"(w)
                     : "memory");
    }
}

/** @c payload[i]=Σ(members)payload[i] over @b [begin,end) on every replica: one switch-reduced load
 *  and one multicast store per four scalars. */
static __global__ void multimem_all_reduce_kernel(nk::tensor_span<f32_t, 1> multicast, std::size_t begin,
                                                  std::size_t end) {
    for (std::size_t scalar = begin + 4 * (blockIdx.x * std::size_t {blockDim.x} + threadIdx.x); scalar < end;
         scalar += 4 * std::size_t {gridDim.x} * blockDim.x) {
        f32_t x, y, z, w;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                     : "l"(&multicast[scalar])
                     : "memory");
        asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(&multicast[scalar]), "f"(x),
                     "f"(y), "f"(z), "f"(w)
                     : "memory");
    }
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
 *  A replica holds @ref payload_scalars scalars, then its arrival flag and its barrier round. Only
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
        // the payload, then the flag and the round
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
        // reductions while every device is idle rather than between barriers that spin.
        for (std::size_t member = 0; member != members; ++member) {
            require(cudaSetDevice(int(member)), "device switch");
            launch(&multimem_reduce_scatter_kernel, 1, block_threads_k, nk::stream_t {}, domain.multicast_payload_(),
                   domain.payload(member), std::size_t(0), std::size_t(0));
            launch(&multimem_all_reduce_kernel, 1, block_threads_k, nk::stream_t {}, domain.multicast_payload_(),
                   std::size_t(0), std::size_t(0));
            require(cudaDeviceSynchronize(), "kernel loading");
        }
        return domain;
    }

    /** The scalars @p member contributes and receives, in its own replica. */
    [[nodiscard]] nk::tensor_span<f32_t, 1> payload(std::size_t member) const noexcept {
        return {pointer_<f32_t>(replicas_base + member * stride), payload_scalars};
    }

    /** Returns on @p member 's stream once every member has entered the same round. */
    void barrier(std::size_t member, nk::stream_t const &stream) noexcept {
        CUdeviceptr const counters = replicas_base + member * stride + payload_scalars * sizeof(f32_t);
        launch(&multimem_barrier_kernel, 1, 1, stream,
               pointer_<std::uint32_t>(multicast_base + payload_scalars * sizeof(f32_t)),
               pointer_<std::uint32_t>(counters), pointer_<std::uint32_t>(counters + sizeof(std::uint32_t)),
               static_cast<std::uint32_t>(members));
    }

    /** In place: @p member 's slice of @p shard_scalars , a multiple of four, at
     *  @c member·shard_scalars of its payload becomes the sum of every payload's same slice. */
    void reduce_scatter(std::size_t member, std::size_t shard_scalars, nk::stream_t const &stream) noexcept {
        std::size_t const begin = member * shard_scalars, end = begin + shard_scalars;
        barrier(member, stream);
        launch(&multimem_reduce_scatter_kernel, blocks_(shard_scalars), block_threads_k, stream, multicast_payload_(),
               payload(member), begin, end);
        barrier(member, stream);
    }

    /** In place: the first @p scalars , a multiple of four, of every payload become their sum, each
     *  member reducing and broadcasting its own share. */
    void all_reduce(std::size_t member, std::size_t scalars, nk::stream_t const &stream) noexcept {
        std::size_t const vectors = scalars / 4, share = divide_round_up(vectors, members);
        std::size_t const begin = std::min(member * share, vectors) * 4,
                          end = std::min(member * share + share, vectors) * 4;
        barrier(member, stream);
        if (begin != end)
            launch(&multimem_all_reduce_kernel, blocks_(end - begin), block_threads_k, stream, multicast_payload_(),
                   begin, end);
        barrier(member, stream);
    }

  private:
    template <typename value_type_>
    [[nodiscard]] static value_type_ *pointer_(CUdeviceptr address) noexcept {
        return reinterpret_cast<value_type_ *>(address);
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
        CUdeviceptr base = 0;
        std::size_t allocation = 0;
        require(cuMemGetAddressRange(&base, &allocation, reinterpret_cast<CUdeviceptr>(pointer)), "allocation range");
        int descriptor = -1;
        require(cuMemGetHandleForAddressRange(&descriptor, base, allocation, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
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

inline constexpr std::uint32_t fused_side_k = 64;                // rows and columns of one fused tile
inline constexpr std::uint32_t fused_warps_k = 4;                // each owns 16 rows of the tile
inline constexpr std::uint32_t fused_pitch_k = fused_side_k + 8; // padded rows keep `ldmatrix` conflict-free

using fused_stage_t = bf16_t[fused_side_k][fused_pitch_k];

/** A fused tile's register state: each lane's 32 cells, as raw dots and running gradients, and
 *  its two rows' summands. */
struct fused_cells_t {
    f32_t raw[8][4] = {}, running[8][4] = {};
    f32_t loss[2] = {}, bias_grad[2] = {}, log_beta_grad[2] = {};
};

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

/** The first row and column of this block's fused tile. */
[[nodiscard]] __device__ inline std::size_t tile_row_() noexcept { return std::size_t {blockIdx.y} * fused_side_k; }
[[nodiscard]] __device__ inline std::size_t tile_column_() noexcept { return std::size_t {blockIdx.x} * fused_side_k; }

/** Starts copying the rows' @b [64,64] and the transposed columns' @b [64,64] at channel
 *  @p depth . */
__device__ inline void load_fused_stage_(fused_stage_t &rows, fused_stage_t &columns,
                                         nk::tensor_view<bf16_t, 2> vertices, nk::tensor_view<bf16_t, 2> transposed,
                                         fused_tile_t const &tile, std::size_t depth) noexcept {
    for (std::uint32_t chunk = threadIdx.x; chunk < fused_side_k * fused_side_k / 8; chunk += blockDim.x) {
        std::uint32_t const line = chunk / 8, part = chunk % 8 * 8;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(shared_address_(&rows[line][part])),
                     "l"(&vertices[tile_row_() + line, depth + part])
                     : "memory");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(shared_address_(&columns[line][part])),
                     "l"(&transposed[depth + line, tile_column_() + part])
                     : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

/** One k16 step of a warp's 16 rows against the stage's 64 columns into @p accumulator , the
 *  rows' sign bits flipped by @p sign , which @c 0x80008000 negates exactly. */
__device__ inline void multiply_fused_step_(fused_stage_t const &rows, fused_stage_t const &columns, std::uint32_t step,
                                            std::uint32_t sign, f32_t (&accumulator)[8][4]) noexcept {
    std::uint32_t const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    std::uint32_t a[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 " //
                 "{%0, %1, %2, %3}, [%4];"                   //
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(shared_address_(&rows[warp * 16 + lane % 16][step * 16 + lane / 16 * 8])));
#pragma unroll
    for (std::uint32_t &word : a) word ^= sign;
#pragma unroll
    for (std::uint32_t pair = 0; pair != 4; ++pair) {
        std::uint32_t b[4];
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " //
            "{%0, %1, %2, %3}, [%4];"                         //
            : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
            : "r"(shared_address_(&columns[step * 16 + lane % 8 + lane / 8 % 2 * 8][pair * 16 + lane / 16 * 8])));
#pragma unroll
        for (std::uint32_t half = 0; half != 2; ++half) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " //
                         "{%0, %1, %2, %3}, "                                   //
                         "{%4, %5, %6, %7}, "                                   //
                         "{%8, %9}, "                                           //
                         "{%0, %1, %2, %3};"                                    //
                         : "+f"(accumulator[pair * 2 + half][0]), "+f"(accumulator[pair * 2 + half][1]),
                           "+f"(accumulator[pair * 2 + half][2]), "+f"(accumulator[pair * 2 + half][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[half * 2]), "r"(b[half * 2 + 1]));
        }
    }
}

/** Scores @p band from the cells' raw dots, adds its gradient to the running suffix, and writes
 *  that suffix through @p gradient_stage in 16-byte rows, with the band's coefficient partials. */
__device__ inline void score_fused_band_(fused_cells_t &cells, fused_tile_t const &tile, fused_stage_t &gradient_stage,
                                         std::uint32_t band) noexcept {
    std::uint32_t const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    f32_t coefficient[2] = {};
#pragma unroll
    for (std::uint32_t column_group = 0; column_group != 8; ++column_group)
#pragma unroll
        for (std::uint32_t element = 0; element != 4; ++element) {
            std::uint32_t const slot = element / 2;
            std::uint32_t const local_row = warp * 16 + lane / 4 + slot * 8;
            std::uint32_t const local_column = column_group * 8 + lane % 4 * 2 + element % 2;
            std::size_t const row = tile_row_() + local_row, column = tile_column_() + local_column;
            bool const own_copy = tile.row_base + row == tile.column_base + column;
            f32_t const raw = cells.raw[column_group][element];
            f32_t const denominator = tile.row_norms[band, row] * tile.column_norms[band, column];
            f32_t const cosine = denominator > cosine_floor_k ? __fdividef(raw, denominator) : 0.0f;
            // A vertex's cosine with its own copy is one, whatever its raw dot rounds to.
            f32_t const logit = tile.beta * (own_copy ? 1.0f : cosine);
            f32_t const target = own_copy ? 1.0f : -1.0f;
            f32_t const signed_logit = target * (logit + tile.bias);
            f32_t const decay = __expf(-fabsf(signed_logit));
            f32_t const complement = __fdividef(signed_logit >= 0.0f ? decay : 1.0f, 1.0f + decay); // 1 - sigmoid
            f32_t const cell_grad = -tile.weight * target * complement;
            cells.loss[slot] += tile.weight * (fmaxf(-signed_logit, 0.0f) + log1pf(decay));
            cells.bias_grad[slot] += cell_grad;
            cells.log_beta_grad[slot] += cell_grad * logit;
            // The mirror cell (j, i) moves v_i the same way, and the own copy moves nothing.
            f32_t const raw_grad = own_copy
                                       ? 0.0f
                                       : __fdividef(2.0f * tile.beta * cell_grad, fmaxf(denominator, cosine_floor_k));
            coefficient[slot] += raw_grad * raw;
            cells.running[column_group][element] += raw_grad;
            gradient_stage[local_row][local_column] = __float2bfloat16(cells.running[column_group][element]);
        }
#pragma unroll
    for (std::uint32_t slot = 0; slot != 2; ++slot) {
#pragma unroll
        for (std::uint32_t offset = 1; offset != 4; offset *= 2)
            coefficient[slot] += __shfl_xor_sync(0xFFFFFFFFu, coefficient[slot], offset);
        std::size_t const row = tile_row_() + warp * 16 + lane / 4 + slot * 8;
        if (lane % 4 == 0) tile.partials[3 + band, row, blockIdx.x] = coefficient[slot];
    }
    __syncthreads();
    for (std::uint32_t chunk = threadIdx.x; chunk < fused_side_k * fused_side_k / 8; chunk += blockDim.x) {
        std::uint32_t const line = chunk / 8, part = chunk % 8 * 8;
        std::uint32_t x, y, z, w;
        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                     : "r"(shared_address_(&gradient_stage[line][part])));
        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                         &tile.gradient[band, tile_row_() + line, tile_column_() + part]),
                     "r"(x), "r"(y), "r"(z), "r"(w)
                     : "memory");
    }
    __syncthreads();
}

/**
 *  @brief One @b [64,64] tile of a subgraph, scored at every Matryoshka prefix, with no Float32
 *      @b [K,R,R] stash.
 *
 *  A forward sweep over the depth gives every cell's full raw dot. A reverse sweep with negated
 *  rows then takes the suffix back off, meeting the band edges widest first, so each band's
 *  suffix-summed gradient needs one running register per cell. Row summands land in the tile's
 *  @b [3+K,R,R/64] partials per column tile, which @ref fold_partials_kernel adds in order.
 */
static __global__ void matryoshka_tile_kernel(nk::tensor_view<bf16_t, 2> vertices,
                                              nk::tensor_view<bf16_t, 2> transposed, fused_tile_t const tile,
                                              matryoshka_t ladder) {
    __shared__ alignas(16) fused_stage_t rows_stages[2], columns_stages[2], gradient_stage;
    std::uint32_t const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    std::uint32_t const stages = static_cast<std::uint32_t>(vertices.extent(1) / fused_side_k);
    std::uint32_t const reversed = ladder.count > 1 ? stages - ladder.ends[0] / fused_side_k : 0;
    std::uint32_t const positions = stages + reversed;
    fused_cells_t cells;

    // Forward positions take stages in order, reverse ones from the last to the narrowest band's.
    auto const depth_of = [&](std::uint32_t position) {
        return std::size_t {position < stages ? position : 2 * stages - 1 - position} * fused_side_k;
    };
    std::uint32_t band = ladder.count - 1;
    load_fused_stage_(rows_stages[0], columns_stages[0], vertices, transposed, tile, depth_of(0));
    for (std::uint32_t position = 0; position != positions; ++position) {
        std::uint32_t const buffer = position % 2;
        if (position + 1 != positions)
            load_fused_stage_(rows_stages[1 - buffer], columns_stages[1 - buffer], vertices, transposed, tile,
                              depth_of(position + 1));
        if (position + 1 != positions) asm volatile("cp.async.wait_group 1;" ::: "memory");
        else asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (position < stages) {
#pragma unroll
            for (std::uint32_t step = 0; step != fused_side_k / 16; ++step)
                multiply_fused_step_(rows_stages[buffer], columns_stages[buffer], step, 0, cells.raw);
        }
        else {
#pragma unroll
            for (std::uint32_t backwards = 0; backwards != fused_side_k / 16; ++backwards) {
                std::uint32_t const step = fused_side_k / 16 - 1 - backwards;
                multiply_fused_step_(rows_stages[buffer], columns_stages[buffer], step, 0x80008000u, cells.raw);
                if (band && depth_of(position) + step * 16 == ladder.ends[band - 1])
                    score_fused_band_(cells, tile, gradient_stage, --band);
            }
        }
        __syncthreads();
        if (position + 1 == stages) score_fused_band_(cells, tile, gradient_stage, band);
    }

#pragma unroll
    for (std::uint32_t slot = 0; slot != 2; ++slot) {
#pragma unroll
        for (std::uint32_t offset = 1; offset != 4; offset *= 2) {
            cells.loss[slot] += __shfl_xor_sync(0xFFFFFFFFu, cells.loss[slot], offset);
            cells.bias_grad[slot] += __shfl_xor_sync(0xFFFFFFFFu, cells.bias_grad[slot], offset);
            cells.log_beta_grad[slot] += __shfl_xor_sync(0xFFFFFFFFu, cells.log_beta_grad[slot], offset);
        }
        std::size_t const row = tile_row_() + warp * 16 + lane / 4 + slot * 8;
        if (lane % 4 != 0) continue;
        tile.partials[0, row, blockIdx.x] = cells.loss[slot];
        tile.partials[1, row, blockIdx.x] = cells.bias_grad[slot];
        tile.partials[2, row, blockIdx.x] = cells.log_beta_grad[slot];
    }
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
    nk::tensor<bf16_t, nk::allocator<bf16_t>, 2> transposed;
    nk::tensor<byte_t, nk::allocator<byte_t>, 1> packed, workspace;

    [[nodiscard]] static streamed_loss_t make(nk::device_t device, nk::stream_t const &stream, std::size_t rows,
                                              std::size_t hidden, matryoshka_t ladder) noexcept {
        streamed_loss_t loss;
        loss.ladder = ladder, loss.rows = rows, loss.hidden = hidden;
        loss.capabilities = take(device.capabilities_enabled(), "capability query");
        std::size_t const k = ladder.count;
        bool const edges_fit = std::all_of(ladder.ends, ladder.ends + k,
                                           [](std::uint32_t end) { return end % 16 == 0; });
        require(rows % fused_side_k == 0 && hidden % fused_side_k == 0 && edges_fit && ladder.ends[k - 1] == hidden
                    ? 0
                    : EINVAL,
                "fused tile shape, which takes rows and channels in multiples of 64 and band edges in 16");
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
        loss.packed = make_tensor<byte_t, 1>(loss.capabilities, stream, {pack_bytes});
        loss.workspace = make_tensor<byte_t, 1>(loss.capabilities, stream, {workspace_bytes});
        return loss;
    }

    /** Device bytes the loss holds, independent of the world's size. */
    [[nodiscard]] std::size_t bytes() const noexcept {
        return (row_norms.numel() + column_norms.numel() + products.numel() + forces.numel() + scalars.numel() +
                self_coefficients.numel() + partials.numel()) *
                   sizeof(f32_t) +
               (gradient.numel() + transposed.numel()) * sizeof(bf16_t) + packed.numel() + workspace.numel();
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
     *  global axis. */
    void tile(nk::tensor_view<bf16_t, 2> vertices, nk::tensor_view<bf16_t, 2> subgraph, std::size_t row_base,
              std::size_t column_base, f32_t beta, f32_t bias, f32_t live, nk::stream_t const &stream) noexcept {
        launch(&prefix_norms_kernel, rows, block_threads_k, stream, subgraph, column_norms.span(), ladder);
        dim3 const transpose_grid {static_cast<std::uint32_t>(hidden / 32), static_cast<std::uint32_t>(rows / 32)};
        launch(&transpose_kernel, transpose_grid, dim3 {32, 8}, stream, subgraph, transposed.span());
        std::uint32_t const tiles = static_cast<std::uint32_t>(rows / fused_side_k);
        fused_tile_t const tile {
            row_norms.view(), column_norms.view(), gradient.span(), partials.span(), row_base, column_base, beta, bias,
            1.0f / live};
        launch(&matryoshka_tile_kernel, dim3 {tiles, tiles}, fused_warps_k * 32, stream, vertices, transposed.view(),
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
