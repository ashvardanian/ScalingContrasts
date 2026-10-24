"""Direct scale-up primitives for training, and a contrastive loss built on them, for NVIDIA and AMD GPUs.

One host's GPUs hold a replica each of a payload they reduce. On NVIDIA the replicas bind into one multicast
object, which any GPU addresses to reduce in the NVLink switch or to store into every replica at once. On AMD
every replica maps into every other GPU, which reads its peers' and stores into all of them. Barrier rounds live
on the GPUs, so consecutive collectives advance with no host writes between them.

The loss is a streamed, row-owned Matryoshka sigmoid loss: each candidate block's tile is scored at every prefix
of the ladder and folded into the query gradient before the next block is touched, so its memory does not grow
with the world, and its gradient exchanges nothing. One kernel per stage compiles for Hopper, Blackwell, CDNA 4
and CDNA 5, its matrix multiplies dispatched to each vendor's warp-level MMA.

File: scalingcontrasts.mojo
Author: Ash Vardanian
Date: October 23, 2026
"""

from std.atomic import Atomic, Ordering
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import exp, fma, sqrt
from std.memory import AddressSpace, bitcast, stack_allocation
from std.sys import inlined_assembly, is_amd_gpu, is_nvidia_gpu

from max.gpu import WARP_SIZE, barrier, block_dim, block_idx, grid_dim, lane_id, thread_idx
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext, DeviceMulticastBuffer
from max.gpu.intrinsics import Scope
from max.gpu.memory import Consistency, ReduceOp, multimem_ld_reduce, multimem_st
from max.gpu.primitives import block


# region Constants

comptime BLOCK_THREADS = 256
"""Threads per block of every collective, payload and loss kernel."""

comptime VECTOR_SCALARS = 4
"""Scalars per switch-reduced load and per multicast store, one 16-byte vector."""

comptime COUNTER_SCALARS = 4
"""Slots trailing each replica's payload: its arrival flag, its round, its count of blocks done, and padding."""

comptime MAX_MEMBERS = 8
"""The most GPUs a peer domain spans, one host's."""

comptime SUPPORTED_ACCELERATOR = (
    DeviceContext.target.is_nvidia_gpu() and DeviceContext.default_device_info.compute >= 9.0
) or DeviceContext.target.is_amd_gpu["gfx950"]() or DeviceContext.target.is_amd_gpu["gfx1250"]()
"""Whether the build targets a GPU the module compiles for: Hopper or Blackwell, CDNA 4 or CDNA 5."""

comptime Scalars = Pointer[Float32, MutUntrackedOrigin]
"""A device address of Float32 scalars, in a replica or in the multicast range."""

comptime Halves = Pointer[BFloat16, MutUntrackedOrigin]
"""A device address of BFloat16 scalars, such as embeddings or their gradient."""

comptime Counter = Pointer[UInt32, MutUntrackedOrigin]
"""A device address of one 32-bit counter."""

comptime Tally = Pointer[UInt64, MutUntrackedOrigin]
"""A device address of one 64-bit count that every GPU adds to over NVLink."""


def divide_round_up(value: Int, divisor: Int) -> Int:
    """The smallest count of `divisor`-sized pieces that covers `value`.

    Args:
        value: The quantity to cover.
        divisor: The size of one piece.

    Returns:
        The count of pieces.
    """
    return value // divisor + Int(value % divisor != 0)


def device_address[
    ScalarType: DType
](buffer: DeviceBuffer[ScalarType]) -> Pointer[Scalar[ScalarType], MutUntrackedOrigin]:
    """The device address of `buffer`'s first scalar, for kernel arguments.

    Parameters:
        ScalarType: The type of the buffer's scalars.

    Args:
        buffer: The buffer to address.

    Returns:
        The buffer's device address.
    """
    return buffer.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()


def collective_blocks(scalars: Int, multiprocessors: Int) -> Int:
    """Enough blocks for one vector per thread, at most one per multiprocessor.

    Args:
        scalars: The scalars a collective covers.
        multiprocessors: The multiprocessors of the GPU.

    Returns:
        The grid size.
    """
    return max(1, min(divide_round_up(scalars // VECTOR_SCALARS, BLOCK_THREADS), multiprocessors))


def share_of(member: Int, members: Int, scalars: Int) -> Tuple[Int, Int]:
    """The share of the first `scalars`, a multiple of four, that `member` reduces and broadcasts in an all-reduce.

    Args:
        member: The member's ordinal.
        members: The count of members.
        scalars: The leading scalars of every payload to sum.

    Returns:
        The first scalar of the share and the scalar past it.
    """
    var vectors = scalars // VECTOR_SCALARS
    var share = divide_round_up(vectors, members)
    return (min(member * share, vectors) * VECTOR_SCALARS, min(member * share + share, vectors) * VECTOR_SCALARS)


# endregion Constants


# region Multimem Primitives


@fieldwise_init
struct MultimemCounters(DevicePassable, TrivialRegisterPassable):
    """One member's counters in the multimem domain: the multicast arrival flag, and its own replica's flag, round
    and count of blocks done."""

    var multicast_flag: Counter
    """The arrival flag at its multicast address, so one reduction lands in every replica."""
    var own_flag: Counter
    """The arrival flag in this member's own replica, which it waits on."""
    var own_round: Counter
    """The arrivals this member's flag held when its current round began."""
    var own_finished: Counter
    """The count of this member's blocks done with the current collective."""
    var members: UInt32
    """The count of GPUs in the domain."""

    comptime device_type: AnyType = Self
    """The counters cross to the device unchanged."""

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Names the type in kernel argument errors.

        Returns:
            The type's name.
        """
        return "MultimemCounters"


def multimem_arrive(counters: MultimemCounters):
    """Adds one arrival for this member to every replica's flag.

    Args:
        counters: This member's counters.
    """
    inlined_assembly[
        "multimem.red.release.sys.global.add.u32 [$0], $1;",
        NoneType,
        constraints="l,r,~{memory}",
        has_side_effect=True,
    ](counters.multicast_flag, UInt32(1))


def multimem_await(counters: MultimemCounters, target: UInt32):
    """Waits until this member's own flag reaches `target`, wrap-safe past 2³² arrivals.

    Args:
        counters: This member's counters.
        target: The arrivals to wait for.
    """
    while True:
        var arrived = inlined_assembly[
            "ld.acquire.sys.global.u32 $0, [$1];",
            UInt32,
            constraints="=r,l,~{memory}",
            has_side_effect=True,
        ](counters.own_flag)
        if (arrived - target).cast[DType.int32]() >= 0:
            return


def multimem_enter(counters: MultimemCounters) -> UInt32:
    """Enters a collective round: block 0 arrives for the member, every block waits for all members, and all of them
    read the round before the last block to leave advances it.

    Args:
        counters: This member's counters.

    Returns:
        The round being entered, for `multimem_leave`.
    """
    var round = counters.own_round[]
    if thread_idx.x == 0 and block_idx.x == 0:
        multimem_arrive(counters)
    if thread_idx.x == 0:
        multimem_await(counters, round + counters.members)
    barrier()
    return round


def multimem_leave(counters: MultimemCounters, round: UInt32):
    """Leaves a collective round: the last block done arrives again, waits for every member's last block, and
    advances the round, so the kernel ends only once no member still reads any payload.

    Args:
        counters: This member's counters.
        round: The round `multimem_enter` returned.
    """
    barrier()
    if thread_idx.x != 0:
        return
    # System scope, as the blocks' multicast stores land on peers before the last block's release.
    var finished = Atomic.fetch_add[ordering=Ordering.ACQUIRE_RELEASE](counters.own_finished, UInt32(1))
    if finished != UInt32(grid_dim.x - 1):
        return
    Atomic.store[ordering=Ordering.RELAXED](counters.own_finished, UInt32(0))
    multimem_arrive(counters)
    multimem_await(counters, round + 2 * counters.members)
    counters.own_round[] = round + 2 * counters.members


def multimem_barrier_kernel(counters: MultimemCounters):
    """A barrier on its own: one arrival, then the round's last.

    Args:
        counters: This member's counters.
    """
    var round = counters.own_round[]
    multimem_arrive(counters)
    multimem_await(counters, round + counters.members)
    counters.own_round[] = round + counters.members


def multimem_reduce_scatter_kernel(
    counters: MultimemCounters, multicast: Scalars, own: Scalars, begin: Int64, end: Int64
):
    """Sets `own[i]` to the sum of every member's `payload[i]` over `[begin, end)`, one switch-reduced load per four
    scalars, between the round's entry and exit.

    Args:
        counters: This member's counters.
        multicast: The payload's multicast address.
        own: This member's own payload.
        begin: The first scalar of this member's slice, a multiple of four.
        end: The scalar past the slice, a multiple of four.
    """
    var round = multimem_enter(counters)
    var reducible = multicast.unsafe_address_space_cast[AddressSpace.GLOBAL]()
    var scalar = Int(begin) + VECTOR_SCALARS * Int(block_idx.x * block_dim.x + thread_idx.x)
    while scalar < Int(end):
        var sums = multimem_ld_reduce[
            DType.float32,
            simd_width=VECTOR_SCALARS,
            reduction=ReduceOp.ADD,
            scope=Scope.SYSTEM,
            consistency=Consistency.RELAXED,
        ](reducible.unsafe_offset(scalar))
        own.unsafe_offset(scalar).unsafe_store[alignment=16](sums)
        scalar += VECTOR_SCALARS * Int(grid_dim.x * block_dim.x)
    multimem_leave(counters, round)


def multimem_all_reduce_kernel(counters: MultimemCounters, multicast: Scalars, begin: Int64, end: Int64):
    """Sets `payload[i]` to the sum of every member's `payload[i]` over `[begin, end)` on every replica, one
    switch-reduced load and one multicast store per four scalars, between the round's entry and exit.

    Args:
        counters: This member's counters.
        multicast: The payload's multicast address.
        begin: The first scalar of this member's share, a multiple of four.
        end: The scalar past the share, a multiple of four.
    """
    var round = multimem_enter(counters)
    var reducible = multicast.unsafe_address_space_cast[AddressSpace.GLOBAL]()
    var scalar = Int(begin) + VECTOR_SCALARS * Int(block_idx.x * block_dim.x + thread_idx.x)
    while scalar < Int(end):
        var sums = multimem_ld_reduce[
            DType.float32,
            simd_width=VECTOR_SCALARS,
            reduction=ReduceOp.ADD,
            scope=Scope.SYSTEM,
            consistency=Consistency.RELAXED,
        ](reducible.unsafe_offset(scalar))
        multimem_st[DType.float32, simd_width=VECTOR_SCALARS, scope=Scope.SYSTEM, consistency=Consistency.RELAXED](
            reducible.unsafe_offset(scalar), sums
        )
        scalar += VECTOR_SCALARS * Int(grid_dim.x * block_dim.x)
    multimem_leave(counters, round)


# endregion Multimem Primitives


# region Domain


trait Collective(Movable):
    """One host's GPUs reducing a payload each: the barrier, reduce-scatter and all-reduce every fabric provides."""

    def __init__(out self, contexts: List[DeviceContext], payload_scalars: Int) raises:
        """Gives every context's GPU a zeroed replica of `payload_scalars`, a multiple of four.

        Args:
            contexts: One context per member, in device order.
            payload_scalars: The scalars of one replica ahead of its counters.

        Raises:
            When the driver cannot map the replicas or load the collectives.
        """
        ...

    def members(self) -> Int:
        """The count of GPUs in the domain.

        Returns:
            The count of members.
        """
        ...

    def replica_bytes(self) -> Int:
        """The bytes of one replica: the payload and its trailing counters.

        Returns:
            The bytes requested per replica.
        """
        ...

    def payload(self, member: Int) -> Scalars:
        """The scalars `member` contributes and receives, in its own replica.

        Args:
            member: The member's ordinal.

        Returns:
            The address of the member's payload.
        """
        ...

    def barrier(self, member: Int) raises:
        """Returns on `member`'s stream once every member has entered the same round.

        Args:
            member: The member's ordinal.

        Raises:
            When the launch fails.
        """
        ...

    def reduce_scatter(self, member: Int, shard_scalars: Int) raises:
        """In place: `member`'s slice of `shard_scalars`, a multiple of four, at `member * shard_scalars` of its
        payload becomes the sum of every payload's same slice.

        Args:
            member: The member's ordinal.
            shard_scalars: The scalars of one member's slice.

        Raises:
            When the launch fails.
        """
        ...

    def all_reduce(self, member: Int, scalars: Int) raises:
        """In place: the first `scalars`, a multiple of four, of every payload become their sum, each member reducing
        and broadcasting its own share.

        Args:
            member: The member's ordinal.
            scalars: The leading scalars of every payload to sum.

        Raises:
            When the launch fails.
        """
        ...


struct MultimemDomain(Collective):
    """One host's GPUs as an NVLink multicast domain: a replica per GPU, all bound into one multicast object that any
    of them addresses to reduce in the switch or to store into every replica at once.

    A replica holds `payload_scalars` scalars, then its arrival flag, its barrier round and its count of blocks done
    with a collective. Only its own GPU reads the round, so barriers advance with no host writes. Every call enqueues
    on the member's own context.
    """

    var contexts: List[DeviceContext]
    """One context per member, in device order."""
    var multicast: DeviceMulticastBuffer[DType.float32]
    """The multicast object every replica is bound into."""
    var replicas: List[DeviceBuffer[DType.float32]]
    """Each member's own replica."""
    var multicasts: List[DeviceBuffer[DType.float32]]
    """The multicast range as each member maps it."""
    var payload_scalars: Int
    """The scalars of one replica ahead of its counters."""
    var multiprocessors: Int
    """The multiprocessors of the first member, which caps the blocks of a collective."""

    def __init__(out self, contexts: List[DeviceContext], payload_scalars: Int) raises:
        """Binds a zeroed replica of `payload_scalars`, a multiple of four, on every context's GPU.

        Args:
            contexts: One context per member, in device order.
            payload_scalars: The scalars of one replica ahead of its counters.

        Raises:
            When the driver cannot bind the multicast object or load the collectives.
        """
        self.contexts = contexts.copy()
        self.multicast = DeviceMulticastBuffer[DType.float32](contexts.copy(), payload_scalars + COUNTER_SCALARS)
        self.replicas = List[DeviceBuffer[DType.float32]]()
        self.multicasts = List[DeviceBuffer[DType.float32]]()
        self.payload_scalars = payload_scalars
        self.multiprocessors = contexts[0].get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        for context in contexts:
            var replica = self.multicast.unicast_buffer_for(context)
            # Blocking, so no member's first arrival can land before another's zeroing.
            context.enqueue_memset(replica, Float32(0))
            context.synchronize()
            self.replicas.append(replica)
            self.multicasts.append(self.multicast.multicast_buffer_for(context))
        # A kernel's first launch loads it lazily, waiting for an idle device, so load the
        # collectives while every device is idle rather than behind a member that spins.
        for context in contexts:
            _ = context.compile_function[multimem_barrier_kernel]()
            _ = context.compile_function[multimem_reduce_scatter_kernel]()
            _ = context.compile_function[multimem_all_reduce_kernel]()

    def members(self) -> Int:
        """The count of GPUs in the domain.

        Returns:
            The count of members.
        """
        return len(self.contexts)

    def replica_bytes(self) -> Int:
        """The bytes of one replica: the payload and its trailing counters.

        Returns:
            The bytes requested per replica.
        """
        return (self.payload_scalars + COUNTER_SCALARS) * 4

    def payload(self, member: Int) -> Scalars:
        """The scalars `member` contributes and receives, in its own replica.

        Args:
            member: The member's ordinal.

        Returns:
            The address of the member's payload.
        """
        return device_address(self.replicas[member])

    def multicast_payload(self, member: Int) -> Scalars:
        """The payload's multicast address, as `member` maps it.

        Args:
            member: The member's ordinal.

        Returns:
            The multicast address of the payload.
        """
        return device_address(self.multicasts[member])

    def counters(self, member: Int) -> MultimemCounters:
        """The counters `member`'s collectives arrive on and wait at.

        Args:
            member: The member's ordinal.

        Returns:
            The member's counters.
        """
        var own = self.payload(member).unsafe_offset(self.payload_scalars).unsafe_bitcast[UInt32]()
        return MultimemCounters(
            multicast_flag=self.multicast_payload(member).unsafe_offset(self.payload_scalars).unsafe_bitcast[UInt32](),
            own_flag=own,
            own_round=own.unsafe_offset(1),
            own_finished=own.unsafe_offset(2),
            members=UInt32(self.members()),
        )

    def barrier(self, member: Int) raises:
        """Returns on `member`'s stream once every member has entered the same round.

        Args:
            member: The member's ordinal.

        Raises:
            When the launch fails.
        """
        self.contexts[member].enqueue_function[multimem_barrier_kernel](self.counters(member), grid_dim=1, block_dim=1)

    def reduce_scatter(self, member: Int, shard_scalars: Int) raises:
        """In place: `member`'s slice of `shard_scalars`, a multiple of four, at `member * shard_scalars` of its
        payload becomes the sum of every payload's same slice.

        Args:
            member: The member's ordinal.
            shard_scalars: The scalars of one member's slice.

        Raises:
            When the launch fails.
        """
        var begin = member * shard_scalars
        self.contexts[member].enqueue_function[multimem_reduce_scatter_kernel](
            self.counters(member),
            self.multicast_payload(member),
            self.payload(member),
            Int64(begin),
            Int64(begin + shard_scalars),
            grid_dim=collective_blocks(shard_scalars, self.multiprocessors),
            block_dim=BLOCK_THREADS,
        )

    def all_reduce(self, member: Int, scalars: Int) raises:
        """In place: the first `scalars`, a multiple of four, of every payload become their sum, each member reducing
        and broadcasting its own share.

        Args:
            member: The member's ordinal.
            scalars: The leading scalars of every payload to sum.

        Raises:
            When the launch fails.
        """
        var begin, end = share_of(member, self.members(), scalars)
        self.contexts[member].enqueue_function[multimem_all_reduce_kernel](
            self.counters(member),
            self.multicast_payload(member),
            Int64(begin),
            Int64(end),
            grid_dim=collective_blocks(end - begin, self.multiprocessors),
            block_dim=BLOCK_THREADS,
        )


@fieldwise_init
struct PeerCounters(DevicePassable, TrivialRegisterPassable):
    """One member's view of a peer domain: every member's replica, each of which ends in its arrival flag, its round
    and its count of blocks done."""

    var replicas: SIMD[DType.uint64, MAX_MEMBERS]
    """Every member's replica address, in member order."""
    var payload_scalars: Int64
    """The scalars of one replica ahead of its counters."""
    var member: UInt32
    """This member's ordinal."""
    var members: UInt32
    """The count of GPUs in the domain."""

    comptime device_type: AnyType = Self
    """The counters cross to the device unchanged."""

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Names the type in kernel argument errors.

        Returns:
            The type's name.
        """
        return "PeerCounters"

    def replica(self, member: Int) -> Scalars:
        """The payload of `member`'s replica, mapped into this GPU.

        Args:
            member: The member's ordinal.

        Returns:
            The address of the member's payload.
        """
        return Scalars(unsafe_from_address=Int(self.replicas[member]))

    def counter(self, member: Int, slot: Int) -> Counter:
        """One of the counters trailing `member`'s payload: its arrival flag, its round or its count of blocks done.

        Args:
            member: The member's ordinal.
            slot: The counter's index past the payload.

        Returns:
            The counter's address.
        """
        return self.replica(member).unsafe_offset(Int(self.payload_scalars) + slot).unsafe_bitcast[UInt32]()


def peer_arrive(counters: PeerCounters):
    """Adds one arrival for this member to every replica's flag, a system-scope release on each.

    Args:
        counters: This member's view of the domain.
    """
    for member in range(Int(counters.members)):
        _ = Atomic.fetch_add[ordering=Ordering.RELEASE](counters.counter(member, 0), UInt32(1))


def peer_await(counters: PeerCounters, target: UInt32):
    """Waits until this member's own flag reaches `target`, wrap-safe past 2³² arrivals.

    Args:
        counters: This member's view of the domain.
        target: The arrivals to wait for.
    """
    var own_flag = counters.counter(Int(counters.member), 0)
    while True:
        var arrived = Atomic.load[ordering=Ordering.ACQUIRE](own_flag)
        if (arrived - target).cast[DType.int32]() >= 0:
            return


def peer_enter(counters: PeerCounters) -> UInt32:
    """Enters a collective round, as `multimem_enter` does, with an arrival on every peer's flag.

    Args:
        counters: This member's view of the domain.

    Returns:
        The round being entered, for `peer_leave`.
    """
    var round = counters.counter(Int(counters.member), 1)[]
    if thread_idx.x == 0 and block_idx.x == 0:
        peer_arrive(counters)
    if thread_idx.x == 0:
        peer_await(counters, round + counters.members)
    barrier()
    return round


def peer_leave(counters: PeerCounters, round: UInt32):
    """Leaves a collective round, as `multimem_leave` does, with an arrival on every peer's flag.

    Args:
        counters: This member's view of the domain.
        round: The round `peer_enter` returned.
    """
    barrier()
    if thread_idx.x != 0:
        return
    var own_finished = counters.counter(Int(counters.member), 2)
    var finished = Atomic.fetch_add[ordering=Ordering.ACQUIRE_RELEASE](own_finished, UInt32(1))
    if finished != UInt32(grid_dim.x - 1):
        return
    Atomic.store[ordering=Ordering.RELAXED](own_finished, UInt32(0))
    peer_arrive(counters)
    peer_await(counters, round + 2 * counters.members)
    counters.counter(Int(counters.member), 1)[] = round + 2 * counters.members


def peer_barrier_kernel(counters: PeerCounters):
    """A barrier on its own: one arrival, then the round's last.

    Args:
        counters: This member's view of the domain.
    """
    var own_round = counters.counter(Int(counters.member), 1)
    var round = own_round[]
    peer_arrive(counters)
    peer_await(counters, round + counters.members)
    own_round[] = round + counters.members


def peer_sums(counters: PeerCounters, scalar: Int) -> SIMD[DType.float32, VECTOR_SCALARS]:
    """The sum of four scalars at `scalar` over every member's replica, read in member order.

    Args:
        counters: This member's view of the domain.
        scalar: The first of the four scalars, a multiple of four.

    Returns:
        The four sums.
    """
    var sums = SIMD[DType.float32, VECTOR_SCALARS](0)
    for member in range(Int(counters.members)):
        sums += counters.replica(member).unsafe_offset(scalar).unsafe_load[width=VECTOR_SCALARS, alignment=16]()
    return sums


def peer_reduce_scatter_kernel(counters: PeerCounters, begin: Int64, end: Int64):
    """Sets this member's `payload[i]` to the sum of every member's `payload[i]` over `[begin, end)`, reading every
    peer's replica, between the round's entry and exit.

    Args:
        counters: This member's view of the domain.
        begin: The first scalar of this member's slice, a multiple of four.
        end: The scalar past the slice, a multiple of four.
    """
    var round = peer_enter(counters)
    var own = counters.replica(Int(counters.member))
    var scalar = Int(begin) + VECTOR_SCALARS * Int(block_idx.x * block_dim.x + thread_idx.x)
    while scalar < Int(end):
        own.unsafe_offset(scalar).unsafe_store[alignment=16](peer_sums(counters, scalar))
        scalar += VECTOR_SCALARS * Int(grid_dim.x * block_dim.x)
    peer_leave(counters, round)


def peer_all_reduce_kernel(counters: PeerCounters, begin: Int64, end: Int64):
    """Sums this member's share `[begin, end)` by reading every peer's replica, then stores the sums into all of
    them, between the round's entry and exit.

    Args:
        counters: This member's view of the domain.
        begin: The first scalar of this member's share, a multiple of four.
        end: The scalar past the share, a multiple of four.
    """
    var round = peer_enter(counters)
    var scalar = Int(begin) + VECTOR_SCALARS * Int(block_idx.x * block_dim.x + thread_idx.x)
    while scalar < Int(end):
        var sums = peer_sums(counters, scalar)
        for member in range(Int(counters.members)):
            counters.replica(member).unsafe_offset(scalar).unsafe_store[alignment=16](sums)
        scalar += VECTOR_SCALARS * Int(grid_dim.x * block_dim.x)
    peer_leave(counters, round)


struct PeerDomain(Collective):
    """One host's GPUs as a peer-to-peer mesh: a replica per GPU, each mapped into every other GPU, for fabrics such
    as AMD's Infinity Fabric that move data but do not reduce it.

    A replica holds `payload_scalars` scalars, then its arrival flag, its barrier round and its count of blocks done
    with a collective, as in the multimem domain. Arrivals are system-scope atomic adds on every peer's flag. Every
    call enqueues on the member's own context, and peer access must be enabled between all of them.
    """

    var contexts: List[DeviceContext]
    """One context per member, in device order."""
    var replicas: List[DeviceBuffer[DType.float32]]
    """Each member's own replica."""
    var payload_scalars: Int
    """The scalars of one replica ahead of its counters."""
    var multiprocessors: Int
    """The multiprocessors of the first member, which caps the blocks of a collective."""

    def __init__(out self, contexts: List[DeviceContext], payload_scalars: Int) raises:
        """Allocates a zeroed replica of `payload_scalars`, a multiple of four, on every context's GPU.

        Args:
            contexts: One context per member, in device order.
            payload_scalars: The scalars of one replica ahead of its counters.

        Raises:
            When there are more members than a domain spans, or an allocation or a kernel load fails.
        """
        if len(contexts) > MAX_MEMBERS:
            raise Error(t"A peer domain spans at most {MAX_MEMBERS} GPUs, not {len(contexts)}")
        self.contexts = contexts.copy()
        self.replicas = List[DeviceBuffer[DType.float32]]()
        self.payload_scalars = payload_scalars
        self.multiprocessors = contexts[0].get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        for context in contexts:
            var replica = context.enqueue_create_buffer[DType.float32](payload_scalars + COUNTER_SCALARS)
            # Blocking, so no member's first arrival can land before another's zeroing.
            context.enqueue_memset(replica, Float32(0))
            context.synchronize()
            self.replicas.append(replica)
        # Loaded while every device is idle, as in the multimem domain.
        for context in contexts:
            _ = context.compile_function[peer_barrier_kernel]()
            _ = context.compile_function[peer_reduce_scatter_kernel]()
            _ = context.compile_function[peer_all_reduce_kernel]()

    def members(self) -> Int:
        """The count of GPUs in the domain.

        Returns:
            The count of members.
        """
        return len(self.contexts)

    def replica_bytes(self) -> Int:
        """The bytes of one replica: the payload and its trailing counters.

        Returns:
            The bytes requested per replica.
        """
        return (self.payload_scalars + COUNTER_SCALARS) * 4

    def payload(self, member: Int) -> Scalars:
        """The scalars `member` contributes and receives, in its own replica.

        Args:
            member: The member's ordinal.

        Returns:
            The address of the member's payload.
        """
        return device_address(self.replicas[member])

    def counters(self, member: Int) -> PeerCounters:
        """`member`'s view of the domain, which its collectives arrive on and wait at.

        Args:
            member: The member's ordinal.

        Returns:
            The member's counters.
        """
        var replicas = SIMD[DType.uint64, MAX_MEMBERS](0)
        for peer in range(self.members()):
            replicas[peer] = UInt64(Int(self.payload(peer)))
        return PeerCounters(
            replicas=replicas,
            payload_scalars=Int64(self.payload_scalars),
            member=UInt32(member),
            members=UInt32(self.members()),
        )

    def barrier(self, member: Int) raises:
        """Returns on `member`'s stream once every member has entered the same round.

        Args:
            member: The member's ordinal.

        Raises:
            When the launch fails.
        """
        self.contexts[member].enqueue_function[peer_barrier_kernel](self.counters(member), grid_dim=1, block_dim=1)

    def reduce_scatter(self, member: Int, shard_scalars: Int) raises:
        """In place: `member`'s slice of `shard_scalars`, a multiple of four, at `member * shard_scalars` of its
        payload becomes the sum of every payload's same slice.

        Args:
            member: The member's ordinal.
            shard_scalars: The scalars of one member's slice.

        Raises:
            When the launch fails.
        """
        var begin = member * shard_scalars
        self.contexts[member].enqueue_function[peer_reduce_scatter_kernel](
            self.counters(member),
            Int64(begin),
            Int64(begin + shard_scalars),
            grid_dim=collective_blocks(shard_scalars, self.multiprocessors),
            block_dim=BLOCK_THREADS,
        )

    def all_reduce(self, member: Int, scalars: Int) raises:
        """In place: the first `scalars`, a multiple of four, of every payload become their sum, each member reducing
        and broadcasting its own share.

        Args:
            member: The member's ordinal.
            scalars: The leading scalars of every payload to sum.

        Raises:
            When the launch fails.
        """
        var begin, end = share_of(member, self.members(), scalars)
        self.contexts[member].enqueue_function[peer_all_reduce_kernel](
            self.counters(member),
            Int64(begin),
            Int64(end),
            grid_dim=collective_blocks(end - begin, self.multiprocessors),
            block_dim=BLOCK_THREADS,
        )


comptime Domain: Collective = MultimemDomain if DeviceContext.target.is_nvidia_gpu() else PeerDomain
"""The domain of the targeted fabric: NVLink switches reduce in the network, AMD meshes only move data."""

# endregion Domain


# region Matryoshka

comptime MAX_PREFIXES = 8
"""The most prefixes a ladder holds."""

comptime TILE_SIDE = 64
"""Query rows and candidate columns of a forward tile, and the rows and channels of a backward tile."""

comptime TILE_DEPTH = 32
"""Channels, or candidates in the backward tile, per shared-memory slab."""

comptime TILE_CELLS = TILE_SIDE * TILE_SIDE // BLOCK_THREADS
"""Cells of a tile each thread accumulates."""

comptime SLAB_STRIDE = TILE_DEPTH + 8
"""BFloat16 scalars between the rows of a slab, padded to stagger shared-memory banks."""

comptime SHEET_STRIDE = TILE_SIDE + 1
"""Float32 scalars between the rows of the sheet a tile's rows are summed on."""

comptime COSINE_FLOOR: Float32 = 1.0e-2
"""The norm product below which a cosine reads as zero."""

comptime INVERSE_COSINE_FLOOR: Float32 = 1.0 / COSINE_FLOOR
"""The largest inverse norm product a cosine scales by."""

comptime LOG1P_LEADING: Float32 = 5.185916197e-03
"""The leading coefficient of the `log1p(x)/x` fit."""

comptime LOG1P_COEFFICIENTS = SIMD[DType.float32, 8](
    -2.920989730e-02,
    7.753973263e-02,
    -1.358388136e-01,
    1.905592322e-01,
    -2.482563861e-01,
    3.331600901e-01,
    -4.999925577e-01,
    9.999999208e-01,
)
"""The remaining coefficients of the `log1p(x)/x` fit, highest degree first."""

comptime Cells = SIMD[DType.float32, TILE_CELLS]
"""One thread's Float32 cells of a tile."""

comptime Coordinates = SIMD[DType.int32, TILE_CELLS]
"""The rows or the columns, within a tile, of one thread's cells."""

comptime SharedHalves = Pointer[BFloat16, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
"""A shared-memory address of BFloat16 scalars, one slab."""

comptime SharedScalars = Pointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
"""A shared-memory address of Float32 scalars, one sheet."""


@fieldwise_init
struct Matryoshka(DevicePassable, TrivialRegisterPassable):
    """The Matryoshka ladder: ascending prefix widths, band `b` covering the channels `[begin(b), end(b))`."""

    var ends: SIMD[DType.uint32, MAX_PREFIXES]
    """The channel past each prefix, narrowest first."""
    var count: UInt32
    """The count of prefixes."""

    comptime device_type: AnyType = Self
    """The ladder crosses to the device unchanged."""

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Names the type in kernel argument errors.

        Returns:
            The type's name.
        """
        return "Matryoshka"

    @staticmethod
    def halvings(hidden: Int, count: Int) -> Self:
        """`count` halvings of `hidden`, narrowest first.

        Args:
            hidden: The widest prefix, every channel.
            count: The count of prefixes, at most eight.

        Returns:
            The ladder.
        """
        var ends = SIMD[DType.uint32, MAX_PREFIXES](0)
        for prefix in range(count):
            ends[prefix] = UInt32(hidden >> (count - 1 - prefix))
        return Self(ends=ends, count=UInt32(count))

    def prefixes(self) -> Int:
        """The count of prefixes.

        Returns:
            The count of prefixes.
        """
        return Int(self.count)

    def begin(self, band: Int) -> Int:
        """The first channel of `band`.

        Args:
            band: The band's ordinal.

        Returns:
            The channel.
        """
        return 0 if band == 0 else Int(self.ends[band - 1])

    def end(self, band: Int) -> Int:
        """The channel past `band`, the width of its prefix.

        Args:
            band: The band's ordinal.

        Returns:
            The channel.
        """
        return Int(self.ends[band])


# endregion Matryoshka


# region Matrix Multiply Fragments

comptime MMA_ROWS = 16
"""Rows of one warp-level MMA."""

comptime MMA_COLUMNS = 8 if is_nvidia_gpu() else 16
"""Columns of one warp-level MMA: `mma.sync.m16n8k16` on NVIDIA, MFMA and WMMA `16x16x32` on AMD."""

comptime MMA_DEPTH = 16 if is_nvidia_gpu() else 32
"""The depth one warp-level MMA reduces over."""

comptime OPERAND_RUN = 2 if is_nvidia_gpu() else 8
"""Consecutive depths of an operand one lane holds together."""

comptime A_SCALARS = MMA_ROWS * MMA_DEPTH // WARP_SIZE
"""BFloat16 scalars of the left operand per lane."""

comptime B_SCALARS = MMA_COLUMNS * MMA_DEPTH // WARP_SIZE
"""BFloat16 scalars of the right operand per lane."""

comptime C_SCALARS = MMA_ROWS * MMA_COLUMNS // WARP_SIZE
"""Float32 accumulators per lane."""

comptime MMA_TILES_ACROSS = TILE_SIDE // MMA_COLUMNS
"""MMA tiles across one row of a tile."""

comptime WARP_TILES = TILE_SIDE // MMA_ROWS * MMA_TILES_ACROSS // (BLOCK_THREADS // WARP_SIZE)
"""MMA tiles each warp accumulates."""


def left_run(lane: Int, run: Int) -> Tuple[Int, Int]:
    """The row and the first depth of a lane's `run` of the left operand.

    On NVIDIA the runs alternate between rows `g` and `g + 8` of the lane's group `g`. On AMD the lanes' sixteen rows
    repeat across depth groups, which CDNA 5 splits into two halves sixteen apart.

    Args:
        lane: The lane in its warp.
        run: The run of `OPERAND_RUN` consecutive depths.

    Returns:
        The row within the MMA, and the first depth within it.
    """
    comptime if is_nvidia_gpu():
        return (lane // 4 + 8 * (run % 2), 2 * (lane % 4) + 8 * (run // 2))
    else:
        return (lane % 16, 8 * (lane // 16) + 16 * run)


def right_run(lane: Int, run: Int) -> Tuple[Int, Int]:
    """The column and the first depth of a lane's `run` of the right operand.

    Args:
        lane: The lane in its warp.
        run: The run of `OPERAND_RUN` consecutive depths.

    Returns:
        The column within the MMA, and the first depth within it.
    """
    comptime if is_nvidia_gpu():
        return (lane // 4, 2 * (lane % 4) + 8 * run)
    else:
        return (lane % 16, 8 * (lane // 16) + 16 * run)


def accumulator_cell(lane: Int, cell: Int) -> Tuple[Int, Int]:
    """The row and column within an MMA of a lane's accumulator `cell`.

    Args:
        lane: The lane in its warp.
        cell: The accumulator's index in the lane.

    Returns:
        The row and the column.
    """
    comptime if is_nvidia_gpu():
        return (lane // 4 + 8 * (cell // 2), 2 * (lane % 4) + cell % 2)
    else:
        return (C_SCALARS * (lane // 16) + cell, lane % 16)


def mma_tile_origin(fragment: Int) -> Tuple[Int, Int]:
    """The first row and column, within a tile, of this warp's MMA tile `fragment`.

    Args:
        fragment: The MMA tile among this warp's.

    Returns:
        The row and the column.
    """
    var mma_tile = Int(thread_idx.x) // WARP_SIZE * WARP_TILES + fragment
    return (mma_tile // MMA_TILES_ACROSS * MMA_ROWS, mma_tile % MMA_TILES_ACROSS * MMA_COLUMNS)


def load_operand[
    Scalars: Int, RunOrigin: def(Int, Int) thin -> Tuple[Int, Int]
](slab: SharedHalves, first_row: Int, first_depth: Int, lane: Int) -> SIMD[DType.bfloat16, Scalars]:
    """Gathers a lane's operand of one MMA from a slab whose rows are consecutive in depth.

    Parameters:
        Scalars: The scalars of the operand per lane.
        RunOrigin: The row and first depth of each of a lane's runs, `left_run` or `right_run`.

    Args:
        slab: The slab, `TILE_SIDE` rows of `SLAB_STRIDE`.
        first_row: The MMA's first row, or first column, within the slab.
        first_depth: The MMA's first depth within the slab.
        lane: The lane in its warp.

    Returns:
        The lane's operand.
    """
    var fragment = SIMD[DType.bfloat16, Scalars]()
    comptime for run in range(Scalars // OPERAND_RUN):
        var row, depth = RunOrigin(lane, run)
        var offset = (first_row + row) * SLAB_STRIDE + first_depth + depth
        fragment = fragment.insert[offset=run * OPERAND_RUN](
            slab.unsafe_offset(offset).unsafe_load[width=OPERAND_RUN, alignment=OPERAND_RUN * 2]()
        )
    return fragment


def cell_coordinates() -> Tuple[Coordinates, Coordinates]:
    """The rows and columns, within a tile, of this thread's cells.

    Returns:
        The rows and the columns.
    """
    var lane = Int(lane_id())
    var rows = Coordinates(0)
    var columns = Coordinates(0)
    comptime for fragment in range(WARP_TILES):
        var first_row, first_column = mma_tile_origin(fragment)
        comptime for cell in range(C_SCALARS):
            var row, column = accumulator_cell(lane, cell)
            rows[fragment * C_SCALARS + cell] = Int32(first_row + row)
            columns[fragment * C_SCALARS + cell] = Int32(first_column + column)
    return (rows, columns)


def stage_slab(slab: SharedHalves, source: Halves, stride: Int, first_row: Int, first_depth: Int):
    """Copies `TILE_SIDE` rows of `TILE_DEPTH` consecutive scalars into a slab, eight per thread.

    Args:
        slab: The slab to fill.
        source: The row-major matrix to copy from.
        stride: The scalars between the matrix's rows.
        first_row: The first row to copy.
        first_depth: The first column of every row to copy.
    """
    comptime parts = TILE_DEPTH // 8
    var row = Int(thread_idx.x) // parts
    var part = Int(thread_idx.x) % parts
    var values = source.unsafe_offset((first_row + row) * stride + first_depth + part * 8).unsafe_load[
        width=8, alignment=16
    ]()
    slab.unsafe_offset(row * SLAB_STRIDE + part * 8).unsafe_store[alignment=16](values)


def multiply_slab(mut accumulator: Cells, rows_slab: SharedHalves, columns_slab: SharedHalves):
    """Adds the products of two staged slabs, rows by columns over their depth, into this thread's cells.

    Args:
        accumulator: This thread's cells.
        rows_slab: The rows of the left operand, consecutive in depth.
        columns_slab: The columns of the right operand, consecutive in depth.
    """
    var lane = Int(lane_id())
    comptime for step in range(TILE_DEPTH // MMA_DEPTH):
        comptime for fragment in range(WARP_TILES):
            var first_row, first_column = mma_tile_origin(fragment)
            var left = load_operand[A_SCALARS, left_run](rows_slab, first_row, step * MMA_DEPTH, lane)
            var right = load_operand[B_SCALARS, right_run](columns_slab, first_column, step * MMA_DEPTH, lane)
            var sums = SIMD[DType.float32, C_SCALARS](0)
            mma(sums, left, right, accumulator.slice[C_SCALARS, offset=fragment * C_SCALARS]())
            accumulator = accumulator.insert[offset=fragment * C_SCALARS](sums)


def fold_rows(cells: Cells, sheet: SharedScalars, rows: Coordinates, columns: Coordinates, destination: Scalars):
    """Sums a tile's cells across each row, in column order, into `destination[row]`.

    Args:
        cells: This thread's cells.
        sheet: Shared memory for `TILE_SIDE` rows of `SHEET_STRIDE`.
        rows: The rows of this thread's cells.
        columns: The columns of this thread's cells.
        destination: The tile's first row of the quantity summed.
    """
    barrier()
    comptime for cell in range(TILE_CELLS):
        sheet[unsafe_offset=Int(rows[cell]) * SHEET_STRIDE + Int(columns[cell])] = cells[cell]
    barrier()
    if Int(thread_idx.x) < TILE_SIDE:
        var total = Float32(0)
        for column in range(TILE_SIDE):
            total += sheet[unsafe_offset=Int(thread_idx.x) * SHEET_STRIDE + column]
        destination[unsafe_offset=Int(thread_idx.x)] = total


# endregion Matrix Multiply Fragments


# region Loss Kernels


def log1p_unit(x: Cells) -> Cells:
    """`log(1+x)` for `x` in `[0, 1]`, as `x` times a degree-8 Chebyshev least-squares fit of `log1p(x)/x` on that
    interval, within 3.2e-7 of it relative in Float32.

    Args:
        x: The arguments.

    Returns:
        The logarithms.
    """
    var fit = Cells(LOG1P_LEADING)
    comptime for index in range(8):
        fit = fma(fit, x, Cells(LOG1P_COEFFICIENTS[index]))
    return fit * x


@fieldwise_init
struct ForwardTile(DevicePassable, TrivialRegisterPassable):
    """What every forward tile of one candidate block shares."""

    var queries: Halves
    """This rank's queries, `[rows, hidden]`."""
    var candidates: Halves
    """The staged candidate block, `[rows, hidden]`."""
    var query_inverse_norms: Scalars
    """The queries' inverse prefix norms, `[prefixes, rows]`."""
    var candidate_inverse_norms: Scalars
    """The candidates' inverse prefix norms, `[prefixes, rows]`."""
    var similarity_gradients: Halves
    """Each band's suffix-summed gradient of the raw dots, `[prefixes, rows, rows]`."""
    var row_partials: Scalars
    """Each row's summands per column tile, `[3 + prefixes, rows / TILE_SIDE, rows]`."""
    var rows: Int64
    """The rows of the queries and of the candidate block."""
    var hidden: Int64
    """The channels of every embedding."""
    var query_base: Int64
    """The first query's index on the world's axis."""
    var candidate_base: Int64
    """The first candidate's index on the world's axis."""
    var beta: Float32
    """The logit scale."""
    var bias: Float32
    """The logit bias."""
    var weight: Float32
    """Each cell's weight, one over the world's candidates."""

    comptime device_type: AnyType = Self
    """The tile crosses to the device unchanged."""

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Names the type in kernel argument errors.

        Returns:
            The type's name.
        """
        return "ForwardTile"


def transpose_kernel(source: Halves, destination: Halves, rows: Int64, columns: Int64):
    """Writes the `[columns, rows]` transpose of `source`, `[rows, columns]`, one 32-square per block.

    Args:
        source: The matrix to transpose.
        destination: The transpose.
        rows: The rows of the source, a multiple of 32.
        columns: The columns of the source, a multiple of 32.
    """
    var tile = stack_allocation[32 * 33, DType.bfloat16, address_space=AddressSpace.SHARED]()
    var column_base = Int(block_idx.x) * 32
    var row_base = Int(block_idx.y) * 32
    var step = Int(thread_idx.y)
    while step < 32:
        tile[unsafe_offset=step * 33 + Int(thread_idx.x)] = source[
            unsafe_offset=(row_base + step) * Int(columns) + column_base + Int(thread_idx.x)
        ]
        step += Int(block_dim.y)
    barrier()
    step = Int(thread_idx.y)
    while step < 32:
        destination[unsafe_offset=(column_base + step) * Int(rows) + row_base + Int(thread_idx.x)] = tile[
            unsafe_offset=Int(thread_idx.x) * 33 + step
        ]
        step += Int(block_dim.y)


def prefix_inverse_norms_kernel(embeddings: Halves, norms: Scalars, rows: Int64, hidden: Int64, ladder: Matryoshka):
    """Writes `norms[band, row]`, the inverse norm of every prefix of every row, one row per block.

    Args:
        embeddings: The rows, `[rows, hidden]`.
        norms: The inverse norms, `[prefixes, rows]`.
        rows: The count of rows.
        hidden: The channels of every row.
        ladder: The prefixes.
    """
    var row = Int(block_idx.x)
    var running = Float32(0)
    for band in range(ladder.prefixes()):
        var partial = Float32(0)
        var channel = ladder.begin(band) + Int(thread_idx.x)
        while channel < ladder.end(band):
            var value = embeddings[unsafe_offset=row * Int(hidden) + channel].cast[DType.float32]()
            partial += value * value
            channel += Int(block_dim.x)
        barrier()
        running += block.sum[block_size=BLOCK_THREADS, broadcast=False](partial)
        if thread_idx.x == 0:
            norms[unsafe_offset=band * Int(rows) + row] = 1 / sqrt(running)


def forward_tile_kernel(tile: ForwardTile, ladder: Matryoshka):
    """One `[64, 64]` tile of queries against candidates, scored at every Matryoshka prefix.

    The sweep over ascending channels snapshots each band's prefix dots in registers as its last slab lands. The
    epilogue then scores the bands widest first, as the CUDA sweep does, adding each band's raw gradient to a running
    suffix written as that band's BFloat16 similarity gradient. Row summands land in the tile's
    `[3 + prefixes, rows / 64, rows]` partials, which `fold_partials_kernel` adds in tile order.

    Args:
        tile: What every tile of this candidate block shares.
        ladder: The prefixes.
    """
    var query_slab = stack_allocation[
        TILE_SIDE * SLAB_STRIDE, DType.bfloat16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var candidate_slab = stack_allocation[
        TILE_SIDE * SLAB_STRIDE, DType.bfloat16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var sheet = stack_allocation[TILE_SIDE * SHEET_STRIDE, DType.float32, address_space=AddressSpace.SHARED]()
    var rows = Int(tile.rows)
    var hidden = Int(tile.hidden)
    var row_base = Int(block_idx.y) * TILE_SIDE
    var column_base = Int(block_idx.x) * TILE_SIDE

    var accumulator = Cells(0)
    var dots = Array[Cells, MAX_PREFIXES](fill=Cells(0))
    for slab in range(hidden // TILE_DEPTH):
        barrier()
        stage_slab(query_slab, tile.queries, hidden, row_base, slab * TILE_DEPTH)
        stage_slab(candidate_slab, tile.candidates, hidden, column_base, slab * TILE_DEPTH)
        barrier()
        multiply_slab(accumulator, query_slab, candidate_slab)
        comptime for band in range(MAX_PREFIXES):
            if band < ladder.prefixes() and ladder.end(band) == (slab + 1) * TILE_DEPTH:
                dots[band] = accumulator

    var cell_rows, cell_columns = cell_coordinates()
    var query_rows = cell_rows.cast[DType.int64]() + Int64(row_base) + tile.query_base
    var candidate_columns = cell_columns.cast[DType.int64]() + Int64(column_base) + tile.candidate_base
    var own_copy = query_rows.eq(candidate_columns)
    var partials_stride = rows // TILE_SIDE * rows
    var partials = tile.row_partials.unsafe_offset(Int(block_idx.x) * rows + row_base)
    var loss = Cells(0)
    var bias_grad = Cells(0)
    var log_beta_grad = Cells(0)
    var suffix = Cells(0)
    comptime for step in range(MAX_PREFIXES):
        comptime band = MAX_PREFIXES - 1 - step
        if band < ladder.prefixes():
            var query_norms = Cells(0)
            var candidate_norms = Cells(0)
            comptime for cell in range(TILE_CELLS):
                query_norms[cell] = tile.query_inverse_norms[
                    unsafe_offset=band * rows + row_base + Int(cell_rows[cell])
                ]
                candidate_norms[cell] = tile.candidate_inverse_norms[
                    unsafe_offset=band * rows + column_base + Int(cell_columns[cell])
                ]
            # The inverse of the norms' product, clamped where they read as a zero cosine.
            var inverse_denominator = min(query_norms * candidate_norms, Cells(INVERSE_COSINE_FLOOR))
            var cosine = inverse_denominator.lt(Cells(INVERSE_COSINE_FLOOR)).select(
                dots[band] * inverse_denominator, Cells(0)
            )
            # A query's cosine with its own copy is one, whatever its raw dot rounds to.
            var logit = tile.beta * own_copy.select(Cells(1), cosine)
            var target = own_copy.select(Cells(1), Cells(-1))
            var signed_logit = target * (logit + tile.bias)
            var decay = exp(-abs(signed_logit))
            var complement = signed_logit.ge(Cells(0)).select(decay, Cells(1)) / (1 + decay)
            var cell_grad = -tile.weight * target * complement
            loss += tile.weight * (max(-signed_logit, Cells(0)) + log1p_unit(decay))
            bias_grad += cell_grad
            log_beta_grad += cell_grad * logit
            # The mirror cell (j, i) moves v_i the same way, and the own copy moves nothing.
            var raw_grad = own_copy.select(Cells(0), 2 * tile.beta * cell_grad * inverse_denominator)
            suffix += raw_grad
            comptime for cell in range(TILE_CELLS):
                tile.similarity_gradients[
                    unsafe_offset=(band * rows + row_base + Int(cell_rows[cell])) * rows
                    + column_base
                    + Int(cell_columns[cell])
                ] = suffix[cell].cast[DType.bfloat16]()
            fold_rows(
                raw_grad * dots[band],
                sheet,
                cell_rows,
                cell_columns,
                partials.unsafe_offset((3 + band) * partials_stride),
            )
    fold_rows(loss, sheet, cell_rows, cell_columns, partials)
    fold_rows(bias_grad, sheet, cell_rows, cell_columns, partials.unsafe_offset(partials_stride))
    fold_rows(log_beta_grad, sheet, cell_rows, cell_columns, partials.unsafe_offset(2 * partials_stride))


def backward_tile_kernel(
    similarity_gradients: Halves,
    transposed_candidates: Halves,
    query_gradient: Scalars,
    rows: Int64,
    hidden: Int64,
    ladder: Matryoshka,
):
    """Adds one `[64, 64]` tile of the candidates' pull on this rank's queries into `query_gradient`: the band's
    BFloat16 similarity gradients times the candidates' channels in that band, over every candidate.

    Args:
        similarity_gradients: Each band's suffix-summed gradient of the raw dots, `[prefixes, rows, rows]`.
        transposed_candidates: The candidate block's channels, `[hidden, rows]`.
        query_gradient: The Float32 gradient of the queries, `[rows, hidden]`.
        rows: The rows of the queries and of the candidate block.
        hidden: The channels of every embedding.
        ladder: The prefixes, whose edges fall on multiples of 64.
    """
    var gradient_slab = stack_allocation[
        TILE_SIDE * SLAB_STRIDE, DType.bfloat16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var channel_slab = stack_allocation[
        TILE_SIDE * SLAB_STRIDE, DType.bfloat16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var channel_base = Int(block_idx.x) * TILE_SIDE
    var row_base = Int(block_idx.y) * TILE_SIDE
    var band = 0
    while ladder.end(band) <= channel_base:
        band += 1
    var gradients = similarity_gradients.unsafe_offset(band * Int(rows) * Int(rows))

    var accumulator = Cells(0)
    for position in range(Int(rows) // TILE_DEPTH):
        barrier()
        stage_slab(gradient_slab, gradients, Int(rows), row_base, position * TILE_DEPTH)
        stage_slab(channel_slab, transposed_candidates, Int(rows), channel_base, position * TILE_DEPTH)
        barrier()
        multiply_slab(accumulator, gradient_slab, channel_slab)

    var cell_rows, cell_columns = cell_coordinates()
    comptime for cell in range(TILE_CELLS):
        var offset = (row_base + Int(cell_rows[cell])) * Int(hidden) + channel_base + Int(cell_columns[cell])
        query_gradient[unsafe_offset=offset] += accumulator[cell]


def fold_partials_kernel(
    row_partials: Scalars, summand_rows: Scalars, self_coefficients: Scalars, quantities: Int64, rows: Int64
):
    """Adds each row's column-tile `row_partials`, in tile order, to its `summand_rows` and `self_coefficients`.

    Args:
        row_partials: Each row's summands per column tile, `[quantities, rows / 64, rows]`.
        summand_rows: Each row's loss, bias gradient and log-β gradient, `[3, rows]`.
        self_coefficients: Each row's coefficient of its own embedding, `[prefixes, rows]`.
        quantities: Three summands, then one coefficient per prefix.
        rows: The count of rows.
    """
    var row = Int(block_idx.x * block_dim.x + thread_idx.x)
    if row >= Int(rows):
        return
    var tiles = Int(rows) // TILE_SIDE
    for quantity in range(Int(quantities)):
        var total = Float32(0)
        for tile in range(tiles):
            total += row_partials[unsafe_offset=(quantity * tiles + tile) * Int(rows) + row]
        if quantity < 3:
            summand_rows[unsafe_offset=quantity * Int(rows) + row] += total
        else:
            self_coefficients[unsafe_offset=(quantity - 3) * Int(rows) + row] += total


def finalize_kernel(
    queries: Halves,
    inverse_norms: Scalars,
    self_coefficients: Scalars,
    query_gradient: Scalars,
    gradient: Halves,
    rows: Int64,
    hidden: Int64,
    ladder: Matryoshka,
):
    """Folds the cosine's self term into the query gradient, then writes its BFloat16 copy, one row per block.

    Args:
        queries: This rank's queries, `[rows, hidden]`.
        inverse_norms: The queries' inverse prefix norms, `[prefixes, rows]`.
        self_coefficients: Each row's coefficient of its own embedding, `[prefixes, rows]`.
        query_gradient: The Float32 gradient of the queries, `[rows, hidden]`.
        gradient: The BFloat16 gradient written, `[rows, hidden]`.
        rows: The count of rows.
        hidden: The channels of every row.
        ladder: The prefixes.
    """
    var row = Int(block_idx.x)
    var channel = Int(thread_idx.x)
    while channel < Int(hidden):
        var fold = Float32(0)
        for band in range(ladder.prefixes()):
            if channel < ladder.end(band):
                var inverse_norm = inverse_norms[unsafe_offset=band * Int(rows) + row]
                fold += self_coefficients[unsafe_offset=band * Int(rows) + row] * min(
                    inverse_norm * inverse_norm, INVERSE_COSINE_FLOOR
                )
        var offset = row * Int(hidden) + channel
        gradient[unsafe_offset=offset] = (
            query_gradient[unsafe_offset=offset] - fold * queries[unsafe_offset=offset].cast[DType.float32]()
        ).cast[DType.bfloat16]()
        channel += Int(block_dim.x)


def sum_summands_kernel(summand_rows: Scalars, summands: Scalars, rows: Int64):
    """Writes the rank's loss, bias gradient and log-β gradient, each summed over its rows in one block.

    Args:
        summand_rows: Each row's loss, bias gradient and log-β gradient, `[3, rows]`.
        summands: The three sums, a domain payload or any device scalars.
        rows: The count of rows.
    """
    for which in range(3):
        var partial = Float32(0)
        var row = Int(thread_idx.x)
        while row < Int(rows):
            partial += summand_rows[unsafe_offset=which * Int(rows) + row]
            row += Int(block_dim.x)
        barrier()
        var total = block.sum[block_size=BLOCK_THREADS, broadcast=False](partial)
        if thread_idx.x == 0:
            summands[unsafe_offset=which] = total


# endregion Loss Kernels


# region Streamed Loss


struct StreamedLoss(Copyable, Movable):
    """A rank's Matryoshka sigmoid contrastive loss, streamed one candidate block at a time and row-owned.

    Every query is a row against all candidates of the world, at every prefix of the ladder, scored by SigLIP's
    sigmoid BCE with weight one over the world's candidates: its own copy is the positive, every other candidate a
    negative. Each candidate block's `[prefixes, rows, rows]` tile is scored and folded into the rank's query
    gradient before the next is touched, so the scratch is one tile whatever the world. Cell `(i, j)` and its mirror
    `(j, i)` share the cosine, target and weight, so the owner of `i` differentiates both, and the gradient exchanges
    nothing.

    Call `begin`, then `score` once per candidate block in any order, then `finish`, which writes the rank's loss,
    bias-gradient and log-β-gradient summands, which add up across ranks. Every call enqueues on the loss's context.
    """

    var context: DeviceContext
    """The context of the GPU whose queries the loss scores."""
    var ladder: Matryoshka
    """The prefixes."""
    var rows: Int
    """The rows of the queries and of every candidate block."""
    var hidden: Int
    """The channels of every embedding."""
    var query_inverse_norms: DeviceBuffer[DType.float32]
    """The queries' inverse prefix norms, `[prefixes, rows]`."""
    var candidate_inverse_norms: DeviceBuffer[DType.float32]
    """The current candidate block's inverse prefix norms, `[prefixes, rows]`."""
    var query_gradient: DeviceBuffer[DType.float32]
    """The Float32 gradient of the queries, `[rows, hidden]`."""
    var summand_rows: DeviceBuffer[DType.float32]
    """Each row's loss, bias gradient and log-β gradient, `[3, rows]`."""
    var self_coefficients: DeviceBuffer[DType.float32]
    """Each row's coefficient of its own embedding, `[prefixes, rows]`."""
    var row_partials: DeviceBuffer[DType.float32]
    """Each row's summands per column tile of the current block, `[3 + prefixes, rows / 64, rows]`."""
    var similarity_gradients: DeviceBuffer[DType.bfloat16]
    """Each band's suffix-summed gradient of the current block's raw dots, `[prefixes, rows, rows]`."""
    var transposed_candidates: DeviceBuffer[DType.bfloat16]
    """The current candidate block's channels, `[hidden, rows]`."""
    var staged_candidates: DeviceBuffer[DType.bfloat16]
    """The current candidate block, copied to this GPU, `[rows, hidden]`."""

    def __init__(out self, context: DeviceContext, *, rows: Int, hidden: Int, prefixes: Int) raises:
        """Allocates the loss's buffers on `context`'s GPU for `prefixes` halvings of `hidden`.

        Args:
            context: The context of the GPU whose queries the loss scores.
            rows: The rows of the queries and of every candidate block, a multiple of 64.
            hidden: The channels of every embedding, a multiple of 64.
            prefixes: The count of Matryoshka prefixes, from one to eight, each a multiple of 64 channels.

        Raises:
            When the shape does not tile, or an allocation fails.
        """
        if prefixes < 1 or prefixes > MAX_PREFIXES:
            raise Error(t"The ladder takes one to {MAX_PREFIXES} prefixes, not {prefixes}")
        var ladder = Matryoshka.halvings(hidden, prefixes)
        for band in range(prefixes):
            if rows % TILE_SIDE != 0 or ladder.end(band) % TILE_SIDE != 0:
                raise Error(t"Tiles take rows, channels and band edges in multiples of {TILE_SIDE}")
        self.context = context
        self.ladder = ladder
        self.rows = rows
        self.hidden = hidden
        self.query_inverse_norms = context.enqueue_create_buffer[DType.float32](prefixes * rows)
        self.candidate_inverse_norms = context.enqueue_create_buffer[DType.float32](prefixes * rows)
        self.query_gradient = context.enqueue_create_buffer[DType.float32](rows * hidden)
        self.summand_rows = context.enqueue_create_buffer[DType.float32](3 * rows)
        self.self_coefficients = context.enqueue_create_buffer[DType.float32](prefixes * rows)
        self.row_partials = context.enqueue_create_buffer[DType.float32]((3 + prefixes) * (rows // TILE_SIDE) * rows)
        self.similarity_gradients = context.enqueue_create_buffer[DType.bfloat16](prefixes * rows * rows)
        self.transposed_candidates = context.enqueue_create_buffer[DType.bfloat16](hidden * rows)
        self.staged_candidates = context.enqueue_create_buffer[DType.bfloat16](rows * hidden)

    def bytes(self) -> Int:
        """Device bytes the loss holds, independent of the world's size.

        Returns:
            The bytes of every buffer.
        """
        var floats = (
            len(self.query_inverse_norms)
            + len(self.candidate_inverse_norms)
            + len(self.query_gradient)
            + len(self.summand_rows)
            + len(self.self_coefficients)
            + len(self.row_partials)
        )
        var halves = len(self.similarity_gradients) + len(self.transposed_candidates) + len(self.staged_candidates)
        return floats * 4 + halves * 2

    def begin(self, queries: Halves) raises:
        """Zeroes the accumulators and takes the queries' inverse prefix norms.

        Args:
            queries: This rank's queries, `[rows, hidden]`, on the loss's GPU.

        Raises:
            When a launch fails.
        """
        self.context.enqueue_memset(self.query_gradient, Float32(0))
        self.context.enqueue_memset(self.summand_rows, Float32(0))
        self.context.enqueue_memset(self.self_coefficients, Float32(0))
        self.context.enqueue_function[prefix_inverse_norms_kernel](
            queries,
            device_address(self.query_inverse_norms),
            Int64(self.rows),
            Int64(self.hidden),
            self.ladder,
            grid_dim=self.rows,
            block_dim=BLOCK_THREADS,
        )

    def score(
        self,
        queries: Halves,
        candidates: Halves,
        *,
        query_base: Int,
        candidate_base: Int,
        beta: Float32,
        bias: Float32,
        live: Float32,
    ) raises:
        """Scores this rank's `queries` against a block of `candidates`, on this GPU or a peer's.

        Args:
            queries: This rank's queries, `[rows, hidden]`, on the loss's GPU.
            candidates: The candidate block, `[rows, hidden]`, copied once to the loss's GPU.
            query_base: The first query's index on the world's axis.
            candidate_base: The first candidate's index on the world's axis.
            beta: The logit scale.
            bias: The logit bias.
            live: The world's count of candidates, whose inverse weighs every cell.

        Raises:
            When a copy or a launch fails.
        """
        var tiles = self.rows // TILE_SIDE
        var staged = device_address(self.staged_candidates)
        var transposed = device_address(self.transposed_candidates)
        self.context.enqueue_copy(staged, candidates, self.rows * self.hidden)
        self.context.enqueue_function[prefix_inverse_norms_kernel](
            staged,
            device_address(self.candidate_inverse_norms),
            Int64(self.rows),
            Int64(self.hidden),
            self.ladder,
            grid_dim=self.rows,
            block_dim=BLOCK_THREADS,
        )
        self.context.enqueue_function[transpose_kernel](
            staged,
            transposed,
            Int64(self.rows),
            Int64(self.hidden),
            grid_dim=(self.hidden // 32, self.rows // 32),
            block_dim=(32, 8),
        )
        var tile = ForwardTile(
            queries=queries,
            candidates=staged,
            query_inverse_norms=device_address(self.query_inverse_norms),
            candidate_inverse_norms=device_address(self.candidate_inverse_norms),
            similarity_gradients=device_address(self.similarity_gradients),
            row_partials=device_address(self.row_partials),
            rows=Int64(self.rows),
            hidden=Int64(self.hidden),
            query_base=Int64(query_base),
            candidate_base=Int64(candidate_base),
            beta=beta,
            bias=bias,
            weight=1 / live,
        )
        self.context.enqueue_function[forward_tile_kernel](
            tile, self.ladder, grid_dim=(tiles, tiles), block_dim=BLOCK_THREADS
        )
        self.context.enqueue_function[fold_partials_kernel](
            device_address(self.row_partials),
            device_address(self.summand_rows),
            device_address(self.self_coefficients),
            Int64(3 + self.ladder.prefixes()),
            Int64(self.rows),
            grid_dim=divide_round_up(self.rows, BLOCK_THREADS),
            block_dim=BLOCK_THREADS,
        )
        self.context.enqueue_function[backward_tile_kernel](
            device_address(self.similarity_gradients),
            transposed,
            device_address(self.query_gradient),
            Int64(self.rows),
            Int64(self.hidden),
            self.ladder,
            grid_dim=(self.hidden // TILE_SIDE, tiles),
            block_dim=BLOCK_THREADS,
        )

    def finish(self, queries: Halves, gradient: Halves, summands: Scalars) raises:
        """Writes the BFloat16 `gradient` and the three `summands`, a domain payload or any device scalars.

        Args:
            queries: This rank's queries, `[rows, hidden]`, on the loss's GPU.
            gradient: The BFloat16 gradient of the queries, `[rows, hidden]`.
            summands: The rank's loss, bias gradient and log-β gradient.

        Raises:
            When a launch fails.
        """
        self.context.enqueue_function[finalize_kernel](
            queries,
            device_address(self.query_inverse_norms),
            device_address(self.self_coefficients),
            device_address(self.query_gradient),
            gradient,
            Int64(self.rows),
            Int64(self.hidden),
            self.ladder,
            grid_dim=self.rows,
            block_dim=BLOCK_THREADS,
        )
        self.context.enqueue_function[sum_summands_kernel](
            device_address(self.summand_rows), summands, Int64(self.rows), grid_dim=1, block_dim=BLOCK_THREADS
        )


# endregion Streamed Loss


# region Keyed Payloads


def splitmix64(value: UInt64) -> UInt64:
    """The splitmix64 finalizer, a keyed hash for inputs every run reproduces.

    Args:
        value: The key.

    Returns:
        The key's hash.
    """
    var mixed = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EB
    return mixed ^ (mixed >> 31)


def keyed_embeddings_kernel(embeddings: Halves, rows: Int64, hidden: Int64, salt: UInt64, scale: Float32):
    """Fills one row of `embeddings` per block with keyed BFloat16 values in `[-scale/2, scale/2)`.

    Args:
        embeddings: The rows, `[rows, hidden]`.
        rows: The count of rows.
        hidden: The channels of every row.
        salt: The key of this rank's rows.
        scale: The width of the values' range.
    """
    var row = Int(block_idx.x)
    var channel = Int(thread_idx.x)
    while channel < Int(hidden):
        var bits = splitmix64((salt * UInt64(rows) + UInt64(row)) * UInt64(hidden) + UInt64(channel))
        var unit = (bits >> 40).cast[DType.float32]() / Float32(1 << 24) - 0.5
        embeddings[unsafe_offset=row * Int(hidden) + channel] = (unit * scale).cast[DType.bfloat16]()
        channel += Int(block_dim.x)


def keyed_integer(scalar: UInt64, member: UInt64, members: UInt64) -> Float32:
    """Keyed integers below 2¹², so any order of summing up to 2¹² members is exact in Float32.

    Args:
        scalar: The scalar's index in the payload.
        member: The contributing member's ordinal.
        members: The count of members.

    Returns:
        The member's integer for that scalar.
    """
    return (splitmix64(scalar * members + member) >> 52).cast[DType.float32]()


def keyed_integers_kernel(payload: Scalars, scalars: Int64, member: Int64, members: Int64):
    """Fills `payload` with keyed integers, whose sums are exact.

    Args:
        payload: The member's payload.
        scalars: The scalars to fill.
        member: The member's ordinal.
        members: The count of members.
    """
    var scalar = Int(block_idx.x * block_dim.x + thread_idx.x)
    if scalar < Int(scalars):
        payload[unsafe_offset=scalar] = keyed_integer(UInt64(scalar), UInt64(member), UInt64(members))


def keyed_fractions_kernel(payload: Scalars, scalars: Int64, member: Int64, members: Int64):
    """Fills `payload` with keyed fractions in `[-1/2, 1/2)`, whose sums round.

    Args:
        payload: The member's payload.
        scalars: The scalars to fill.
        member: The member's ordinal.
        members: The count of members.
    """
    var scalar = Int(block_idx.x * block_dim.x + thread_idx.x)
    if scalar < Int(scalars):
        var bits = splitmix64(UInt64(scalar) * UInt64(members) + UInt64(member)) >> 40
        payload[unsafe_offset=scalar] = bits.cast[DType.float32]() / Float32(1 << 24) - 0.5


def count_wrong_sums_kernel(payload: Scalars, begin: Int64, end: Int64, members: Int64, wrong: Tally):
    """Counts into `wrong` the scalars of `[begin, end)` that differ from the sum of every member's keyed integers.

    Args:
        payload: The reduced payload.
        begin: The first scalar to check.
        end: The scalar past the last to check.
        members: The count of members whose integers were summed.
        wrong: The count to add mismatches to.
    """
    var scalar = Int(begin) + Int(block_idx.x * block_dim.x + thread_idx.x)
    if scalar >= Int(end):
        return
    var expected = Float32(0)
    for member in range(Int(members)):
        expected += keyed_integer(UInt64(scalar), UInt64(member), UInt64(members))
    if payload[unsafe_offset=scalar] != expected:
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](wrong, UInt64(1))


def count_differences_kernel(actual: Scalars, reference: Scalars, scalars: Int64, differing: Tally):
    """Counts into `differing` the scalars of `actual` whose bits differ from `reference`.

    Args:
        actual: The payload to compare.
        reference: The payload to compare against.
        scalars: The scalars to compare.
        differing: The count to add differences to.
    """
    var scalar = Int(block_idx.x * block_dim.x + thread_idx.x)
    if scalar >= Int(scalars):
        return
    var actual_bits = bitcast[DType.uint32](actual[unsafe_offset=scalar])
    if actual_bits != bitcast[DType.uint32](reference[unsafe_offset=scalar]):
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](differing, UInt64(1))


# endregion Keyed Payloads
