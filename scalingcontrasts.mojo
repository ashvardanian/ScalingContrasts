"""Direct NVLink scale-up primitives for training: multimem barriers and switch-reduced collectives.

One host's GPUs bind a replica each into one multicast object, which any of them addresses to
reduce in the NVLink switch or to store into every replica at once. Barrier rounds live on the
GPUs, so consecutive collectives advance with no host writes between them.

File: scalingcontrasts.mojo
Author: Ash Vardanian
Date: October 23, 2026
"""

from std.atomic import Atomic, Ordering
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.memory import AddressSpace, bitcast
from std.sys import inlined_assembly

from max.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext, DeviceMulticastBuffer
from max.gpu.intrinsics import Scope
from max.gpu.memory import Consistency, ReduceOp, multimem_ld_reduce, multimem_st


# region Constants

comptime BLOCK_THREADS = 256
"""Threads per block of every collective and payload kernel."""

comptime VECTOR_SCALARS = 4
"""Scalars per switch-reduced load and per multicast store, one 16-byte vector."""

comptime COUNTER_SCALARS = 4
"""Slots trailing each replica's payload: its arrival flag, its round, its count of blocks done, and padding."""

comptime Scalars = Pointer[Float32, MutUntrackedOrigin]
"""A device address of Float32 scalars, in a replica or in the multicast range."""

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


# endregion Constants


# region Device Primitives


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


# endregion Device Primitives


# region Domain


struct MultimemDomain(Movable):
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
        return self.replicas[member].unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()

    def multicast_payload(self, member: Int) -> Scalars:
        """The payload's multicast address, as `member` maps it.

        Args:
            member: The member's ordinal.

        Returns:
            The multicast address of the payload.
        """
        return self.multicasts[member].unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()

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

    def blocks(self, scalars: Int) -> Int:
        """Enough blocks for one vector per thread, at most one per multiprocessor.

        Args:
            scalars: The scalars a collective covers.

        Returns:
            The grid size.
        """
        return max(1, min(divide_round_up(scalars // VECTOR_SCALARS, BLOCK_THREADS), self.multiprocessors))

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
            grid_dim=self.blocks(shard_scalars),
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
        var vectors = scalars // VECTOR_SCALARS
        var share = divide_round_up(vectors, self.members())
        var begin = min(member * share, vectors) * VECTOR_SCALARS
        var end = min(member * share + share, vectors) * VECTOR_SCALARS
        self.contexts[member].enqueue_function[multimem_all_reduce_kernel](
            self.counters(member),
            self.multicast_payload(member),
            Int64(begin),
            Int64(end),
            grid_dim=self.blocks(end - begin),
            block_dim=BLOCK_THREADS,
        )


# endregion Domain


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
