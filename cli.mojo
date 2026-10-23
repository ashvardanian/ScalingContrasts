"""Benchmarks of the multimem primitives, each printing CSV rows.

    scalingcontrasts_mojo collectives [--max-mib M] [--repeats R] [--windows W]

The rows share their columns with the native binary's, so both backends' output pastes into one table.

File: cli.mojo
Author: Ash Vardanian
Date: October 23, 2026
"""

from std.builtin.sort import sort
from std.sys import argv, has_nvidia_gpu_accelerator, size_of
from std.time import perf_counter_ns

from comm import MAX_GPUS, Signal
from comm.allreduce import allreduce
from comm.sync import init_signal_buffer
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext

from scalingcontrasts import (
    BLOCK_THREADS,
    MultimemDomain,
    Scalars,
    count_differences_kernel,
    count_wrong_sums_kernel,
    divide_round_up,
    keyed_fractions_kernel,
    keyed_integers_kernel,
)

comptime COLLECTIVE_CALLS = 20
"""Collectives per timed window, nccl-tests' default."""

comptime SMALLEST_SCALARS = 1 << 18
"""The payload of the smallest timed collective, one MiB of Float32."""


# region Timing


@fieldwise_init
struct Timing(Copyable, Movable):
    """The minimum and median milliseconds per call of a timed run."""

    var min_ms: Float64
    """The fastest window's milliseconds per call."""
    var median_ms: Float64
    """The median window's milliseconds per call."""


def drain(contexts: List[DeviceContext]) raises:
    """Waits for every context's stream.

    Args:
        contexts: The contexts to drain.

    Raises:
        When a stream reports a failed launch.
    """
    for context in contexts:
        context.synchronize()


def time_steps[
    Step: def() raises -> None
](step: Step, contexts: List[DeviceContext], windows: Int, calls: Int) raises -> Timing:
    """Times `step` over windows of `calls` calls, each drained at its end.

    Parameters:
        Step: The type of the timed step.

    Args:
        step: Enqueues one call on every member.
        contexts: The contexts drained at the end of every window.
        windows: The count of timed windows.
        calls: The calls per window.

    Returns:
        The fastest and the median window's milliseconds per call.

    Raises:
        When a step or a drain fails.
    """
    var per_step = List[Float64]()
    for _ in range(windows):
        var start = perf_counter_ns()
        for _ in range(calls):
            step()
        drain(contexts)
        per_step.append(Float64(perf_counter_ns() - start) / 1e6 / Float64(calls))
    sort(per_step)
    return Timing(per_step[0], per_step[windows // 2])


def fixed(value: Float64, digits: Int) -> String:
    """Formats `value` rounded to `digits` decimals, the way `%.Nf` does.

    Args:
        value: The number to format.
        digits: The decimals to keep.

    Returns:
        The formatted number.
    """
    var scale = 10**digits
    var units = Int(abs(value) * Float64(scale) + 0.5)
    var sign = "-" if value < 0 and units != 0 else ""
    if digits == 0:
        return String(t"{sign}{units}")
    var fraction = String(units % scale)
    while fraction.byte_length() < digits:
        fraction = "0" + fraction
    return String(t"{sign}{units // scale}.{fraction}")


def bandwidth_row(name: String, members: Int, bytes: Int, timing: Timing, bus_factor: Float64) -> String:
    """One bandwidth row: latencies in microseconds, then algorithm and bus bandwidths in GB/s.

    Args:
        name: The collective, the row's first column.
        members: The count of GPUs.
        bytes: The payload every GPU contributes.
        timing: The collective's timing.
        bus_factor: The ratio of bus to algorithm bandwidth, as nccl-tests defines it.

    Returns:
        The CSV row.
    """
    var bandwidth = Float64(bytes) / (timing.min_ms * 1e6)
    return String(
        t"{name},{members},{bytes},{fixed(timing.min_ms * 1e3, 2)},{fixed(timing.median_ms * 1e3, 2)},"
        t"{fixed(bandwidth, 1)},{fixed(bandwidth * bus_factor, 1)}"
    )


# endregion Timing


# region Collectives


def fill[
    KeyedKernel: def(Scalars, Int64, Int64, Int64) thin -> None
](contexts: List[DeviceContext], domain: MultimemDomain, scalars: Int) raises:
    """Fills every member's payload with its keyed values.

    Parameters:
        KeyedKernel: The kernel writing one member's keyed values.

    Args:
        contexts: One context per member.
        domain: The domain whose payloads to fill.
        scalars: The scalars to fill per payload.

    Raises:
        When a launch fails.
    """
    for member in range(len(contexts)):
        contexts[member].enqueue_function[KeyedKernel](
            domain.payload(member),
            Int64(scalars),
            Int64(member),
            Int64(len(contexts)),
            grid_dim=divide_round_up(scalars, BLOCK_THREADS),
            block_dim=BLOCK_THREADS,
        )


def run_max_all_reduce[
    Members: Int
](contexts: List[DeviceContext], domain: MultimemDomain, max_scalars: Int, windows: Int) raises:
    """Bus bandwidths of MAX's own multimem all-reduce over the same payloads, out of place into one buffer per
    member.

    Parameters:
        Members: The count of GPUs, which MAX fixes at compile time.

    Args:
        contexts: One context per member.
        domain: The domain whose multicast payload is the input.
        max_scalars: The largest payload to time.
        windows: The count of timed windows per payload.

    Raises:
        When an allocation, a launch or a drain fails.
    """
    var signals = List[DeviceBuffer[DType.uint8]]()
    var outputs = List[DeviceBuffer[DType.float32]]()
    var output_addresses = List[Scalars]()
    var rank_signals = Array[Pointer[Signal, MutAnyOrigin], MAX_GPUS](uninitialized=True)
    for member in range(Members):
        signals.append(contexts[member].create_buffer_sync[DType.uint8](size_of[Signal]() + max_scalars * 4))
        outputs.append(contexts[member].enqueue_create_buffer[DType.float32](max_scalars))
        output_addresses.append(outputs[member].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
        init_signal_buffer(signals[member], contexts[member])
        rank_signals[member] = signals[member].unsafe_ptr().unsafe_bitcast[Signal]().unsafe_origin_cast[MutAnyOrigin]()
    drain(contexts)

    var ring = Float64(Members - 1) / Float64(Members)
    var scalars = SMALLEST_SCALARS
    while scalars <= max_scalars:
        var elements = scalars // Members // 4 * 4 * Members
        comptime Input = TileTensor[DType.float32, type_of(row_major(elements)), ImmutAnyOrigin]
        var inputs = Array[Input, 1](uninitialized=True)
        inputs[0] = Input(rebind[Pointer[Float32, ImmutAnyOrigin]](domain.multicast_payload(0)), row_major(elements))

        def reductions() raises {imm} -> None:
            comptime for member in range(Members):
                allreduce[ngpus=Members, use_multimem=True](
                    inputs,
                    TileTensor(output_addresses[member], row_major(elements)),
                    rank_signals,
                    contexts[member],
                )

        # The first call loads each member's kernels while every device is still idle.
        reductions()
        drain(contexts)
        var timing = time_steps(reductions, contexts, windows, COLLECTIVE_CALLS)
        print(bandwidth_row("max_all_reduce", Members, elements * 4, timing, 2 * ring))
        scalars *= 2


def run_collectives(max_mib: Int, repeats: Int, windows: Int) raises:
    """Barrier latency, then reduce-scatter and all-reduce bus bandwidths from 1 MiB to `--max-mib` as nccl-tests
    defines them, then exact sums of keyed integers, and bitwise repeats of an all-reduce of keyed fractions. Each
    phase is enqueued on every GPU before the next, so no first launch waits behind a spinning barrier.

    Args:
        max_mib: The largest payload in MiB.
        repeats: The all-reduces whose bits must all match the first.
        windows: The count of timed windows per collective.

    Raises:
        When the driver cannot bind the domain, or a launch or a drain fails.
    """
    var count = DeviceContext.number_of_devices()
    var contexts = List[DeviceContext]()
    for ordinal in range(count):
        contexts.append(DeviceContext(device_id=ordinal))
    for context in contexts:
        if not context.supports_multicast():
            print(t"no multicast-capable GPU: {context.name()} has no NVLink multicast, skipping the collectives")
            return
    DeviceContext.enable_all_peer_access()

    var max_scalars = max_mib << 18
    var domain = MultimemDomain(contexts, max_scalars)

    def barriers() raises {imm} -> None:
        for member in range(count):
            domain.barrier(member)

    var barrier = time_steps(barriers, contexts, windows, COLLECTIVE_CALLS)
    print(
        t"barrier,{count},{domain.replica_bytes()},{fixed(barrier.min_ms * 1e3, 2)},{fixed(barrier.median_ms * 1e3, 2)}"
    )

    var ring = Float64(count - 1) / Float64(count)
    var scalars = SMALLEST_SCALARS
    while scalars <= max_scalars:
        var shard = scalars // count // 4 * 4

        def scatters() raises {imm} -> None:
            for member in range(count):
                domain.reduce_scatter(member, shard)

        def reductions() raises {imm} -> None:
            for member in range(count):
                domain.all_reduce(member, shard * count)

        var scatter = time_steps(scatters, contexts, windows, COLLECTIVE_CALLS)
        var reduce = time_steps(reductions, contexts, windows, COLLECTIVE_CALLS)
        var bytes = shard * count * 4
        print(bandwidth_row("reduce_scatter", count, bytes, scatter, ring))
        print(bandwidth_row("all_reduce", count, bytes, reduce, 2 * ring))
        scalars *= 2

    comptime for exponent in range(1, 4):
        if count == 1 << exponent:
            run_max_all_reduce[1 << exponent](contexts, domain, max_scalars, windows)

    # Counters on the first GPU, which every GPU adds to over NVLink: wrong scatters, wrong sums,
    # differing repeats.
    var first = contexts[0]
    var counters = first.enqueue_create_buffer[DType.uint64](3)
    var reference = first.enqueue_create_buffer[DType.float32](max_scalars)
    first.enqueue_memset(counters, UInt64(0))
    drain(contexts)
    var tallies = counters.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var shard = max_scalars // count // 4 * 4

    fill[keyed_integers_kernel](contexts, domain, max_scalars)
    for member in range(count):
        domain.reduce_scatter(member, shard)
    for member in range(count):
        contexts[member].enqueue_function[count_wrong_sums_kernel](
            domain.payload(member),
            Int64(member * shard),
            Int64(member * shard + shard),
            Int64(count),
            tallies,
            grid_dim=divide_round_up(shard, BLOCK_THREADS),
            block_dim=BLOCK_THREADS,
        )
    fill[keyed_integers_kernel](contexts, domain, max_scalars)
    for member in range(count):
        domain.all_reduce(member, max_scalars)
    for member in range(count):
        contexts[member].enqueue_function[count_wrong_sums_kernel](
            domain.payload(member),
            Int64(0),
            Int64(max_scalars),
            Int64(count),
            tallies.unsafe_offset(1),
            grid_dim=divide_round_up(max_scalars, BLOCK_THREADS),
            block_dim=BLOCK_THREADS,
        )
    fill[keyed_fractions_kernel](contexts, domain, max_scalars)
    for member in range(count):
        domain.all_reduce(member, max_scalars)
    first.enqueue_copy(reference.unsafe_ptr(), domain.payload(0), max_scalars)
    for _ in range(1, repeats):
        fill[keyed_fractions_kernel](contexts, domain, max_scalars)
        for member in range(count):
            domain.all_reduce(member, max_scalars)
        first.enqueue_function[count_differences_kernel](
            domain.payload(0),
            reference.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            Int64(max_scalars),
            tallies.unsafe_offset(2),
            grid_dim=divide_round_up(max_scalars, BLOCK_THREADS),
            block_dim=BLOCK_THREADS,
        )
    var counted = first.enqueue_create_host_buffer[DType.uint64](3)
    first.enqueue_copy(counted, counters)
    drain(contexts)
    print(t"validation,{count},{max_scalars * 4},{counted[0]},{counted[1]},{repeats},{counted[2]}")


# endregion Collectives


# region Arguments


def parse_int_arg(args: Span[StaticString, ImmStaticOrigin], flag: String, default: Int) raises -> Int:
    """Parses an integer flag.

    Args:
        args: The command line.
        flag: The flag to look for.
        default: The value when the flag is absent.

    Returns:
        The flag's value.

    Raises:
        When the flag has no value or a malformed one.
    """
    for index in range(len(args)):
        if String(args[index]) != flag:
            continue
        if index + 1 >= len(args):
            raise Error(t"{flag} needs a value")
        try:
            return Int(String(args[index + 1]))
        except:
            raise Error(t"Invalid value for {flag}: {args[index + 1]}")
    return default


def reject_unknown_flags(args: Span[StaticString, ImmStaticOrigin]) raises:
    """Rejects unrecognized flags, so a typo cannot silently use defaults.

    Args:
        args: The command line.

    Raises:
        When a flag is not one the binary takes.
    """
    var valued: List[String] = ["--max-mib", "--repeats", "--windows"]
    var index = 2
    while index < len(args):
        var argument = String(args[index])
        if argument not in valued:
            raise Error(t"Unknown flag {argument}")
        index += 2


def main() raises:
    """Runs the mode the first argument names.

    Raises:
        When the command line is malformed or a run fails.
    """
    var args = argv()
    if len(args) < 2 or String(args[1]) != "collectives":
        raise Error("Usage: scalingcontrasts_mojo collectives [--max-mib M] [--repeats R] [--windows W]")
    reject_unknown_flags(args)
    var max_mib = parse_int_arg(args, "--max-mib", 128)
    var repeats = parse_int_arg(args, "--repeats", 1000)
    var windows = parse_int_arg(args, "--windows", 10)
    if max_mib < 1 or repeats < 1 or windows < 1:
        raise Error("Flags --max-mib, --repeats and --windows must be positive")
    comptime if has_nvidia_gpu_accelerator() and DeviceContext.default_device_info.compute >= 9.0:
        run_collectives(max_mib, repeats, windows)
    else:
        print("no multicast-capable GPU: this build targets no NVIDIA GPU of sm_90 or newer, skipping the collectives")


# endregion Arguments
