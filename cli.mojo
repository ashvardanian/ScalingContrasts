"""Benchmarks of the collectives and the streamed loss, each printing CSV rows.

    scalingcontrasts_mojo collectives [--max-mib M] [--repeats N] [--windows W]
    scalingcontrasts_mojo box         [--rows R] [--hidden H] [--prefixes K] [--windows W]
    scalingcontrasts_mojo simulate    [--rows R] --gpus S
    scalingcontrasts_mojo oracle      [--rows R]

The rows share their columns with the native binary's, so both backends' output pastes into one table.

File: cli.mojo
Author: Ash Vardanian
Date: October 23, 2026
"""

from std.builtin.sort import sort
from std.math import exp, floor, log10, log1p, sqrt
from std.sys import argv, size_of
from std.time import perf_counter_ns

from comm import Signal
from comm.allreduce import allreduce
from comm.sync import init_signal_buffer
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from scalingcontrasts import (
    BLOCK_THREADS,
    COSINE_FLOOR,
    Domain,
    Matryoshka,
    MultimemDomain,
    SUPPORTED_ACCELERATOR,
    Scalars,
    StreamedLoss,
    count_differences_kernel,
    count_wrong_sums_kernel,
    device_address,
    divide_round_up,
    keyed_embeddings_kernel,
    keyed_fractions_kernel,
    keyed_integers_kernel,
)

comptime COLLECTIVE_CALLS = 20
"""Collectives per timed window, nccl-tests' default."""

comptime SMALLEST_SCALARS = 1 << 18
"""The payload of the smallest timed collective, one MiB of Float32."""

comptime STEP_CALLS = 3
"""Loss steps per timed window."""

comptime SUMMAND_SCALARS = 4
"""The loss's three summands, padded to one collective vector."""

comptime BETA: Float32 = 10
"""SigLIP's initial logit scale."""

comptime BIAS: Float32 = -10
"""SigLIP's initial logit bias."""

comptime ORACLE_SCALE: Float32 = 1
"""The scale of the oracle's embeddings, which keeps its narrowest prefixes off the cosine floor."""

comptime BENCHMARK_SCALE: Float32 = 0.1
"""The scale of the embeddings the benchmarks score."""

comptime MULTICAST = DeviceContext.target.is_nvidia_gpu()
"""Whether the domain reduces in an NVLink switch, rather than over peer reads."""

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


def scientific(value: Float64, digits: Int) -> String:
    """Formats `value` with one integer digit, `digits` decimals and a signed two-digit exponent, the way `%.Ne` does.

    Args:
        value: The number to format.
        digits: The decimals of the mantissa.

    Returns:
        The formatted number.
    """
    var exponent = 0 if value == 0 else Int(floor(log10(abs(value))))
    var mantissa = value / 10.0**exponent
    if abs(mantissa) * Float64(10**digits) + 0.5 >= Float64(10 ** (digits + 1)):
        mantissa /= 10
        exponent += 1
    var sign = "-" if exponent < 0 else "+"
    var magnitude = String(abs(exponent))
    if magnitude.byte_length() < 2:
        magnitude = "0" + magnitude
    return String(t"{fixed(mantissa, digits)}e{sign}{magnitude}")


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
](contexts: List[DeviceContext], domain: Domain, scalars: Int) raises:
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
](contexts: List[DeviceContext], domain: Domain, max_scalars: Int, windows: Int) raises:
    """Bus bandwidths of MAX's own all-reduce over the same payloads, out of place into one buffer per member: its
    multimem path on NVIDIA, its peer-to-peer path on AMD.

    Parameters:
        Members: The count of GPUs, which MAX fixes at compile time.

    Args:
        contexts: One context per member.
        domain: The domain whose payloads are the input.
        max_scalars: The largest payload to time.
        windows: The count of timed windows per payload.

    Raises:
        When an allocation, a launch or a drain fails.
    """
    var signals = List[DeviceBuffer[DType.uint8]]()
    var outputs = List[DeviceBuffer[DType.float32]]()
    var output_addresses = List[Scalars]()
    var rank_signals = Array[MutPointer[Signal, MutAnyOrigin], Members](uninitialized=True)
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
        var inputs = Array[Input, 1 if MULTICAST else Members](uninitialized=True)
        comptime if MULTICAST:
            var multicast = rebind[MultimemDomain](domain).multicast_payload(0)
            inputs[0] = Input(rebind[Pointer[Float32, ImmutAnyOrigin]](multicast), row_major(elements))
        else:
            comptime for member in range(Members):
                inputs[member] = Input(
                    rebind[Pointer[Float32, ImmutAnyOrigin]](domain.payload(member)), row_major(elements)
                )

        def reductions() raises {imm} -> None:
            comptime for member in range(Members):
                allreduce[ngpus=Members, use_multimem=MULTICAST](
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
    comptime if MULTICAST:
        for context in contexts:
            if not context.supports_multicast():
                print(t"no multicast-capable GPU: {context.name()} has no NVLink multicast, skipping the collectives")
                return
    DeviceContext.enable_all_peer_access()

    var max_scalars = max_mib << 18
    var domain = Domain(contexts, max_scalars)

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


# region Loss


@fieldwise_init
struct Settings(Copyable, Movable):
    """The flags every mode reads."""

    var rows: Int
    """The rows of every GPU's embeddings."""
    var hidden: Int
    """The channels of every embedding."""
    var prefixes: Int
    """The count of Matryoshka prefixes."""
    var gpus: Int
    """The GPUs a simulated world spans."""
    var windows: Int
    """The count of timed windows."""
    var max_mib: Int
    """The largest collective payload in MiB."""
    var repeats: Int
    """The all-reduces whose bits must all match the first."""


struct Box(Movable):
    """Every local GPU with peer access to the others: its keyed embeddings, their gradient, its loss, and the domain
    every GPU's summands all-reduce over."""

    var contexts: List[DeviceContext]
    """One context per GPU, in device order."""
    var embeddings: List[DeviceBuffer[DType.bfloat16]]
    """Each GPU's keyed embeddings, `[rows, hidden]`."""
    var gradients: List[DeviceBuffer[DType.bfloat16]]
    """Each GPU's gradient of its embeddings, `[rows, hidden]`."""
    var losses: List[StreamedLoss]
    """Each GPU's loss."""
    var domain: Domain
    """The domain the summands all-reduce over."""
    var rows: Int
    """The rows of every GPU's embeddings."""

    def __init__(out self, settings: Settings, scale: Float32) raises:
        """Keys every local GPU's embeddings, allocates its loss, and binds the domain.

        Args:
            settings: The rows, channels and prefixes.
            scale: The width of the embeddings' range.

        Raises:
            When an allocation, a launch or the domain's binding fails.
        """
        var count = DeviceContext.number_of_devices()
        self.contexts = List[DeviceContext]()
        self.embeddings = List[DeviceBuffer[DType.bfloat16]]()
        self.gradients = List[DeviceBuffer[DType.bfloat16]]()
        self.losses = List[StreamedLoss]()
        self.rows = settings.rows
        for ordinal in range(count):
            var context = DeviceContext(device_id=ordinal)
            var embeddings = context.enqueue_create_buffer[DType.bfloat16](settings.rows * settings.hidden)
            context.enqueue_function[keyed_embeddings_kernel](
                device_address(embeddings),
                Int64(settings.rows),
                Int64(settings.hidden),
                UInt64(ordinal),
                scale,
                grid_dim=settings.rows,
                block_dim=BLOCK_THREADS,
            )
            self.contexts.append(context)
            self.embeddings.append(embeddings)
            self.gradients.append(context.enqueue_create_buffer[DType.bfloat16](settings.rows * settings.hidden))
            self.losses.append(
                StreamedLoss(context, rows=settings.rows, hidden=settings.hidden, prefixes=settings.prefixes)
            )
        DeviceContext.enable_all_peer_access()
        self.domain = Domain(self.contexts, SUMMAND_SCALARS)


def step_box(box: Box) raises:
    """Every local GPU's step: a barrier once all embeddings are complete, the peers' read in place, each candidate
    block streamed and folded, then one all-reduce of the summands, whose barriers end the peer reads. Every barrier
    is enqueued before any work, as a kernel's first, lazily loaded launch waits for its device to idle, which a
    barrier spinning on a GPU not yet enqueued never does.

    Args:
        box: The GPUs to step.

    Raises:
        When a copy or a launch fails.
    """
    var count = len(box.contexts)
    for member in range(count):
        box.domain.barrier(member)
    for member in range(count):
        ref loss = box.losses[member]
        var queries = device_address(box.embeddings[member])
        loss.begin(queries)
        for peer in range(count):
            loss.score(
                queries,
                device_address(box.embeddings[peer]),
                query_base=member * box.rows,
                candidate_base=peer * box.rows,
                beta=BETA,
                bias=BIAS,
                live=Float32(count * box.rows),
            )
        loss.finish(queries, device_address(box.gradients[member]), box.domain.payload(member))
        box.domain.all_reduce(member, SUMMAND_SCALARS)


def oracle_loss(
    embeddings: List[Float64], queries: Int, hidden: Int, ladder: Matryoshka, mut gradient: List[Float64]
) -> Float64:
    """The world's Float64 loss and every query's gradient: the sum over prefixes and cells of
    `w * softplus(-t * (beta * cos + b))`.

    Args:
        embeddings: Every query of the world, `[queries, hidden]`.
        queries: The count of queries.
        hidden: The channels of every query.
        ladder: The prefixes.
        gradient: The gradient of every query, `[queries, hidden]`, overwritten.

    Returns:
        The loss.
    """
    for index in range(queries * hidden):
        gradient[index] = 0
    var beta = Float64(BETA)
    var bias = Float64(BIAS)
    var weight = 1.0 / Float64(queries)
    var loss = 0.0
    var norms = List[Float64](length=queries, fill=0)
    for band in range(ladder.prefixes()):
        var width = ladder.end(band)
        for i in range(queries):
            var sum = 0.0
            for h in range(width):
                sum += embeddings[i * hidden + h] * embeddings[i * hidden + h]
            norms[i] = sqrt(sum)
        for i in range(queries):
            for j in range(queries):
                var dot = 0.0
                for h in range(width):
                    dot += embeddings[i * hidden + h] * embeddings[j * hidden + h]
                var denominator = norms[i] * norms[j]
                var active = denominator > Float64(COSINE_FLOOR)
                var cosine = Float64(1) if i == j else dot / denominator if active else Float64(0)
                var target = 1.0 if i == j else -1.0
                var signed_logit = target * (beta * cosine + bias)
                loss += weight * (max(-signed_logit, 0.0) + log1p(exp(-abs(signed_logit))))
                if i == j or not active:
                    continue
                var cell_grad = weight * target * (1.0 / (1.0 + exp(-signed_logit)) - 1.0) * beta
                for h in range(width):
                    gradient[i * hidden + h] += cell_grad * (
                        embeddings[j * hidden + h] / (norms[i] * norms[j])
                        - cosine * embeddings[i * hidden + h] / (norms[i] * norms[i])
                    )
                    gradient[j * hidden + h] += cell_grad * (
                        embeddings[i * hidden + h] / (norms[i] * norms[j])
                        - cosine * embeddings[j * hidden + h] / (norms[j] * norms[j])
                    )
    return loss


def run_simulate(settings: Settings) raises:
    """One rank of a `--gpus` world on the first GPU, its own embeddings standing in for every candidate block.

    Args:
        settings: The rows, channels, prefixes, simulated GPUs and timed windows.

    Raises:
        When an allocation, a launch or a drain fails.
    """
    var context = DeviceContext()
    var embeddings = context.enqueue_create_buffer[DType.bfloat16](settings.rows * settings.hidden)
    var gradient = context.enqueue_create_buffer[DType.bfloat16](settings.rows * settings.hidden)
    var summands = context.enqueue_create_buffer[DType.float32](SUMMAND_SCALARS)
    var queries = device_address(embeddings)
    context.enqueue_function[keyed_embeddings_kernel](
        queries,
        Int64(settings.rows),
        Int64(settings.hidden),
        UInt64(0),
        BENCHMARK_SCALE,
        grid_dim=settings.rows,
        block_dim=BLOCK_THREADS,
    )
    var loss = StreamedLoss(context, rows=settings.rows, hidden=settings.hidden, prefixes=settings.prefixes)

    def step() raises {imm} -> None:
        loss.begin(queries)
        for candidate_block in range(settings.gpus):
            loss.score(
                queries,
                queries,
                query_base=0,
                candidate_base=candidate_block * settings.rows,
                beta=BETA,
                bias=BIAS,
                live=Float32(settings.gpus * settings.rows),
            )
        loss.finish(queries, device_address(gradient), device_address(summands))

    var contexts: List[DeviceContext] = [context]
    step()
    drain(contexts)
    var timing = time_steps(step, contexts, settings.windows, STEP_CALLS)
    print(
        t"simulate,{settings.rows},{settings.hidden},{settings.prefixes},{settings.gpus},{loss.bytes()},"
        t"{fixed(timing.min_ms, 3)},{fixed(timing.median_ms, 3)}"
    )


def run_box(settings: Settings) raises:
    """Every local GPU's step time, and the device bytes of one GPU's loss.

    Args:
        settings: The rows, channels, prefixes and timed windows.

    Raises:
        When the box cannot be built, or a launch or a drain fails.
    """
    var box = Box(settings, BENCHMARK_SCALE)

    def step() raises {imm} -> None:
        step_box(box)

    step()
    drain(box.contexts)
    var timing = time_steps(step, box.contexts, settings.windows, STEP_CALLS)
    print(
        t"box,{settings.rows},{settings.hidden},{settings.prefixes},{len(box.contexts)},{box.losses[0].bytes()},"
        t"{fixed(timing.min_ms, 3)},{fixed(timing.median_ms, 3)}"
    )


def run_oracle(settings: Settings) raises:
    """One step against a Float64 evaluation of the whole world: every GPU's embeddings, gradient and summands, each
    downloaded on its GPU's stream.

    Args:
        settings: The rows, channels and prefixes.

    Raises:
        When the box cannot be built, or a launch, a copy or a drain fails.
    """
    var box = Box(settings, ORACLE_SCALE)
    step_box(box)
    drain(box.contexts)
    var count = len(box.contexts)
    var elements = settings.rows * settings.hidden
    var embeddings = List[HostBuffer[DType.bfloat16]]()
    var gradients = List[HostBuffer[DType.bfloat16]]()
    var summands = List[HostBuffer[DType.float32]]()
    for member in range(count):
        var context = box.contexts[member]
        embeddings.append(context.enqueue_create_host_buffer[DType.bfloat16](elements))
        gradients.append(context.enqueue_create_host_buffer[DType.bfloat16](elements))
        summands.append(context.enqueue_create_host_buffer[DType.float32](3))
        context.enqueue_copy(embeddings[member], box.embeddings[member])
        context.enqueue_copy(gradients[member], box.gradients[member])
        context.enqueue_copy(
            summands[member], DeviceBuffer[DType.float32](context, box.domain.payload(member), 3, owning=False)
        )
    drain(box.contexts)

    var world = List[Float64](length=count * elements, fill=0)
    for member in range(count):
        for index in range(elements):
            world[member * elements + index] = embeddings[member][index].cast[DType.float64]()
    var disagreeing = 0
    for member in range(count):
        disagreeing += Int(summands[member][0] != summands[0][0])
    var exact = List[Float64](length=count * elements, fill=0)
    var ladder = Matryoshka.halvings(settings.hidden, settings.prefixes)
    var exact_loss = oracle_loss(world, count * settings.rows, settings.hidden, ladder, exact)
    var difference = 0.0
    var reference = 0.0
    for member in range(count):
        for index in range(elements):
            var expected = exact[member * elements + index]
            var delta = gradients[member][index].cast[DType.float64]() - expected
            difference += delta * delta
            reference += expected * expected
    print(
        t"oracle,{settings.rows},{settings.hidden},{settings.prefixes},{count},{fixed(exact_loss, 9)},"
        t"{fixed(Float64(summands[0][0]), 9)},{scientific(sqrt(difference / reference), 3)},{disagreeing}"
    )


# endregion Loss


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
    var valued: List[String] = ["--rows", "--hidden", "--prefixes", "--gpus", "--windows", "--max-mib", "--repeats"]
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
    var modes: List[String] = ["collectives", "box", "simulate", "oracle"]
    if len(args) < 2 or String(args[1]) not in modes:
        raise Error(
            "Usage: scalingcontrasts_mojo collectives|box|simulate|oracle [--rows R] [--hidden H] [--prefixes K]"
            " [--gpus S] [--windows W] [--max-mib M] [--repeats N]"
        )
    reject_unknown_flags(args)
    var mode = String(args[1])
    var settings = Settings(
        rows=parse_int_arg(args, "--rows", 1024),
        hidden=parse_int_arg(args, "--hidden", 1024),
        prefixes=parse_int_arg(args, "--prefixes", 5),
        gpus=parse_int_arg(args, "--gpus", 0),
        windows=parse_int_arg(args, "--windows", 10),
        max_mib=parse_int_arg(args, "--max-mib", 128),
        repeats=parse_int_arg(args, "--repeats", 1000),
    )
    if settings.rows < 1 or settings.hidden < 1 or settings.windows < 1 or settings.max_mib < 1 or settings.repeats < 1:
        raise Error("Flags --rows, --hidden, --windows, --max-mib and --repeats must be positive")
    if mode == "simulate" and settings.gpus < 1:
        raise Error("Mode simulate needs a positive --gpus")
    comptime if SUPPORTED_ACCELERATOR:
        if mode == "collectives":
            run_collectives(settings.max_mib, settings.repeats, settings.windows)
        elif mode == "box":
            run_box(settings)
        elif mode == "simulate":
            run_simulate(settings)
        else:
            run_oracle(settings)
    else:
        print(t"no supported GPU: this build targets no Hopper, Blackwell, CDNA 4 or CDNA 5 GPU, skipping {mode}")


# endregion Arguments
