"""Checks zero gradients for collinear rows away from the cosine cutoff."""

from max.gpu.host import DeviceContext
from scalingcontrasts import StreamedLoss, device_address


def check(context: DeviceContext, scale: Float32, hidden: Int = 64, prefixes: Int = 1) raises:
    comptime rows = 128
    var inputs = context.enqueue_create_host_buffer[DType.bfloat16](rows * hidden)
    var outputs = context.enqueue_create_host_buffer[DType.bfloat16](rows * hidden)
    var embeddings = context.enqueue_create_buffer[DType.bfloat16](rows * hidden)
    var gradient = context.enqueue_create_buffer[DType.bfloat16](rows * hidden)
    var summands = context.enqueue_create_buffer[DType.float32](3)
    for row in range(rows):
        for channel in range(hidden):
            var value = (scale if row == 0 else Float32(1)) if channel == 0 else Float32(0)
            inputs[row * hidden + channel] = value.cast[DType.bfloat16]()
    context.enqueue_copy(embeddings, inputs)
    var loss = StreamedLoss(context, rows=rows, hidden=hidden, prefixes=prefixes)
    var queries = device_address(embeddings)
    loss.begin(queries)
    loss.score(
        queries, queries, query_base=0, candidate_base=0, beta=Float32(10), bias=Float32(-10), live=Float32(rows)
    )
    loss.finish(queries, device_address(gradient), device_address(summands))
    context.enqueue_copy(outputs, gradient)
    context.synchronize()
    for index in range(rows * hidden):
        var value = outputs[index].cast[DType.float32]()
        if value != 0:
            raise Error(t"Expected zero gradient at {index} for scale={scale}, prefixes={prefixes}; got {value}")
    print(t"scale={scale}, prefixes={prefixes}: passed")


def main() raises:
    var context = DeviceContext()
    check(context, 1)
    check(context, 0.0625)
    check(context, 0.00390625)
    check(context, 0)
    check(context, 0.0625, hidden=256, prefixes=3)
    check(context, 0, hidden=256, prefixes=3)
