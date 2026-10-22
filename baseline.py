# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "numpy", "open_clip_torch"]
# ///
"""The PyTorch baseline: open_clip's SigLIP and CLIP losses, timed and measured like `scalingcontrasts`.

Two ways to run it:

    torchrun --nproc-per-node 8 baseline.py --pairs 1024 --prefixes 5 --impl bidir   # a real 8-GPU world
    python baseline.py --pairs 1024 --prefixes 5 --simulate 72                        # one rank of a 72-GPU world

The simulated world keeps what a rank of the real one keeps: open_clip's `_loss` per peer chunk, each
chunk a distinct received copy of the text features, as the neighbour exchange's autograd saves them.
Prints one CSV row:

    pairs,hidden,prefixes,world,impl,dtype,saved_bytes,peak_bytes,min_ms,median_ms

`saved_bytes` is what the forward leaves alive for the backward; `peak_bytes` is the step's peak above
the inputs.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from open_clip.loss import ClipLoss, SigLipLoss


def prefix_widths(hidden: int, prefixes: int) -> list[int]:
    return [hidden >> (prefixes - 1 - prefix) for prefix in range(prefixes)]


def simulated_siglip(loss: SigLipLoss, image, text, scale, bias, world: int):
    """One rank of a `world`-GPU SigLIP: its own block, then a negative-only block per received chunk."""
    total = loss._loss(image, text, scale, bias)
    for _ in range(world - 1):
        received = text.detach().clone().requires_grad_(True)
        total = total + loss._loss(image, received, scale, bias, negative_only=True)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--prefixes", type=int, default=1)
    parser.add_argument("--impl", choices=["bidir", "shift", "gather", "clip"], default="bidir")
    parser.add_argument("--simulate", type=int, default=0, help="one rank of this many GPUs, on one GPU")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--calls-per-window", type=int, default=5)
    parser.add_argument("--header", action="store_true")
    arguments = parser.parse_args()

    distributed = "WORLD_SIZE" in os.environ and arguments.simulate == 0
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1)) if distributed else max(arguments.simulate, 1)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if distributed:
        torch.distributed.init_process_group("nccl", device_id=device)
    dtype = torch.bfloat16 if arguments.dtype == "bf16" else torch.float32

    generator = torch.Generator(device=device).manual_seed(rank)
    shape = (arguments.pairs, arguments.hidden)
    image = (torch.rand(shape, device=device, generator=generator, dtype=dtype) - 0.5).requires_grad_(True)
    text = (torch.rand(shape, device=device, generator=generator, dtype=dtype) - 0.5).requires_grad_(True)
    scale = torch.tensor(10.0, device=device, requires_grad=True)
    bias = torch.tensor(-10.0, device=device, requires_grad=True)
    widths = prefix_widths(arguments.hidden, arguments.prefixes)

    siglip = SigLipLoss(
        rank=rank,
        world_size=world if distributed else 1,
        dist_impl=arguments.impl if arguments.impl != "clip" else "bidir",
    )
    clip = ClipLoss(gather_with_grad=True, rank=rank, world_size=world if distributed else 1)

    def forward():
        total = 0.0
        for width in widths:
            image_prefix = F.normalize(image[:, :width], dim=-1)
            text_prefix = F.normalize(text[:, :width], dim=-1)
            if arguments.impl == "clip":
                total = total + clip(image_prefix, text_prefix, scale)
            elif distributed or world == 1:
                total = total + siglip(image_prefix, text_prefix, scale, bias)
            else:
                total = total + simulated_siglip(siglip, image_prefix, text_prefix, scale, bias, world)
        return total

    def step():
        image.grad = text.grad = scale.grad = bias.grad = None
        forward().backward()

    step()
    torch.cuda.synchronize()
    image.grad = text.grad = scale.grad = bias.grad = None
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    total = forward()
    saved = torch.cuda.memory_allocated() - baseline
    total.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline

    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    per_call_ms = []
    for _ in range(arguments.windows):
        if distributed:
            torch.distributed.barrier()
        start.record()
        for _ in range(arguments.calls_per_window):
            step()
        stop.record()
        stop.synchronize()
        per_call_ms.append(start.elapsed_time(stop) / arguments.calls_per_window)
    per_call_ms.sort()

    if rank == 0:
        if arguments.header:
            print("pairs,hidden,prefixes,world,impl,dtype,saved_bytes,peak_bytes,min_ms,median_ms")
        impl = arguments.impl if distributed or world == 1 else f"{arguments.impl}-simulated"
        print(
            f"{arguments.pairs},{arguments.hidden},{arguments.prefixes},{world},{impl},{arguments.dtype},"
            f"{saved},{peak},{per_call_ms[0]:.4f},{per_call_ms[len(per_call_ms) // 2]:.4f}",
            flush=True,
        )
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
