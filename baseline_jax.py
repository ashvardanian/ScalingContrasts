# /// script
# requires-python = ">=3.12"
# dependencies = ["jax[cuda13]", "numpy"]
# ///
"""The JAX baseline: big_vision's SigLIP loss, timed and measured like `scalingcontrasts`.

big_vision computes the loss under `jax.jit` on a batch sharded across devices, so XLA gathers the
text embeddings and builds each device's logits against the whole world. The loss function below is
its `loss_fn` from `big_vision/trainers/proj/image_text/siglip.py`, applied to every Matryoshka
prefix of the embeddings, with BF16 embeddings and Float32 logits, as `scalingcontrasts` scores them.
Run one process per host:

    python baseline_jax.py --pairs 1024 --prefixes 5                                     # every local GPU
    python baseline_jax.py --pairs 1024 --prefixes 5 --coordinator <host 0>:<port> --hosts 2 --host 0

Prints one CSV row from the first process:

    pairs,hidden,prefixes,world,impl,dtype,saved_bytes,peak_bytes,min_ms,median_ms

`peak_bytes` is the first device's peak above the inputs; JAX keeps no saved tensors to report.

File: baseline_jax.py
Author: Ash Vardanian
Date: October 22, 2026
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def big_vision_loss(zimg, ztxt, t, b):
    """big_vision's SigLIP loss: every positive on the diagonal, every other pair a negative."""
    logits = jnp.dot(zimg, ztxt.T, preferred_element_type=jnp.float32)
    logits = logits * t + b
    eye = jnp.eye(zimg.shape[0])
    m1_diag1 = -jnp.ones_like(logits) + 2 * eye
    loglik = jax.nn.log_sigmoid(m1_diag1 * logits)
    nll = -jnp.sum(loglik, axis=-1)
    return jnp.mean(nll)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1024, help="pairs per GPU")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--prefixes", type=int, default=1)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--calls-per-window", type=int, default=5)
    parser.add_argument("--coordinator", default="")
    parser.add_argument("--hosts", type=int, default=1)
    parser.add_argument("--host", type=int, default=0)
    parser.add_argument("--header", action="store_true")
    arguments = parser.parse_args()

    if arguments.hosts > 1:
        jax.distributed.initialize(arguments.coordinator, arguments.hosts, arguments.host)
    devices = jax.devices()
    world = len(devices)
    mesh = Mesh(np.array(devices), ("data",))
    rows = NamedSharding(mesh, PartitionSpec("data", None))
    replicated = NamedSharding(mesh, PartitionSpec())
    widths = [arguments.hidden >> (arguments.prefixes - 1 - prefix) for prefix in range(arguments.prefixes)]

    def loss(zimg, ztxt, t, b):
        total = 0.0
        for width in widths:
            image = zimg[:, :width] / jnp.linalg.norm(zimg[:, :width], axis=-1, keepdims=True)
            text = ztxt[:, :width] / jnp.linalg.norm(ztxt[:, :width], axis=-1, keepdims=True)
            total = total + big_vision_loss(image, text, t, b)
        return total

    step = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1, 2, 3)),
        in_shardings=(rows, rows, replicated, replicated),
        out_shardings=(replicated, (rows, rows, replicated, replicated)),
    )

    shape = (world * arguments.pairs, arguments.hidden)
    keys = jax.random.split(jax.random.key(0), 2)
    make = jax.jit(
        lambda key: jax.random.uniform(key, shape, jnp.float32, -0.5, 0.5).astype(jnp.bfloat16),
        out_shardings=rows,
    )
    zimg, ztxt = make(keys[0]), make(keys[1])
    t = jax.device_put(jnp.float32(10.0), replicated)
    b = jax.device_put(jnp.float32(-10.0), replicated)
    jax.block_until_ready((zimg, ztxt))
    baseline = devices[0].memory_stats()["bytes_in_use"] if devices[0] in jax.local_devices() else 0

    jax.block_until_ready(step(zimg, ztxt, t, b))
    per_call_ms = []
    for _ in range(arguments.windows):
        start = time.perf_counter()
        for _ in range(arguments.calls_per_window):
            result = step(zimg, ztxt, t, b)
        jax.block_until_ready(result)
        per_call_ms.append((time.perf_counter() - start) * 1e3 / arguments.calls_per_window)
    per_call_ms.sort()

    if jax.process_index() == 0:
        peak = devices[0].memory_stats()["peak_bytes_in_use"] - baseline
        if arguments.header:
            print("pairs,hidden,prefixes,world,impl,dtype,saved_bytes,peak_bytes,min_ms,median_ms")
        print(
            f"{arguments.pairs},{arguments.hidden},{arguments.prefixes},{world},big_vision,bf16,"
            f"0,{peak},{per_call_ms[0]:.4f},{per_call_ms[len(per_call_ms) // 2]:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
