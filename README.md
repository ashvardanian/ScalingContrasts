# Scaling Contrasts

Direct NVLink scale-up primitives for ML and AI training, in one header and one benchmark binary.
A contrastive loss is the case study, because its batch, and with it the number of negatives every pair learns from, is bounded by how cheaply GPUs see each other's embeddings.
The same ideas ship in [UForm](https://github.com/unum-cloud/UForm), where they train multimodal embedding models.

Every number below comes from this repository, on one or two 8× B300 SXM6 boxes, from unit benchmarks with no training runs.

## The Primitives

- __Peer reads in place.__
  A GPU's tensor-core product reads a peer's embeddings straight over NVLink, with no gather and no copy.
- __Multimem barrier.__
  One `multimem.red.release` adds an arrival to every GPU's flag through the NVSwitch, then each GPU acquire-spins on its own flag.
  Rounds live on the device, so replays need no host writes.
- __Multimem reduce-scatter and all-reduce.__
  `multimem.ld_reduce.v4.f32` sums four scalars across every GPU in the switch, and `multimem.st.v4.f32` stores the sum into every GPU's replica at once.
  That is the NVLink SHARP algorithm, written in a dozen lines of PTX.
- __RDMA rails.__
  Each GPU owns one RoCE port and an RC queue pair to the same GPU on every other host, writing its embeddings from GPU memory straight into GPU memory, registered as dma-buf.
  The landed embeddings are double-buffered by step parity and read in place by the host's other GPUs over NVLink, so each crosses the wire once.
- __Matryoshka in one product.__
  A `mma.sync` tensor-core kernel sweeps the depth forward for every cell's full dot, then back with negated rows, meeting the band edges widest first.
  Each band's gradient is scored and suffix-summed in registers as it is met, so no Float32 `[K, R, R]` of products ever reaches memory.
- __Streamed, row-owned sigmoid loss.__
  Each peer's `[K, R, R]` tile of Matryoshka cosines is scored, lossed and folded into the GPU's own forces before the next one, so memory does not grow with the world.
  Cell `(i, j)` and its mirror `(j, i)` share the cosine, target and weight, so the owner of `i` differentiates both, and the backward exchanges nothing.

The code depends on [NumKong](https://github.com/ashvardanian/NumKong) for tensors, allocators and products, on the CUDA runtime and driver, and on libibverbs.
PyTorch, open_clip and NCCL appear only as baselines.

## Contrastive Loss

Shapes: hidden width H = 1024, K = 5 Matryoshka prefixes {64, 128, 256, 512, 1024}, R rows per GPU, BF16 features.
The baseline is open_clip 3.3.0's `SigLipLoss` with its bidirectional neighbour exchange, under PyTorch 2.14.1 and `torchrun`.
Times are the minimum of 10 windows of back-to-back steps.
Memory is the loss's own allocation, or PyTorch's peak above its inputs.

| 8 GPUs           | Streamed | PyTorch |
| :--------------- | -------: | ------: |
| R = 1024 step    |  4.58 ms | 26.2 ms |
| R = 1024 memory  |    39 MB |  241 MB |
| R = 4096 step    |  11.4 ms | 26.9 ms |
| R = 4096 memory  |   239 MB | 3.03 GB |
| R = 16384 step   |   121 ms |         |
| R = 16384 memory |  3.00 GB |         |

Across two such boxes, one process per box and eight 100 Gb/s RoCE rails between them:

| 16 GPUs          | Streamed | PyTorch |
| :--------------- | -------: | ------: |
| R = 1024 step    |  9.98 ms | 47.3 ms |
| R = 1024 memory  |    44 MB |  474 MB |
| R = 4096 step    |  23.9 ms | 53.1 ms |
| R = 4096 memory  |   256 MB | 5.97 GB |
| R = 16384 step   |   241 ms |         |
| R = 16384 memory |  3.07 GB |         |

Each GPU scores its host's eight towers while the other host's eight are in flight, then scores those once they land.
The memory adds the two landings per GPU to the loss's own tile.

One rank of a 72-GPU world, simulated on one GPU at R = 4096, takes 85.7 ms and 239 MB.
PyTorch's simulated rank takes 101 ms and 26.6 GB in BF16.
The streamed loss holds the same 239 MB at 8, 72 or 144 GPUs.

Against a Float64 evaluation of the whole 8-GPU world at R = 64, the loss matches to 5e-8 relative, and the gradient to 0.24% relative L2, the BF16 rounding of its operands.
The summands reach every GPU through one multimem all-reduce, and every GPU holds the same loss.
Across two hosts, both check the 16-GPU world at R = 64 to the same precision, and agree on the loss bit for bit, as each adds the hosts' summands in host order.

## Multimem Collectives

Eight GPUs, Float32 sums in place, nccl-tests' bus bandwidth.
NCCL is 2.30.7, the one PyTorch bundles, under nccl-tests v2.21.1 with NVLink SHARP on and off.

| Payload | All-reduce | NCCL NVLS | NCCL without NVLS | Reduce-scatter | NCCL NVLS |
| ------: | ---------: | --------: | ----------------: | -------------: | --------: |
|   1 MiB |    66.5 µs |   55.1 µs |           53.4 µs |        72.3 µs |   56.6 µs |
|  16 MiB |   437 GB/s |  266 GB/s |          346 GB/s |       215 GB/s |  257 GB/s |
|  64 MiB |   718 GB/s |  416 GB/s |          461 GB/s |       492 GB/s |  412 GB/s |
| 128 MiB |   771 GB/s |  587 GB/s |          617 GB/s |       549 GB/s |  560 GB/s |

- The barrier takes 27.6 µs, against 43–47 µs for NCCL's smallest all-reduce.
- From 16 MiB up the all-reduce moves 1.25–1.56× NCCL's best.
- The reduce-scatter trails NCCL at 16 MiB, leads it by 1.19× at 64 MiB, and matches it at 128 MiB.
- Below 16 MiB NCCL wins: each collective here costs two barriers and three launches per GPU from one host thread, about 66 µs, which fusing them into one kernel would cut.
- Keyed integer inputs sum exactly in both collectives, and 1,000 all-reduces of 128 MiB of keyed fractions gave bit-identical results.

## Reproducing

Build with CMake, which fetches NumKong:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release
cmake --build build_release
```

The binary prints one CSV row per measurement:

```sh
build_release/scalingcontrasts oracle --rows 64   # every local GPU against a Float64 evaluation
build_release/scalingcontrasts box --rows 4096    # every local GPU, peers read in place
build_release/scalingcontrasts simulate --rows 4096 --world 72
build_release/scalingcontrasts collectives --max-mib 128
```

Across hosts, every host runs one process, naming its index, one NIC per local GPU, and a directory all hosts share for the rendezvous, fresh for every run:

```sh
build_release/scalingcontrasts hosts --rows 4096 --host 0 --hosts 2 --nics <nic0>,<nic1>,... --exchange <shared directory>
build_release/scalingcontrasts oracle --rows 64 --host 0 --hosts 2 --nics <nic0>,<nic1>,... --exchange <shared directory>
```

The baseline runs under `uv` or any environment with PyTorch and open_clip:

```sh
torchrun --nproc-per-node 8 baseline.py --pairs 4096 --prefixes 5 --impl bidir
python baseline.py --pairs 4096 --prefixes 5 --simulate 72
```

The NCCL rows come from nccl-tests' `all_reduce_perf` and `reduce_scatter_perf`, with `-g 8 -n 20 -w 5 -c 0`, each under `NCCL_NVLS_ENABLE=1` and `0`.
