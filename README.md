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
  A `tcgen05` kernel for B300 streams both operands through TMA into a ring of 128-byte swizzled stages and accumulates every cell's full dot in tensor memory, then sweeps back with negated rows, meeting the band edges widest first.
  Between edges the MMA warp sweeps the next segment into one of two tensor-memory buffers while eight warps fold the last one into the raw dots and score its band, keeping each band's suffix-summed gradient in tensor memory too, so no Float32 `[K, R, R]` of products ever reaches HBM.
  The gradient's products take the same path: TMA streams each band's BF16 gradient matrix and the transposed embeddings into `tcgen05` MMAs, which add straight into the Float32 forces.
- __Streamed, row-owned sigmoid loss.__
  Each peer's `[K, R, R]` tile of Matryoshka cosines is scored, lossed and folded into the GPU's own forces before the next one, so memory does not grow with the world.
  Cell `(i, j)` and its mirror `(j, i)` share the cosine, target and weight, so the owner of `i` differentiates both, and the backward exchanges nothing.

The code depends on [NumKong](https://github.com/ashvardanian/NumKong) for tensors, allocators and products, on the CUDA runtime and driver, and on libibverbs.
PyTorch, open_clip, JAX, big_vision and NCCL appear only as baselines.

## Contrastive Loss

Shapes: hidden width H = 1024, K = 5 Matryoshka prefixes {64, 128, 256, 512, 1024}, R rows per GPU, BF16 features.
Times are the minimum of 10 or more windows of back-to-back steps.
Memory is the loss's own allocation, or the baseline's peak above its inputs.
The baselines, every one differentiating the same Matryoshka prefixes:

- __open_clip bidir:__ open_clip 3.3.0's `SigLipLoss` under PyTorch 2.14.1, exchanging text chunks with both neighbours, as SigLIP trains in open_clip.
- __open_clip gather:__ the same loss over all-gathered features.
- __open_clip CLIP:__ the softmax InfoNCE `ClipLoss`, gathering with gradients.
- __big_vision:__ big_vision's SigLIP `loss_fn` under JAX 0.12 and `jax.jit`, with the batch sharded over every GPU, so XLA gathers the text embeddings, and with Float32 logits from the BF16 embeddings, as every other implementation here scores them.

| 8 GPUs, step | Streamed | open_clip bidir | open_clip gather | open_clip CLIP | big_vision |
| :----------- | -------: | --------------: | ---------------: | -------------: | ---------: |
| R = 1024     |  1.87 ms |         26.4 ms |          22.8 ms |        11.0 ms |    3.27 ms |
| R = 4096     |  6.46 ms |         27.3 ms |          26.1 ms |        83.0 ms |    8.18 ms |
| R = 16384    |  69.1 ms |          142 ms |           153 ms | out of memory  |     106 ms |

| 8 GPUs, memory | Streamed | open_clip bidir | open_clip gather | open_clip CLIP | big_vision |
| :------------- | -------: | --------------: | ---------------: | -------------: | ---------: |
| R = 1024       |    19 MB |          241 MB |           270 MB |        1.71 GB |     484 MB |
| R = 4096       |   210 MB |         3.03 GB |          3.40 GB |        26.2 GB |    6.19 GB |
| R = 16384      |  2.95 GB |         45.1 GB |          50.6 GB | out of memory  |    60.8 GB |

Across two such boxes, one process per box. NVLink joins only the eight GPUs inside a box, so between boxes each GPU writes into its counterpart's memory through its own 100 Gb/s RoCE port, about 70 times less bandwidth than its NVLink:

| 16 GPUs, step | Streamed | open_clip bidir | open_clip gather | open_clip CLIP | big_vision |
| :------------ | -------: | --------------: | ---------------: | -------------: | ---------: |
| R = 1024      |  4.16 ms |         45.7 ms |          39.3 ms |        25.4 ms |    4.57 ms |
| R = 4096      |  13.1 ms |         49.4 ms |          42.0 ms |         333 ms |    20.9 ms |
| R = 16384     |   139 ms |          362 ms |           305 ms | out of memory  |     223 ms |

| 16 GPUs, memory | Streamed | open_clip bidir | open_clip gather | open_clip CLIP | big_vision |
| :-------------- | -------: | --------------: | ---------------: | -------------: | ---------: |
| R = 1024        |    24 MB |          474 MB |           503 MB |        6.64 GB |     883 MB |
| R = 4096        |   227 MB |         5.97 GB |          6.34 GB |         104 GB |    11.1 GB |
| R = 16384       |  3.02 GB |         89.1 GB |          94.6 GB | out of memory  |     105 GB |

The streamed loss is the fastest at every size, holding 21 to 49 times less memory than big_vision, the fastest baseline, whose XLA program turns each device's whole loss into one large product.
open_clip's CLIP loss runs out of 268 GB at R = 16384, where its logits and their softmax grow with the world.

Each GPU scores its host's eight towers while the other host's eight are in flight, then scores those once they land.
The memory adds the two landings per GPU to the loss's own tile.

One rank of a 72-GPU world, simulated on one GPU at R = 4096, takes 50.5 ms and 210 MB.
PyTorch's simulated rank takes 101 ms and 26.6 GB in BF16.
The streamed loss holds the same 210 MB at 8, 72 or 144 GPUs.

Against a Float64 evaluation of the whole 8-GPU world at R = 128, the loss matches to 2e-7 relative, and the gradient to 0.23% relative L2, the BF16 rounding of its operands.
The summands reach every GPU through one multimem all-reduce, and every GPU holds the same loss.
Across two hosts, both check the 16-GPU world at R = 128 to the same precision, and agree on the loss bit for bit, as each adds the hosts' summands in host order.

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

## Footprint and Portability

The binary is 13.4 MB with the C++ runtime linked in, and needs only the CUDA driver, the 0.8 MB `libcudart` shipped beside it, and libibverbs.
The general-purpose stacks it replaces weigh far more: cuBLASLt with cuBLAS 603 MB, NCCL 242 MB, NVSHMEM 78 MB.

That size buys their portability, which this code gives up:

- It compiles for B300 alone, `sm_103a`, with no PTX to fall back on, as `tcgen05` and TMA need the architecture-specific target.
- It needs an NVSwitch domain with multicast for the multimem collectives, and NICs with dma-buf RDMA and RoCE v2 IPv4 GIDs across hosts.
- The libraries ship code for every architecture and pick their algorithms and transports at runtime, down to sockets, so they run almost anywhere.

## Reproducing

Build with CMake, which fetches NumKong:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release
cmake --build build_release
```

The binary prints one CSV row per measurement:

```sh
build_release/scalingcontrasts oracle --rows 128   # every local GPU against a Float64 evaluation
build_release/scalingcontrasts box --rows 4096    # every local GPU, peers read in place
build_release/scalingcontrasts simulate --rows 4096 --world 72
build_release/scalingcontrasts collectives --max-mib 128
```

Across hosts, every host runs one process, naming its index, one NIC per local GPU, and a directory all hosts share for the rendezvous, fresh for every run:

```sh
build_release/scalingcontrasts hosts --rows 4096 --host 0 --hosts 2 --nics <nic0>,<nic1>,... --exchange <shared directory>
build_release/scalingcontrasts oracle --rows 128 --host 0 --hosts 2 --nics <nic0>,<nic1>,... --exchange <shared directory>
```

The baselines run under `uv` or any environment with PyTorch and open_clip, or JAX:

```sh
torchrun --nproc-per-node 8 baseline_torch.py --pairs 4096 --prefixes 5 --impl bidir
python baseline_torch.py --pairs 4096 --prefixes 5 --simulate 72
python baseline_jax.py --pairs 4096 --prefixes 5
python baseline_jax.py --pairs 4096 --prefixes 5 --coordinator <host 0>:<port> --hosts 2 --host 0
```

The NCCL rows come from nccl-tests' `all_reduce_perf` and `reduce_scatter_perf`, with `-g 8 -n 20 -w 5 -c 0`, each under `NCCL_NVLS_ENABLE=1` and `0`.
