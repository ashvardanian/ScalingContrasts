# Scaling Contrasts

![Scaling Contrasts Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/ScalingContrasts.jpg?raw=true)

Training primitives for GPUs spread over several boxes, where NVLink joins the GPUs inside a box and an RDMA network joins the boxes.
Inside a box, NVLink multicast reduces in the switch, and between boxes, RDMA rails write each GPU's embeddings straight into its counterpart's memory, all in one CUDA header, one Mojo module and one benchmark binary.

A contrastive loss is the case study, because its batch, and with it the number of negatives every pair learns from, is bounded by how cheaply GPUs see each other's embeddings across boxes.
The case study is one of many possible setups: two 8× B300 SXM6 boxes, joined by one 100 Gb/s RoCE v2 rail per GPU, one of the four ports each GPU's ConnectX-8 card exposes:

- __2.9× big_vision's speed on 16 GPUs__, 77.7 against 223 ms per step at R = 16384, with 35× less memory, 3.02 against 105 GB per GPU.
- __The same pace per GPU on two boxes as on one__, 276 G scored cells/s, or 138 G distinct pairs/s against big_vision's 96.3 G, as every embedding crosses between boxes once and no gradient crosses at all.
- __Memory that barely grows with the world__, 2.95 GB per GPU on one box and 3.02 GB on two, the difference being two landing buffers for the other box's embeddings.

Inside each box, the multimem all-reduce reaches 791 GB/s of bus bandwidth, 1.28× NCCL's best, behind a 25.5 µs barrier.
The same ideas train multimodal embedding models in [UForm](https://github.com/unum-cloud/UForm) v4.

## Usage

Build with CMake, which fetches NumKong:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release
cmake --build build_release
```

The Mojo module ports the single-box loss and collectives to Nvidia Hopper and Blackwell and to AMD CDNA 4 and CDNA 5, and builds with Mojo 1.2 and MAX 26.7 through [pixi](https://pixi.sh):

```sh
pixi run build-cli   # builds build/scalingcontrasts_mojo
pixi run check       # cross-compiles for sm_90a, sm_100a, sm_103a, gfx950 and gfx1250
pixi run bench
```

Other Mojo projects import the module from a checkout, pointing the compiler at it:

```sh
git clone https://github.com/ashvardanian/ScalingContrasts
mojo build -I ScalingContrasts main.mojo   # `from scalingcontrasts import Domain, StreamedLoss`
```

The CUDA binary prints one CSV row per measurement:

```sh
build_release/scalingcontrasts oracle --rows 128    # every local GPU against a Float64 evaluation
build_release/scalingcontrasts box --rows 4096      # every local GPU, peers staged over NVLink
build_release/scalingcontrasts simulate --rows 4096 --gpus 72
build_release/scalingcontrasts collectives --max-mib 128
```

Every mode also takes `--hidden H` and `--prefixes K` for the embeddings' shape and `--windows W` for the timing, and `collectives` takes `--repeats N` for its bitwise check.

Across hosts, every host runs one process, naming its index, one NIC per local GPU, and a directory all hosts share for the rendezvous, fresh for every run:

```sh
build_release/scalingcontrasts hosts --rows 4096 --host 0 --hosts 2 --nics <nic0>,<nic1>,... --exchange <shared directory>
build_release/scalingcontrasts oracle --rows 128 --host 0 --hosts 2 --nics <nic0>,<nic1>,... --exchange <shared directory>
```

The baselines run under `uv` or any environment with PyTorch and open_clip, or JAX, and the NCCL rows come from nccl-tests' `all_reduce_perf` and `reduce_scatter_perf` with `-g 8 -n 20 -w 5 -c 0`, under `NCCL_NVLS_ENABLE=1` and `0`:

```sh
torchrun --nproc-per-node 8 baseline_torch.py --pairs 4096 --prefixes 5 --impl bidir
python baseline_torch.py --pairs 4096 --prefixes 5 --simulate 72
python baseline_jax.py --pairs 4096 --prefixes 5
python baseline_jax.py --pairs 4096 --prefixes 5 --coordinator <host 0>:<port> --hosts 2 --host 0
```

## What's Inside

- __Peer staging over NVLink.__
  Each GPU copies a peer's candidate block once into its own memory, where TMA can read it, and scores it before touching the next.
- __Multimem barrier, reduce-scatter and all-reduce.__
  `multimem.red`, `multimem.ld_reduce.v4.f32` and `multimem.st.v4.f32` add arrivals and sum payloads inside the NVSwitch, each collective in one kernel, with rounds kept on the device.
- __RDMA rails.__
  Between boxes, each GPU writes its embeddings straight into its counterpart's memory through its own RoCE port, registered as dma-buf, double-buffered by step parity.
- __Matryoshka in one product.__
  A `tcgen05` kernel accumulates every cell's full dot product in tensor memory, then sweeps back with negated rows, so the prefixes arrive widest first and each band's gradient builds up as a running sum, with no Float32 `[K, R, R]` cube in HBM.
- __Streamed, row-owned sigmoid loss.__
  Each peer's tile is scored and folded into the GPU's own gradient before the next one, so memory does not grow with the world, and the gradient never leaves its GPU.

The code depends on [NumKong](https://github.com/ashvardanian/NumKong) for tensors, allocators and devices, on the CUDA runtime and driver, and on libibverbs.
PyTorch, open_clip, JAX, big_vision and NCCL appear only as baselines.

## One Set or Two Towers

ScalingContrasts and the baselines score grids of the same size, but not the same pairs:

- __Two towers__, as open_clip and big_vision train CLIP and SigLIP, score R images per GPU against the world's S·R captions.
  Every cell is a distinct image-caption pair, and its gradient reaches two vectors, usually on two GPUs.
- __One set__, as this loss and one-tower models like UForm train, scores R embeddings per GPU against the world's S·R embeddings of the same set.
  Cell `(i, j)` and its mirror `(j, i)` are one pair, scored once by each owner, so each owner applies both to its own vector and no gradient crosses GPUs.

Both score R · S·R · K cells per GPU per step, at K Matryoshka prefixes.
The one-set grid holds every distinct pair twice, and its backward runs one product where two towers run two, 3,008 against 5,952 depth units per cell.
So the tables count both: scored cells per second, the same work for every column, and distinct pairs per second, which halves the one-set rate.

## Contrastive Loss

Hidden width H = 1024, K = 5 prefixes {64, 128, 256, 512, 1024}, R rows per GPU, BF16 features with Float32 scores.
Each cell holds the minimum step time over 10 or more windows, then the loss's own memory, or the baseline's peak above its inputs.

- __open_clip bidir:__ open_clip 3.3.0's `SigLipLoss` under PyTorch 2.14.1, exchanging text chunks with both neighbours.
- __open_clip gather:__ the same loss over all-gathered features.
- __open_clip CLIP:__ the softmax InfoNCE `ClipLoss`, gathering with gradients.
- __big_vision:__ big_vision's SigLIP `loss_fn` under JAX 0.12 and `jax.jit`, with the batch sharded over every GPU.

Across two boxes, one process per box, each GPU writing to its counterpart through its own 100 Gb/s RoCE port:

| 16 GPUs   |  ScalingContrasts |   open_clip bidir |  open_clip gather |    open_clip CLIP |        big_vision |
| :-------- | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: |
| R = 1024  |   3.62 ms · 24 MB |  45.7 ms · 474 MB |  39.3 ms · 503 MB | 25.4 ms · 6.64 GB |  4.57 ms · 883 MB |
| R = 4096  |  9.72 ms · 227 MB | 49.4 ms · 5.97 GB | 42.0 ms · 6.34 GB |   333 ms · 104 GB | 20.9 ms · 11.1 GB |
| R = 16384 | 77.7 ms · 3.02 GB |  362 ms · 89.1 GB |  305 ms · 94.6 GB |     out of memory |   223 ms · 105 GB |
| Cells/s   |             276 G |            59.3 G |            70.4 G |                 … |            96.3 G |
| Pairs/s   |             138 G |            59.3 G |            70.4 G |                 … |            96.3 G |

Within one box, where NVLink joins all eight GPUs:

| 8 GPUs    |  ScalingContrasts |   open_clip bidir |  open_clip gather |    open_clip CLIP |        big_vision |
| :-------- | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: |
| R = 1024  |   1.89 ms · 19 MB |  26.4 ms · 241 MB |  22.8 ms · 270 MB | 11.0 ms · 1.71 GB |  3.27 ms · 484 MB |
| R = 4096  |  4.17 ms · 210 MB | 27.3 ms · 3.03 GB | 26.1 ms · 3.40 GB | 83.0 ms · 26.2 GB | 8.18 ms · 6.19 GB |
| R = 16384 | 39.0 ms · 2.95 GB |  142 ms · 45.1 GB |  153 ms · 50.6 GB |     out of memory |  106 ms · 60.8 GB |
| Cells/s   |             275 G |            75.6 G |            70.2 G |                 … |             101 G |
| Pairs/s   |             138 G |            75.6 G |            70.2 G |                 … |             101 G |

Cells/s and Pairs/s are per GPU at R = 16384.
The loss matches a Float64 evaluation of the whole world to 2e-7 in the loss and 0.23% in the gradient, and every GPU of both boxes holds the same loss bit for bit.
A simulated rank of a 72-GPU world takes 29.5 ms and 210 MB at R = 4096, against PyTorch's 101 ms and 26.6 GB.

## Multimem Collectives

Eight GPUs, Float32 sums in place, nccl-tests' bus bandwidth, against NCCL 2.30.7, the one PyTorch bundles, with NVLink SHARP on and off.

| Payload | All-reduce | NCCL NVLS | NCCL without NVLS | Reduce-scatter | NCCL NVLS |
| ------: | ---------: | --------: | ----------------: | -------------: | --------: |
|   1 MiB |    26.7 µs |   55.1 µs |           53.4 µs |        27.0 µs |   56.6 µs |
|  16 MiB |   548 GB/s |  266 GB/s |          346 GB/s |       356 GB/s |  257 GB/s |
|  64 MiB |   746 GB/s |  416 GB/s |          461 GB/s |       521 GB/s |  412 GB/s |
| 128 MiB |   791 GB/s |  587 GB/s |          617 GB/s |       565 GB/s |  560 GB/s |

- The barrier takes 25.5 µs, against 43–47 µs for NCCL's smallest all-reduce.
- Keyed integers sum exactly, and 1,000 all-reduces of 128 MiB of keyed fractions repeat bit for bit.

## Footprint & Portability

- The binary is 13.4 MB and needs only the CUDA driver, the 0.8 MB `libcudart` beside it, and libibverbs, where cuBLASLt with cuBLAS weighs 603 MB, NCCL 242 MB and NVSHMEM 78 MB.
- It compiles for B300 alone, `sm_103a`, as `tcgen05` and TMA need the architecture-specific target.
- It needs an NVSwitch domain with multicast, and NICs with dma-buf RDMA and RoCE v2 IPv4 GIDs across boxes.
