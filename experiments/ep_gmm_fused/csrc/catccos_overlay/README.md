<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Modified by zhu-mingzhe71 2026
-->

# CatCCOS UDMA overlay

The build uses the official CatCCOS repository at revision
`878c2e0a504be8c49c03b5e1157a7c714a628e29`.

The headers under `include/catccos` are the minimal all-to-all-v UDMA delta
required by this operator and are placed before the official CatCCOS include
directory. They preserve the validated UDMA peer-serial scheduling and
layout-aware two-dimensional copy path while all other CatCCOS, CATLASS, and
SHMEM sources continue to come from the official third-party checkout.

The overlay also carries the follow-up UDMA MoE delta from CatCCOS commit
`1d96ae736f1054a28b0ddf3cbb96cd0c64c81a21`: the optional AllToAllV export
variant and the GroupedMatmul-AllToAllV non-communication-core deadlock fix.
CatCCOS commit `2444c7ee5acee145c8a296e3462b4bcf487571b8` further pipelines
the optional export: AIV subcore 0 exports the previous staging buffer while
subcore 1 submits the current local MTE and remote UDMA communication.
The GroupedMatMul-AllToAllV overlay additionally carries separate input and
output row extents so unbalanced routes can consume `sum(global_tokens)` rows
and return `sum(local_tokens)` rows without requiring those totals to match.
It also carries the target-order AIC/AIV schedulers from CatCCOS commit
`f03b24518e25d9965977b3d70813238f799da2bb`. Complete N-width row ranges are
staged in destination order, reducing each source-rank/round transfer to one
contiguous UDMA request; the final scheduler coordinate-type fixes are included.
The overlaid problem-shape ABI reads both route-count tables from contiguous
signed 64-bit storage, matching the public PyTorch `torch.int64` interface.

CatCCOS delivery commit `54522865d82ecf988d306fa42329d77cca9a179e`
adds the physical-tiling foundation used by both kernels: explicit staging-row
capacity, two/three-buffer kernel templates, configurable export/self-copy AIV
worker pools, and the historical `256x256x256` GMM-AllToAllV template. For
AllToAllV-GroupedMatmul it also aggregates all source-rank fragments of one
expert into a single AIC task domain and supports the `mSplit` M-axis swizzle.
The optional route-metadata ABI remains nullable in this PyTorch integration so
device route tensors do not force a synchronous host round trip per invocation.
Repository-local fixes layered after the CatCCOS branch—dynamic output extents,
UDMA completion publication after queue quiet, and guarded workspace-loop
subtraction—remain applied on top of this delivery update.

The physical staging and parallel self-copy updates from ops-transformer MR
11602 at `3ab6b8bba2cbdaf3751596c3e4b0488e1991df5f` are integrated for
GroupedMatmul-AllToAllV. Both directions instantiate `128x256x256`,
`256x128x256` and `256x256x256`; the inverse supports two or three workspace
stages, explicit stage-row capacity and configurable self-copy workers.
Each remote peer retains one completion owner. Only the primary self worker
publishes receive completion; all self workers finish their MTE block loops
and join the AIV barrier before workspace reuse. Peer-scoped UDMA completion
and `aclshmemx_mte_quiet()` remain in place.
Legacy five-field inverse tilings retain derived staging, two workspace stages
and one self-copy worker. The EP8/256-expert defaults are unchanged; new EP4/
28-expert profiles are selected by the host for the documented shapes.
Host parsing and Python search reject all other MMAD shapes.
Local copy staging uses the pinned CATLASS Ascend950 allocation of 248 KiB,
including a 256-byte UDMA scratch reservation, rather than physical 256 KiB UB.
