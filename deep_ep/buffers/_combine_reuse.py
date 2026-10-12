# Copyright (c) 2026, Lu Lu

"""Capacity fallback for server Combine; normal handles retain their original tables."""

from ._elastic_layout import combine_layout, integer


def select_window(key, capacity, forced_tokens=0):
    world, tokens, topk, rows, gathered, chunk, hidden = key
    integer("reuse tokens", forced_tokens, 0, tokens)
    full = combine_layout(*key)
    if not forced_tokens and full["workspace_bytes"] <= capacity:
        return 0, full
    maximum = forced_tokens or min(tokens, chunk)

    def layout(window):
        return combine_layout(world, tokens, topk, window, window * topk, chunk, hidden)

    if layout(1)["workspace_bytes"] > capacity:
        raise ValueError("Combine fixed buffers and one reuse token exceed its reserved SHMEM half")
    if forced_tokens:
        result = layout(forced_tokens)
        if result["workspace_bytes"] > capacity:
            raise ValueError("Requested Combine reuse window exceeds its reserved SHMEM half")
        return forced_tokens, result
    low, high = 1, maximum
    while low < high:
        middle = (low + high + 1) // 2
        if layout(middle)["workspace_bytes"] <= capacity:
            low = middle
        else:
            high = middle - 1
    return low, layout(low)


def build_plan(handle, window):
    import numpy as np
    import torch

    cached = getattr(handle, "_combine_reuse_plan", None)
    if cached is not None and cached[0] == window:
        return cached[1:]
    world = handle.forward_list.shape[0]
    tokens = handle.server_mask.numel()
    topk = handle.topk_idx.shape[1]
    capacity = window * topk
    ids = handle.weight_return_meta.detach().cpu().numpy().astype(np.int64, copy=False)
    counts = handle.forward_counts.detach().cpu().tolist()
    # Gather rows are a dense prefix over (source, token, secondary slot).
    # A single source/window therefore spans at most window*topk rows. Verify
    # this before applying modulo; arbitrary sparse row IDs cannot be wrapped.
    for source, count in enumerate(counts):
        rows = handle.forward_list[source, :count].detach().cpu().numpy()
        remote = rows[rows[:, 5] >= 0]
        if len(remote):
            gather = remote[:, 5].astype(np.int64)
            if np.any(np.diff(gather) != 1) or np.any(np.diff(remote[:, 1]) < 0):
                raise ValueError("Combine reuse requires dense, token-ordered gather rows")
            periods, size = np.unique(remote[:, 1] // window, return_counts=True)
            if np.any(size > capacity) or np.any(periods < 0):
                raise ValueError("Combine reuse gather window overflows")
    blocks, offsets = [], [0]
    for lane, count in enumerate(handle.backward_counts.detach().cpu().tolist()):
        rows = handle.backward_list[lane, :count].detach().cpu().numpy().copy()
        if count:
            source = rows[:, 0] & 0xFFFFFFFF
            expert = rows[:, 1]
            if np.any(expert < 0) or np.any(expert >= len(ids)):
                raise ValueError("Combine reuse expert row is out of bounds")
            route = ids[expert]
            if np.any(route < 0) or np.any(route // (tokens * topk) != source):
                raise ValueError("Combine reuse backward route identity differs")
            token = route // topk % tokens
            epoch = token // window * world + source + 1
            rows[:, 2] = ((rows[:, 2] & 0xFFFFFFFF) % capacity) | (epoch << 32)
            rows = rows[np.argsort(epoch, kind="stable")]
        blocks.append(rows)
        offsets.append(offsets[-1] + count)
    device = handle.backward_list.device
    backward = torch.from_numpy(np.concatenate(blocks, axis=0)).to(device=device)
    boundaries = torch.tensor(offsets, dtype=torch.int64, device=device)
    handle._combine_reuse_plan = (window, backward, boundaries)
    return backward, boundaries
