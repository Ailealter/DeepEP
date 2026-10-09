# Copyright (c) 2026, Lu Lu

"""Forward-list contract checks and installed-wheel, bitwise NPU dispatch validation.

CPU: python -m pytest tests/elastic/test_dispatch.py -q
NPU: python tests/elastic/test_dispatch.py --ep-size 2 --nproc-per-node 2 \
     --output /tmp/dispatch-check-001
"""

import argparse
import csv
import math
import statistics
import json
import os
from pathlib import Path
import importlib
from importlib.machinery import PathFinder
import sys
import sysconfig
import time
import traceback


_ROUTES = ("normal", "random", "duplicates", "invalid", "empty", "hotspot")


def _progress_logger(operation, rank):
    started = previous = time.perf_counter()

    def log(message):
        nonlocal previous
        now = time.perf_counter()
        print(
            f"[{operation}][rank={rank}] {time.strftime('%H:%M:%S')} "
            f"elapsed={now - started:.3f}s delta={now - previous:.3f}s {message}",
            flush=True,
        )
        previous = now

    return log


def _log_failure(log):
    # Cleanup may wait for device work or peers; expose the original error first.
    log("FAILED: traceback before buffer.destroy")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()


def group_peer(rank, world, group, index):
    if world == 128:
        toggle, subround = divmod(group, 4)
        row, side = rank % 64 // 32, rank % 8 // 4
        peer_side = side ^ (subround % 2)
        peer_row = (row + subround // 2) % 2
        frame = rank // 64 ^ side ^ toggle
        return frame * 64 + (peer_row * 4 + index // 4) * 8 + peer_side * 4 + index % 4
    width, half = min(world, 16), world // 2
    half_width, groups = width // 2, world // width
    own = rank % half // half_width
    target = (own + group) % groups
    return index // half_width * half + target * half_width + index % half_width


def group_steps(rank, world):
    steps = [0] * world
    for group in range(world // min(world, 16)):
        for index in range(min(world, 16)):
            steps[group_peer(rank, world, group, index)] = (group * min(world, 16) + index) // min(world, 64)
    return steps


def build_notify_golden(routes, experts, chunk_tokens):
    world = len(routes)
    if world not in (2, 4, 8, 16, 32, 64, 128, 256) or experts <= 0 or experts % world:
        raise ValueError("unsupported Notify shape")
    tokens = len(routes[0])
    topk = len(routes[0][0]) if tokens else 0
    if (
        chunk_tokens <= 0
        or not 1 <= topk <= 16
        or any(len(rows) != tokens or any(len(row) != topk for row in rows) for rows in routes)
    ):
        raise ValueError("routes must have a common [tokens, topk] shape")
    local_experts = experts // world
    splits = min(4, tokens, chunk_tokens)
    first = min(tokens, chunk_tokens)
    chunks = (tokens + chunk_tokens - 1) // chunk_tokens + splits - 1

    def token_chunk(token):
        return token * splits // first if token < first else token // chunk_tokens + splits - 1

    def chunk_end(chunk):
        return (
            (chunk + 1) * first // splits + bool((chunk + 1) * first % splits)
            if chunk < splits
            else min((chunk + 2 - splits) * chunk_tokens, tokens)
        )

    counts = [0] * experts
    for rows in routes:
        for row in rows:
            for expert in row:
                if 0 <= expert < experts:
                    counts[expert] += 1
    starts = [0] * experts
    rank_rows = [0] * world
    for expert, count in enumerate(counts):
        peer = expert // local_experts
        starts[expert] = rank_rows[peer]
        rank_rows[peer] += count
    stride = max(1, max(rank_rows, default=0))
    invalid = -(1 << 63)
    plans = []
    for source in range(world):
        plans.append(
            {
                "server_mask": [0] * tokens,
                "dst": [invalid] * (tokens * topk),
                "weight_return_meta": [-1] * rank_rows[source],
                "forward_list": [[] for _ in range(world)],
                "backward_list": [[] for _ in range(min(world, 8))],
                "chunk_ranges": [[(0, 0) for _ in range(chunks)] for _ in range(world)],
                "chunk_masks": [0] * (world // min(world, 64) * chunks),
            }
        )

    expert_cursors = [0] * experts
    gathers = [0] * world
    for source, rows in enumerate(routes):
        for token, route in enumerate(rows):
            by_server = {}
            for slot, expert in enumerate(route):
                if not 0 <= expert < experts:
                    continue
                peer = expert // local_experts
                output_row = starts[expert] + expert_cursors[expert]
                expert_cursors[expert] += 1
                plans[peer]["weight_return_meta"][output_row] = (source * tokens + token) * topk + slot
                server = peer // 8
                plans[source]["server_mask"][token] |= 1 << server
                by_server.setdefault(server, []).append((slot, peer, output_row))
            for records in by_server.values():
                primary_slot, gateway, primary_row = records[0]
                for slot, peer, output_row in records:
                    encoded = peer * stride + output_row
                    value = encoded if slot == primary_slot else ~encoded
                    plans[source]["dst"][token * topk + slot] = value
                forward = plans[gateway]["forward_list"][source]
                if len(records) == 1:
                    forward.append((source, token, primary_row, -1, -1, -1))
                    continue
                for _, peer, output_row in records[1:]:
                    reduce_row = -1
                    if peer != gateway:
                        reduce_row = gathers[gateway]
                        gathers[gateway] += 1
                        plans[peer]["backward_list"][gateway % 8].append((source, output_row, reduce_row, token))
                    forward.append((source, token, primary_row, peer, output_row, reduce_row))

    for gateway, plan in enumerate(plans):
        for source, block in enumerate(plan["forward_list"]):
            begin = 0
            while begin < len(block):
                end = begin + 1
                while end < len(block) and block[end][1] == block[begin][1]:
                    end += 1
                for row in range(begin, end):
                    block[row] = (end, *block[row][1:])
                begin = end
            cursor = 0
            for chunk in range(chunks):
                begin = cursor
                token_end = chunk_end(chunk)
                while cursor < len(block) and block[cursor][1] < token_end:
                    peer = block[cursor][3]
                    if peer >= 0 and peer != gateway:
                        plan["chunk_masks"][group_steps(gateway, world)[source] * chunks + chunk] |= 1 << (peer % 8)
                    cursor += 1
                plan["chunk_ranges"][source][chunk] = (begin, cursor)
        for lane, block in enumerate(plan["backward_list"]):
            gateway_peer = gateway // 8 * 8 + lane
            gateway_steps = group_steps(gateway_peer, world)
            ordered = list(enumerate(block))
            ordered.sort(
                key=lambda item: (
                    token_chunk(item[1][3]),
                    gateway_steps[item[1][0]],
                    item[1][0],
                    item[0],
                )
            )
            block[:] = [item[1] for item in ordered]
            for index, (source, row, reduce_row, token) in enumerate(block):
                key = gateway_steps[source] * chunks + token_chunk(token) + 1
                next_key = 0
                if index + 1 == len(block):
                    next_key = key
                else:
                    next_source, _, _, next_token = block[index + 1]
                    candidate = gateway_steps[next_source] * chunks + token_chunk(next_token) + 1
                    if candidate != key:
                        next_key = key
                block[index] = (source, row, reduce_row | (next_key << 32))
        plan["forward_counts"] = [len(block) for block in plan["forward_list"]]
        plan["backward_counts"] = [len(block) for block in plan["backward_list"]]
    for plan in plans:
        plan["rank_rows"] = rank_rows
        plan["gather_rows"] = gathers
    return {"plans": plans, "rank_rows": rank_rows, "gather_rows": plans[0]["gather_rows"], "address_stride": stride}


def make_routes(world, tokens, topk, experts, generation, mode, *, seed=1):
    """Generate deterministic routes, including duplicate expert slots.

    Seed/source/token determine routing; generation only changes normal fixtures.
    Keep random routes fixed across payload generations for cached dispatch.
    """
    result = []
    for source in range(world):
        rows = []
        for token in range(tokens):
            row = [(source * 17 + token * 7 + slot * 131 + generation) % experts for slot in range(topk)]
            if mode == "random":
                state = (seed ^ (source * 0x9E3779B9) ^ (token * 0x85EBCA6B)) & 0xFFFFFFFF
                row = []
                for _ in range(topk):
                    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                    row.append((state >> 8) % experts)
            if mode == "duplicates" and topk > 1:
                row[1] = row[0]
            if mode == "invalid":
                row = [
                    (experts + 17 if (source + token + slot) % 2 else -1) if (source + token + slot) % 3 == 0 else e
                    for slot, e in enumerate(row)
                ]
            if mode == "empty":
                row = [-1] * topk
            if mode == "hotspot":
                row = [experts - 1] * topk
            if mode == "sq_pressure":
                row = [0 if source in (8, 20, 48, 52, 80, 84, 112, 116) else -1] * topk
            if mode in ("local_duplicates", "local_experts"):
                local = experts // world
                peer = (source + token) % world
                row = [peer * local + (slot % local if mode == "local_experts" else 0) for slot in range(topk)]
            rows.append(row)
        result.append(rows)
    return result


def receive_capacity(routes, experts):
    """Exact maximum expanded rows, without constructing all Notify metadata."""
    world = len(routes)
    if experts < world or experts % world:
        raise ValueError("experts must be divisible by EP and at least EP")
    local = experts // world
    counts = [0] * world
    for source in routes:
        for token in source:
            for expert in token:
                if 0 <= expert < experts:
                    counts[expert // local] += 1
    return max(1, max(counts))


def test_random_routes_match_reference_and_preserve_cached_routing():
    import runpy

    # Compare with the CPU reference's independent random route implementation.
    reference = runpy.run_path(str(Path(__file__).with_name("combine_reference.py")))
    for world in (2, 8, 256):
        for seed in (0, 1, 2026, -1, 1 << 40):
            case = reference["CombineCase"](world, 7, 6, world * 2, "random", seed)
            actual = make_routes(world, 7, 6, world * 2, 0, "random", seed=seed)
            expected = [
                [reference["route_experts"](source, token, case) for token in range(7)] for source in range(world)
            ]
            assert actual == expected
            assert actual == make_routes(world, 7, 6, world * 2, 1, "random", seed=seed)
            assert all(0 <= expert < world * 2 for rows in actual for row in rows for expert in row)
    routes = make_routes(8, 32, 6, 16, 0, "random", seed=1)
    assert routes != make_routes(8, 32, 6, 16, 0, "random", seed=2026)
    assert routes[0] != routes[1]
    assert any(len(set(row)) < len(row) for rows in routes for row in rows)
    counts = [len(expert_rows(routes, 16, rank)) for rank in range(8)]
    assert len(set(counts)) > 1  # No balancing pass or per-token deduplication.
    assert receive_capacity(routes, 16) == max(counts)
    assert receive_capacity([[[-1, 2]], [[-1, 2]]], 2) == 1
    assert make_routes(8, 7, 6, 16, 0, "normal", seed=1) == make_routes(8, 7, 6, 16, 0, "normal", seed=2026)


def test_random_dispatch_cli(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_dispatch.py",
            "--ep-size",
            "256",
            "--nproc-per-node",
            "8",
            "--node-rank",
            "31",
            "--tokens",
            "4096",
            "--topk",
            "6",
            "--experts",
            "512",
            "--route",
            "random",
            "--seed",
            "2026",
            "--output",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(sys.modules[__name__], "_launch", lambda args, worker: calls.append(args))
    main()
    (args,) = calls
    assert (args.ep_size, args.node_rank, args.route, args.seed) == (256, 31, "random", 2026)


def check_handle(handle, golden, rank, routes, experts):
    import torch

    plan = golden["plans"][rank]
    if handle.metadata_only or handle.metadata_version != 7:
        raise AssertionError("Expected expanded dispatch handle metadata v6")
    for name in (
        "server_mask",
        "dst",
        "forward_counts",
        "backward_counts",
        "chunk_ranges",
        "chunk_masks",
        "rank_rows",
        "gather_rows",
    ):
        actual = getattr(handle, name)
        if actual.device.type != "npu":
            raise AssertionError(f"{name} is not an NPU tensor")
        expected = torch.tensor(plan[name], dtype=actual.dtype).reshape(actual.shape)
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError(f"rank {rank}: {name} mismatch")
    for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
        actual = getattr(handle, name).cpu()
        for row, count in enumerate(plan[counts]):
            expected = torch.tensor(plan[name][row], dtype=actual.dtype).reshape(count, actual.shape[-1])
            if not torch.equal(actual[row, :count], expected):
                raise AssertionError(f"rank {rank}: {name}[{row}] mismatch")
    weight_meta = handle.weight_return_meta
    weight_meta_cpu = weight_meta.cpu()
    expected_weight_meta = torch.tensor(
        [(s * handle.num_tokens + t) * handle.num_topk + k for s, t, k in expert_rows(routes, experts, rank)],
        dtype=torch.int32,
    )
    if (
        weight_meta.dtype != torch.int32
        or weight_meta.device.type != "npu"
        or not weight_meta.is_contiguous()
        or not torch.equal(weight_meta_cpu, expected_weight_meta)
    ):
        raise AssertionError(f"rank {rank}: weight_return_meta mismatch")
    totals = [0] * experts
    for source in routes:
        for row in source:
            for expert in row:
                if 0 <= expert < experts:
                    totals[expert] += 1
    local = experts // len(routes)
    expected_counts = totals[rank * local : (rank + 1) * local]
    if handle.num_recv_tokens_per_expert_list != expected_counts:
        raise AssertionError("Expert counts mismatch")
    for name, expected in (
        ("expert_totals", torch.tensor(totals, dtype=torch.int32)),
        ("num_unaligned_recv_tokens_per_expert", torch.tensor(expected_counts, dtype=torch.int32)),
        (
            "psum_num_recv_tokens_per_expert",
            torch.tensor(expected_counts, dtype=torch.int32).cumsum(0, dtype=torch.int32),
        ),
    ):
        if not torch.equal(getattr(handle, name).cpu(), expected):
            raise AssertionError(f"rank {rank}: {name} mismatch")
    if handle.num_recv_tokens is not None or handle.dst_buffer_slot_idx is not None:
        raise AssertionError("Expanded dispatch must leave ordinary-layout fields unset")
    if handle.address_stride != golden["address_stride"]:
        raise AssertionError("Destination stride mismatch")
    if handle.num_expanded_tokens != golden["rank_rows"][rank]:
        raise AssertionError("Expanded token count mismatch")


def _import_installed_deep_ep():
    # Launching from the checkout or inheriting PYTHONPATH must not shadow the wheel.
    locations = list(dict.fromkeys(sysconfig.get_path(name) for name in ("purelib", "platlib")))
    spec = PathFinder.find_spec("deep_ep", locations)
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"deep_ep is not installed in {sys.executable}; "
            "run python -m pip install --no-build-isolation --no-deps . first"
        )
    expected = Path(spec.origin).resolve()
    sys.path.insert(0, str(Path(spec.origin).parent.parent))
    package = importlib.import_module("deep_ep")
    actual = Path(package.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"deep_ep was already loaded from {actual}; expected installed package {expected}")
    return package


def _add_launch_args(parser):
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8899)
    parser.add_argument("--shmem-ip-port", help="Defaults to DEEPEP_SHMEM_ENDPOINT or master-port + 1")


def _launch(args, worker):
    # torchrun may still supply process identities, but no process group is created.
    launched = [name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    if any(launched) and not all(launched):
        raise ValueError("RANK, WORLD_SIZE and LOCAL_RANK must be supplied together")
    if all(launched):
        args.ep_size = int(os.environ["WORLD_SIZE"])
    elif (
        args.nproc_per_node < 1
        or args.ep_size % args.nproc_per_node
        or not 0 <= args.node_rank < args.ep_size // args.nproc_per_node
    ):
        raise ValueError("EP must be divisible by nproc-per-node; node-rank must be within the job")
    if args.ep_size not in (2, 4, 8, 16, 32, 64, 128, 256):
        raise ValueError("Dispatch/combine EP must be a power of two from 2 to 256")
    endpoint = args.shmem_ip_port or os.environ.get("DEEPEP_SHMEM_ENDPOINT")
    if not endpoint:
        master = os.environ.get("MASTER_ADDR", args.master_addr) if all(launched) else args.master_addr
        port = int(os.environ.get("MASTER_PORT", args.master_port)) if all(launched) else args.master_port
        endpoint = f"tcp://{master}:{port + 1}"
    os.environ["DEEPEP_SHMEM_ENDPOINT"] = endpoint
    if all(launched):
        worker(int(os.environ["LOCAL_RANK"]), args)
    else:
        import torch.multiprocessing as mp

        mp.spawn(worker, args=(args,), nprocs=args.nproc_per_node, join=True)


def _worker_identity(local_rank, args):
    rank = int(os.environ.get("RANK", args.node_rank * args.nproc_per_node + local_rank))
    if not 0 <= rank < args.ep_size or local_rank < 0:
        raise ValueError("Invalid rank or local rank")
    return rank, args.ep_size


def test_ep256_launcher(monkeypatch):
    from types import ModuleType

    parser = argparse.ArgumentParser()
    _add_launch_args(parser)
    args = parser.parse_args(
        [
            "--ep-size",
            "256",
            "--nproc-per-node",
            "8",
            "--node-rank",
            "31",
            "--shmem-ip-port",
            "tcp://192.0.2.1:19091",
        ]
    )
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    torch = ModuleType("torch")
    mp = ModuleType("torch.multiprocessing")
    mp.spawn = lambda worker, args, nprocs, join: calls.append((args, nprocs, join))
    torch.multiprocessing = mp
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", mp)
    # Let monkeypatch restore the endpoint written by the launcher.
    monkeypatch.setenv("DEEPEP_SHMEM_ENDPOINT", "tcp://192.0.2.2:1")
    _launch(args, lambda *_: None)
    assert calls == [((args,), 8, True)]
    assert os.environ["DEEPEP_SHMEM_ENDPOINT"] == "tcp://192.0.2.1:19091"
    assert [_worker_identity(local, args)[0] for local in range(8)] == list(range(248, 256))


def expert_rows(routes, experts, rank):
    """Independent destination order: local expert, source rank, token, top-k slot."""
    local = experts // len(routes)
    begin = rank * local
    buckets = [[] for _ in range(local)]
    for source, tokens in enumerate(routes):
        for token, row in enumerate(tokens):
            for slot, selected in enumerate(row):
                if begin <= selected < begin + local:
                    buckets[selected - begin].append((source, token, slot))
    return [entry for bucket in buckets for entry in bucket]


def test_forward_list_covers_unique_expert_rows():
    # Exercise both primary core maps, cross-server forwarding, same-rank
    # duplicates, singleton records, empty ranks and non-power-of-two strides.
    for world in (2, 8, 16, 32, 64, 128, 256):
        for mode in _ROUTES:
            tokens, topk, experts = 3, 6, world * 2
            routes = make_routes(world, tokens, topk, experts, 0, mode)
            golden = build_notify_golden(routes, experts, 256)
            stride = golden["address_stride"]
            output = [[None] * n for n in golden["rank_rows"]]

            def write(peer, row, value):
                assert output[peer][row] is None, "Two tasks write the same output row"
                output[peer][row] = value

            for source, plan in enumerate(golden["plans"]):
                for route, encoded in enumerate(plan["dst"]):
                    if encoded >= 0:
                        peer, row = divmod(encoded, stride)
                        write(peer, row, (source, route // topk))
            primary = [list(rows) for rows in output]
            workers = 16 if world == 128 else 24
            for gateway, plan in enumerate(golden["plans"]):
                for source, records in enumerate(plan["forward_list"]):
                    covered = []
                    for worker in range(workers):
                        begin, end = len(records) * worker // workers, len(records) * (worker + 1) // workers
                        covered.extend(range(begin, end))
                        for record in records[begin:end]:
                            _, token, primary_row, peer, row, _ = record
                            assert primary[gateway][primary_row] == (source, token)
                            if peer >= 0:
                                assert gateway // 8 == peer // 8
                                write(peer, row, primary[gateway][primary_row])
                    assert covered == list(range(len(records)))
            for rank in range(world):
                assert output[rank] == [(source, token) for source, token, _ in expert_rows(routes, experts, rank)]


def _host_elastic_buffer(monkeypatch):
    """Exercise Python orchestration with real CPU tensors and a fake native backend."""
    from collections import Counter
    from contextlib import nullcontext
    from threading import RLock
    from types import SimpleNamespace
    import pytest
    from deep_ep.buffers import elastic
    from deep_ep.utils import event

    torch = pytest.importorskip("torch")
    calls = Counter()
    device = SimpleNamespace(type="npu", index=0)
    empty = torch.empty

    class Tensor(torch.Tensor):
        @property
        def device(self):
            return device

        def record_stream(self, stream):
            calls["record_stream"] += 1

        def cpu(self):
            calls["cpu"] += 1
            return super().cpu()

    def allocate(*args, **kwargs):
        if kwargs.get("device") is device:
            kwargs["device"] = "cpu"
            return torch.Tensor._make_subclass(Tensor, empty(*args, **kwargs), False)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    stream = SimpleNamespace(npu_stream=1, synchronize=lambda: calls.update(["sync"]))
    npu = SimpleNamespace(
        stream=lambda _: nullcontext(),
        current_device=lambda: 0,
        current_stream=lambda _: stream,
        Event=lambda: SimpleNamespace(record=lambda _: None, wait=lambda _: calls.update(["event_wait"])),
    )

    def notify_layout(rank, world, tokens, topk, experts, chunk, **kwargs):
        shapes = {name: ((16,), "int32") for name in elastic._PERSISTENT}
        shapes.update(
            dst=((tokens, topk), "int64"),
            rank_rows=((world,), "int32"),
            gather_rows=((world,), "int32"),
            expert_totals=((experts,), "int32"),
            weight_return_meta=((world * tokens * topk,), "int32"),
            _peer_plan=((16,), "uint8"),
        )
        return {
            "workspace_bytes": elastic.workspace_layout(world, tokens, topk, experts, chunk)["workspace_bytes"],
            "tensors": shapes,
        }

    def notify(session, indices, outputs, *args):
        calls["notify"] += 1
        for tensor in outputs.values():
            assert not torch.is_inference(tensor)
            tensor.zero_()
        outputs["rank_rows"].fill_(2)
        outputs["expert_totals"].fill_(2)
        outputs["gather_rows"].fill_(1)
        return None, None

    def layout(kind, args):
        calls[kind + "_layout"] += 1
        return getattr(elastic, kind + "_layout")(*args[1:])

    native = SimpleNamespace(
        notify_check_mapping=lambda _: None,
        notify_layout=notify_layout,
        notify=notify,
        dispatch_layout=lambda *args: layout("dispatch", args),
        combine_layout=lambda *args: layout("combine", args),
        dispatch=lambda *args: calls.update(["dispatch"]),
        combine=lambda *args: calls.update(["combine"]),
    )
    for module in (elastic, event):
        monkeypatch.setattr(module, "require_native", lambda _: native)
        monkeypatch.setattr(module, "npu_api", lambda: npu)
    buffer = elastic.ElasticBuffer.__new__(elastic.ElasticBuffer)
    buffer.hidden = 0
    buffer.num_max_tokens_per_rank, buffer.num_topk = 16, 1
    buffer._handle_token = object()
    buffer._runtime = SimpleNamespace(
        ensure_alive=lambda: None,
        notify_failed=False,
        notify_lock=RLock(),
        group=None,
        device=0,
        rank=0,
        world_size=2,
        stream=stream,
        native=SimpleNamespace(peer_table=True),
        capacity=1 << 32,
        combine_only=False,
        closed=True,
    )
    return buffer, lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device), calls


def test_inference_dispatch_combine_and_cached_host_work(monkeypatch):
    import pytest
    from deep_ep.buffers.elastic import _PERSISTENT

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    with torch.inference_mode():
        x, indices = tensor((2, 7168), torch.bfloat16), tensor((2, 1), torch.int64)
        recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=2)
        assert torch.is_inference(x) and torch.is_inference(indices)
        assert handle.topk_idx.data_ptr() != indices.data_ptr()
        for name in ("topk_idx", "dispatch_dst", *_PERSISTENT, "_peer_plan"):
            value = getattr(handle, name)
            assert not torch.is_inference(value)
            assert handle._routing_versions[name] == (value.data_ptr(), value._version)
        assert calls["cpu"] == 1
        assert handle.address_stride == 2 and handle._gather_stride == 1
        calls.clear()
        for _ in range(2):
            buffer.combine(recv, handle, async_with_compute_stream=True)
            buffer.dispatch(x, handle=handle, async_with_compute_stream=True)
        assert calls["combine"] == calls["dispatch"] == 2
        assert calls["cpu"] == calls["sync"] == calls["notify"] == 0
        assert calls["combine_layout"] == 1 and calls["dispatch_layout"] == 0
        assert calls["event_wait"] and calls["record_stream"]
        handle.forward_counts.add_(1)
        with pytest.raises(ValueError, match="Cached routing was modified"):
            buffer.combine(recv, handle)
        with pytest.raises(ValueError, match="Cached routing was modified"):
            buffer.dispatch(x, handle=handle)
        assert calls["combine"] == calls["dispatch"] == 2


def test_inference_routing_copy_policy(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    ordinary_indices = tensor((2, 1), torch.int64)
    with torch.inference_mode():
        x, indices = tensor((2, 7168), torch.bfloat16), tensor((2, 1), torch.int64)
        with pytest.raises(ValueError, match="do_handle_copy=True"):
            buffer.dispatch(x, topk_idx=indices, num_experts=2, do_handle_copy=False)
        assert calls["notify"] == calls["dispatch"] == 0
        assert not buffer._runtime.notify_failed
        recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=ordinary_indices, num_experts=2, do_handle_copy=False)
        assert handle.topk_idx is ordinary_indices
    buffer.combine(recv, handle)
    with torch.inference_mode():
        handle.forward_list = tensor(handle.forward_list.shape, handle.forward_list.dtype)
        with pytest.raises(ValueError, match="Cached routing was modified"):
            buffer.combine(recv, handle)
    assert calls["combine"] == 1


def test_cached_layout_variants_and_capacity_checks(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    x, indices = tensor((2, 7168), torch.bfloat16), tensor((2, 1), torch.int64)
    recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=2)
    weights = tensor((2, 1), torch.float32)
    buffer.dispatch(x, handle=handle, topk_weights=weights)
    buffer.dispatch(x, handle=handle, topk_weights=weights)
    fp8 = (tensor((2, 7168), torch.float8_e4m3fn), tensor((2, 56), torch.float32))
    buffer.dispatch(fp8, handle=handle, topk_weights=weights)
    buffer.dispatch(x, handle=handle)
    assert calls["dispatch_layout"] == 3
    buffer.combine(recv, handle)
    buffer._runtime.capacity = 0
    with pytest.raises(ValueError, match="SHMEM"):
        buffer.combine(recv, handle)
    with pytest.raises(ValueError, match="SHMEM"):
        buffer.dispatch(x, handle=handle)
    assert calls["combine"] == 1 and calls["dispatch"] == 5
    assert not buffer._runtime.notify_failed


def test_hidden_shapes_and_cached_contract(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    buffer, tensor, calls = _host_elastic_buffer(monkeypatch)
    indices = tensor((2, 1), torch.int64)
    for width in (16, 48, 128, 144, 1024, 4096, 6144, 7040, 7152, 7168):
        x = tensor((2, width), torch.bfloat16)
        recv, _, _, handle, _ = buffer.dispatch(x, topk_idx=indices, num_experts=2)
        assert recv.shape == (2, width) and handle.hidden == width
        out, _, _ = buffer.combine(recv, handle)
        assert out.shape == (2, width)
        buffer.dispatch(x, handle=handle)
        if width % 128 == 0:
            fp8 = (tensor((2, width), torch.float8_e4m3fn), tensor((2, width // 128), torch.float32))
            recv_fp8, _, _, _, _ = buffer.dispatch(fp8, handle=handle)
            assert recv_fp8[0].shape == (2, width) and recv_fp8[1].shape == (2, width // 128)
        with pytest.raises(ValueError, match="original hidden"):
            buffer.dispatch(tensor((2, 32 if width == 16 else 16), torch.bfloat16), handle=handle)
        with pytest.raises(ValueError, match="matching the handle"):
            buffer.combine(tensor((2, width + 16), torch.bfloat16), handle)
    launched = calls["dispatch"]
    for width in (0, 15, 17, 7169, 7296):
        with pytest.raises(ValueError, match="hidden"):
            buffer.dispatch(tensor((2, width), torch.bfloat16), topk_idx=indices, num_experts=2)
    with pytest.raises(ValueError, match="scales"):
        buffer.dispatch(
            (tensor((2, 128), torch.float8_e4m3fn), tensor((2, 56), torch.float32)), topk_idx=indices, num_experts=2
        )
    buffer.hidden = 4096
    with pytest.raises(ValueError, match="buffer configuration"):
        buffer.dispatch(tensor((2, 7168), torch.bfloat16), topk_idx=indices, num_experts=2)
    assert calls["dispatch"] == launched


def test_scale_packet_scatter_hidden_sizes(tmp_path):
    import shutil
    import subprocess
    import pytest

    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    header = (Path(__file__).resolve().parents[2] / "csrc/kernels/dispatch/scale_pipeline.hpp").read_text(
        encoding="utf-8"
    )
    expand = header[
        header.index("__simt_vf__ __launch_bounds__(128) inline void Expand") : header.index(
            "__aicore__ inline void WaitPack"
        )
    ]
    source = tmp_path / "scale.cpp"
    source.write_text(
        """
#include <cstdint>
#include <cassert>
#include <vector>
#define __simt_vf__
#define __launch_bounds__(n)
#define __ubuf__
#define __gm__
struct { unsigned x; } threadIdx;
void asc_threadfence() {}
"""
        + expand
        + """
int main() {
    for (unsigned words = 1; words <= 56; ++words) {
        std::vector<uint32_t> packed(3 * 128, 0xdeadbeef), errors(128, 0);
        auto* slots = reinterpret_cast<int32_t*>(packed.data());
        const int32_t targets[] = {2, 0, 1, -1, 3, -1};
        for (unsigned row = 0; row < 6; ++row) {
            slots[row / 2 * 128 + 112 + row % 2] = targets[row];
            for (unsigned col = 0; col < words; ++col)
                packed[row / 2 * 128 + row % 2 * 56 + col] = (row + 1) * 1000 + col;
        }
        std::vector<uint32_t> output(4 * words + 16, 0xabcddcba), expected = output;
        for (unsigned row = 0; row < 6; ++row) if (targets[row] >= 0)
            for (unsigned col = 0; col < words; ++col)
                expected[8 + targets[row] * words + col] = (row + 1) * 1000 + col;
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            Expand(packed.data(), slots, output.data() + 8, 6, words, 56, 4, errors.data());
        assert(output == expected);
        output[8] ^= 1;
        assert(output != expected);
        for (auto error : errors) assert(error == 0);
        slots[112] = 4;
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
            Expand(packed.data(), slots, output.data() + 8, 6, words, 56, 4, errors.data());
        unsigned bad = 0;
        for (auto error : errors) bad += error;
        assert(bad == 1);
    }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("scale.exe" if sys.platform == "win32" else "scale")
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=30)


def test_collective_check_copies_one_summary(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from deep_ep.buffers.elastic import _collective_check

    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    seen = []
    fail_rank = None

    def gather(peers, local, group):
        assert not torch.is_inference(local)
        assert len({peer.untyped_storage().data_ptr() for peer in peers}) == 1
        # Backend workers may write after leaving the caller's inference context.
        with torch.inference_mode(False):
            for rank, peer in enumerate(peers):
                peer.copy_(local)
                if rank == fail_rank:
                    peer[0] = 1
        seen.append(len(peers))

    monkeypatch.setattr(dist, "get_backend", lambda _: "gloo")
    monkeypatch.setattr(dist, "all_gather", gather)
    runtime = SimpleNamespace(group=object(), world_size=4)
    with torch.inference_mode():
        _collective_check(runtime, None, [2, 1, 2])
        fail_rank = 2
        with pytest.raises(ValueError, match=r"ranks \[2\]"):
            _collective_check(runtime, None, [2, 1, 2])
    assert seen == [4, 4]


def test_dispatch_destination_initialization():
    import ast
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[2] / "deep_ep/buffers/elastic.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Execute the production initialization block without requiring an NPU.
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "dispatch_dst" for target in statement.targets)
            for statement in node.body
        )
    )
    code = compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec")

    class ScalarTensor(int):
        def to(self, dtype):
            return self

    torch = SimpleNamespace(
        int32="int32", int64="int64", where=lambda condition, yes, no: ScalarTensor(yes if condition else no)
    )
    for raw, expected in ((0, 0), (2, 2), (3, 4), (5, 6), (-1, -1), (-4, -5), (-6, -7), (-(1 << 63), -(1 << 63))):
        handle = SimpleNamespace(dispatch_dst=None, dst=ScalarTensor(raw))
        scope = dict(handle=handle, rows=3, layout={"address_stride": 4}, torch=torch)
        exec(code, scope)
        assert isinstance(handle.dispatch_dst, ScalarTensor)
        assert handle.dispatch_dst == expected
        cached = handle.dispatch_dst
        handle.dst = object()  # Cached calls must not read/re-encode the Notify tensor.
        exec(code, scope)
        assert handle.dispatch_dst is cached


def test_dispatch_regions_and_encoding_bounds():
    import pytest
    from deep_ep.buffers._elastic_layout import dispatch_layout, workspace_layout

    for world in (2, 8, 16, 64, 128, 256):
        for fp8 in (False, True):
            for rows in (1, 3, world * 33 * 6):
                layout = dispatch_layout(world, 33, 6, 1024, rows, fp8)
                assert layout["sync"] == workspace_layout(world, 33, 6)["workspace_bytes"]
                assert rows <= layout["address_stride"] < 2 * rows
                offsets = [
                    layout[name]
                    for name in ("sync", "book", "input", "output", "weights", "scales", "inbox", "peer_book", "done")
                ]
                assert offsets == sorted(offsets) and all(n % 512 == 0 for n in offsets)
                row_bytes = 7168 if fp8 else 14336
                assert layout["weights"] - layout["output"] >= rows * row_bytes
                assert layout["peer_book"] == layout["inbox"] + world * layout["inbox_stride"]
                assert layout["done"] == layout["peer_book"] + world * 512
                assert layout["workspace_bytes"] == layout["done"] + world * 512
                assert layout["num_bytes"] >= layout["workspace_bytes"]
                assert layout["num_bytes"] % (2 << 20) == 0
    for arguments in ((1024, 1, 1, 1024, 1), (2, 1, 1, 1024, 3), (2, 1, 1, 1024, 0)):
        with pytest.raises(ValueError):
            dispatch_layout(*arguments)


def assert_bytes(actual, expected, name, progress=None):
    import torch

    if progress:
        progress(f"{name}: readback.begin (waiting for device result)")
    actual = actual.cpu().contiguous()
    if progress:
        progress(f"{name}: readback.done; precision.begin")
    expected = expected.cpu().contiguous()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape/dtype {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
    a, b = actual.view(torch.uint8).flatten(), expected.view(torch.uint8).flatten()
    bad = torch.nonzero(a != b).flatten()
    if bad.numel():
        first = bad[:8]
        raise AssertionError(
            f"{name}: {bad.numel()} mismatched bytes; offsets={first.tolist()}, "
            f"actual={a[first].tolist()}, expected={b[first].tolist()}"
        )
    if progress:
        progress(f"{name}: precision.passed")


def payload(source, tokens, generation, fp8, width=7168):
    import torch

    # Bounded independently of EP, token count and hidden size. Neither BF16
    # nor FP8 construction can overflow; no arithmetic runs in dispatch.
    rows = torch.arange(tokens, dtype=torch.int32)[:, None]
    columns = torch.arange(width, dtype=torch.int32)[None, :]
    hidden = ((source * 13 + rows * 7 + columns * 3 + generation * 11) % 127 - 63).float() / 16
    hidden = hidden.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
    scales = ((source + rows + torch.arange(width // 128)[None, :] + generation) % 7 + 1).float() / 8
    return hidden, scales


def weights_for(source, tokens, topk, generation):
    import torch

    return ((torch.arange(tokens * topk).reshape(tokens, topk) + source * 3 + generation) % 31 + 1).float() / 32


def expected_payload(routes, experts, rank, generation, fp8, *, entries=None, width=7168):
    import torch

    if entries is None:
        entries = expert_rows(routes, experts, rank)
    topk = len(routes[0][0])
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    hidden = torch.empty((len(entries), width), dtype=dtype)
    scales = torch.empty((len(entries), width // 128), dtype=torch.float32)
    weights = torch.empty((len(entries),), dtype=torch.float32)
    if not entries:
        return hidden, scales, weights
    columns = torch.arange(width, dtype=torch.int32)[None, :]
    # Quantize the 127 distinct rows once; gather bytes to preserve FP8 rounding.
    phases = torch.arange(127, dtype=torch.int32)[:, None]
    lookup = (((phases + columns * 3) % 127 - 63).float() / 16).to(dtype).view(torch.uint8)
    scale_columns = torch.arange(width // 128, dtype=torch.int32)[None, :]
    for begin in range(0, len(entries), 256):
        end = min(begin + 256, len(entries))
        coordinates = torch.tensor(entries[begin:end], dtype=torch.int32)
        source, token, slot = (coordinates[:, i : i + 1] for i in range(3))
        phase = ((source * 13 + token * 7 + generation * 11) % 127).flatten().long()
        hidden[begin:end].view(torch.uint8).copy_(lookup.index_select(0, phase))
        scales[begin:end] = ((source + token + scale_columns + generation) % 7 + 1).float() / 8
        weights[begin:end] = ((token * topk + slot + source * 3 + generation) % 31 + 1).flatten().float() / 32
    return hidden, scales, weights


_PROFILE_KERNELS = {"notify": "notify_full_kernel", "dispatch": "dispatch_kernel", "combine": "combine_kernel"}


def add_profile_args(parser, operations):
    parser.add_argument("--profile-op", choices=operations, help="Profile device kernels after precision validation")
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=10)


def validate_profile_args(parser, args):
    if args.profile_warmup < 0 or args.profile_iters < 1:
        parser.error("--profile-warmup must be >=0 and --profile-iters must be >=1")


def _statistics(kernel, durations):
    return dict(
        kernel=kernel,
        unit="us",
        samples=len(durations),
        durations_us=durations,
        avg_us=statistics.mean(durations),
        median_us=statistics.median(durations),
        min_us=min(durations),
        max_us=max(durations),
    )


def read_kernel_summary(directory, operator, expected_samples):
    """Select one CSV source only; retain all samples and reject incomplete captures."""
    kernel = _PROFILE_KERNELS[operator]
    wanted = {kernel}
    if operator == "combine":
        wanted.add("combine_prepare_kernel")
    for pattern in ("kernel_details*.csv", "op_summary*.csv", "task_time*.csv"):
        tables = []
        for path in sorted(Path(directory).rglob(pattern)):
            values = {name: [] for name in wanted}
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                headers = {key.strip(): key for key in reader.fieldnames or ()}
                name_key = next((headers[k] for k in ("Name", "Op Name", "Kernel Name") if k in headers), None)
                units = [
                    (f"{prefix}({unit})", scale)
                    for unit, scale in (("us", 1.0), ("ns", 0.001), ("ms", 1000.0))
                    for prefix in ("Duration", "Task Duration")
                ]
                duration = next(((headers[k], scale) for k, scale in units if k in headers), None)
                if name_key is None or duration is None:
                    continue
                duration_key, scale = duration
                for row in reader:
                    name = (row.get(name_key) or "").strip()
                    if name not in wanted:
                        continue
                    try:
                        value = float(row[duration_key]) * scale
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"Invalid duration for {name} in {path}") from exc
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"Invalid duration for {name} in {path}: {value}")
                    values[name].append(value)
            if values[kernel]:
                tables.append((path, values))
        if not tables:
            continue
        if len(tables) != 1:
            raise ValueError(f"Ambiguous {kernel} capture: {len(tables)} {pattern} tables under {directory}")
        path, values = tables[0]
        for name in wanted:
            if len(values[name]) != expected_samples:
                raise ValueError(f"{name}: expected {expected_samples} samples, found {len(values[name])} in {path}")
        summary = _statistics(kernel, values[kernel])
        summary.update(
            expected_samples=expected_samples, source_csv=str(path.resolve()), metric="device_kernel_duration"
        )
        if operator == "combine":
            summary["prepare"] = _statistics("combine_prepare_kernel", values["combine_prepare_kernel"])
        return summary
    raise ValueError(f"No {kernel} duration records found in {directory}")


def capture_calls(profiler, invoke, synchronize, warmup, iterations, log):
    """One warmup step and one active step, with exactly the requested call counts."""
    last = None
    if warmup:
        log(f"profile warmup.begin calls={warmup}")
        for _ in range(warmup):
            last = invoke()
        synchronize()
        profiler.step()
    log(f"profile capture.begin calls={iterations}")
    for _ in range(iterations):
        last = invoke()
    synchronize()
    profiler.step()
    return last


def profile_operator(args, rank, scenario, invoke, validate, build_info, log):
    """Capture rank-local kernel durations after warmup."""
    import torch
    import torch_npu

    directory = args.output / "profile" / args.profile_op / scenario / f"rank_{rank}" / str(time.time_ns())
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        operator=args.profile_op,
        rank=rank,
        world_size=args.ep_size,
        scenario=scenario,
        seed=args.seed,
        tokens=args.tokens,
        topk=args.topk,
        experts=args.experts,
        hidden=getattr(args, "hidden", 7168),
        warmup=args.profile_warmup,
        iterations=args.profile_iters,
        torch_version=str(torch.__version__),
        torch_npu_version=str(torch_npu.__version__),
        ascend_home_path=os.environ.get("ASCEND_HOME_PATH"),
        build_info=build_info,
    )
    toolkit = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    metadata["cann_version_files"] = {
        str(path): path.read_text(encoding="utf-8", errors="replace")
        for path in (toolkit / "version.cfg", toolkit / "version.info")
        if path.is_file()
    }
    (directory / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log(f"profile op={args.profile_op} scenario={scenario} output={directory}")
    torch.npu.synchronize()
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=int(args.profile_warmup > 0), active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=profiler._ExperimentalConfig(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level0,
            aic_metrics=profiler.AiCMetrics.AiCoreNone,
            data_simplification=False,
        ),
    ) as capture:
        last = capture_calls(capture, invoke, torch.npu.synchronize, args.profile_warmup, args.profile_iters, log)
    torch.npu.synchronize()
    validate(last)
    summary = read_kernel_summary(directory, args.profile_op, args.profile_iters)
    summary.update(metadata, precision_checked=True)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(
        f"profile kernel={summary['kernel']} samples={summary['samples']} avg_us={summary['avg_us']:.3f} "
        f"median_us={summary['median_us']:.3f} min_us={summary['min_us']:.3f} max_us={summary['max_us']:.3f}"
    )
    if "prepare" in summary:
        log(f"profile combine_prepare_kernel avg_us={summary['prepare']['avg_us']:.3f} excluded_from_combine=1")
    return summary


def notify_profile_call(buffer, handle, experts):
    """Prepare the existing native Notify entry independently of payload dispatch."""
    import torch
    from deep_ep._native import require_native
    from deep_ep._runtime import runtime_of

    runtime = runtime_of(buffer)
    native = require_native("notify profiling")
    specification = native.notify_layout(runtime.rank, runtime.world_size, handle.num_tokens, handle.num_topk, experts)
    if specification["workspace_bytes"] > runtime.capacity // 2:
        raise ValueError("Notify profile exceeds the reserved SHMEM half")
    native.notify_check_mapping(runtime.native)
    indices = handle.topk_idx
    with torch.npu.stream(runtime.stream):
        outputs = {
            name: torch.empty(shape, dtype=getattr(torch, dtype), device=indices.device)
            for name, (shape, dtype) in specification["tensors"].items()
        }
    torch.npu.synchronize()

    def invoke():
        with torch.npu.stream(runtime.stream):
            native.notify(
                runtime.native, indices, outputs, handle.num_tokens, handle.num_topk, experts, runtime.stream.npu_stream
            )
        return outputs

    def validate(result):
        # The baseline handle has already passed the independent CPU golden.
        for name in (
            "server_mask",
            "dst",
            "forward_counts",
            "backward_counts",
            "chunk_ranges",
            "chunk_masks",
            "rank_rows",
            "gather_rows",
            "expert_totals",
        ):
            if not torch.equal(result[name].cpu(), getattr(handle, name).cpu()):
                raise AssertionError(f"profile notify: {name} differs from validated metadata")
        for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
            for lane, count in enumerate(getattr(handle, counts).cpu().tolist()):
                if not torch.equal(result[name][lane, :count].cpu(), getattr(handle, name)[lane, :count].cpu()):
                    raise AssertionError(f"profile notify: {name}[{lane}] differs from validated metadata")
        if not torch.equal(
            result["weight_return_meta"][: handle.num_expanded_tokens].cpu(), handle.weight_return_meta.cpu()
        ):
            raise AssertionError("profile notify: weight_return_meta differs from validated metadata")

    return invoke, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _add_launch_args(parser)
    parser.add_argument("--tokens", type=int, default=33)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bf16", "fp8", "both"), default="both")
    parser.add_argument("--route", choices=(*_ROUTES, "all"), default="all")
    parser.add_argument("--seed", type=int, default=1, help="Random routing seed; identical on every rank")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    add_profile_args(parser, ("notify", "dispatch"))
    args = parser.parse_args()
    validate_profile_args(parser, args)
    if not 0 < args.hidden <= 7168 or args.hidden % (16 if args.dtype == "bf16" else 128):
        parser.error("hidden must be <=7168 and divisible by 16 (BF16) or 128 (FP8)")
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE") or args.repeat < 2:
        parser.error("Enable assertions and use --repeat >=2")
    _launch(args, worker)


def worker(local_rank, args):
    rank, world = _worker_identity(local_rank, args)
    log = _progress_logger("dispatch", rank)
    log(f"worker.begin route={args.route} seed={args.seed}; import installed package")
    deep_ep = _import_installed_deep_ep()
    import torch
    import torch_npu  # noqa: F401
    from deep_ep.buffers._elastic_layout import dispatch_layout

    info = json.loads(Path(deep_ep.__file__).with_name("_build_info.json").read_text())
    if "elastic-dispatch-expanded-v1" not in info.get("capabilities", []):
        raise RuntimeError("Installed wheel does not contain payload dispatch")
    if "elastic-notify-metadata-v6" not in info.get("capabilities", []):
        raise RuntimeError("Rebuild/install the wheel with Notify metadata v6")
    torch.npu.set_device(local_rank)
    modes = _ROUTES if args.route == "all" else (args.route,)
    dtypes = (False, True) if args.dtype == "both" else (args.dtype == "fp8",)
    if local_rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    # Small correctness runs reserve the exact all-to-one bound. The public
    # hint budgets four times average rows and is intentionally smaller at EP>4.
    log("capacity.begin")
    # Random routing is reproducible, so reserve its actual maximum rather than
    # the all-to-one bound (which overflows address/workspace limits at EP256).
    random_routes = (
        make_routes(world, args.tokens, args.topk, args.experts, 0, "random", seed=args.seed)
        if args.route == "random"
        else None
    )
    rows = (
        receive_capacity(random_routes, args.experts) if random_routes is not None else world * args.tokens * args.topk
    )
    capacity = 2 * max(
        dispatch_layout(world, args.tokens, args.topk, args.experts, rows, fp8, weights=True, hidden=args.hidden)[
            "num_bytes"
        ]
        for fp8 in dtypes
    )
    log(f"buffer.init.begin bytes={capacity}")
    buffer = deep_ep.ElasticBuffer(
        rank=rank,
        world_size=world,
        num_bytes=capacity,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.topk,
        explicitly_destroy=True,
    )
    log("buffer.init.done")
    checks, previous_output = 0, None
    try:
        for mode in modes:
            log(f"route={mode} routes_and_notify_golden.begin (CPU)")
            routes = (
                random_routes
                if random_routes is not None
                else make_routes(world, args.tokens, args.topk, args.experts, 0, mode, seed=args.seed)
            )
            notify_golden = build_notify_golden(routes, args.experts, 256)
            entries = expert_rows(routes, args.experts, rank)
            log(f"route={mode} routes_and_notify_golden.done")
            for fp8 in dtypes:
                context = f"dtype={'fp8' if fp8 else 'bf16'} route={mode}"
                for with_weights in (False, True):
                    handle = None
                    for generation in range(args.repeat):
                        use_weights = with_weights if generation % 2 == 0 else not with_weights
                        iteration = (
                            f"{context} weights_case={with_weights} weights={use_weights} "
                            f"iter={generation + 1}/{args.repeat}"
                        )
                        log(f"{iteration} inputs.begin")
                        local, scales = payload(rank, args.tokens, generation, fp8, args.hidden)
                        local_weights = weights_for(rank, args.tokens, args.topk, generation)
                        producer = torch.npu.Stream()
                        with torch.npu.stream(producer):
                            x = (local.npu(), scales.npu()) if fp8 else local.npu()
                            w = local_weights.npu() if use_weights else None
                            indices = (
                                torch.tensor(routes[rank], dtype=torch.int64, device="npu") if handle is None else None
                            )
                            before = buffer.capture()
                        old_handle = handle
                        log(f"{iteration} dispatch.enter mode={'fresh' if handle is None else 'cached'}")
                        output, index, weights, handle, event = buffer.dispatch(
                            x,
                            topk_idx=indices,
                            topk_weights=w,
                            num_experts=args.experts if handle is None else None,
                            handle=handle,
                            previous_event=before,
                            async_with_compute_stream=True,
                            do_expand=True,
                        )
                        log(f"{iteration} dispatch.return (async; device completion not yet confirmed)")
                        event.current_stream_wait()
                        if index is not None or handle.metadata_only:
                            raise AssertionError("Expected expanded payload output")
                        if old_handle is None:
                            log(f"{iteration} handle_check.begin (includes device readback)")
                            check_handle(handle, notify_golden, rank, routes, args.experts)
                            log(f"{iteration} handle_check.done")
                        if old_handle is not None and old_handle is not handle:
                            raise AssertionError("Cached dispatch replaced its routing handle")
                        log(f"{iteration} payload_golden.begin (CPU)")
                        expected, expected_scales, expected_weights = expected_payload(
                            routes, args.experts, rank, generation, fp8, entries=entries, width=args.hidden
                        )
                        log(f"{iteration} payload_golden.done")
                        data = output[0] if fp8 else output
                        assert_bytes(data, expected, f"{iteration} dispatch hidden", progress=log)
                        log(f"{iteration} remaining_checks.begin")
                        if fp8:
                            assert_bytes(output[1], expected_scales, "scales")
                        if use_weights:
                            assert_bytes(weights, expected_weights, "weights")
                        elif weights is not None:
                            raise AssertionError("Weights returned when none were supplied")
                        if args.profile_op and generation == 0 and (args.profile_op != "notify" or not with_weights):
                            if args.profile_op == "notify":
                                invoke, validate = notify_profile_call(buffer, handle, args.experts)
                            else:

                                def invoke():
                                    result = buffer.dispatch(
                                        x,
                                        topk_weights=w,
                                        handle=handle,
                                        do_expand=True,
                                        async_with_compute_stream=True,
                                    )
                                    result[-1].current_stream_wait()
                                    return result

                                def validate(result):
                                    received, _, returned_weights, returned_handle, _ = result
                                    if returned_handle is not handle:
                                        raise AssertionError("Profile dispatch replaced the routing handle")
                                    assert_bytes(received[0] if fp8 else received, expected, "profile dispatch hidden")
                                    if fp8:
                                        assert_bytes(received[1], expected_scales, "profile dispatch scales")
                                    if use_weights:
                                        assert_bytes(returned_weights, expected_weights, "profile dispatch weights")
                                    elif returned_weights is not None:
                                        raise AssertionError("Profile dispatch unexpectedly returned weights")

                            profile_operator(
                                args,
                                rank,
                                f"{mode}-{'fp8' if fp8 else 'bf16'}-weights{int(use_weights)}",
                                invoke,
                                validate,
                                info,
                                log,
                            )
                        if previous_output is not None:
                            for prior, expected_prior in previous_output:
                                assert_bytes(prior, expected_prior, "retained output across cached/fresh calls")
                        previous_output = [(data, expected)]
                        if fp8:
                            previous_output.append((output[1], expected_scales))
                        if use_weights:
                            previous_output.append((weights, expected_weights))
                        if indices is not None:
                            indices.fill_(-1)
                            assert_bytes(handle.topk_idx, torch.tensor(routes[rank], dtype=torch.int64), "routing copy")
                        if expected.numel():
                            corrupt = expected.view(torch.uint8).clone()
                            corrupt.flatten()[0] ^= 1
                            try:
                                assert_bytes(data, corrupt.view(expected.dtype), "deliberate corruption")
                            except AssertionError:
                                pass
                            else:
                                raise AssertionError("Precision comparator failed to detect corruption")
                        checks += 1
                        log(f"{iteration} iteration.passed")
                    print(
                        f"rank={rank} dtype={'fp8' if fp8 else 'bf16'} route={mode} weights={with_weights} passed",
                        flush=True,
                    )
    except BaseException:
        _log_failure(log)
        raise
    finally:
        log("buffer.destroy.begin (may wait for device work/peers)")
        buffer.destroy()
        log("buffer.destroy.done")
    # Returned tensors must also remain valid after workspace destruction.
    if previous_output is not None:
        for prior, expected_prior in previous_output:
            assert_bytes(prior, expected_prior, "retained output after destroy")
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                world_size=world,
                checks=checks,
                passed=True,
                source_sha256=info["source_sha256"],
                tokens=args.tokens,
                topk=args.topk,
                experts=args.experts,
                hidden=args.hidden,
                dtype=args.dtype,
                route=args.route,
                seed=args.seed,
            ),
            indent=2,
        )
        + "\n"
    )


def test_direct_and_external_launcher(monkeypatch):
    from types import SimpleNamespace
    import pytest

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "DEEPEP_SHMEM_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    parser = argparse.ArgumentParser()
    _add_launch_args(parser)
    args = parser.parse_args(
        ["--ep-size", "64", "--nproc-per-node", "8", "--node-rank", "5", "--master-addr", "141.61.54.91"]
    )
    calls = []

    def spawn(worker, args, nprocs, join):
        assert nprocs == 8 and join
        for local_rank in range(nprocs):
            worker(local_rank, *args)

    mp = SimpleNamespace(spawn=spawn)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(multiprocessing=mp))
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", mp)

    def worker(local_rank, args):
        calls.append(_worker_identity(local_rank, args))

    _launch(args, worker)
    assert calls == [(rank, 64) for rank in range(40, 48)]
    assert os.environ["DEEPEP_SHMEM_ENDPOINT"] == "tcp://141.61.54.91:8900"
    calls.clear()
    monkeypatch.setenv("RANK", "23")
    monkeypatch.setenv("WORLD_SIZE", "32")
    monkeypatch.setenv("LOCAL_RANK", "7")
    _launch(args, worker)
    assert calls == [(23, 32)]
    assert os.environ["DEEPEP_SHMEM_ENDPOINT"] == "tcp://141.61.54.91:8900"
    monkeypatch.delenv("WORLD_SIZE")
    with pytest.raises(ValueError, match="supplied together"):
        _launch(args, worker)


def test_hand_computed_routes():
    golden = build_notify_golden([[[0, 2]], [[1, 2]]], 4, 256)
    assert golden["rank_rows"] == [2, 2]
    assert golden["plans"][0]["dst"] == [0, ~2]
    assert golden["plans"][1]["dst"] == [1, ~3]
    duplicate = build_notify_golden([[[0, 0]], [[-1, -1]]], 4, 256)
    assert duplicate["rank_rows"] == [2, 0]
    assert duplicate["plans"][0]["dst"] == [0, ~1]


def test_group_chunk_order_and_token_ends():
    # Every source sends six tokens to gateway 0, with two contributions on
    # rank 1 and one extra local contribution. Two tokens per raw chunk.
    routes = [[[0, 2, 0, 2] for _ in range(6)] for _ in range(32)]
    golden = build_notify_golden(routes, 64, 2)
    for source in range(32):
        forward = golden["plans"][0]["forward_list"][source]
        assert [row[0] for row in forward] == [end for end in (3, 6, 9, 12, 15, 18) for _ in range(3)]
    # One 32-rank group. First raw chunk [0,2) splits into [0,1), [1,2).
    expected = []
    for token_ids in ((0,), (1,), (2, 3), (4, 5)):
        for source in range(32):
            expected.extend((source, token, slot) for token in token_ids for slot in (1, 3))
    backward = golden["plans"][1]["backward_list"][0]
    entries = expert_rows(routes, 64, 1)
    assert [entries[row] for _, row, _ in backward] == expected
    tails = {63: 1, 127: 2, 255: 3, 383: 4}
    assert [packed >> 32 for _, _, packed in backward] == [tails.get(i, 0) for i in range(len(backward))]


def test_installed_import_ignores_checkout_and_pythonpath(tmp_path):
    import subprocess

    installed = tmp_path / "site-packages"
    checkout = tmp_path / "checkout"
    for root, marker in ((installed, "installed"), (checkout, "source")):
        (root / "deep_ep").mkdir(parents=True)
        (root / "deep_ep/__init__.py").write_text(f"marker = {marker!r}\n", encoding="utf-8")
    # A separate process keeps pytest's already-imported source package intact.
    script = """
import runpy, sys, sysconfig
namespace = runpy.run_path(sys.argv[1])
sysconfig.get_path = lambda name: sys.argv[2]
mode = sys.argv[3]
if mode == "cached":
    import deep_ep
    assert deep_ep.marker == "source"
try:
    package = namespace["_import_installed_deep_ep"]()
except RuntimeError as error:
    assert mode != "installed", str(error)
    assert ("already loaded" if mode == "cached" else "not installed") in str(error)
else:
    assert mode == "installed"
    assert package.marker == "installed"
"""
    for mode in ("installed", "cached", "missing"):
        location = installed if mode != "missing" else tmp_path / "empty-site-packages"
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve()), str(location), mode],
            cwd=checkout,
            env={**os.environ, "PYTHONPATH": str(checkout)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_dispatch_barrier_timing_and_profile_versions():
    import runpy

    import pytest

    metrics = runpy.run_path(str(Path(__file__).with_name("dispatch_metrics.py")))
    records = []
    for core in range(64):
        words = [0] * 32
        words[:8] = range(100, 108)
        words[26:31] = [0x44504550, 0, core, 1, 8]
        if core == 0:
            words[8:14] = [104, 105, 105, 1, 1, 1]
        records.append(words)
    current = metrics["decode_profile"](records, 0, fp8=False, world=2)
    assert all("store_drain_ms" in r["phases_ms"] and "finish_wait_ms" not in r["phases_ms"] for r in current)
    for words in records:
        words[30] = 7
    legacy = metrics["decode_profile"](records, 0, fp8=False, world=2)
    assert current[0]["phases_ms"]["store_drain_ms"] == legacy[0]["phases_ms"]["finish_wait_ms"]

    ranks = [
        dict(
            rank=rank,
            world=2,
            passed=True,
            tokens=4096,
            topk=6,
            experts=384,
            dtype="bf16",
            weights=False,
            backend="udma",
            source_sha256="test",
            clean_device_ms=[[0.1, 1.0, 0.2, 1.3]],
            kernel_profiles=[],
            event_only=True,
        )
        for rank in range(2)
    ]
    old_report = metrics["summarize"](ranks)
    assert old_report["input_copy"]["mean_ms"] == 0.1
    columns = ["entry_barrier", "kernel", "exit_barrier", "device_total"]
    for rank in ranks:
        rank["device_time_columns"] = columns
    report = metrics["summarize"](ranks)
    assert "input_copy" not in report and "output_copy" not in report
    assert report["entry_barrier"]["mean_ms"] == 0.1
    assert report["exit_barrier"]["mean_ms"] == 0.2
    assert report["algorithm_bandwidth"] == old_report["algorithm_bandwidth"]
    ranks[1].pop("device_time_columns")
    with pytest.raises(ValueError, match="Mixed device timing columns"):
        metrics["summarize"](ranks)


if __name__ == "__main__":
    main()
