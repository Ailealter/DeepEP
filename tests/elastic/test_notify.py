"""Standalone Notify precision, profiling and metadata regression checks."""

import json
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import pytest
import argparse
import csv
import math
import statistics
import os
import importlib
from importlib.machinery import PathFinder
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
    if world not in (2, 4, 8, 16, 32, 64, 128, 256, 512) or experts <= 0 or experts % world:
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
                plans[source]["server_mask"][token] |= (1 << server) if server < 63 else -(1 << 63)
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


def check_notify_outputs(outputs, golden, rank, routes, experts):
    """Compare native metadata with the independent CPU golden, without Dispatch."""
    import torch

    plan = golden["plans"][rank]
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
        actual = outputs[name]
        if actual.device.type != "npu":
            raise AssertionError(f"{name} is not an NPU tensor")
        expected = torch.tensor(plan[name], dtype=actual.dtype).reshape(actual.shape)
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError(f"rank {rank}: {name} mismatch")
    for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
        actual = outputs[name].cpu()
        for row, count in enumerate(plan[counts]):
            expected = torch.tensor(plan[name][row], dtype=actual.dtype).reshape(count, actual.shape[-1])
            if not torch.equal(actual[row, :count], expected):
                raise AssertionError(f"rank {rank}: {name}[{row}] mismatch")
    weight_meta = outputs["weight_return_meta"][: golden["rank_rows"][rank]]
    weight_meta_cpu = weight_meta.cpu()
    expected_weight_meta = torch.tensor(
        [(s * len(routes[0]) + t) * len(routes[0][0]) + k for s, t, k in expert_rows(routes, experts, rank)],
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
    if not torch.equal(outputs["expert_totals"].cpu(), torch.tensor(totals, dtype=torch.int32)):
        raise AssertionError(f"rank {rank}: expert_totals mismatch")
    return totals


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
    if getattr(args, "bootstrap", "default") == "mpi":
        try:
            rank, world = (int(os.environ[name]) for name in ("PMI_RANK", "PMI_SIZE"))
        except (KeyError, ValueError) as exc:
            raise ValueError("--bootstrap mpi requires an MPICH mpiexec launcher") from exc
        if (
            world != args.ep_size
            or args.nproc_per_node < 1
            or world % args.nproc_per_node
            or not 0 <= rank < world
            or rank // args.nproc_per_node != args.node_rank
        ):
            raise ValueError("MPI ranks must match EP/node-rank and use contiguous ranks per node")
        local_rank = rank % args.nproc_per_node
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local_rank))
        # mpiexec already created one process per NPU. Never fork/spawn another
        # set of processes, which would lose the MPI_COMM_WORLD membership.
        worker(local_rank, args)
        return
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
    if args.ep_size not in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        raise ValueError("Dispatch/combine EP must be a power of two from 2 to 512")
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
        hidden=7168,
        chunk_tokens=getattr(args, "chunk_tokens", 256),
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
    if args.profile_op == "combine":
        payload_bytes = args.tokens * args.topk * 7168 * 2
        summary.update(
            payload_bytes=payload_bytes,
            bandwidth_gbps=payload_bytes / (summary["avg_us"] * 1000),
            bandwidth_metric="logical_bf16_payload_per_rank_over_mean_kernel_duration",
        )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(
        f"profile kernel={summary['kernel']} samples={summary['samples']} avg_us={summary['avg_us']:.3f} "
        f"median_us={summary['median_us']:.3f} min_us={summary['min_us']:.3f} max_us={summary['max_us']:.3f}"
    )
    if "prepare" in summary:
        log(f"profile combine_prepare_kernel avg_us={summary['prepare']['avg_us']:.3f} excluded_from_combine=1")
    return summary


def profile_notify(args, rank, scenario, invoke, validate, info, log):
    """Profile the clean kernel, then collect exactly one separate diagnostic call."""
    import torch
    from deep_ep._native import require_native

    view = runpy.run_path(str(Path(__file__).with_name("notify_profile.py")))
    native = require_native("Notify diagnostics")
    if not hasattr(native, "notify_profile_layout"):
        raise RuntimeError("Rebuild/install the wheel with Notify diagnostics v1")
    version, cores, words = native.notify_profile_layout()
    if (version, cores, words) != (view["VERSION"], 32, view["WORDS"]):
        raise RuntimeError("Notify profile ABI differs; update the test and installed wheel together")
    clean = profile_operator(args, rank, scenario, invoke, validate, info, log)
    if clean["kernel"] != "notify_full_kernel":
        raise RuntimeError("Notify performance must use the uninstrumented kernel")
    profile = torch.zeros((cores, words), dtype=torch.int64, device="npu")
    torch.npu.synchronize()  # Allocation/zeroing on default stream precedes comm-stream export.
    log("diagnostic.begin kernel=notify_profile_kernel (outside profiler)")
    result = invoke(profile=profile, sample=1)
    torch.npu.synchronize()
    validate(result)
    data = view["decode"](
        profile.cpu().tolist(),
        rank=rank,
        world=args.ep_size,
        tokens=args.tokens,
        topk=args.topk,
        experts=args.experts,
        chunk_tokens=args.chunk_tokens,
        sample=1,
    )
    data.update(scenario=scenario, seed=args.seed, clean_profile=clean, precision_checked=True, build_info=info)
    paths = view["write_timeline"](data, args.output / "diagnostics" / scenario / f"rank-{rank}.notify.json")
    log(f"diagnostic.done span_us={data['span_us']:.3f} svg={paths['svg']}")
    return dict(profile=clean, diagnostics=paths)


def notify_only_worker(local_rank, args):
    """Native Notify + CPU golden only; no payload or Dispatch peer-plan allocation."""
    rank, world = _worker_identity(local_rank, args)
    log = _progress_logger("notify", rank)
    deep_ep = _import_installed_deep_ep()
    import torch
    import torch_npu  # noqa: F401
    from deep_ep._native import require_native
    from deep_ep._runtime import runtime_of

    info = json.loads(Path(deep_ep.__file__).with_name("_build_info.json").read_text())
    native = require_native("Notify-only validation")
    specification = native.notify_layout(
        rank, world, args.tokens, args.topk, args.experts, args.chunk_tokens, combine_only=True
    )
    if "_peer_plan" in specification["tensors"]:
        raise RuntimeError("Notify-only must not build Dispatch peer plans")
    torch.npu.set_device(local_rank)
    if local_rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    capacity = 2 * ((specification["workspace_bytes"] + (2 << 20) - 1) // (2 << 20) * (2 << 20))
    log(f"buffer.init.begin bytes={capacity} dispatch_plan=0")
    buffer = deep_ep.ElasticBuffer(
        rank=rank,
        world_size=world,
        num_bytes=capacity,
        num_max_tokens_per_rank=args.tokens,
        hidden=7168,
        num_topk=args.topk,
        explicitly_destroy=True,
        combine_only=True,
    )
    log("buffer.init.done")
    results = []
    try:
        runtime = runtime_of(buffer)
        native.notify_check_mapping(runtime.native)
        # Reuse all output buffers, including across routes, to expose stale-count bugs.
        with torch.npu.stream(runtime.stream):
            outputs = {
                name: torch.empty(shape, dtype=getattr(torch, dtype), device="npu")
                for name, (shape, dtype) in specification["tensors"].items()
            }
        for mode in _ROUTES if args.route == "all" else (args.route,):
            log(f"route={mode} golden.begin (CPU)")
            routes = make_routes(world, args.tokens, args.topk, args.experts, 0, mode, seed=args.seed)
            golden = build_notify_golden(routes, args.experts, args.chunk_tokens)
            with torch.npu.stream(runtime.stream):
                indices = torch.tensor(routes[rank], dtype=torch.int64, device="npu")
            torch.npu.synchronize()

            def invoke(*, profile=None, sample=1):
                call_outputs = outputs if profile is None else dict(outputs, profile=profile, profile_sample=sample)
                with torch.npu.stream(runtime.stream):
                    native.notify(
                        runtime.native,
                        indices,
                        call_outputs,
                        args.tokens,
                        args.topk,
                        args.experts,
                        runtime.stream.npu_stream,
                        args.chunk_tokens,
                    )
                return outputs

            def validate(result):
                check_notify_outputs(result, golden, rank, routes, args.experts)

            for iteration in range(args.repeat):
                log(f"route={mode} notify.begin iteration={iteration + 1}")
                result = invoke()
                torch.npu.synchronize()
                validate(result)
                log(f"route={mode} notify.precision.passed iteration={iteration + 1}")
            entry = dict(route=mode, checks=args.repeat, passed=True)
            if args.profile_op:
                entry.update(profile_notify(args, rank, mode, invoke, validate, info, log))
            results.append(entry)
    except BaseException:
        _log_failure(log)
        raise
    finally:
        log("buffer.destroy.begin")
        buffer.destroy()
        log("buffer.destroy.done")
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                world_size=world,
                operator="notify",
                notify_only=True,
                dispatch_plan=False,
                tokens=args.tokens,
                topk=args.topk,
                experts=args.experts,
                chunk_tokens=args.chunk_tokens,
                seed=args.seed,
                cases=results,
                passed=True,
                source_sha256=info["source_sha256"],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_native_notify_combine_only_layout():
    """Host-only layout check when a compiled wheel is available; no NPU allocation."""
    import pytest
    from deep_ep._native import require_native

    try:
        native = require_native("Notify layout check")
    except NotImplementedError:
        pytest.skip("Notify layout check requires a compiled wheel")
    for world in (2, 128, 256, 512):
        for tokens in (4096, 16384):
            for chunk in (256, 512, 1024):
                normal = native.notify_layout(0, world, tokens, 6, world, chunk)
                lean = native.notify_layout(0, world, tokens, 6, world, chunk, combine_only=True)
                assert "_peer_plan" in normal["tensors"]
                assert "_peer_plan" not in lean["tensors"]
                assert normal["workspace_bytes"] == lean["workspace_bytes"]
                assert {k: v for k, v in normal["tensors"].items() if k != "_peer_plan"} == lean["tensors"]


def test_notify_fused_chunk_metadata(tmp_path):
    """Run the production SIMT routine against an independent host oracle."""
    import shutil
    import subprocess
    import pytest

    root = Path(__file__).resolve().parents[2]
    source = root / "csrc/kernels/notify/notify.cpp"
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not source.is_file() or compiler is None:
        pytest.skip("Notify SIMT host check requires source and a C++17 compiler")
    text = source.read_text(encoding="utf-8")
    begin = text.index("template <uint32_t CoreCount = kSimtCores>", text.index("inline void NotifyBuildForwardRowsVf"))
    end = text.index("template <uint32_t CoreCount = kSimtCores>", begin + 1)
    routine = text[begin:end]
    code = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <vector>
#include "kernels/notify/tiling.hpp"
#define __simt_vf__
#define __launch_bounds__(...)
#define __gm__
constexpr uint32_t kSimtCores = 32, kNotifySimtThreads = 128;
struct { uint32_t x; } threadIdx;
template<class T> T asc_ldcg(const T* p) { return *p; }
template<class T> void asc_stcg(T* p, T value) { *p = value; }
void asc_threadfence() {}
"""
        + routine
        + r"""
void check(uint32_t world, uint32_t tokens, uint32_t chunk_tokens) {
    const uint32_t rank = world - 1, lanes = std::min(world, 8U), capacity = 2048;
    auto t = ascend_deepep::MakeNotifyTiling(rank, world, tokens, 6, world, chunk_tokens, 7168);
    const uint32_t chunks = t.chunks;
    const uint64_t words = 16 + capacity * 6, stride = uint64_t(world) * chunks;
    std::vector<int32_t> packets(world * words, -97), forward(world * capacity * 6, -97);
    std::vector<int32_t> expected_forward = forward, ranges(stride * 2, -97), masks(stride, -97);
    std::vector<int32_t> counts(lanes * stride, -97);
    // Independent explicit boundaries, including the four startup subchunks.
    std::vector<uint32_t> bounds{0};
    const uint32_t first = std::min(tokens, chunk_tokens), splits = std::min(first, 4U);
    for (uint32_t i = 1; i <= splits; ++i) bounds.push_back((first * i + splits - 1) / splits);
    for (uint32_t i = first; i < tokens;) { i = std::min(tokens, i + chunk_tokens); bounds.push_back(i); }
    assert(bounds.size() == chunks + 1);
    std::set<uint32_t> selected;
    for (auto b : bounds) {
        if (b < tokens) selected.insert(b);
        if (b) selected.insert(b - 1);
    }
    // A dense run followed by an empty run must clear every count/range/mask.
    for (bool empty : {false, true}) {
        std::vector<int32_t> want_ranges(stride * 2, 0), want_masks(stride, 0), want_counts(lanes * stride, 0);
        for (uint32_t s = 0; s < world; ++s) {
            auto* packet = packets.data() + s * words;
            std::vector<uint32_t> row_tokens;
            if (!empty && s % 5 != 0) {
                for (auto token : selected) {
                    const uint32_t begin = row_tokens.size();
                    // Sentinel, local duplicate, and remote secondaries in every lane.
                    for (int32_t peer : {-1, int32_t(rank), int32_t(rank / 8 * 8 + (token + s) % lanes),
                                         int32_t(rank / 8 * 8 + (token + s + 1) % lanes)}) {
                        const uint32_t row = row_tokens.size();
                        assert(row < capacity);
                        row_tokens.push_back(token);
                        auto* p = packet + 16 + row * 6;
                        p[1] = token; p[3] = peer;
                        // Gather condition from NotifyGatherCountTilesVf; packet field 5 is not a gather index.
                        p[5] = -1;
                        const uint32_t c = std::upper_bound(bounds.begin(), bounds.end(), token) - bounds.begin() - 1;
                        if (peer >= 0 && peer != int32_t(rank)) {
                            want_masks[s * chunks + c] |= 1 << (peer % 8);
                            want_counts[(peer % 8) * stride + s * chunks + c]++;
                        }
                    }
                    for (uint32_t row = begin; row < row_tokens.size(); ++row)
                        expected_forward[(s * capacity + row) * 6] = row_tokens.size();
                }
            }
            packet[0] = row_tokens.size();
            for (uint32_t c = 0; c < chunks; ++c) {
                want_ranges[(s * chunks + c) * 2] = std::lower_bound(row_tokens.begin(), row_tokens.end(), bounds[c]) - row_tokens.begin();
                want_ranges[(s * chunks + c) * 2 + 1] = std::lower_bound(row_tokens.begin(), row_tokens.end(), bounds[c + 1]) - row_tokens.begin();
            }
        }
        // Reverse core order stresses independence from other cores' forward fields.
        for (uint32_t core = kSimtCores; core-- > 0;)
            for (threadIdx.x = 0; threadIdx.x < kNotifySimtThreads; ++threadIdx.x)
                NotifyChunkRangesVf(packets.data(), forward.data(), capacity, ranges.data(), masks.data(), counts.data(),
                                    rank, world, chunks, chunk_tokens, tokens, words, core);
        assert(ranges == want_ranges && masks == want_masks && counts == want_counts);
        assert(forward == expected_forward);
        auto corrupted = counts;
        corrupted.back() ^= 1;
        assert(corrupted != want_counts);
    }
}
int main() {
    for (uint32_t world : {2U, 4U, 8U, 64U, 128U, 256U, 512U})
        for (uint32_t tokens : {1U, 3U, 257U, 4096U, 16384U})
            for (uint32_t chunk : {256U, 512U, 1024U}) check(world, tokens, chunk);
}
"""
    )
    path = tmp_path / "notify-chunks.cpp"
    path.write_text(code, encoding="utf-8")
    binary = tmp_path / ("notify-chunks.exe" if sys.platform == "win32" else "notify-chunks")
    subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(root / "csrc"), str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=60)


def test_notify_forward_ub_paths(tmp_path):
    """Execute production forward VFs on host, including newly enabled 16K caches."""
    import shutil
    import subprocess
    import pytest

    root = Path(__file__).resolve().parents[2]
    source = root / "csrc/kernels/notify/notify.cpp"
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not source.is_file() or compiler is None:
        pytest.skip("Notify SIMT host check requires source and a C++17 compiler")
    text = source.read_text(encoding="utf-8")
    begin = text.index("template <bool Write, bool Cached")
    end = text.rfind("template <", begin, text.index("inline void NotifyGatherCountTilesVf"))
    code = (
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#include "kernels/notify/tiling.hpp"
#define __simt_vf__
#define __launch_bounds__(...)
#define __gm__
#define __ubuf__
constexpr uint32_t kSimtCores = 32, kNotifySimtThreads = 128, kForwardTile = 32;
constexpr uint32_t kWirePrimaryTopkShift = 9, kWireSecondaryTopkShift = 13;
struct { uint32_t x; } threadIdx;
template<class T> T asc_ldcg(const T* p) { return *p; }
template<class T> void asc_stcg(T* p, T value) { *p = value; }
void asc_threadfence() {}
"""
        + text[begin:end]
        + r"""
using namespace ascend_deepep;
template<uint32_t Cores> void check(uint32_t world, uint32_t tokens, uint32_t topk, uint32_t experts) {
    const uint32_t rank = world - 1, tiles = (tokens + 31) / 32, per_rank = experts / world;
    const auto t = MakeNotifyTiling(rank, world, tokens, topk, experts, 256, 7168, Cores);
    const size_t meta_words = kNotifyForwardMetaUbBytes / 4, base_words = kNotifyForwardBasesUbBytes / 4;
    std::vector<std::vector<int32_t>> meta_ub(Cores, std::vector<int32_t>(meta_words + 16, -97));
    std::vector<std::vector<int32_t>> bases_ub(Cores, std::vector<int32_t>(base_words + 16, -97));
    std::vector<int32_t> meta(tokens * topk * 6, -97), bases(experts), offsets(world * tiles, -97);
    for (uint32_t e = 0; e < experts; ++e) bases[e] = e * (tokens * topk + 17);
    // Reuse dirty UB for an all-invalid round; no old row/count may leak through.
    for (bool empty : {false, true}) {
        std::vector<std::vector<int32_t>> expected(world);
        std::vector<int32_t> ordinals(experts, 0), want_counts(world * tiles, 0);
        for (uint32_t token = 0; token < tokens; ++token) {
            std::vector<int32_t> es(topk, -1), rows(topk, -1);
            for (uint32_t k = 0; k < topk; ++k) {
                int32_t e = (token * 41 + k * 73 + token / 7) % experts;
                if (k == 1 && token % 5 == 0) e = es[0]; // duplicate expert
                if (k == 2 && es[0] >= 0 && token % 7 == 0)
                    e = (es[0] / (per_rank * 8) * (per_rank * 8) + 1) % experts;
                if (empty || token % 11 == 0 || (token + k) % 19 == 0) e = -1;
                es[k] = e;
                auto* m = meta.data() + (token * topk + k) * 6;
                std::fill(m, m + 6, -1);
                if (e >= 0) {
                    m[0] = e; m[4] = ordinals[e]++;
                    rows[k] = bases[e] + m[4];
                }
            }
            // Independent server grouping and source order oracle.
            for (uint32_t k = 0; k < topk; ++k) {
                if (es[k] < 0) continue;
                std::vector<uint32_t> slots;
                for (uint32_t j = 0; j < topk; ++j)
                    if (es[j] >= 0 && es[j] / per_rank / 8 == es[k] / per_rank / 8) slots.push_back(j);
                const uint32_t first = slots.front(), gateway = es[first] / per_rank;
                auto* m = meta.data() + (token * topk + k) * 6;
                m[1] = gateway; m[2] = first;
                const bool direct = slots.size() == 1;
                if (!direct && k == first) continue;
                const std::array<int32_t, 6> row = {
                    int32_t(rank | (first << 9) | (k << 13)), int32_t(token), rows[first],
                    direct ? -1 : es[k] / int32_t(per_rank), direct ? -1 : rows[k], -1};
                expected[gateway].insert(expected[gateway].end(), row.begin(), row.end());
                ++want_counts[gateway * tiles + token / 32];
            }
        }
        size_t words = 16;
        for (auto& rows : expected) words = std::max(words, 16 + rows.size());
        words += 16; // Guard every packet, not just the last gateway.
        std::vector<int32_t> packets(world * words, -97), want_packets = packets;
        for (uint32_t g = 0; g < world; ++g) {
            want_packets[g * words] = expected[g].size() / 6;
            std::copy(expected[g].begin(), expected[g].end(), want_packets.begin() + g * words + 16);
        }
        // Reverse core and thread order to expose accidental scheduling dependence.
        for (uint32_t core = Cores; core-- > 0;) {
            auto* m = meta_ub[core].data(); auto* b = bases_ub[core].data();
            auto* plan = m + t.forward_plan_offset_words;
            auto* tp = reinterpret_cast<uint16_t*>(m + t.forward_token_prefix_offset_words);
            auto* counts = b + t.forward_count_offset_words;
            const uint32_t local_tiles = core < tiles ? (tiles - core + Cores - 1) / Cores : 0;
            if (t.cache_forward_inputs) {
                std::copy(bases.begin(), bases.end(), b);
                for (uint32_t tile = core; tile < tiles; tile += Cores) {
                    const uint32_t n = std::min(32U, tokens - tile * 32) * topk * 6;
                    std::copy_n(meta.data() + tile * 32 * topk * 6, n, m + tile / Cores * 32 * topk * 6);
                }
            }
            if (t.forward_token_prefix_offset_words) std::fill_n(tp, local_tiles * world * 32, 0);
            if (t.forward_count_offset_words) std::fill_n(counts, local_tiles * world, 0);
            for (threadIdx.x = 128; threadIdx.x-- > 0;) {
                if (t.forward_token_prefix_offset_words)
                    NotifyForwardTokensVf<Cores>(tokens, topk, world, core, m, plan, tp);
                else if (t.forward_count_offset_words)
                    NotifyForwardTilesVf<false, true, true, Cores, true>(meta.data(), bases.data(), offsets.data(),
                        packets.data(), rank, world, experts, tokens, topk, words, core, m, b, plan, counts);
                else if (t.forward_plan_offset_words)
                    NotifyForwardTilesVf<false, true, true, Cores>(meta.data(), bases.data(), offsets.data(),
                        packets.data(), rank, world, experts, tokens, topk, words, core, m, b, plan);
                else if (t.cache_forward_inputs)
                    NotifyForwardTilesVf<false, true, false, Cores>(meta.data(), bases.data(), offsets.data(),
                        packets.data(), rank, world, experts, tokens, topk, words, core, m, b);
                else
                    NotifyForwardTilesVf<false, false, false, Cores>(meta.data(), bases.data(), offsets.data(),
                        packets.data(), rank, world, experts, tokens, topk, words, core, m, b);
            }
            for (threadIdx.x = 128; threadIdx.x-- > 0;) {
                if (t.forward_token_prefix_offset_words)
                    NotifyForwardTokenPrefixVf<Cores>(offsets.data(), tokens, world, core, tp);
                else if (t.forward_count_offset_words)
                    NotifyExportForwardCountsVf<Cores>(offsets.data(), tokens, world, core, counts);
            }
        }
        assert(offsets == want_counts);
        for (uint32_t core = 0; core < Cores; ++core)
            for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
                NotifyForwardPrefixVf<false, Cores>(offsets.data(), packets.data(), world, tokens, words, core);
        for (uint32_t core = Cores; core-- > 0;) {
            auto* m = meta_ub[core].data(); auto* b = bases_ub[core].data();
            auto* plan = m + t.forward_plan_offset_words;
            auto* tp = reinterpret_cast<uint16_t*>(m + t.forward_token_prefix_offset_words);
            for (threadIdx.x = 128; threadIdx.x-- > 0;) {
                if (t.forward_token_prefix_offset_words)
                    NotifyForwardRowsVf<true, Cores>(offsets.data(), packets.data(), rank, world, experts, tokens,
                        topk, words, core, m, b, plan, tp);
                else if (t.forward_plan_offset_words)
                    NotifyForwardRowsVf<false, Cores>(offsets.data(), packets.data(), rank, world, experts, tokens,
                        topk, words, core, m, b, plan);
                else if (t.cache_forward_inputs)
                    NotifyForwardTilesVf<true, true, false, Cores>(meta.data(), bases.data(), offsets.data(),
                        packets.data(), rank, world, experts, tokens, topk, words, core, m, b);
                else
                    NotifyForwardTilesVf<true, false, false, Cores>(meta.data(), bases.data(), offsets.data(),
                        packets.data(), rank, world, experts, tokens, topk, words, core, m, b);
            }
            assert(std::all_of(m + meta_words, m + meta_words + 16, [](int32_t x) { return x == -97; }));
            assert(std::all_of(b + base_words, b + base_words + 16, [](int32_t x) { return x == -97; }));
            if (t.cache_forward_inputs) assert(std::equal(bases.begin(), bases.end(), b));
        }
        assert(packets == want_packets);
        auto corrupted = packets; corrupted[0] ^= 1;
        assert(corrupted != want_packets);
    }
}
int main() {
    static_assert(kNotifyDynamicUbBytes == 128U * 1024U);
    for (uint32_t world : {128U, 256U}) {
        const auto t = MakeNotifyTiling(0, world, 16384, 6, 512, 256, 7168);
        assert(t.cache_parse_hist && t.cache_parse_routes && t.cache_local_prefix && t.cache_forward_inputs);
        assert(t.forward_plan_offset_words && t.forward_count_offset_words && !t.forward_token_prefix_offset_words);
    }
    assert(!NotifyForwardCountOffsetWords(16384, 6, 512, 512)); // Preserve safe GM fallback.
    for (uint32_t world : {2U, 8U, 128U, 256U, 512U})
        for (uint32_t tokens : {0U, 1U, 33U, 1057U, 4096U, 16383U, 16384U}) check<32>(world, tokens, 6, 512);
    for (uint32_t topk : {1U, 8U, 16U}) {
        check<32>(128, 16384, topk, 1024); // Partial caches and GM fallback at large top-k.
        check<64>(256, 16384, topk, 512);
    }
}
"""
    )
    path = tmp_path / "notify-forward.cpp"
    path.write_text(code, encoding="utf-8")
    binary = tmp_path / ("notify-forward.exe" if sys.platform == "win32" else "notify-forward")
    subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(root / "csrc"), str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=90)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _add_launch_args(parser)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--chunk-tokens", type=int, choices=(256, 512, 1024), default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--route", choices=(*_ROUTES, "all"), default="random")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    add_profile_args(parser, ("notify",))
    args = parser.parse_args()
    validate_profile_args(parser, args)
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE") or args.repeat < 1:
        parser.error("Enable assertions and use --repeat >=1")
    if not 1 <= args.tokens <= 16384 or not 1 <= args.topk <= 16:
        parser.error("tokens1..16384 and topk1..16 required")
    if args.experts < args.ep_size or args.experts > 1024 or args.experts % args.ep_size:
        parser.error("experts must be divisible by EP and <=1024")
    _launch(args, notify_only_worker)


_trace_ROOT = Path(__file__).resolve().parents[2]

_trace_VIEW = runpy.run_path(str(_trace_ROOT / "tests/elastic/notify_profile.py"))

_trace_DISPATCH = globals()

_trace_CONFIG = dict(rank=1, world=128, tokens=4096, topk=6, experts=512, chunk_tokens=256, sample=3)


def _trace_records():
    result = []
    for core in range(32):
        header = [
            _trace_VIEW["MAGIC"],
            _trace_VIEW["VERSION"],
            1,
            core,
            128,
            4096,
            6,
            512,
            256,
            3,
            32,
            len(_trace_VIEW["PHASES"]),
            0,
            0,
            0,
            0,
        ]
        stamps = [2**54 + core * 100 + i * 2000 for i in range(len(_trace_VIEW["PHASES"]) + 1)]
        header[13:16] = [stamps[0] - 5000, stamps[0], 1]
        row = header + stamps
        result.append(row + [0] * (_trace_VIEW["WORDS"] - len(row)))
    return result


def test_trace_notify_trace_abi_and_views(tmp_path):
    text = (_trace_ROOT / "csrc/kernels/notify/profile.hpp").read_text(encoding="utf-8")
    phases = re.findall("^\\s+(\\w+) = (\\d+)U,", text, re.M)
    assert [name for name, _ in phases] == [*_trace_VIEW["PHASES"], "End"]
    assert [int(value) for _, value in phases] == list(range(len(phases)))
    data = _trace_VIEW["decode"](_trace_records(), **_trace_CONFIG)
    assert data["events"][0]["duration_us"] == 2.0
    assert data["span_us"] == len(_trace_VIEW["PHASES"]) * 2 + 3.1
    assert all((e["mean_us"] == 2 for e in data["phases"]))
    assert data["alignment"]["wait_us_by_core"] == [5.0] * 32
    assert data["alignment"]["exit_skew_us"] == 3.1
    assert data["alignment"]["calibrated_clocks"] is False
    paths = _trace_VIEW["write_timeline"](data, tmp_path / "rank-1.notify.json")
    ET.parse(paths["svg"])
    raw = json.loads(Path(paths["json"]).read_text())
    assert raw["raw"] == _trace_records()
    trace = json.loads(Path(paths["perfetto"]).read_text())
    slices = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    assert len(slices) == 32 * len(_trace_VIEW["PHASES"])
    assert slices[0]["dur"] == 2 and slices[0]["ts"] == 0
    assert {e["tid"] for e in slices} == set(range(32))
    assert len(Path(paths["csv"]).read_text().splitlines()) == len(slices) + 1
    assert len([e for e in slices if e["cat"] == "global_wait"]) == 3 * 32


@pytest.mark.parametrize(
    "field,value",
    [(0, 0), (1, 9), (2, 0), (3, 31), (9, 2), (11, 0), (12, 64), (13, 0), (14, 1), (15, 0), (16, 0), (17, 1), (63, 1)],
)
def test_trace_notify_rejects_incomplete_stale_or_invalid_trace(field, value):
    rows = _trace_records()
    rows[0][field] = value
    with pytest.raises(ValueError):
        _trace_VIEW["decode"](rows, **_trace_CONFIG)


@pytest.mark.parametrize("flags", [0, 16, 31, 63])
def test_trace_notify_cache_flags_accept_old_and_count_cached_traces(flags):
    rows = _trace_records()
    for row in rows:
        row[12] = flags
    data = _trace_VIEW["decode"](rows, **_trace_CONFIG)
    assert data["cache_flags"]["forward_counts"] == bool(flags & 32)
    assert data["cache_flags"]["parse_hist"] == bool(flags & 1)


def test_trace_notify_profile_is_outside_clean_capture(monkeypatch, tmp_path):
    scope = _trace_DISPATCH["profile_notify"].__globals__
    calls = []
    rows = _trace_records()
    for row in rows:
        row[9] = 1
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: rows))
    fake_torch = SimpleNamespace(
        int64="int64", zeros=lambda *a, **kw: tensor, npu=SimpleNamespace(synchronize=lambda: calls.append("sync"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "deep_ep._native",
        SimpleNamespace(require_native=lambda _: SimpleNamespace(notify_profile_layout=lambda: (1, 32, 64))),
    )

    def invoke(**kwargs):
        calls.append("trace" if kwargs.get("profile") is tensor else "clean")
        return "outputs"

    def capture(*args):
        calls.append("capture.begin")
        args[3]()
        calls.append("capture.end")
        return dict(kernel="notify_full_kernel")

    monkeypatch.setitem(scope, "profile_operator", capture)
    args = SimpleNamespace(output=tmp_path, ep_size=128, tokens=4096, topk=6, experts=512, chunk_tokens=256, seed=1)
    result = _trace_DISPATCH["profile_notify"](
        args, 1, "random", invoke, lambda _: calls.append("validate"), {}, lambda _: None
    )
    assert calls == ["capture.begin", "clean", "capture.end", "sync", "trace", "sync", "validate"]
    assert Path(result["diagnostics"]["svg"]).is_file()


def test_trace_notify_metadata_checker_detects_wrong_outputs():
    torch = pytest.importorskip("torch")

    class NpuView:
        device = SimpleNamespace(type="npu")

        def __init__(self, tensor):
            self.tensor, self.dtype, self.shape = (tensor, tensor.dtype, tensor.shape)

        def cpu(self):
            return self.tensor

        def __getitem__(self, key):
            return NpuView(self.tensor[key])

        def is_contiguous(self):
            return self.tensor.is_contiguous()

    routes = [[[0, 2], [0, 0]], [[1, 2], [-1, -1]]]
    golden = _trace_DISPATCH["build_notify_golden"](routes, 4, 256)
    plan = golden["plans"][0]
    outputs = {}
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
        outputs[name] = NpuView(
            torch.tensor(plan[name], dtype=torch.int64 if name in ("server_mask", "dst") else torch.int32)
        )
    for name, counts in (("forward_list", "forward_counts"), ("backward_list", "backward_counts")):
        width = 6 if name == "forward_list" else 3
        tensor = torch.full((2, 8, width), -987, dtype=torch.int32 if width == 6 else torch.int64)
        for i, n in enumerate(plan[counts]):
            if n:
                tensor[i, :n] = torch.tensor(plan[name][i], dtype=tensor.dtype)
        outputs[name] = NpuView(tensor)
    outputs["weight_return_meta"] = NpuView(torch.tensor([*plan["weight_return_meta"], -987, -987], dtype=torch.int32))
    outputs["expert_totals"] = NpuView(torch.tensor([3, 1, 2, 0], dtype=torch.int32))
    check = _trace_DISPATCH["check_notify_outputs"]
    check(outputs, golden, 0, routes, 4)
    for name, index in (("dst", 0), ("forward_counts", 0), ("weight_return_meta", 0), ("expert_totals", 0)):
        outputs[name].tensor.flatten()[index] += 1
        with pytest.raises(AssertionError, match=name):
            check(outputs, golden, 0, routes, 4)
        outputs[name].tensor.flatten()[index] -= 1


def test_notify_cli_runs_only_metadata(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sys.modules[__name__], "_launch", lambda args, worker: calls.append((args, worker)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_notify.py",
            "--tokens",
            "16384",
            "--chunk-tokens",
            "512",
            "--profile-op",
            "notify",
            "--output",
            str(tmp_path),
        ],
    )
    main()
    args, entry = calls.pop()
    assert entry is notify_only_worker and args.repeat == 1 and args.tokens == 16384
    for flags in (("--repeat", "0"), ("--tokens", "32768"), ("--profile-op", "dispatch")):
        monkeypatch.setattr(sys, "argv", ["test_notify.py", "--output", str(tmp_path), *flags])
        with pytest.raises(SystemExit):
            main()
    assert not calls


@pytest.mark.parametrize("with_masks", [False, True])
def test_shmem_masks_require_explicit_dense_fallback(tmp_path, with_masks):
    import shutil
    import subprocess

    compiler = shutil.which("g++")
    if sys.platform != "linux" or compiler is None:
        pytest.skip("SHMEM host doubles require Linux and a C++17 compiler")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "session_masks.cpp"
    source.write_text(
        r"""
#define main lifecycle_main
#include "session_test.cpp"
#undef main
#include "kernels/common.hpp"
#include "runtime/udma_peer_masks.hpp"
bool expected_combine = false, expected_dense = false;
#ifdef WITH_MASKS
extern "C" int aclshmemx_set_udma_peer_qp_masks(const uint64_t* masks, uint32_t count) {
    if (expected_dense) { assert(!masks && !count); return 0; }
    assert(count == uint32_t(expected_world) && masks[0] == 0);
    for (uint32_t peer=1; peer<count; ++peer)
        assert(masks[peer] == (expected_combine ? first_hit_schedule::CombineQpMask(0, peer, count)
                                              : deepep::udma_admission::PeerMask(0, peer, count)));
    return 0;
}
#endif
int main() {
    for (int world : {128, 256, 512}) {
        expected_world = world;
        for (bool dense : {false, true}) for (bool combine : {true, false}) {
            expected_combine = combine;
            expected_dense = dense;
            setenv("DEEPEP_SHMEM_DENSE_QPS", dense ? "1" : "0", 1);
#ifndef WITH_MASKS
            if (!dense) {
                const int previous_allocations = allocations;
                try {
                    deepep::Session s(0, world, expected_bytes, "tcp://127.0.0.1:19091", 30,
                                      true, false, false, combine);
                    assert(false);
                } catch (const std::runtime_error& error) {
                    assert(std::string(error.what()).find("aclshmemx_set_udma_peer_qp_masks") != std::string::npos);
                }
                assert(!deepep::Session::busy() && initialized == 0 && allocations == previous_allocations);
                continue;
            }
#endif
            deepep::Session s(0, world, expected_bytes, "tcp://127.0.0.1:19091", 30, true, false, false, combine);
            assert(s.combine_only() == combine);
#ifdef WITH_MASKS
            assert(s.sparse_qp_enabled() == !dense);
#else
            assert(!s.sparse_qp_enabled());
#endif
            s.close();
            assert(!deepep::Session::busy());
        }
    }
    setenv("DEEPEP_SHMEM_DENSE_QPS", "invalid", 1);
    fails([]() { deepep::Session s(0, 2, expected_bytes, "tcp://127.0.0.1:19091", 30, true); });
    assert(!deepep::Session::busy());
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "session_masks"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-rdynamic",
        "-I" + str(root / "tests/runtime/stubs"),
        "-I" + str(root / "tests/runtime"),
        "-I" + str(root / "csrc"),
        str(root / "csrc/runtime/session.cpp"),
        str(source),
        "-ldl",
        "-o",
        str(binary),
    ]
    if with_masks:
        command.append("-DWITH_MASKS")
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)


if __name__ == "__main__":
    main()
