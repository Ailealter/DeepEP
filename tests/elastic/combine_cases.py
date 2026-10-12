"""Ordered Notify/Combine cases sharing one process group and symmetric heap."""

import argparse
import json
from pathlib import Path


def add_arguments(parser, default_tokens):
    tokens = parser.add_mutually_exclusive_group()
    tokens.add_argument("--tokens", type=int, default=default_tokens)
    tokens.add_argument("--tokens-list", type=int, nargs="+", help="Token counts tested in one SHMEM session")
    chunks = parser.add_mutually_exclusive_group()
    chunks.add_argument("--chunk-tokens", type=int, choices=(256, 512, 1024), default=256)
    chunks.add_argument("--chunk-tokens-list", type=int, nargs="+", choices=(256, 512, 1024))


def make_cases(tokens, chunks, notify_cores=None):
    pairs = [(t, c) for t in tokens for c in chunks]
    if not pairs:
        raise ValueError("Combine case list must not be empty")
    if any(
        type(t) is not int or not 1 <= t <= 16384 or type(c) is not int or c not in (256, 512, 1024) for t, c in pairs
    ):
        raise ValueError("Cases require tokens 1..16384 and chunks 256/512/1024")
    if notify_cores is None:
        return [dict(case_id=f"case-{i:03d}-t{t}-c{c}", tokens=t, chunk_tokens=c) for i, (t, c) in enumerate(pairs)]
    if not notify_cores or any(type(n) is not int or n not in (32, 64) for n in notify_cores):
        raise ValueError("Notify cores must be 32 or 64")
    return [
        dict(case_id=f"case-{i:03d}-t{t}-c{c}-n{n}", tokens=t, chunk_tokens=c, notify_num_cores=n)
        for i, (t, c, n) in enumerate((t, c, n) for t, c in pairs for n in notify_cores)
    ]


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Combine case list must not be empty")
    for i, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) not in (
            {"case_id", "tokens", "chunk_tokens"},
            {"case_id", "tokens", "chunk_tokens", "notify_num_cores"},
        ):
            raise ValueError("Invalid Combine case fields")
        cores = [case["notify_num_cores"]] if "notify_num_cores" in case else None
        expected = make_cases([case["tokens"]], [case["chunk_tokens"]], cores)[0]
        expected["case_id"] = expected["case_id"].replace("case-000-", f"case-{i:03d}-", 1)
        if case != expected:
            raise ValueError("Invalid Combine case ID/order")
    return cases


def from_args(args):
    if getattr(args, "cases_file", None):
        return validate_cases(json.loads(Path(args.cases_file).read_text(encoding="utf-8")))
    return make_cases(
        getattr(args, "tokens_list", None) or [args.tokens],
        getattr(args, "chunk_tokens_list", None) or [args.chunk_tokens],
        getattr(args, "notify_num_cores_list", None),
    )


def case_args(args, case, multiple):
    result = argparse.Namespace(**(vars(args) | case))
    result.output = args.output / "cases" / case["case_id"] if multiple else args.output
    return result
