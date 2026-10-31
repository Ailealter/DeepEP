# Copyright (c) 2026, Lu Lu
# Modified by zhu-mingzhe71 2026

"""Route construction and assertions, independent of the operator implementation."""

import numpy as np


def assignments(case, source):
    """[token, TopK] distinct global expert IDs; input rows are expert sorted."""
    s = case.scenario or {}
    q, e, p = s["top_k"], case.local_experts, case.ep_size
    total, tokens = e * p, case.m // q
    groups, selected_groups = s.get("n_group", 1), s.get("topk_group", 1)
    if not 0 <= source < p:
        raise ValueError("source rank must be inside the EP group")
    if "receive_ratio" in s:
        if case.route != "hot_rank":
            raise ValueError("receive_ratio requires the hot_rank route")
        hot = round(s["receive_ratio"] * case.m / p)
        cumulative = np.arange(tokens + 1, dtype=np.int64) * hot // tokens
        hot_per_token = np.diff(cumulative)
        if hot_per_token.max() > e or (q - hot_per_token).max() > total - e:
            raise ValueError("hot route cannot preserve distinct experts per token")
        slots = np.arange(q)[None, :]
        hot_ids = (cumulative[:-1, None] + slots + source) % e
        other_start = np.arange(tokens) * q - cumulative[:-1]
        other_ids = e + (other_start[:, None] + slots - hot_per_token[:, None] + source) % (total - e)
        output = np.where(slots < hot_per_token[:, None], hot_ids, other_ids)
        if groups > 1 and selected_groups < groups:
            group_ids = np.sort(output // (total // groups), axis=1)
            if np.any(1 + np.count_nonzero(np.diff(group_ids, axis=1), axis=1) > selected_groups):
                raise ValueError("hot_rank assignments exceed topk_group")
        return output
    if case.route == "balanced":
        output = np.arange(case.m).reshape(tokens, q)
        output += source * e
        if groups > 1 and selected_groups < groups:
            group_size = total // groups
            start = output[:, :1] // group_size * group_size
            # Fold only positions outside the selected contiguous group window.
            # q <= window capacity keeps each token's experts distinct; already
            # legal balanced assignments retain their original expert IDs.
            output -= start
            output %= selected_groups * group_size
            output += start
        output %= total
        return output
    if case.route != "random":
        raise ValueError("TopK routing supports balanced, random, or hot_rank with receive_ratio")
    rng = np.random.default_rng(s.get("routing_seed", case.seed) + source * 101)
    output = np.empty((tokens, q), dtype=np.int64)
    for begin in range(0, tokens, 2048):
        size = min(2048, tokens - begin)
        scores = rng.random((size, total))
        if groups > 1:
            chosen = np.argpartition(rng.random((size, groups)), selected_groups - 1, axis=1)[:, :selected_groups]
            allowed = np.zeros((size, groups), dtype=bool)
            np.put_along_axis(allowed, chosen, True, axis=1)
            scores[~np.repeat(allowed, total // groups, axis=1)] = np.inf
        output[begin : begin + size] = np.argpartition(scores, q - 1, axis=1)[:, :q]
    return output


def token_order(case, source):
    expert_ids = assignments(case, source).reshape(-1)
    return np.argsort(expert_ids, kind="stable") // case.scenario["top_k"]


def special_route(case):
    s = case.scenario or {}
    p, e = case.ep_size, case.local_experts
    if "top_k" in s:
        return np.stack([np.bincount(assignments(case, source).reshape(-1), minlength=p * e) for source in range(p)])
    if s.get("category") == "segment_boundary":
        return np.full((p, p * e), s["segment_rows"], dtype=np.int64)
    if s.get("category") == "zero_receive":
        table = np.zeros((p, p * e), dtype=np.int64)
        receivers = 1 if s["pattern"] == "one_receiver" else p - 1
        for source in range(p):
            table[source, : receivers * e] = case.m // (receivers * e)
            table[source, : case.m % (receivers * e)] += 1
        return table
    return None


def route_statistics(case, table):
    p, e = case.ep_size, case.local_experts
    if table.shape != (p, p * e) or (table < 0).any():
        raise AssertionError("invalid route table")
    sent, expert = table.sum(axis=1), table.sum(axis=0).reshape(p, e)
    received = expert.sum(axis=1)
    if not np.all(sent == case.m) or sent.sum() != received.sum():
        raise AssertionError("route conservation failed")
    total = int(received.sum())
    result = dict(
        sent_rows=sent.tolist(),
        received_rows=received.tolist(),
        expert_rows=expert.tolist(),
        zero_receive_ranks=np.flatnonzero(received == 0).tolist(),
        empty_experts=np.argwhere(expert == 0).tolist(),
        total_rows=total,
        max_rank_share=float(received.max() / total) if total else None,
        max_expert_share=float(expert.max() / total) if total else None,
    )
    s = case.scenario or {}
    if s.get("category") == "zero_receive":
        expected = p - 1 if s["pattern"] == "one_receiver" else 1
        assert len(result["zero_receive_ranks"]) == expected, "zero receive target lost"
    if "segment_rows" in s:
        assert np.all(table == s["segment_rows"]), "segment boundary not reached"
    if "receive_ratio" in s:
        assert abs(received[0] - s["receive_ratio"] * case.m) <= p, "hot rank target not reached"
        assert np.all(received > 0), "positive imbalance created empty rank"
    if case.route == "empty_experts":
        assert np.count_nonzero(expert) <= p, "sparse route target lost"
    if case.route in ("hot_rank", "hot_expert") and "top_k" not in s and p > 1:
        metric = "max_rank_share" if case.route == "hot_rank" else "max_expert_share"
        uniform = 1 / p if case.route == "hot_rank" else 1 / (p * e)
        assert result[metric] > uniform * 1.5, "hot distribution not realized"
    result["distribution_assertions_passed"] = True
    return result
