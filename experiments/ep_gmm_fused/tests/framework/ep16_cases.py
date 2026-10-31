"""EP16 G/K/N/TopK cases shared by accuracy and native profiling."""

from .cases import CaseSpec


EP_SIZE = 16


TOKEN_UNIT = 1024


STAGES = ("smoke", "topk", "scale")


OPERATORS = ("alltoallv_gmm", "gmm_alltoallv", "composition")


SMALL_TOPK = (1, 2, 3, 4, 8, 16, 31, 32)


SMALL_TOKENS = tuple(value * TOKEN_UNIT for value in (1, 2, 4, 8, 16, 32))


LARGE_TOKENS = tuple(value * TOKEN_UNIT for value in (9, 16, 32, 64, 128, 256, 300))


SHAPE_FAMILIES = (
    (3328, 2560, (16, 8), SMALL_TOKENS),
    (1280, 3328, (16, 8), SMALL_TOKENS),
    (1024, 384, (32, 8, 4), LARGE_TOKENS),
    (384, 1024, (32, 8, 4), LARGE_TOKENS),
)


def _stage_cases(stage, seed):
    for k, n, expert_counts, token_counts in SHAPE_FAMILIES:
        for local_experts in expert_counts:
            for global_tokens in token_counts:
                smallest = global_tokens == token_counts[0]
                for top_k in SMALL_TOPK if smallest else (1, 32):
                    for route in ("balanced", "random"):
                        qualification_stage = "scale"
                        if smallest:
                            qualification_stage = "smoke" if top_k == 1 and route == "balanced" else "topk"
                        if qualification_stage != stage:
                            continue
                        local_tokens = global_tokens // EP_SIZE
                        expanded_rows = local_tokens * top_k
                        for operator in OPERATORS:
                            scenario = dict(
                                category="ep16_gkn",
                                qualification_stage=qualification_stage,
                                label=(
                                    f"ep16-{operator}-g{local_experts}-k{k}-n{n}"
                                    f"-t{global_tokens}-q{top_k}-{route}-s{seed}"
                                ),
                                required_world_size=EP_SIZE,
                                token_scope="global_original",
                                token_unit=TOKEN_UNIT,
                                global_tokens=global_tokens,
                                local_tokens=local_tokens,
                                expanded_rows=expanded_rows,
                                global_expanded_rows=global_tokens * top_k,
                                global_experts=EP_SIZE * local_experts,
                                top_k=top_k,
                                routing_seed=seed,
                                n_group=1,
                                topk_group=1,
                                drop_tokens=False,
                            )
                            if operator == "composition":
                                scenario["second_weight_shape"] = [local_experts, n, k]
                            yield CaseSpec(
                                operator=operator,
                                ep_size=EP_SIZE,
                                local_experts=local_experts,
                                m=expanded_rows,
                                k=k,
                                n=n,
                                route=route,
                                seed=seed,
                                preallocate=operator != "composition",
                                scenario=scenario,
                            )


def generate_ep16_cases(stage="smoke", seed=20260920):
    """Return replayable cases; M is (global original tokens / 16) * TopK.

    Counts are smoke=30, topk=450, scale=672 and all=1152. The smoke stage
    exercises every GKN/operator at the smallest token count and TopK=1;
    topk adds the remaining smallest-input routing combinations, and scale
    covers all larger token counts at TopK=1 and TopK=32.
    """
    if stage not in (*STAGES, "all"):
        raise ValueError(f"unknown EP16 stage: {stage}; expected smoke, topk, scale or all")
    stages = STAGES if stage == "all" else (stage,)
    return [case for selected in stages for case in _stage_cases(selected, seed)]
