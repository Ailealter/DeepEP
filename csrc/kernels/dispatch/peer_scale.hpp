// Copyright (c) 2026, Lu Lu

#pragma once

#include "peer_address.hpp"

// Queue owners publish terminal notifications after their URMA SO/quiet.
// Local MTE stores are drained before AIV7's incoming flag wait.
template <bool Profile>
__aicore__ inline void DrainDispatchStores(const DispatchTiling& t, uint64_t* detail)
{
    const uint64_t stamp = DetailClock<Profile>(t.detail_mode);
    aclshmemx_mte_quiet();
    DetailAdd<Profile>(detail, 12U, DetailClock<Profile>(t.detail_mode) - stamp);
}

// Peer-double pipeline: SIMT decode, vector compaction,
// self-contained 512B data-with-flag packets, per-peer ping-pong and SIMT scatter.
#include "scale_pipeline.hpp"
#include "aux_pipeline.hpp"
template <bool Profile>
__aicore__ inline void PeerScales(GM_ADDR workspace, GM_ADDR input, GM_ADDR destinations, const DispatchTiling& t,
                                  TBuf<QuePosition::VECCALC>& metadata, TBuf<QuePosition::VECCALC>& staging,
                                  TBuf<TPosition::VECOUT>& wqe_scratch)
{
    PeerScalePlan plan{destinations, input, reinterpret_cast<GM_ADDR>(t.output_scales_address), t.address_stride};
    DispatchScalePipeline::Run<Profile>(workspace, t, plan, staging, metadata, wqe_scratch);
}
