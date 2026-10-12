// Copyright (c) 2026, Lu Lu

#pragma once

// Read Notify's public six-column forward_list directly. Compact only the
// current UB tile; no persistent FM directory or additional Notify pass.
// Remote: SIMT builds RO WQEs against published ordinary GM virtual addresses on private QP0.
// Local: MTE ping-pong, never interpreted as a remote address.
// Both own-source and received-source expansion use the same MTE double buffer.
// The caller acquires the appropriate source-ready flag before entering here.
__aicore__ inline void CopyLocalForwardRecords(DeepepSecondaryState& s, uint32_t count, LocalTensor<uint32_t> records,
                                               TBuf<QuePosition::VECCALC>& staging, uint32_t& stage)
{
    for (uint32_t row = 0; row < count; row += 4U) {
        const uint32_t n = count - row < 4U ? count - row : 4U;
        uint32_t source_slot[4];
        const TEventID event = stage ? EVENT_ID1 : EVENT_ID0;
        auto ub = staging.GetWithOffset<uint8_t>(kDuplicateHiddenStageBytes, stage * kDuplicateHiddenStageBytes);
        WaitFlag<HardEvent::MTE3_MTE2>(event);
        for (uint32_t i = 0; i < n; ++i) {
            source_slot[i] = i;
            const uint32_t primary = records.GetValue((row + i) * 2U);
            for (uint32_t j = 0; j < i; ++j)
                if (records.GetValue((row + j) * 2U) == primary) {
                    source_slot[i] = source_slot[j];
                    break;
                }
            if (source_slot[i] == i)
                IssueGmBytesToUb(ub[i * s.row_bytes], s.hidden + uint64_t(primary) * s.row_bytes, s.row_bytes);
        }
        SetFlag<HardEvent::MTE2_MTE3>(event);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        for (uint32_t i = 0; i < n; ++i)
            IssueUbBytesToGm(s.hidden + uint64_t(records.GetValue((row + i) * 2U + 1U)) * s.row_bytes,
                             ub[source_slot[i] * s.row_bytes], s.row_bytes);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        stage ^= 1U;
    }
}

// Each thread decodes one original row. Repeat the target for both words so
// one vector GatherMask produces interleaved (primary row, output row) pairs.
__simt_vf__ __launch_bounds__(128) inline void DecodeForwardRowsVf(__ubuf__ int32_t* input, __ubuf__ int32_t* targets,
                                                                   __ubuf__ uint32_t* pairs, uint32_t count)
{
    for (uint32_t row = threadIdx.x; row < 128U; row += 128U) {
        const int32_t target = row < count ? input[row * 6U + 3U] : -1;
        targets[row * 2U] = target;
        targets[row * 2U + 1U] = target;
        pairs[row * 2U] = row < count ? uint32_t(input[row * 6U + 2U]) : 0U;
        pairs[row * 2U + 1U] = row < count ? uint32_t(input[row * 6U + 4U]) : 0U;
    }
}

// Two bounded metadata banks, as in Combine's GatewayTaskCache. Prefetch only
// this source/core's owned range; an unused prefetch is drained on parking.
struct ForwardMetadataCache {
    LocalTensor<int32_t> banks;
    __gm__ int32_t* source;
    bool pending[2]{};

    __aicore__ inline void Load(uint32_t slot, uint32_t row, uint32_t count)
    {
        const auto event = slot ? EVENT_ID5 : EVENT_ID4;
        SetFlag<HardEvent::V_MTE2>(event);
        WaitFlag<HardEvent::V_MTE2>(event);
        GlobalTensor<int32_t> gm;
        gm.SetGlobalBuffer(source + uint64_t(row) * 6U);
        DataCopyPad(banks[slot * 128U * 6U], gm, DataCopyExtParams(1U, count * 24U, 0U, 0U, 0U),
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(event);
        pending[slot] = true;
    }
    __aicore__ inline LocalTensor<int32_t> Wait(uint32_t slot)
    {
        if (pending[slot]) {
            const auto event = slot ? EVENT_ID5 : EVENT_ID4;
            WaitFlag<HardEvent::MTE2_V>(event);
            pending[slot] = false;
        }
        return banks[slot * 128U * 6U];
    }
    __aicore__ inline void Finish()
    {
        Wait(0U);
        Wait(1U);
        SetFlag<HardEvent::V_S>(EVENT_ID2);
        WaitFlag<HardEvent::V_S>(EVENT_ID2);
    }
};

__aicore__ inline uint32_t SelectForwardRows(LocalTensor<int32_t> input, uint32_t count, uint32_t peer,
                                             TBuf<QuePosition::VECCALC>& metadata)
{
    // [0,1024): selected pairs; [1024,2304): up to 31 carried + 128 new pairs.
    // Two raw banks occupy [4416,10560); source arrays start at 12288.
    auto pairs = metadata.GetWithOffset<uint32_t>(256U, 2304U);
    auto targets = metadata.GetWithOffset<int32_t>(256U, 3328U);
    auto mask = metadata.GetWithOffset<uint8_t>(32U, 4352U);
    asc_vf_call<DecodeForwardRowsVf>(dim3(128), reinterpret_cast<__ubuf__ int32_t*>(input.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ int32_t*>(targets.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ uint32_t*>(pairs.GetPhyAddr()), count);
    PipeBarrier<PIPE_V>();
    Compares(mask, targets, int32_t(peer), CMPMODE::EQ, 256U);
    PipeBarrier<PIPE_V>();
    uint64_t words = 0U;
    GatherMask(metadata.Get<uint32_t>(), pairs, mask.ReinterpretCast<uint32_t>(), true, count * 2U, {1U, 1U, 0U, 0U},
               words);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    return uint32_t(words / 2U);
}

// Append only acquired sources. Carrying across raw tiles (and ready sources)
// keeps the 32-WQE builder full; filtering does not change the 4-WQE DB batch.
__simt_vf__ __launch_bounds__(128) inline void AppendForwardPairsVf(__ubuf__ uint32_t* accumulated,
                                                                    __ubuf__ uint32_t* selected, uint32_t pending,
                                                                    uint32_t count)
{
    for (uint32_t word = threadIdx.x; word < count * 2U; word += 128U)
        accumulated[pending * 2U + word] = selected[word];
}

__simt_vf__ __launch_bounds__(128) inline void CarryForwardPairsVf(__ubuf__ uint32_t* accumulated, uint32_t first,
                                                                   uint32_t count)
{
    // The source starts at >=32 pairs; the retained tail has <32 pairs, so
    // input and output cannot overlap, even across SIMT threads.
    for (uint32_t word = threadIdx.x; word < count * 2U; word += 128U)
        accumulated[word] = accumulated[first * 2U + word];
}

// Read each source until the first matching row before waiting. A primary-only
// source need not publish an early SO; waiting on every forward_counts entry
// would deadlock. A parked source remembers its tile and is not rescanned
// while its ready flag remains absent.
template <bool Profile, bool Local>
__aicore__ inline void ForwardPeer(DeepepSecondaryState& s, const DispatchTiling& t, GM_ADDR forward, GM_ADDR counts,
                                   TBuf<QuePosition::VECCALC>& metadata, TBuf<QuePosition::VECCALC>& staging,
                                   __ubuf__ uint8_t* wqe)
{
    if constexpr (!Local) {
        if (s.core + 1U >= s.lanes) return;
    }
    const uint32_t lane = Local ? s.rank % 8U : (s.rank % 8U + s.core + 1U) % s.lanes;
    const uint32_t peer = s.rank / 8U * 8U + lane;
    const uint32_t capacity = t.tokens * t.topk;
    const uint64_t profile = t.profile + uint64_t(s.core) * 256U;
    auto cursors = metadata.GetWithOffset<uint32_t>(s.world, 14336U);
    auto ends = metadata.GetWithOffset<uint32_t>(s.world, 12288U);
    auto records = metadata.Get<uint32_t>();
    auto accumulated = metadata.GetWithOffset<uint32_t>(320U, 1024U);
    uint32_t carried = 0U;
    NativeLoad(ends, reinterpret_cast<__gm__ uint32_t*>(counts), s.world);
    uint32_t seen[16]{}, parked[16]{};
    for (uint32_t source = 0U; source < s.world; ++source) {
        const int32_t count = int32_t(ends.GetValue(source));
        if (count < 0 || uint32_t(count) > capacity) {
            aclshmemi_kernel_abort("FM invalid forward count");
            trap();
            return;
        }
        uint32_t first = 0U, last = uint32_t(count);
        if constexpr (Local) {
            // Rotate raw-row partitions across sources; every public record has
            // exactly one local reader. Filtering never changes Combine's table.
            const uint32_t base = source * capacity;
            ascend_deepep::scale_clos::SplitLocalForwardRange(s.world, s.core, base, base + last, first, last);
            first -= base;
            last -= base;
        } else {
            // Rank dedup has only local expansion: all cross-rank payloads
            // were sent by the primary path, with no FM relay work.
            if (t.dedup_group_size == 1U) last = 0U;
        }
        cursors.SetValue(source, first);
        ends.SetValue(source, last);
        if (first == last) seen[source / 32U] |= 1U << (source % 32U);
    }
    if constexpr (Local) {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        DispatchProfileMark<Profile>(profile, 8U);
        DispatchProfileMark<Profile>(profile, 9U);
    }
    uint32_t stage = 0U;
    uint64_t issued = 0U, batches = 0U, scan_ticks = 0U, scanned = 0U;
    uint64_t local_work_ticks = 0U, local_rows = 0U;
    __gm__ aclshmemi_udma_queue_state_t* fm_queue = nullptr;
    if constexpr (!Local) {
        auto* info = aclshmemi_udma_qp_info_fetch();
        const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
        fm_queue = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(
            aclshmemi_udma_get_qp_ctx(info, slot, 0U)->state_addr);
    }
    bool own_done = !Local;
    while (true) {
        bool pending = false, progress = false;
        // Local expansion visits its own original source first. Sources after
        // that remain independent: an absent SO never blocks another source.
        for (uint32_t visit = 0U; visit < s.world; ++visit) {
            const uint32_t source = Local ? (s.rank + visit) % s.world : visit;
            const uint32_t bit = 1U << (source % 32U), word = source / 32U;
            if (seen[word] & bit) continue;
            pending = true;
            bool ready = IsSourceReadyForForwarding(source, GetGroupRankIndex(s.rank, s.world), s.magic, 2U, s.sync);
            if ((parked[word] & bit) && !ready) continue;
            uint32_t at = cursors.GetValue(source), end = ends.GetValue(source);
            ForwardMetadataCache cache{metadata.GetWithOffset<int32_t>(2U * 128U * 6U, 4416U),
                                       reinterpret_cast<__gm__ int32_t*>(forward) + uint64_t(source) * capacity * 6U};
            uint32_t bank = 0U;
            cache.Load(bank, at, end - at < 128U ? end - at : 128U);
            while (at < end) {
                const uint32_t tile = end - at < 128U ? end - at : 128U;
                const uint64_t scan_start = DispatchProfileClock<Profile>();
                auto input = cache.Wait(bank);
                const uint32_t next = at + tile;
                if (next < end) cache.Load(bank ^ 1U, next, end - next < 128U ? end - next : 128U);
                const uint32_t count = SelectForwardRows(input, tile, peer, metadata);
                if constexpr (Profile) {
                    scan_ticks += DispatchProfileClock<true>() - scan_start;
                    scanned += tile;
                }
                if (count && !ready) {
                    parked[word] |= bit;
                    break;
                }
                const uint64_t copy_start = DispatchProfileClock < Profile && Local > ();
                if constexpr (Local) {
                    CopyLocalForwardRecords(s, count, records, staging, stage);
                    if constexpr (Profile) {
                        local_work_ticks += DispatchProfileClock<true>() - copy_start;
                        local_rows += count;
                    }
                } else {
                    if (count) {
                        PipeBarrier<PIPE_V>();
                        asc_vf_call<AppendForwardPairsVf>(
                            dim3(128), reinterpret_cast<__ubuf__ uint32_t*>(accumulated.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t*>(records.GetPhyAddr()), carried, count);
                        PipeBarrier<PIPE_V>();
                        const uint32_t ready_rows = (carried + count) / 32U * 32U;
                        if (ready_rows) AdmitFmRows(s, peer, wqe, fm_queue, ready_rows);
                        for (uint32_t row = 0U; row < ready_rows; row += 32U)
                            IssueFmRows(s, peer, wqe, accumulated, row, 32U);
                        issued += ready_rows;
                        batches += ready_rows / 4U;
                        carried += count;
                        carried -= ready_rows;
                        if (ready_rows && carried) {
                            PipeBarrier<PIPE_V>();
                            asc_vf_call<CarryForwardPairsVf>(
                                dim3(128), reinterpret_cast<__ubuf__ uint32_t*>(accumulated.GetPhyAddr()), ready_rows,
                                carried);
                            PipeBarrier<PIPE_V>();
                        }
                    }
                }
                at += tile;
                bank ^= 1U;
                progress = true;
            }
            cache.Finish();
            cursors.SetValue(source, at);
            if (at == end) {
                seen[word] |= bit;
                if constexpr (Local) {
                    if (source == s.rank && !own_done) {
                        DispatchProfileMark<Profile>(profile, 10U);
                        if constexpr (Profile) reinterpret_cast<__gm__ uint64_t*>(profile)[22] = local_rows;
                        own_done = true;
                    }
                }
            }
        }
        if (!pending) break;
        if (!progress) {
            if constexpr (!Local) {
                // As in the earlier FM path, never stall a ready short tail
                // just to fill a batch while other sources are still in flight.
                if (carried) {
                    AdmitFmRows(s, peer, wqe, fm_queue, carried);
                    IssueFmRows(s, peer, wqe, accumulated, 0U, carried);
                    issued += carried;
                    batches += (carried + 3U) / 4U;
                    carried = 0U;
                }
            }
            PauseClosAckPoll();
        }
    }
    if constexpr (Local) {
        if (!own_done) DispatchProfileMark<Profile>(profile, 10U);
        const uint64_t drain_start = DispatchProfileClock<Profile>();
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        aclshmemx_mte_quiet();
        if constexpr (Profile) {
            auto* record = reinterpret_cast<__gm__ uint64_t*>(profile);
            record[18] = local_work_ticks;
            record[19] = local_rows;
            record[21] = DispatchProfileClock<true>() - drain_start;
        }
    } else {
        if (carried) {
            AdmitFmRows(s, peer, wqe, fm_queue, carried);
            IssueFmRows(s, peer, wqe, accumulated, 0U, carried);
            issued += carried;
            batches += (carried + 3U) / 4U;
        }
        DispatchProfileMark<Profile>(profile, 8U);
        FinishFmRows(s, peer, wqe);
        DispatchProfileMark<Profile>(profile, 9U);
        aclshmemx_udma_qp_quiet(peer, 0U);
        // The final SO/CQE completed primary + relay writes on this private SQ.
        // Publish separately from ready/checkpoint SO sinks; one peer uses UB.
        auto targets = metadata.Get<uint64_t>();
        targets.SetValue(0U, reinterpret_cast<uint64_t>(
                                 aclshmem_ptr(DispatchCompletionFlag(s.completion, s.world, s.rank, 2U), peer)));
        PublishDispatchCompletionTargets(targets, 1U, s.magic, metadata.GetWithOffset<uint64_t>(4U, 1024U));
        DispatchProfileMark<Profile>(profile, 10U);
        if constexpr (Profile) {
            auto* record = reinterpret_cast<__gm__ uint64_t*>(profile);
            record[11] = issued;
            record[12] = batches;
            record[13] = peer;
            record[24] = s.fm_checkpoints;
            record[25] = 1U;
        }
    }
    if constexpr (Profile) {
        auto* record = reinterpret_cast<__gm__ uint64_t*>(profile);
        record[14] = scan_ticks;
        record[15] = scanned;
    }
}
