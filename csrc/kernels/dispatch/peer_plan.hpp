// Copyright (c) 2026, Lu Lu
#pragma once

// Directory [phase,peer,QP,{begin,count}], scratch [peer,ceil(T/16),4].
// Notify's existing barriers separate classification, prefix, directory, emit.
template <bool Write, uint32_t Cores, bool Cached = false, bool SavedPlan = false>
__simt_vf__ __launch_bounds__(128) inline void NotifyDispatchRowsVf(
    __gm__ int32_t* routes, __gm__ int32_t* hist, __gm__ int32_t* segments, __gm__ int32_t* gateways,
    __gm__ int32_t* bases, __gm__ int32_t* directory, __gm__ int32_t* rows, __gm__ int32_t* scratch, uint32_t rank,
    uint32_t world, uint32_t tokens, uint32_t topk, uint32_t experts, uint32_t group, uint32_t core,
    __ubuf__ int32_t* cached_routes = nullptr, __ubuf__ int32_t* cached_plan = nullptr)
{
    const uint32_t tiles = (tokens + 15U) / 16U, stride = (experts + world + 15U) / 16U * 16U, local = experts / world;
    const uint32_t forward_tiles = (tokens + 31U) / 32U;
    const uint32_t local_tiles = core < forward_tiles ? (forward_tiles - core + Cores - 1U) / Cores : 0U;
    const uint32_t tasks = Cached ? local_tiles * 2U * world : tiles * world;
    for (uint32_t task = Cached ? threadIdx.x : threadIdx.x * Cores + core; task < tasks;
         task += Cached ? 128U : Cores * 128U) {
        const uint32_t peer = task % world;
        const uint32_t tile = Cached ? (core + task / world / 2U * Cores) * 2U + task / world % 2U : task / world;
        if (tile >= tiles) continue;
        auto* prefix = scratch + (peer * tiles + tile) * 4U;
        uint32_t cursor[4]{};
        if constexpr (Write) {
            for (uint32_t b = 0U; b < 4U; ++b)
                cursor[b] = uint32_t(asc_ldcg(directory + (b / 2U * world + peer) * 4U + b % 2U * 2U)) +
                            uint32_t(asc_ldcg(prefix + b));
        }
        const uint32_t total = uint32_t(asc_ldcg(gateways + peer)), col = experts + peer, next = tile + 1U;
        uint32_t ordinal =
            uint32_t(asc_ldcg(hist + tile * stride + col)) + uint32_t(asc_ldcg(segments + tile / 16U * stride + col));
        const uint32_t end_ordinal = next == tiles ? total
                                                   : uint32_t(asc_ldcg(hist + next * stride + col)) +
                                                         uint32_t(asc_ldcg(segments + next / 16U * stride + col));
        const uint32_t full = total / 64U * 64U, tail = total - full - (total - full) / 4U;
        const uint32_t end = tokens < next * 16U ? tokens : next * 16U;
        // Original-input FM ranges are QP0-only, separate from Clos splitting.
        // Self copies never consume this directory; no FM forwarding index.
        const bool fm = peer / 8U == rank / 8U;
        if (peer != rank && ordinal != end_ordinal) {
            for (uint32_t token = tile * 16U; token < end; ++token) {
                for (uint32_t slot = 0U; slot < topk; ++slot) {
                    auto* route = routes + (token * topk + slot) * 6U;
                    const uint32_t cached_token = tile / 2U / Cores * 32U + token % 32U;
                    auto* cached = Cached ? cached_routes + (cached_token * topk + slot) * 6U : nullptr;
                    // Notify already resolved the gateway rank. Avoid repeating
                    // expert/rank division for every peer's scan of this tile.
                    int32_t expert = -1;
                    if constexpr (SavedPlan) {
                        if (cached[1] != int32_t(peer) || (cached[5] & 2) == 0) continue;
                    } else {
                        expert = Cached ? cached[0] : asc_ldcg(route);
                        const int32_t flags = Cached ? cached[5] : asc_ldcg(route + 5U);
                        if (expert < 0 || uint32_t(expert) / local != peer || (flags & 2) == 0) continue;
                    }
                    // Use the whole-peer ordinal before dividing into phases.
                    const uint32_t qp = !fm && (ordinal < full ? ordinal % 64U >= 48U : ordinal - full >= tail);
                    ++ordinal;
                    bool relay = false;
                    if constexpr (SavedPlan) {
                        // Notify omits the primary forward record iff another
                        // expert shares it. Reuse that classification.
                        relay = cached_plan[(cached_token * topk + slot) * 2U] < 0;
                    } else {
                        for (uint32_t k = 0U; k < topk; ++k) {
                            auto* sibling = routes + (token * topk + k) * 6U;
                            auto* sibling_ub = Cached ? cached_routes + (cached_token * topk + k) * 6U : nullptr;
                            const int32_t target = Cached ? sibling_ub[0] : asc_ldcg(sibling);
                            const int32_t sibling_flags = Cached ? sibling_ub[5] : asc_ldcg(sibling + 5U);
                            relay |= target >= 0 && (sibling_flags & 2) == 0 &&
                                     uint32_t(target) / local / group == peer / group;
                        }
                    }
                    const uint32_t bucket = (relay ? 0U : 2U) + qp, at = cursor[bucket]++;
                    if constexpr (SavedPlan && !Write) {
                        // Bit zero remains Notify's direct marker. Upper bits
                        // retain this route's class and tile-local row ordinal.
                        auto* saved = cached_plan + (cached_token * topk + slot) * 2U + 1U;
                        *saved = (*saved & 1) | int32_t((at * 4U + bucket + 1U) << 1U);
                    }
                    if constexpr (Write) {
                        asc_stcg(rows + at * 2U, int32_t(token));
                        asc_stcg(rows + at * 2U + 1U,
                                 asc_ldcg(bases + expert) + (Cached ? cached[4] : asc_ldcg(route + 4U)));
                    }
                }
            }
        }
        if constexpr (!Write)
            for (uint32_t b = 0U; b < 4U; ++b) asc_stcg(prefix + b, int32_t(cursor[b]));
    }
    asc_threadfence();
}

// The cached count pass retained class/ordinal in spare bits of Notify's UB
// plan. Each route now owns its final pair; no second serial peer/tile scan.
template <uint32_t Cores>
__simt_vf__ __launch_bounds__(128) inline void NotifyDispatchEmitCachedVf(
    __ubuf__ int32_t* meta, __ubuf__ int32_t* plan, __ubuf__ int32_t* bases, __gm__ int32_t* directory,
    __gm__ int32_t* rows, __gm__ int32_t* prefixes, uint32_t rank, uint32_t world, uint32_t tokens, uint32_t topk,
    uint32_t experts, uint32_t core)
{
    const uint32_t tiles = (tokens + 15U) / 16U, forward_tiles = (tokens + 31U) / 32U;
    const uint32_t local_tiles = core < forward_tiles ? (forward_tiles - core + Cores - 1U) / Cores : 0U;
    for (uint32_t route = threadIdx.x; route < local_tiles * 32U * topk; route += 128U) {
        const uint32_t local_token = route / topk;
        const uint32_t token = (core + local_token / 32U * Cores) * 32U + local_token % 32U;
        if (token >= tokens) continue;
        auto* m = meta + route * 6U;
        const int32_t expert = m[0];
        if (expert < 0 || (m[5] & 2) == 0) continue;
        const uint32_t peer = uint32_t(expert) / (experts / world);
        if (peer == rank) continue;
        const uint32_t encoded = (uint32_t(plan[route * 2U + 1U]) >> 1U) - 1U;
        const uint32_t bucket = encoded % 4U;
        const uint32_t at = uint32_t(asc_ldcg(directory + (bucket / 2U * world + peer) * 4U + bucket % 2U * 2U)) +
                            uint32_t(asc_ldcg(prefixes + (peer * tiles + token / 16U) * 4U + bucket)) + encoded / 4U;
        asc_stcg(rows + at * 2U, int32_t(token));
        asc_stcg(rows + at * 2U + 1U, bases[expert] + m[4]);
    }
    asc_threadfence();
}

// Four UB columns with one AIV owner per peer. Blocks fit in 2 KiB.
__simt_vf__ __launch_bounds__(128) inline void NotifyDispatchPrefixBlockVf(__ubuf__ int32_t* block,
                                                                           __ubuf__ int32_t* carry, uint32_t count,
                                                                           bool first, bool last,
                                                                           __gm__ int32_t* directory, uint32_t peer,
                                                                           uint32_t world)
{
    // One warp per class, four consecutive tiles per lane. All original
    // values stay in registers until the warp prefix is complete.
    const uint32_t bucket = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    int32_t values[4], sum = 0;
    const int32_t base = first ? 0 : carry[bucket];
    for (uint32_t i = 0U; i < 4U; ++i) {
        const uint32_t tile = lane * 4U + i;
        values[i] = tile < count ? block[tile * 4U + bucket] : 0;
        sum += values[i];
    }
    int32_t inclusive = sum;
    for (uint32_t delta = 1U; delta < 32U; delta *= 2U) {
        const int32_t preceding = asc_shfl_up(inclusive, delta, 32);
        if (lane >= delta) inclusive += preceding;
    }
    int32_t prefix = base + inclusive - sum;
    for (uint32_t i = 0U; i < 4U; ++i) {
        const uint32_t tile = lane * 4U + i;
        if (tile < count) block[tile * 4U + bucket] = prefix;
        prefix += values[i];
    }
    if (lane == 31U) {
        carry[bucket] = base + inclusive;
        if (last) asc_stcg(directory + (bucket / 2U * world + peer) * 4U + bucket % 2U * 2U + 1U, base + inclusive);
    }
    asc_threadfence();
}

// Read immutable counts; write disjoint begin fields. No second barrier.
template <uint32_t Cores>
__simt_vf__ __launch_bounds__(128) inline void NotifyDispatchDirectoryBlockVf(__ubuf__ int32_t* block,
                                                                              __ubuf__ int32_t* next_sum,
                                                                              __gm__ int32_t* directory, uint32_t begin,
                                                                              uint32_t count, uint32_t base,
                                                                              uint32_t core)
{
    for (uint32_t local = threadIdx.x * Cores + core; local < count; local += Cores * 128U) {
        uint32_t sum = base;
        for (uint32_t i = 0U; i < local; ++i) sum += uint32_t(block[i * 2U + 1U]);
        asc_stcg(directory + (begin + local) * 2U, int32_t(sum));
    }
    if (threadIdx.x == 0U) {
        uint32_t sum = base;
        for (uint32_t i = 0U; i < count; ++i) sum += uint32_t(block[i * 2U + 1U]);
        next_sum[0] = int32_t(sum);
    }
    asc_threadfence();
}

template <uint32_t Cores>
__simt_vf__ __launch_bounds__(128) inline void NotifyDispatchInitVf(__gm__ uint32_t* plan, uint32_t header_words,
                                                                    uint64_t mailbox_offset, uint32_t mailbox_words,
                                                                    uint32_t phase_words, uint32_t options,
                                                                    uint32_t core)
{
    const uint32_t worker = threadIdx.x * Cores + core;
    for (uint32_t i = worker; i < header_words; i += Cores * 128U) {
        uint32_t value = 0U;
        if (i == 4U) value = 3U;
        if (i == 5U) value = phase_words;
        if (i == 6U) value = uint32_t(mailbox_offset / 4U);
        if (i == 7U) value = 8U;
        if (i == 12U) value = options;
        asc_stcg(plan + i, value);
    }
    auto* mailbox = reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(plan) + mailbox_offset);
    for (uint32_t i = worker; i < mailbox_words; i += Cores * 128U) asc_stcg(mailbox + i, 0U);
    asc_threadfence();
}

template <uint32_t Cores>
__aicore__ inline void NotifyDispatchPrefix(const ascend_deepep::NotifyTiling& t, LocalTensor<int32_t> ub)
{
    const auto& p = t.dispatch_plan;
    const uint32_t tiles = (t.tokens + 15U) / 16U;
    GlobalTensor<int32_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(p.address + p.scratch_offset));
    auto* directory = reinterpret_cast<__gm__ int32_t*>(p.address + 64U);
    for (uint32_t peer = GetBlockIdx(); peer < t.world; peer += Cores) {
        for (uint32_t at = 0U; at < tiles; at += 128U) {
            const uint32_t n = tiles - at < 128U ? tiles - at : 128U;
            const DataCopyExtParams copy(1U, n * 16U, 0U, 0U, 0U);
            DataCopyPad(ub, gm[(peer * tiles + at) * 4U], copy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            asc_vf_call<NotifyDispatchPrefixBlockVf>(dim3(128), reinterpret_cast<__ubuf__ int32_t*>(ub.GetPhyAddr()),
                                                     reinterpret_cast<__ubuf__ int32_t*>(ub[512U].GetPhyAddr()), n,
                                                     at == 0U, at + n == tiles, directory, peer, t.world);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopyPad(gm[(peer * tiles + at) * 4U], ub, copy);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        }
    }
}

template <uint32_t Cores>
__aicore__ inline void NotifyDispatchDirectory(const ascend_deepep::NotifyTiling& t, LocalTensor<int32_t> ub)
{
    auto* directory = reinterpret_cast<__gm__ int32_t*>(t.dispatch_plan.address + 64U);
    GlobalTensor<int32_t> gm;
    gm.SetGlobalBuffer(directory);
    uint32_t base = 0U;
    for (uint32_t begin = 0U; begin < t.world * 4U; begin += 128U) {
        const uint32_t n = t.world * 4U - begin < 128U ? t.world * 4U - begin : 128U;
        DataCopyPad(ub, gm[begin * 2U], DataCopyExtParams(1U, n * 8U, 0U, 0U, 0U),
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        asc_vf_call<NotifyDispatchDirectoryBlockVf<Cores>>(
            dim3(128), reinterpret_cast<__ubuf__ int32_t*>(ub.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(ub[512U].GetPhyAddr()), directory, begin, n, base, GetBlockIdx());
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        base = uint32_t(ub.GetValue(512U));
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }
    if (base > t.tokens * t.topk) {
        aclshmemi_kernel_abort("Notify Dispatch plan exceeds route capacity");
        trap();
    }
}
