// Copyright (c) 2026, Lu Lu

#pragma once

// A sender's final CQE covers its outgoing SQ, not this rank's incoming data.
// After observing that CQE, publish a separate generation word to every peer
// covered by the SQ. Never reuse intermediate ready or SQ-pressure SO sinks.
constexpr uint64_t kDispatchPollCyclesPerUs = 1000ULL;
constexpr uint64_t kDispatchCompletionTimeoutCycles = 10000000ULL * kDispatchPollCyclesPerUs;

__aicore__ inline void PauseDispatchCompletionPoll()
{
    const uint64_t begin = GetSystemCycle();
    do {
        Nop<64>();
    } while (GetSystemCycle() - begin < kDispatchPollCyclesPerUs);
}

__aicore__ inline void WaitDispatchGeneration(__gm__ uint64_t* flag, uint64_t generation, uint32_t rank, uint32_t peer,
                                              uint32_t lane)
{
    const uint64_t begin = GetSystemCycle();
    for (;;) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(flag));
        const uint64_t seen = *reinterpret_cast<__gm__ volatile uint64_t*>(flag);
        if (seen == generation) break;
        if (GetSystemCycle() - begin >= kDispatchCompletionTimeoutCycles) {
            aclshmemi_kernel_abort("Dispatch receive timeout rank=%u peer=%u lane=%u expected=%llu seen=%llu\n", rank,
                                   peer, lane, static_cast<unsigned long long>(generation),
                                   static_cast<unsigned long long>(seen));
            trap();
            return;
        }
        PauseDispatchCompletionPoll();
    }
    DataSyncBarrier<MemDsbT::DDR>();
}

__aicore__ inline __gm__ uint64_t* DispatchCompletionFlag(GM_ADDR base, uint32_t world, uint32_t source, uint32_t lane)
{
    // Lane-major slots let AIV7 load consecutive sources with one strided MTE
    // operation. Match the existing 512B packet-tail vector polling layout.
    return reinterpret_cast<__gm__ uint64_t*>(base + (uint64_t(lane) * world + source) * 512U + 504U);
}

__simt_vf__ __launch_bounds__(128) inline void PublishDispatchCompletionVf(__ubuf__ uint64_t* targets, uint32_t count,
                                                                           uint64_t generation)
{
    for (uint32_t i = threadIdx.x; i < count; i += 128U)
        asc_stcg(reinterpret_cast<__gm__ uint64_t*>(targets[i]), generation);
    asc_threadfence();
}

__aicore__ inline void PublishDispatchCompletionTargets(LocalTensor<uint64_t> targets, uint32_t count,
                                                        uint64_t generation, LocalTensor<uint64_t> payload)
{
    // All targets are mapped SHMEM control words, never exchanged ordinary GM addresses.
    if (count <= 4U) {
        payload.SetValue(0U, generation);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
        for (uint32_t i = 0U; i < count; ++i) {
            GlobalTensor<uint64_t> target;
            target.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(targets.GetValue(i)));
            DataCopyPad(target, payload, DataCopyExtParams(1, sizeof(uint64_t), 0, 0, 0));
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    } else {
        SetFlag<HardEvent::S_V>(EVENT_ID3);
        WaitFlag<HardEvent::S_V>(EVENT_ID3);
        asc_vf_call<PublishDispatchCompletionVf>(dim3(128), reinterpret_cast<__ubuf__ uint64_t*>(targets.GetPhyAddr()),
                                                 count, generation);
        SetFlag<HardEvent::V_S>(EVENT_ID3);
        WaitFlag<HardEvent::V_S>(EVENT_ID3);
    }
    DataSyncBarrier<MemDsbT::DDR>();
}

__aicore__ inline void WaitDispatchPeerEpoch(GM_ADDR workspace, const DispatchTiling& t, uint32_t peer)
{
    // Publishing the next descriptor occurs after the peer's previous kernel
    // has consumed its inbox. Gate new scale/weight writes, including empty
    // headers, on this existing handshake instead of overwriting a slow reader.
    auto* flag = reinterpret_cast<__gm__ uint64_t*>(workspace + t.peer_book) +
                 uint64_t(peer) * ascend_deepep::kPeerAddressRowWords + ascend_deepep::kPeerAddressGenerationWord;
    WaitDispatchGeneration(flag, t.generation, t.rank, peer, ascend_deepep::kDispatchCompletionLanes);
}
