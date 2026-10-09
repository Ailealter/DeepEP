#pragma once
#include <cstdint>

namespace ascend_deepep::combine_chunks {
constexpr uint32_t kVersion = 2, kMagic = 0x43484b50, kWords = 36;
// One reusable, 288-byte UB record per AIV. GM slots are disjoint by core,
// producer group/chunk, local-return chunk, and final-reduce chunk. No per-peer
// tensor and no new symmetric heap storage or cross-core synchronization.
enum Kind : uint32_t { Gather = 1, Gateway = 2, Cross = 3, LocalReturn = 4, FinalReduce = 5 };
enum Field : uint32_t {
    Magic,
    Version,
    Sample,
    KindId,
    Core,
    Group,
    Chunk,
    Peer,
    Begin,
    WaitBegin,
    Ready,
    WorkBegin,
    WorkEnd,
    End,
    Tokens,
    Rows,
    DirectRows,
    ReducedRows,
    FirstSend0,
    LastSend0,
    FirstSend1,
    LastSend1,
    FirstDirect,
    LastDirect,
    FirstReduced,
    LastReduced,
    PayloadWqes,
    Sos,
    PeerMask,
    WindowEnd,
    Flags,
    Reserved,
    BarrierBegin,
    BarrierEnd,
    Lane0Rows,
    Lane1Rows
};
inline constexpr uint64_t RecordsPerCore(uint32_t groups, uint32_t chunks)
{
    return uint64_t(groups + 2U) * chunks;
}
}  // namespace ascend_deepep::combine_chunks
