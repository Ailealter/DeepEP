#pragma once
#include <cstdint>

namespace ascend_deepep::notify_diagnostics {
inline constexpr uint64_t kMagic = 0x4E544659;
inline constexpr uint32_t kVersion = 2U;
inline constexpr uint32_t kWords = 64U;
inline constexpr uint32_t kHeaderWords = 16U;
inline constexpr uint32_t kUbBytes = kWords * sizeof(uint64_t);
// Header words 13/14: diagnostic entry barrier begin/end; 15: barrier enabled.
// Each stamp starts this phase; the next stamp ends it. End is a sentinel.
enum Phase : uint32_t {
    Setup = 0U,
    ParseTopk = 1U,
    LocalWaitParseTopk = 2U,
    LocalPrefixSegments = 3U,
    LocalWaitLocalPrefixSegments = 4U,
    LocalPrefixTotals = 5U,
    LocalWaitLocalPrefixTotals = 6U,
    FinalizeTopk = 7U,
    PackCounts = 8U,
    LocalWaitFinalizeTopk = 9U,
    CountExchange = 10U,
    GlobalWaitCountExchange = 11U,
    ExpertTotals = 12U,
    LocalWaitExpertTotals = 13U,
    RankBases = 14U,
    LocalWaitRankBases = 15U,
    AddressStride = 16U,
    LocalWaitAddressStride = 17U,
    ResolveDst = 18U,
    ForwardCount = 19U,
    LocalWaitResolveDst = 20U,
    ForwardPrefix = 21U,
    LocalWaitForwardPrefix = 22U,
    ForwardWrite = 23U,
    LocalWaitForwardWrite = 24U,
    ForwardExchange = 25U,
    GlobalWaitForwardExchange = 26U,
    GatherCount = 27U,
    LocalWaitGatherCount = 28U,
    GatherSourcePrefix = 29U,
    LocalWaitGatherSourcePrefix = 30U,
    GatherPrefix = 31U,
    LocalWaitGatherPrefix = 32U,
    BuildForwardRows = 33U,
    ChunkRangesBackwardCount = 34U,
    LocalWaitBuildForwardRows = 35U,
    BackwardChunkPrefix = 36U,
    LocalWaitBackwardChunkPrefix = 37U,
    BackwardTaskPrefix = 38U,
    LocalWaitBackwardTaskPrefix = 39U,
    BackwardWrite = 40U,
    LocalWaitBackwardWrite = 41U,
    BackwardGatherExchange = 42U,
    GlobalWaitBackwardGatherExchange = 43U,
    CollectBackward = 44U,
    FinalizeOutputs = 45U,
    DispatchPlan = 46U,
    End = 47U,
};
static_assert(kHeaderWords + End + 1U <= kWords);
static_assert(kUbBytes % 32U == 0U);
}  // namespace ascend_deepep::notify_diagnostics
