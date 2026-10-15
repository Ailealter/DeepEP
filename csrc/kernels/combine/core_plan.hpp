#pragma once
#include <cstdint>

namespace ascend_deepep {
// Physical cores are compact; completion slots keep the peer/lane ABI.
struct Ep8Owner {
    uint32_t peer, lane;
};
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
    inline constexpr uint32_t
    Ep8SourceBegin(uint32_t cores, bool final_window)
{
    return final_window ? 11U : cores / 2U;
}
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
    inline constexpr uint32_t
    Ep8FirstSourceToken(uint32_t begin, uint32_t core, uint32_t source_begin, uint32_t stride)
{
    return begin + (core - source_begin + stride - begin % stride) % stride;
}
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
    inline constexpr Ep8Owner
    Ep8ReturnOwner(uint32_t rank, uint32_t core)
{
    if (core < 7U) return {core < rank ? core : core + 1U, 0U};
    return {rank, core - 7U};
}
}  // namespace ascend_deepep
