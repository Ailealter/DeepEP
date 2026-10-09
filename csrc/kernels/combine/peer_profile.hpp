#pragma once
#include <cstdint>

namespace ascend_deepep::combine_peers {
constexpr uint32_t kVersion = 1, kMagic = 0x43505250, kWords = 16;
enum Direction : uint32_t { Receive = 0, Send = 1 };
enum Field : uint32_t {
    Magic,
    Version,
    Sample,
    Kind,
    Rank,
    Core,
    Peer,
    Chunk,
    Lane,
    Qp,
    Begin,
    Observed,
    FlagsReady,
    SyncEnd,
    Polls,
    Flags
};
// Each of 64 pollers owns at most eight gateways at EP512, two flags each.
// EP64 parity windows can require four flags, still below this bound.
constexpr uint32_t kMaxWaits = 16, kWaitWords = 20;
constexpr uint32_t kSlot = 16, kExpected = 17;
constexpr uint32_t kUbBytes = (kWords + kMaxWaits * kWaitWords) * sizeof(uint64_t);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
    inline constexpr uint64_t
    Offset(uint32_t direction, uint32_t world, uint32_t chunks, uint32_t peer, uint32_t chunk, uint32_t lane)
{
    return (((uint64_t(direction) * world + peer) * chunks + chunk) * 2U + lane) * kWords;
}
static_assert(kWaitWords * sizeof(uint64_t) % 32U == 0U);
}  // namespace ascend_deepep::combine_peers
