// Copyright (c) 2026, Lu Lu

#pragma once
#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define DEEPEP_SCHEDULE_INLINE __aicore__ inline constexpr
#else
#define DEEPEP_SCHEDULE_INLINE inline constexpr
#endif

// Scheduling helpers shared by Notify and Combine.
// MoonEP's Legacy half-world mapping for EP<=64 and EP256, and Staged 4x4
// two-frame mapping for EP128. Keep pure arithmetic host-testable.
namespace first_hit_schedule {
DEEPEP_SCHEDULE_INLINE uint32_t Width(uint32_t world)
{
    return world < 16U ? world : 16U;
}
DEEPEP_SCHEDULE_INLINE uint32_t Index(uint32_t rank, uint32_t world)
{
    return world == 128U ? (rank / 8U % 4U) * 4U + rank % 4U
                         : (rank / (world / 2U)) * (Width(world) / 2U) + rank % (Width(world) / 2U);
}
DEEPEP_SCHEDULE_INLINE uint32_t Peer(uint32_t rank, uint32_t world, uint32_t group, uint32_t index,
                                     bool incoming = false)
{
    if (world == 128U) {
        const uint32_t toggle = group / 4U, subround = group % 4U;
        const uint32_t row = rank % 64U / 32U;
        const uint32_t side = rank % 8U / 4U;
        const uint32_t peer_side = side ^ (subround % 2U);
        const uint32_t peer_row = (row + subround / 2U) % 2U;
        const uint32_t frame = rank / 64U ^ (incoming ? peer_side : side) ^ toggle;
        return frame * 64U + (peer_row * 4U + index / 4U) * 8U + peer_side * 4U + index % 4U;
    }
    const uint32_t half = world / 2U, half_width = Width(world) / 2U;
    const uint32_t groups = world / Width(world), own = rank % half / half_width;
    const uint32_t peer_group = (own + (incoming ? groups - group : group)) % groups;
    return index / half_width * half + peer_group * half_width + index % half_width;
}
// Peer 的出向逆映射：expert 发送 backward 时必须传目的 gateway rank。
DEEPEP_SCHEDULE_INLINE uint32_t Group(uint32_t rank, uint32_t world, uint32_t peer)
{
    if (world == 128U) {
        const uint32_t side = rank % 8U / 4U, peer_side = peer % 8U / 4U;
        const uint32_t row = rank % 64U / 32U, peer_row = peer % 64U / 32U;
        const uint32_t toggle = rank / 64U ^ peer / 64U ^ side;
        return toggle * 4U + (row ^ peer_row) * 2U + (side ^ peer_side);
    }
    const uint32_t half = world / 2U, half_width = Width(world) / 2U;
    const uint32_t groups = world / Width(world), own = rank % half / half_width;
    return (peer % half / half_width + groups - own) % groups;
}
// Combine aggregates the existing topology order into groups of up to 64.
DEEPEP_SCHEDULE_INLINE uint32_t CombineGroup(uint32_t rank, uint32_t world, uint32_t peer)
{
    return (Group(rank, world, peer) * Width(world) + Index(peer, world)) / (world < 64U ? world : 64U);
}

// Shared by the host channel mask and the device scheduler. Cross peers only.
DEEPEP_SCHEDULE_INLINE uint32_t CombineOwner(uint32_t sender, uint32_t peer, uint32_t world, uint32_t owners = 16U,
                                             uint32_t chunk = 0U)
{
    const uint32_t order = Group(sender, world, peer) * Width(world) + Index(peer, world);
    uint32_t index = order;
    if (world == 128U) {
        // The sender's MESH occupies two four-rank intervals in the Staged order.
        const uint32_t begin = (sender % 8U / 4U) * 64U + (sender / 8U % 4U) * 4U;
        index -= 4U * uint32_t(order >= begin + 4U) + 4U * uint32_t(order >= begin + 20U);
    } else if (world >= 256U) {
        // In the half-world order the local MESH is one contiguous interval.
        const uint32_t begin = sender / (world / 2U) * 8U;
        index -= 8U * uint32_t(order >= begin + 8U);
    } else {
        for (uint32_t local = sender / 8U * 8U; local < sender / 8U * 8U + 8U; ++local) {
            if (Group(sender, world, local) * Width(world) + Index(local, world) < order) --index;
        }
    }
    index = index / 8U * 8U + (index % 8U + 8U - sender % 8U) % 8U;
    if (world == 64U && owners == 16U && index >= 48U) return index % 8U + chunk % 2U * 8U;
    return (index + (owners == 16U ? 8U : 0U)) % owners;
}

DEEPEP_SCHEDULE_INLINE uint32_t CombineQpMask(uint32_t rank, uint32_t peer, uint32_t world)
{
    if (rank == peer) return 0;
    uint32_t mask = 1U;  // Default operations and local MESH retain QP0.
    if (rank / 8U != peer / 8U) {
        for (uint32_t parity = 0; parity < 2; ++parity) {
            mask |= 3U << (2U * CombineOwner(rank, peer, world, 16U, parity));
            mask |= 3U << (2U * CombineOwner(peer, rank, world, 16U, parity));
        }
    }
    return mask;
}

}  // namespace first_hit_schedule
#undef DEEPEP_SCHEDULE_INLINE
