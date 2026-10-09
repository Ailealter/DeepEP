// Copyright (c) 2026, Lu Lu

#pragma once

// Private FM QP0. Use the existing 8 KiB scratch: header [0,64),
// 32 prebuilt WQEs [256,2304). No payload staging or extra workspace.
constexpr uint32_t kFmBuildRows = 32U;
constexpr uint32_t kFmSimtWqeOffset = 256U;
static_assert(kFmSimtWqeOffset + kFmBuildRows * 64U <= kUdmaWqeScratchBytes);

// Each pair is (received hidden row, final peer row), NOT (input token, row).
// No per-WQE branch. Only the last RO of each DB batch requests a CQE.
__simt_vf__ __launch_bounds__(32) inline void DispatchBuildFmWqesVf(__ubuf__ uint64_t* wqes, __ubuf__ uint64_t* header,
                                                                    __ubuf__ int32_t* rows, uint32_t first,
                                                                    uint32_t count, uint32_t batch, uint32_t row_bytes,
                                                                    uint64_t source, uint64_t target, uint64_t cqe_mask)
{
    for (uint32_t i = threadIdx.x; i < count; i += 32U) {
        auto* sqe = wqes + uint64_t(i) * 8U;
        const uint32_t row = (first + i) * 2U;
        const uint64_t terminal = uint64_t(((i + 1U) & (batch - 1U)) == 0U) | uint64_t(i + 1U == count);
        sqe[0] = header[0] | (terminal * cqe_mask);
        sqe[1] = header[1];
        sqe[2] = header[2];
        sqe[3] = header[3];
        sqe[4] = header[4];
        sqe[5] = target + uint64_t(rows[row + 1U]) * row_bytes;
        sqe[6] = uint64_t(row_bytes);
        sqe[7] = source + uint64_t(rows[row]) * row_bytes;
    }
}

// Consume only row pairs whose source-ready has already been acquired.
__aicore__ inline void IssueFmRows(DeepepSecondaryState& s, uint32_t peer, __ubuf__ uint8_t* scratch,
                                   LocalTensor<uint32_t> rows, uint32_t first, uint32_t count)
{
    if (!count || count > 32U) {
        aclshmemi_kernel_abort("FM SIMT invalid batch size");
        trap();
        return;
    }
    auto* header = reinterpret_cast<__ubuf__ uint64_t*>(scratch);
    auto* wqes = reinterpret_cast<__ubuf__ uint64_t*>(scratch + kFmSimtWqeOffset);
    if (!s.fm_context) {
        EnsureDeepepPeerAddress(s, peer);
        auto* info = aclshmemi_udma_qp_info_fetch();
        const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
        auto* ctx = aclshmemi_udma_get_qp_ctx(info, slot, 0U);
        if (!ctx || !ctx->state_addr || !ctx->buf_addr || ctx->depth != shm::UDMA_SQ_BASKBLK_CNT ||
            ctx->wqe_size != 64U) {
            aclshmemi_kernel_abort("FM SIMT SQ ABI mismatch");
            trap();
            return;
        }
        auto* target = PeerHiddenAddress(s.book, peer, s.book_words);
        if (!target) {
            aclshmemi_kernel_abort("FM SIMT missing live output");
            trap();
            return;
        }
        aclshmemi_udma_fill_sqe_base_ctx<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                         kUdmaDataRoNoCqeConfig>(
            reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(header), target,
            aclshmemi_udma_get_mem_info(info, slot, 0U), 0U);
        // WQE word5 caches the live peer base for this invocation only.
        header[5] = reinterpret_cast<uint64_t>(target);
        s.fm_context = reinterpret_cast<uint64_t>(ctx);
        s.fm_queue = ctx->state_addr;
        s.fm_slot = slot;
    }
    auto* ctx = reinterpret_cast<__gm__ aclshmemi_udma_wq_ctx_t*>(s.fm_context);
    auto* state = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(s.fm_queue);
    aclshmemi_udma_poll_before_aggregate_flush(state, s.fm_slot, 0U, count);
    if (!aclshmemi_udma_sq_has_capacity(state, count)) {
        aclshmemi_kernel_abort("FM SIMT SQ capacity exhausted");
        trap();
        return;
    }
    constexpr uint32_t ro = aclshmemi_udma_build_flag<aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kUdmaDataRoNoCqeConfig>();
    constexpr uint32_t cqe = aclshmemi_udma_build_flag<aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kUdmaDataRoCqeConfig>();
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    asc_vf_call<DispatchBuildFmWqesVf>(dim3(32), wqes, header, reinterpret_cast<__ubuf__ int32_t*>(rows.GetPhyAddr()),
                                       first, count, 4U, s.row_bytes, reinterpret_cast<uint64_t>(s.hidden), header[5],
                                       uint64_t(ro ^ cqe) << 16);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (uint32_t at = 0U; at < count; at += 4U) {
        const uint32_t n = count - at < 4U ? count - at : 4U;
        PublishFmSimtWqes(reinterpret_cast<__ubuf__ uint8_t*>(wqes) + at * 64U, n, ctx, state);
    }
}

// One terminal SO on the same private QP0 orders all forwarded RO writes.
// Use an otherwise unused tail of the SHMEM sync area, NOT a framework VA
// and NOT a primary source-ready flag. The caller drains this QP; the wrapper's
// standalone exit barrier then joins all senders before exposing live outputs.
constexpr uint32_t kFmFenceOffset = 57344U;
static_assert(kFmFenceOffset + 8U * 512U <= 65536U);
__aicore__ inline void FinishFmRows(DeepepSecondaryState& s, uint32_t peer, __ubuf__ uint8_t* scratch)
{
    auto* flag = reinterpret_cast<__gm__ uint64_t*>(s.sync + uint64_t(s.world) * kRankSyncSlotBytes + kFmFenceOffset +
                                                    uint64_t(s.rank % 8U) * 512U);
    auto* value = GetPeerCompletionFlag(s.rank, GetGroupRankIndex(peer, s.world), kPeerCompletionValueSlot, s.sync);
    aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kUdmaSyncSoCqeConfig>(
        flag, value, reinterpret_cast<__ubuf__ uint64_t*>(scratch), 1U, peer, 0U, EVENT_ID0);
}
