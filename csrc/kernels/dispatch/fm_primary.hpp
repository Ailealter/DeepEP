// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// The caller owns one private FM QP0 and has acquired the live peer address.
// Keep the original selected-route order; selection, SO and source-ready are unchanged.
__simt_vf__ __launch_bounds__(32) inline void BuildFmPrimaryWqesVf(__ubuf__ uint64_t* wqes, __ubuf__ uint64_t* header,
                                                                   __ubuf__ int32_t* selected, __ubuf__ int32_t* rows,
                                                                   uint32_t first, uint32_t count, uint32_t chunk_start,
                                                                   uint32_t topk, uint32_t address_stride,
                                                                   uint32_t row_bytes, uint64_t source, uint64_t target,
                                                                   uint64_t cqe_mask)
{
    for (uint32_t i = threadIdx.x; i < count; i += 32U) {
        const uint32_t route = selected[first + i];
        auto* sqe = wqes + uint64_t(i) * 8U;
        const uint64_t terminal = uint64_t(((i + 1U) & 3U) == 0U) | uint64_t(i + 1U == count);
        sqe[0] = header[0] | terminal * cqe_mask;
        sqe[1] = header[1];
        sqe[2] = header[2];
        sqe[3] = header[3];
        sqe[4] = header[4];
        sqe[5] = target + uint64_t(uint32_t(rows[route - chunk_start]) % address_stride) * row_bytes;
        sqe[6] = row_bytes;
        sqe[7] = source + uint64_t(route / topk) * row_bytes;
    }
}

// One DB per batch, including SQ wrap. This helper is also used by the
// secondary builder. Only the terminal RO of each batch requests a CQE.
__aicore__ inline void PublishFmSimtWqes(__ubuf__ uint8_t* wqes, uint32_t count, __gm__ aclshmemi_udma_wq_ctx_t* ctx,
                                         __gm__ aclshmemi_udma_queue_state_t* state)
{
    const uint32_t head = state->sq_head, index = head % ctx->depth;
    const uint32_t first = count < ctx->depth - index ? count : ctx->depth - index;
    AscendC::LocalTensor<uint8_t> view;
    view.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes);
    aclshmemi_udma_copy_wqe_from_ub(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr + uint64_t(index) * ctx->wqe_size),
                                    view, first * ctx->wqe_size, EVENT_ID0);
    if (first != count) {
        view.address_.bufferAddr = reinterpret_cast<uint64_t>(wqes + first * ctx->wqe_size);
        aclshmemi_udma_copy_wqe_from_ub(reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr), view,
                                        (count - first) * ctx->wqe_size, EVENT_ID0);
    }
    aclshmemi_udma_post_send_update_info(head + count, ctx, state);
    ++state->cqe_cnt;
}

__aicore__ inline void IssueFmPrimaryRows(DeepepPrimaryState& p, uint32_t peer, __ubuf__ uint8_t* scratch,
                                          LocalTensor<int32_t> selected, LocalTensor<int32_t> rows,
                                          uint32_t chunk_start, uint32_t count)
{
    if (!count) return;
    auto& s = *p.secondary;
    auto* header = reinterpret_cast<__ubuf__ uint64_t*>(scratch);
    auto* wqes = reinterpret_cast<__ubuf__ uint64_t*>(scratch + 256U);
    static_assert(256U + 32U * 64U <= kUdmaWqeScratchBytes);
    if (!p.fm_context) {
        auto* info = aclshmemi_udma_qp_info_fetch();
        const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
        auto* ctx = aclshmemi_udma_get_qp_ctx(info, slot, 0U);
        if (!ctx || !ctx->state_addr || !ctx->buf_addr || ctx->depth != shm::UDMA_SQ_BASKBLK_CNT ||
            ctx->wqe_size != 64U) {
            aclshmemi_kernel_abort("FM primary SQ ABI mismatch");
            trap();
            return;
        }
        auto* target = PeerHiddenAddress(s.book, peer, s.book_words);
        if (!target) {
            aclshmemi_kernel_abort("FM primary missing live output");
            trap();
            return;
        }
        aclshmemi_udma_fill_sqe_base_ctx<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                         kUdmaDataRoNoCqeConfig>(
            reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(header), target,
            aclshmemi_udma_get_mem_info(info, slot, 0U), 0U);
        header[5] = reinterpret_cast<uint64_t>(target);
        p.fm_context = reinterpret_cast<uint64_t>(ctx);
        p.fm_queue = ctx->state_addr;
        p.fm_slot = slot;
    }
    auto* ctx = reinterpret_cast<__gm__ aclshmemi_udma_wq_ctx_t*>(p.fm_context);
    auto* queue = reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(p.fm_queue);
    constexpr uint32_t ro = aclshmemi_udma_build_flag<aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kUdmaDataRoNoCqeConfig>();
    constexpr uint32_t cqe = aclshmemi_udma_build_flag<aclshmemi_udma_opcode_t::UDMA_OP_WRITE, kUdmaDataRoCqeConfig>();
    for (uint32_t first = 0U; first < count; first += 32U) {
        const uint32_t n = count - first < 32U ? count - first : 32U;
        aclshmemi_udma_poll_before_aggregate_flush(queue, p.fm_slot, 0U, n);
        if (!aclshmemi_udma_sq_has_capacity(queue, n)) {
            aclshmemi_kernel_abort("FM primary SQ capacity exhausted");
            trap();
            return;
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        asc_vf_call<BuildFmPrimaryWqesVf>(
            dim3(32), wqes, header, reinterpret_cast<__ubuf__ int32_t*>(selected.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(rows.GetPhyAddr()), first, n, chunk_start, p.topk, p.address_stride,
            s.row_bytes, reinterpret_cast<uint64_t>(p.input), header[5], uint64_t(ro ^ cqe) << 16);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t at = 0U; at < n; at += 4U)
            PublishFmSimtWqes(reinterpret_cast<__ubuf__ uint8_t*>(wqes) + at * 64U, n - at < 4U ? n - at : 4U, ctx,
                              queue);
    }
}
