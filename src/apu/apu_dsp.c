/*
 * MCPX APU DSP (GP/EP)
 *
 * Copyright (c) 2012 espes
 * Copyright (c) 2018-2019 Jannik Vogel
 * Copyright (c) 2019-2025 Matt Borgerson
 *
 * Ported from xemu's hw/xbox/mcpx/apu/dsp/gp_ep.c. The DSP56300 core it
 * drives lives in dsp/ and is GPL-2.0-or-later; see the NOTICE file for what
 * that means for the linked binary.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * ── What differs from xemu, and why ─────────────────────────────────────
 *
 * 1. No asserts on the scatter-gather path. In QEMU a title that programs a
 *    bad page table aborts the emulator; here it would kill the game the user
 *    is playing. Out-of-range transfers are dropped and named once.
 *
 * 2. The MMIO handlers take byte and word accesses, not just aligned dwords.
 *    Guest bulk copies into the APU window go through recomp_guest_copy,
 *    which walks them a byte at a time so the fault handler can decode each
 *    one -- so a dword of DSP program arrives here as four one-byte writes.
 *
 * 3. If neither processor is out of reset, the mixbins are passed straight to
 *    the monitor buffer, which is what this file did in full before the DSP
 *    was ported. Titles that never start the DSP keep the sound they had.
 */

#include "apu_state.h"
#include "fpconv.h"

#include "dsp/dsp.h"
#include "dsp/dsp_cpu.h"
#include "dsp/dsp_cpu_regs.h"
#include "dsp/dsp_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Upper bound of what apu_phys_off can return: the end of the contiguous
 * window, which sits above the 64 MB of RAM in the flat guest view. */
#define APU_ADDR_LIMIT 0x84000000u

static const int16_t ep_silence[256][2] = { { 0 } };

/* ── Options ─────────────────────────────────────────────────────────────
 *
 * RECOMP_APU_DSP=1 runs the emulated DSP56300 cores; otherwise the VP mixbins
 * go straight to the sound card and the cores stay out of the way.
 *
 * The passthrough is the default because it is the path that currently
 * produces sound. The voices decode real audio and mix into the front bins on
 * their own; what the DSP adds is effects and the Dolby encode. Def Jam's
 * encode processor still parks itself during its own init (it calls p:$04d0,
 * gets a non-zero status back and spins on a branch-to-self), so with the DSP
 * on, nothing reaches the sound card at all. Until that is understood, the
 * DSP is opt-in.
 */
static int dsp_opt(const char *name, int dflt)
{
    const char *v = getenv(name);
    if (!v || !*v)
        return dflt;
    return atoi(v) != 0;
}

static int dsp_enabled(void)
{
    static int cached = -1;
    if (cached < 0)
        cached = dsp_opt("RECOMP_APU_DSP", 0);
    return cached;
}

static int dsp_traced(void)
{
    static int cached = -1;
    if (cached < 0)
        cached = dsp_opt("RECOMP_APU_TRACE", 0);
    return cached;
}

/* ── DSP command doorbell acknowledgement ────────────────────────────────
 *
 * Predates the DSP port: RECOMP_APU_DSP_ACK=<addr>[,<addr>...] clears guest
 * dwords once per APU frame, which is what "the command completed" looks like
 * to a title spinning on a DSP handshake. It computes nothing, so anything
 * whose *result* the title reads back is still wrong -- with a real DSP behind
 * it now, this should only be needed for a title whose command block we have
 * not tracked down. Off unless the variable is set.
 */
#define APU_DSP_ACK_MAX 8
static uint32_t s_dsp_ack[APU_DSP_ACK_MAX];
static int s_dsp_ack_count = -1;

static void dsp_ack_init(void)
{
    const char *spec = getenv("RECOMP_APU_DSP_ACK");
    char buf[256], *p, *end;

    s_dsp_ack_count = 0;
    if (!spec || !*spec)
        return;
    strncpy(buf, spec, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;
    for (p = buf; *p && s_dsp_ack_count < APU_DSP_ACK_MAX; ) {
        unsigned long v = strtoul(p, &end, 0);
        if (end == p)
            break;
        if (v)
            s_dsp_ack[s_dsp_ack_count++] = (uint32_t)v;
        p = (*end == ',') ? end + 1 : end;
    }
    if (s_dsp_ack_count)
        fprintf(stderr, "[APU] DSP doorbell ack: %d address(es), first 0x%08X\n",
                s_dsp_ack_count, s_dsp_ack[0]);
}

static void dsp_ack_frame(MCPXAPUState *d)
{
    int i;

    if (s_dsp_ack_count < 0)
        dsp_ack_init();
    if (!d->ram_ptr)
        return;
    for (i = 0; i < s_dsp_ack_count; i++) {
        uint32_t *slot = (uint32_t *)(d->ram_ptr + s_dsp_ack[i]);
        if (*slot) {
            static int shown[APU_DSP_ACK_MAX];
            if (shown[i]++ < 3)
                fprintf(stderr, "[APU] DSP doorbell 0x%08X: command 0x%08X"
                                " acknowledged\n", s_dsp_ack[i], *slot);
            *slot = 0;
        }
    }
}

/* ── Scatter-gather ──────────────────────────────────────────────────────
 *
 * DSP scratch and frame memory are page tables in guest RAM, the same shape
 * the voices use: entry N of the table at sge_base holds the physical address
 * of page N, eight bytes per entry.
 */
/* A second's worth of DSP activity, printed as one line under the trace. Per
 * event printing is useless here: these run 1504 times a second. */
static struct {
    unsigned gp_overrun, ep_overrun;
    unsigned gp_pc, ep_pc;
    unsigned gp_fifo_in, gp_fifo_out;
    unsigned ep_fifo_in, ep_fifo_out;
    unsigned gp_scratch, ep_scratch;
    unsigned ep_sink;
    unsigned clipped;
    float loudest;
} g_dsp_stat;

static void sge_complain(const char *what, uint32_t addr, size_t len)
{
    static int shown;
    if (shown++ < 8)
        fprintf(stderr, "[APU] DSP %s out of range: addr 0x%X len 0x%X"
                        " -- transfer dropped\n",
                what, addr, (unsigned)len);
}

static void scatter_gather_rw(MCPXAPUState *d, hwaddr sge_base,
                              unsigned int max_sge, uint8_t *ptr, uint32_t addr,
                              size_t len, bool dir)
{
    unsigned int page_entry = addr / TARGET_PAGE_SIZE;
    unsigned int offset_in_page = addr % TARGET_PAGE_SIZE;
    unsigned int bytes_to_copy = TARGET_PAGE_SIZE - offset_in_page;

    if (!d->ram_ptr)
        return;

    while (len > 0) {
        if (page_entry > max_sge) {
            sge_complain("page entry", addr, len);
            return;
        }

        uint32_t prd_address = ldl_le_phys(address_space_memory,
                                           sge_base + page_entry * 8 + 0);

        /* apu_phys_off, not a mask: these pages usually come from the
         * contiguous window, which lives above RAM in the flat view. */
        hwaddr paddr = apu_phys_off(prd_address + offset_in_page);

        if (bytes_to_copy > len)
            bytes_to_copy = (unsigned int)len;

        if (paddr + bytes_to_copy > APU_ADDR_LIMIT) {
            sge_complain("page address", prd_address, bytes_to_copy);
            return;
        }

        if (dir) {
            /* Where DSP output actually lands, so it can be matched against
             * where the voices read. Rate-limited: this is a hot path. */
            if (dsp_traced()) {
                static int shown;
                static uint32_t last_dest;
                uint32_t dest = prd_address + offset_in_page;
                if (shown < 12 && dest != last_dest) {
                    last_dest = dest;
                    shown++;
                    fprintf(stderr, "[APU] DSP writes %u bytes to 0x%08X"
                                    " (scratch offset 0x%X)\n",
                            bytes_to_copy, dest, addr);
                }
            }
            memcpy(&d->ram_ptr[paddr], ptr, bytes_to_copy);
        } else {
            memcpy(ptr, &d->ram_ptr[paddr], bytes_to_copy);
        }

        ptr += bytes_to_copy;
        len -= bytes_to_copy;

        /* After the first iteration, we are page aligned */
        page_entry += 1;
        bytes_to_copy = TARGET_PAGE_SIZE;
        offset_in_page = 0;
    }
}

static void gp_scratch_rw(void *opaque, uint8_t *ptr, uint32_t addr, size_t len,
                          bool dir)
{
    MCPXAPUState *d = opaque;
    g_dsp_stat.gp_scratch++;
    scatter_gather_rw(d, d->regs[NV_PAPU_GPSADDR], d->regs[NV_PAPU_GPSMAXSGE],
                      ptr, addr, len, dir);
}

static void ep_scratch_rw(void *opaque, uint8_t *ptr, uint32_t addr, size_t len,
                          bool dir)
{
    MCPXAPUState *d = opaque;
    g_dsp_stat.ep_scratch++;
    scatter_gather_rw(d, d->regs[NV_PAPU_EPSADDR], d->regs[NV_PAPU_EPSMAXSGE],
                      ptr, addr, len, dir);
}

static uint32_t circular_scatter_gather_rw(MCPXAPUState *d, hwaddr sge_base,
                                           unsigned int max_sge, uint8_t *ptr,
                                           uint32_t base, uint32_t end,
                                           uint32_t cur, size_t len, bool dir)
{
    if (end <= base)
        return cur;

    while (len > 0) {
        unsigned int bytes_to_copy = end - cur;

        if (bytes_to_copy > len)
            bytes_to_copy = (unsigned int)len;

        if (cur < base || cur + bytes_to_copy > end) {
            sge_complain("fifo cursor", cur, bytes_to_copy);
            return cur;
        }

        scatter_gather_rw(d, sge_base, max_sge, ptr, cur, bytes_to_copy, dir);

        ptr += bytes_to_copy;
        len -= bytes_to_copy;

        /* After the first iteration we might have to wrap */
        cur += bytes_to_copy;
        if (cur >= end)
            cur = base;
    }

    return cur;
}

static void gp_fifo_rw(void *opaque, uint8_t *ptr, unsigned int index,
                       size_t len, bool dir)
{
    MCPXAPUState *d = opaque;
    uint32_t base;
    uint32_t end;
    hwaddr cur_reg;

    if (dir) {
        if (index >= GP_OUTPUT_FIFO_COUNT)
            return;
        base = GET_MASK(d->regs[NV_PAPU_GPOFBASE0 + 0x10 * index],
                        NV_PAPU_GPOFBASE0_VALUE);
        end = GET_MASK(d->regs[NV_PAPU_GPOFEND0 + 0x10 * index],
                       NV_PAPU_GPOFEND0_VALUE);
        cur_reg = NV_PAPU_GPOFCUR0 + 0x10 * index;
    } else {
        if (index >= GP_INPUT_FIFO_COUNT)
            return;
        base = GET_MASK(d->regs[NV_PAPU_GPIFBASE0 + 0x10 * index],
                        NV_PAPU_GPOFBASE0_VALUE);
        end = GET_MASK(d->regs[NV_PAPU_GPIFEND0 + 0x10 * index],
                       NV_PAPU_GPOFEND0_VALUE);
        cur_reg = NV_PAPU_GPIFCUR0 + 0x10 * index;
    }

    uint32_t cur = GET_MASK(d->regs[cur_reg], NV_PAPU_GPOFCUR0_VALUE);

    if (dir)
        g_dsp_stat.gp_fifo_out++;
    else
        g_dsp_stat.gp_fifo_in++;

    /* DSP hangs if current >= end; but forces current >= base */
    if (cur >= end)
        return;
    if (cur < base)
        cur = base;

    cur = circular_scatter_gather_rw(d,
        d->regs[NV_PAPU_GPFADDR], d->regs[NV_PAPU_GPFMAXSGE],
        ptr, base, end, cur, len, dir);

    SET_MASK(d->regs[cur_reg], NV_PAPU_GPOFCUR0_VALUE, cur);
}

static bool ep_sink_samples(MCPXAPUState *d, uint8_t *ptr, size_t len)
{
    if (d->monitor.point == MCPX_APU_DEBUG_MON_AC97) {
        return false;
    } else if ((d->monitor.point == MCPX_APU_DEBUG_MON_EP) ||
               (d->monitor.point == MCPX_APU_DEBUG_MON_GP_OR_EP)) {
        if (len > sizeof(d->monitor.frame_buf))
            len = sizeof(d->monitor.frame_buf);
        memcpy(d->monitor.frame_buf, ptr, len);
    }

    return true;
}

static void ep_fifo_rw(void *opaque, uint8_t *ptr, unsigned int index,
                       size_t len, bool dir)
{
    MCPXAPUState *d = opaque;
    uint32_t base;
    uint32_t end;
    hwaddr cur_reg;

    if (dir) {
        if (index >= EP_OUTPUT_FIFO_COUNT)
            return;
        base = GET_MASK(d->regs[NV_PAPU_EPOFBASE0 + 0x10 * index],
                        NV_PAPU_GPOFBASE0_VALUE);
        end = GET_MASK(d->regs[NV_PAPU_EPOFEND0 + 0x10 * index],
                       NV_PAPU_GPOFEND0_VALUE);
        cur_reg = NV_PAPU_EPOFCUR0 + 0x10 * index;
    } else {
        if (index >= EP_INPUT_FIFO_COUNT)
            return;
        base = GET_MASK(d->regs[NV_PAPU_EPIFBASE0 + 0x10 * index],
                        NV_PAPU_GPOFBASE0_VALUE);
        end = GET_MASK(d->regs[NV_PAPU_EPIFEND0 + 0x10 * index],
                       NV_PAPU_GPOFEND0_VALUE);
        cur_reg = NV_PAPU_EPIFCUR0 + 0x10 * index;
    }

    uint32_t cur = GET_MASK(d->regs[cur_reg], NV_PAPU_GPOFCUR0_VALUE);

    if (dir)
        g_dsp_stat.ep_fifo_out++;
    else
        g_dsp_stat.ep_fifo_in++;

    /* Output FIFO 0 is the encoded frame: that is the sound card's copy. */
    if (dir && index == 0) {
        if (dsp_traced()) {
            static int shown;
            static uint32_t next_ms;
            uint32_t now = (uint32_t)qemu_clock_get_ms(QEMU_CLOCK_REALTIME);
            if (now >= next_ms) {
                const int16_t *s = (const int16_t *)ptr;
                int n = (int)(len / 2), peak = 0;
                for (int i = 0; i < n; i++) {
                    int a = s[i] < 0 ? -s[i] : s[i];
                    if (a > peak)
                        peak = a;
                }
                next_ms = now + 1000;
                fprintf(stderr, "[APU] EP output FIFO: %u bytes, peak %d"
                                " (sink #%d)\n",
                        (unsigned)len, peak, ++shown);
            }
        }
        g_dsp_stat.ep_sink++;
        if (ep_sink_samples(d, ptr, len)) {
            /* Since we are sinking, push silence out */
            if (len <= sizeof(ep_silence))
                ptr = (uint8_t *)ep_silence;
        }
    }

    /* DSP hangs if current >= end; but forces current >= base */
    if (cur >= end && end > base)
        cur = base + (cur % (end - base));
    if (cur < base)
        cur = base;

    cur = circular_scatter_gather_rw(d,
        d->regs[NV_PAPU_EPFADDR], d->regs[NV_PAPU_EPFMAXSGE],
        ptr, base, end, cur, len, dir);

    SET_MASK(d->regs[cur_reg], NV_PAPU_GPOFCUR0_VALUE, cur);
}

/* ── Reset and bootstrap ─────────────────────────────────────────────────
 *
 * Holding either reset line low resets the core; releasing both runs the
 * bootstrap, which is what pulls the title's DSP program out of scratch
 * memory and into program RAM. This is how Def Jam uploads its program --
 * not by writing PMEM through MMIO.
 */
static void proc_rst_write(MCPXAPUState *d, DSPState *dsp, const char *who,
                           uint32_t oldval, uint32_t val)
{
    if (!dsp)
        return;

    if (!(val & NV_PAPU_GPRST_GPRST) || !(val & NV_PAPU_GPRST_GPDSPRST)) {
        dsp_reset(dsp);
        if (dsp_traced())
            fprintf(stderr, "[APU] %s DSP reset (RST=0x%X)\n", who, val);
    } else if (
        (!(oldval & NV_PAPU_GPRST_GPRST) || !(oldval & NV_PAPU_GPRST_GPDSPRST))
        && ((val & NV_PAPU_GPRST_GPRST) && (val & NV_PAPU_GPRST_GPDSPRST))) {
        dsp_bootstrap(dsp);
        if (dsp_traced()) {
            int words = 0;
            for (int i = 0; i < 0x800; i++)
                if (dsp->core.pram[i] != 0 && dsp->core.pram[i] != 0x00ACACAC)
                    words++;
            fprintf(stderr, "[APU] %s DSP bootstrapped: %d words of program"
                            " loaded, pram[0..3]=%06X %06X %06X %06X\n",
                    who, words, dsp->core.pram[0], dsp->core.pram[1],
                    dsp->core.pram[2], dsp->core.pram[3]);

            /* Where the bootstrap read from: the scratch page table, and what
             * the first pages actually resolve to. A program of zeros means
             * either the table is not filled in yet or it points elsewhere. */
            hwaddr sge = who[0] == 'G' ? d->regs[NV_PAPU_GPSADDR]
                                       : d->regs[NV_PAPU_EPSADDR];
            unsigned max = who[0] == 'G' ? d->regs[NV_PAPU_GPSMAXSGE]
                                         : d->regs[NV_PAPU_EPSMAXSGE];
            fprintf(stderr, "[APU] %s scratch: SGE table 0x%08X, max %u,"
                            " pages:", who, (uint32_t)sge, max);
            for (int i = 0; i < 4; i++) {
                uint32_t prd = ldl_le_phys(address_space_memory, sge + i * 8);
                fprintf(stderr, " 0x%08X", prd);
            }
            fprintf(stderr, "\n");
        }
    }
}

/* ── GP and EP register windows ──────────────────────────────────────────
 *
 * Sub-dword accesses are read-modify-write on the containing dword, because
 * that is how a guest memcpy into this window reaches us.
 */
/* MSVC has no case ranges, so the windows are spelled out. Each is a
 * count of 32-bit words at a base offset in the processor's register
 * space. */
#define IN_WINDOW(a, base, words) \
    ((a) >= (base) && (a) < (base) + (words) * 4u)

static uint32_t dsp_space_read(DSPState *dsp, char space, uint32_t word)
{
    return dsp ? dsp_read_memory(dsp, space, word) : 0;
}

static void dsp_space_write(DSPState *dsp, char space, uint32_t word,
                            uint32_t val, unsigned int size, unsigned int shift)
{
    if (!dsp)
        return;
    if (size < 4) {
        uint32_t mask = (size == 1 ? 0xFFu : 0xFFFFu) << shift;
        uint32_t cur = dsp_read_memory(dsp, space, word);
        val = (cur & ~mask) | ((val << shift) & mask);
    }
    dsp_write_memory(dsp, space, word, val);
}

static void reg_write(uint32_t *regs, hwaddr addr, uint64_t val,
                      unsigned int size, unsigned int shift)
{
    if (size < 4) {
        uint32_t mask = (size == 1 ? 0xFFu : 0xFFFFu) << shift;
        regs[addr] = (regs[addr] & ~mask) | (((uint32_t)val << shift) & mask);
    } else {
        regs[addr] = (uint32_t)val;
    }
}

uint64_t mcpx_apu_gp_read(MCPXAPUState *d, hwaddr addr, unsigned int size)
{
    unsigned int shift = (addr & 3) * 8;
    hwaddr word_addr = addr & ~(hwaddr)3;
    uint64_t r;

    if (IN_WINDOW(word_addr, NV_PAPU_GPXMEM, 0x1000)) {
        r = dsp_space_read(d->gp.dsp, 'X', (word_addr - NV_PAPU_GPXMEM) / 4);
    } else if (IN_WINDOW(word_addr, NV_PAPU_GPMIXBUF, 0x400)) {
        r = dsp_space_read(d->gp.dsp, 'X', GP_DSP_MIXBUF_BASE +
                           (word_addr - NV_PAPU_GPMIXBUF) / 4);
    } else if (IN_WINDOW(word_addr, NV_PAPU_GPYMEM, 0x800)) {
        r = dsp_space_read(d->gp.dsp, 'Y', (word_addr - NV_PAPU_GPYMEM) / 4);
    } else if (IN_WINDOW(word_addr, NV_PAPU_GPPMEM, 0x1000)) {
        r = dsp_space_read(d->gp.dsp, 'P', (word_addr - NV_PAPU_GPPMEM) / 4);
    } else {
        r = d->gp.regs[word_addr];
    }

    return size < 4 ? (r >> shift) & (size == 1 ? 0xFFu : 0xFFFFu) : r;
}

void mcpx_apu_gp_write(MCPXAPUState *d, hwaddr addr, uint64_t val,
                       unsigned int size)
{
    unsigned int shift = (addr & 3) * 8;
    hwaddr word_addr = addr & ~(hwaddr)3;

    if (IN_WINDOW(word_addr, NV_PAPU_GPXMEM, 0x1000)) {
        dsp_space_write(d->gp.dsp, 'X', (word_addr - NV_PAPU_GPXMEM) / 4,
                        (uint32_t)val, size, shift);
    } else if (IN_WINDOW(word_addr, NV_PAPU_GPMIXBUF, 0x400)) {
        dsp_space_write(d->gp.dsp, 'X', GP_DSP_MIXBUF_BASE +
                        (word_addr - NV_PAPU_GPMIXBUF) / 4,
                        (uint32_t)val, size, shift);
    } else if (IN_WINDOW(word_addr, NV_PAPU_GPYMEM, 0x800)) {
        dsp_space_write(d->gp.dsp, 'Y', (word_addr - NV_PAPU_GPYMEM) / 4,
                        (uint32_t)val, size, shift);
    } else if (IN_WINDOW(word_addr, NV_PAPU_GPPMEM, 0x1000)) {
        dsp_space_write(d->gp.dsp, 'P', (word_addr - NV_PAPU_GPPMEM) / 4,
                        (uint32_t)val, size, shift);
    } else if (word_addr == NV_PAPU_GPRST) {
        uint32_t old = d->gp.regs[NV_PAPU_GPRST];
        reg_write(d->gp.regs, NV_PAPU_GPRST, val, size, shift);
        proc_rst_write(d, d->gp.dsp, "GP", old, d->gp.regs[NV_PAPU_GPRST]);
    } else {
        reg_write(d->gp.regs, word_addr, val, size, shift);
    }
}

uint64_t mcpx_apu_ep_read(MCPXAPUState *d, hwaddr addr, unsigned int size)
{
    unsigned int shift = (addr & 3) * 8;
    hwaddr word_addr = addr & ~(hwaddr)3;
    uint64_t r;

    if (IN_WINDOW(word_addr, NV_PAPU_EPXMEM, 0xC00)) {
        r = dsp_space_read(d->ep.dsp, 'X', (word_addr - NV_PAPU_EPXMEM) / 4);
    } else if (IN_WINDOW(word_addr, NV_PAPU_EPYMEM, 0x100)) {
        r = dsp_space_read(d->ep.dsp, 'Y', (word_addr - NV_PAPU_EPYMEM) / 4);
    } else if (IN_WINDOW(word_addr, NV_PAPU_EPPMEM, 0x1000)) {
        r = dsp_space_read(d->ep.dsp, 'P', (word_addr - NV_PAPU_EPPMEM) / 4);
    } else {
        r = d->ep.regs[word_addr];
    }

    return size < 4 ? (r >> shift) & (size == 1 ? 0xFFu : 0xFFFFu) : r;
}

void mcpx_apu_ep_write(MCPXAPUState *d, hwaddr addr, uint64_t val,
                       unsigned int size)
{
    unsigned int shift = (addr & 3) * 8;
    hwaddr word_addr = addr & ~(hwaddr)3;

    if (IN_WINDOW(word_addr, NV_PAPU_EPXMEM, 0xC00)) {
        dsp_space_write(d->ep.dsp, 'X', (word_addr - NV_PAPU_EPXMEM) / 4,
                        (uint32_t)val, size, shift);
    } else if (IN_WINDOW(word_addr, NV_PAPU_EPYMEM, 0x100)) {
        dsp_space_write(d->ep.dsp, 'Y', (word_addr - NV_PAPU_EPYMEM) / 4,
                        (uint32_t)val, size, shift);
    } else if (IN_WINDOW(word_addr, NV_PAPU_EPPMEM, 0x1000)) {
        dsp_space_write(d->ep.dsp, 'P', (word_addr - NV_PAPU_EPPMEM) / 4,
                        (uint32_t)val, size, shift);
    } else if (word_addr == NV_PAPU_EPRST) {
        uint32_t old = d->ep.regs[NV_PAPU_EPRST];
        reg_write(d->ep.regs, NV_PAPU_EPRST, val, size, shift);
        proc_rst_write(d, d->ep.dsp, "EP", old, d->ep.regs[NV_PAPU_EPRST]);
        d->ep_frame_div = 0; /* FIXME: Still unsure about frame sync */
    } else {
        reg_write(d->ep.regs, word_addr, val, size, shift);
    }
}

/* ── Per-frame work ──────────────────────────────────────────────────────*/

static bool proc_running(uint32_t rst)
{
    return (rst & NV_PAPU_GPRST_GPRST) && (rst & NV_PAPU_GPRST_GPDSPRST);
}

/* Run one processor until its program parks itself.
 *
 * xemu spins here until the core goes idle. A frame is 32 samples at 48 kHz,
 * so this thread gets through 1504 of them a second and a program that never
 * idles takes the audio thread down with it -- which is what a half-wired DSP
 * does. The budget is generous next to what the title's program actually
 * needs, and overrunning it costs one frame, not the run.
 */
/* The budget has to be generous: the encode processor works on 256 samples
 * at a time, which is 5.3 ms of audio, and the hardware DSP runs at roughly
 * 100 MHz -- so hundreds of thousands of cycles are a normal frame for it,
 * not a runaway. A budget below that starves the program in the middle of a
 * filter loop and it never reaches its output stage. This exists only so a
 * program that truly never parks itself cannot take the audio thread with
 * it. RECOMP_APU_DSP_BUDGET overrides it. */
#define DSP_FRAME_CYCLE_BUDGET 1000000

static int dsp_budget(void)
{
    static int cached;
    if (!cached) {
        const char *v = getenv("RECOMP_APU_DSP_BUDGET");
        cached = (v && *v) ? atoi(v) : DSP_FRAME_CYCLE_BUDGET;
        if (cached < 1000)
            cached = DSP_FRAME_CYCLE_BUDGET;
    }
    return cached;
}

static void dsp_run_frame(DSPState *dsp, bool realtime, const char *who)
{
    int spent = 0;
    int budget = dsp_budget();

    do {
        dsp_run(dsp, 1000);
        spent += 1000;
    } while (!dsp->core.is_idle && realtime && spent < budget);

    if (!dsp->core.is_idle && realtime) {
        if (who[0] == 'G')
            g_dsp_stat.gp_overrun++, g_dsp_stat.gp_pc = dsp->core.pc;
        else
            g_dsp_stat.ep_overrun++, g_dsp_stat.ep_pc = dsp->core.pc;

        /* One dump per distinct address, not one per processor: the first
         * overrun is rarely the one that settles in. */
        if (dsp_traced()) {
            static uint32_t seen[8];
            static int seen_n;
            int known = 0;
            for (int i = 0; i < seen_n; i++)
                if (seen[i] == dsp->core.pc)
                    known = 1;
            if (!known && seen_n < 8) {
                seen[seen_n++] = dsp->core.pc;
                uint32_t pc = dsp->core.pc;
                fprintf(stderr, "[APU] %s DSP stuck at p:%04X, around it:\n",
                        who, pc);
                dsp56k_disasm_range(&dsp->core, pc > 6 ? pc - 6 : 0, pc + 6);
                fprintf(stderr, "      SR=%06X OMR=%06X A=%06X B=%06X"
                                " X0=%06X Y0=%06X\n",
                        dsp->core.registers[DSP_REG_SR],
                        dsp->core.registers[DSP_REG_OMR],
                        dsp->core.registers[DSP_REG_A1],
                        dsp->core.registers[DSP_REG_B1],
                        dsp->core.registers[DSP_REG_X0],
                        dsp->core.registers[DSP_REG_Y0]);
                fprintf(stderr, "      R0-R3 = %06X %06X %06X %06X,"
                                " DMA control %08X config %08X\n",
                        dsp->core.registers[DSP_REG_R0],
                        dsp->core.registers[DSP_REG_R0 + 1],
                        dsp->core.registers[DSP_REG_R0 + 2],
                        dsp->core.registers[DSP_REG_R0 + 3],
                        dsp->dma.control, dsp->dma.configuration);
                fflush(stderr);
            }
        }
    }
}

/* What this file did before the DSP was ported: mixbin 0 and 1 (front left
 * and right) straight into the monitor buffer. Still the right answer for a
 * title that leaves the DSP in reset, and for RECOMP_APU_DSP=0. */
static void dsp_passthrough(MCPXAPUState *d,
                            float mixbins[NUM_MIXBINS][NUM_SAMPLES_PER_FRAME])
{
    int off = (d->ep_frame_div % 8) * NUM_SAMPLES_PER_FRAME;

    if (d->monitor.point == MCPX_APU_DEBUG_MON_VP)
        return;

    for (int i = 0; i < NUM_SAMPLES_PER_FRAME; i++) {
        float left = mixbins[0][i];
        float right = mixbins[1][i];
        /* Clipping is worth counting, not just clamping: a mix that arrives
         * above full scale comes out as crackle, and the number says whether
         * that is what is being heard. */
        if (left > 1.0f || left < -1.0f || right > 1.0f || right < -1.0f) {
            g_dsp_stat.clipped++;
            float mag = left < 0 ? -left : left;
            float magr = right < 0 ? -right : right;
            if (magr > mag)
                mag = magr;
            if (mag > g_dsp_stat.loudest)
                g_dsp_stat.loudest = mag;
        }
        if (left > 1.0f) left = 1.0f;
        if (left < -1.0f) left = -1.0f;
        if (right > 1.0f) right = 1.0f;
        if (right < -1.0f) right = -1.0f;
        d->monitor.frame_buf[off + i][0] = (int16_t)(left * 32767.0f);
        d->monitor.frame_buf[off + i][1] = (int16_t)(right * 32767.0f);
    }
}

/* The scratch page table is the bootstrap's source. If it reads as zeros at
 * reset time but fills in later, the program we loaded is empty and the real
 * one never arrives -- so watch it for the first few seconds. */
/* One-shot: for each scatter-gather table the title programmed, show the first
 * entries read both ways -- as a contiguous-window address (what apu_phys_off
 * does now) and as a 64 MB-masked RAM address (what it used to do). Whichever
 * yields plausible page addresses is the right reading for that table. */
static void dsp_dump_tables(MCPXAPUState *d)
{
    static const struct { const char *name; uint32_t reg; } tables[] = {
        { "VPSGEADDR", NV_PAPU_VPSGEADDR },
        { "VPVADDR",   NV_PAPU_VPVADDR },
        { "GPSADDR",   NV_PAPU_GPSADDR },
        { "EPSADDR",   NV_PAPU_EPSADDR },
    };

    for (unsigned t = 0; t < sizeof tables / sizeof tables[0]; t++) {
        uint32_t base = d->regs[tables[t].reg];
        fprintf(stderr, "[APU] %s = 0x%08X\n", tables[t].name, base);
        if (!base)
            continue;
        fprintf(stderr, "        contiguous:");
        for (int i = 0; i < 4; i++)
            fprintf(stderr, " %08X",
                    *(uint32_t *)(d->ram_ptr + apu_phys_off(base + i * 8)));
        fprintf(stderr, "\n        masked RAM:");
        for (int i = 0; i < 4; i++)
            fprintf(stderr, " %08X",
                    *(uint32_t *)(d->ram_ptr + ((base + i * 8) & 0x03FFFFFF)));
        fprintf(stderr, "\n");
    }
}

/* RECOMP_APU_DSP_DISASM=GP:0100-0180,EP:04d0-0520 prints those spans of
 * program memory once the title's program is loaded. There is no other way to
 * read the code a DSP is running: it arrives as data at run time. */
static void dsp_disasm_option(MCPXAPUState *d)
{
    const char *spec = getenv("RECOMP_APU_DSP_DISASM");
    char buf[256], *p;

    if (!spec || !*spec)
        return;
    strncpy(buf, spec, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;

    for (p = strtok(buf, ","); p; p = strtok(NULL, ",")) {
        DSPState *dsp;
        char *range = strchr(p, ':');
        unsigned from, to;

        if (!range)
            continue;
        *range++ = 0;
        dsp = (p[0] == 'G' || p[0] == 'g') ? d->gp.dsp : d->ep.dsp;
        if (!dsp || sscanf(range, "%x-%x", &from, &to) != 2)
            continue;

        fprintf(stderr, "[APU] %s DSP p:%04X-%04X:\n", p, from, to);
        dsp56k_disasm_range(&dsp->core, from, to);
        fflush(stderr);
    }
}

static void dsp_watch_scratch(MCPXAPUState *d)
{
    static int frames;
    static uint32_t last_gp, last_ep;

    if (!dsp_traced())
        return;
    frames++;

    if (frames == 3000 && d->ram_ptr) {
        dsp_dump_tables(d);
        dsp_disasm_option(d);
    }

    if (frames % 1500 == 0) {
        fprintf(stderr, "[APU] DSP/s: GP scratch %u fifo in %u out %u,"
                        " EP scratch %u fifo in %u out %u, sink %u;"
                        " overruns GP %u (pc %04X) EP %u (pc %04X);"
                        " clipped %u samples, loudest %.2f\n",
                g_dsp_stat.gp_scratch, g_dsp_stat.gp_fifo_in,
                g_dsp_stat.gp_fifo_out, g_dsp_stat.ep_scratch,
                g_dsp_stat.ep_fifo_in, g_dsp_stat.ep_fifo_out,
                g_dsp_stat.ep_sink, g_dsp_stat.gp_overrun, g_dsp_stat.gp_pc,
                g_dsp_stat.ep_overrun, g_dsp_stat.ep_pc,
                g_dsp_stat.clipped, g_dsp_stat.loudest);
        fflush(stderr);
        memset(&g_dsp_stat, 0, sizeof g_dsp_stat);
    }

    if (frames > 6000)
        return;

    uint32_t gp = ldl_le_phys(address_space_memory, d->regs[NV_PAPU_GPSADDR]);
    uint32_t ep = ldl_le_phys(address_space_memory, d->regs[NV_PAPU_EPSADDR]);
    if (gp == last_gp && ep == last_ep)
        return;
    last_gp = gp;
    last_ep = ep;

    fprintf(stderr, "[APU] scratch page table changed at frame %d:"
                    " GP page0 0x%08X, EP page0 0x%08X\n", frames, gp, ep);
}

void mcpx_apu_dsp_frame(MCPXAPUState *d,
                        float mixbins[NUM_MIXBINS][NUM_SAMPLES_PER_FRAME])
{
    dsp_ack_frame(d);
    dsp_watch_scratch(d);

    bool gp_enabled = d->gp.dsp && dsp_enabled() &&
                      proc_running(d->gp.regs[NV_PAPU_GPRST]);
    bool ep_enabled = d->ep.dsp && dsp_enabled() &&
                      proc_running(d->ep.regs[NV_PAPU_EPRST]);

    if (!gp_enabled && !ep_enabled) {
        dsp_passthrough(d, mixbins);
        g_dbg.gp.cycles = 0;
        g_dbg.ep.cycles = 0;
        return;
    }

    /* Write VP results to the GP DSP MIXBUF */
    if (d->gp.dsp) {
        for (int mixbin = 0; mixbin < NUM_MIXBINS; mixbin++) {
            uint32_t base = GP_DSP_MIXBUF_BASE + mixbin * NUM_SAMPLES_PER_FRAME;
            for (int sample = 0; sample < NUM_SAMPLES_PER_FRAME; sample++) {
                dsp_write_memory(d->gp.dsp, 'X', base + sample,
                                 float_to_24b(mixbins[mixbin][sample]));
            }
        }
    }

    /* Run GP */
    if (gp_enabled) {
        dsp_start_frame(d->gp.dsp);
        d->gp.dsp->core.is_idle = false;
        d->gp.dsp->core.cycle_count = 0;
        dsp_run_frame(d->gp.dsp, d->gp.realtime, "GP");
        g_dbg.gp.cycles = d->gp.dsp->core.cycle_count;

        if ((d->monitor.point == MCPX_APU_DEBUG_MON_GP) ||
            (d->monitor.point == MCPX_APU_DEBUG_MON_GP_OR_EP && !ep_enabled)) {
            int off = (d->ep_frame_div % 8) * NUM_SAMPLES_PER_FRAME;
            for (int i = 0; i < NUM_SAMPLES_PER_FRAME; i++) {
                uint32_t l = dsp_read_memory(d->gp.dsp, 'X', 0x1400 + i);
                d->monitor.frame_buf[off + i][0] = l >> 8;
                uint32_t r =
                    dsp_read_memory(d->gp.dsp, 'X', 0x1400 + 1 * 0x20 + i);
                d->monitor.frame_buf[off + i][1] = r >> 8;
            }
        }
    }

    /* Run EP, once every eight VP frames: it encodes 256 samples at a time. */
    if (ep_enabled) {
        if (d->ep_frame_div % 8 == 0) {
            dsp_start_frame(d->ep.dsp);
            d->ep.dsp->core.is_idle = false;
            d->ep.dsp->core.cycle_count = 0;
            dsp_run_frame(d->ep.dsp, d->ep.realtime, "EP");
            g_dbg.ep.cycles = d->ep.dsp->core.cycle_count;
        }
    }
}

/* ── Setup ───────────────────────────────────────────────────────────────*/

void mcpx_apu_update_dsp_preference(MCPXAPUState *d)
{
    static int last_known_preference = -1;
    int want = dsp_enabled();

    if (last_known_preference == want)
        return;

    if (want) {
        /* The EP's encoded frame is the sound card's copy; the monitor point
         * says so, and ep_sink_samples takes it on its way out. */
        d->monitor.point = MCPX_APU_DEBUG_MON_GP_OR_EP;
        d->gp.realtime = true;
        d->ep.realtime = true;
    } else {
        d->monitor.point = MCPX_APU_DEBUG_MON_AC97;
        d->gp.realtime = false;
        d->ep.realtime = false;
    }

    last_known_preference = want;
}

void mcpx_apu_dsp_init(MCPXAPUState *d)
{
    d->gp.dsp = dsp_init(d, gp_scratch_rw, gp_fifo_rw);
    if (d->gp.dsp) {
        for (int i = 0; i < DSP_PRAM_SIZE; i++)
            d->gp.dsp->core.pram[i] = 0xCACACACA;
        memset((void *)d->gp.dsp->core.pram_opcache, 0,
               sizeof(d->gp.dsp->core.pram_opcache));
        d->gp.dsp->is_gp = true;
        d->gp.dsp->core.is_gp = true;
        d->gp.dsp->core.is_idle = false;
        d->gp.dsp->core.cycle_count = 0;
    }

    d->ep.dsp = dsp_init(d, ep_scratch_rw, ep_fifo_rw);
    if (d->ep.dsp) {
        for (int i = 0; i < DSP_PRAM_SIZE; i++)
            d->ep.dsp->core.pram[i] = 0xCACACACA;
        memset((void *)d->ep.dsp->core.pram_opcache, 0,
               sizeof(d->ep.dsp->core.pram_opcache));
        for (int i = 0; i < DSP_XRAM_SIZE; i++)
            d->ep.dsp->core.xram[i] = 0xCACACACA;
        for (int i = 0; i < DSP_YRAM_SIZE; i++)
            d->ep.dsp->core.yram[i] = 0xCACACACA;
        d->ep.dsp->is_gp = false;
        d->ep.dsp->core.is_gp = false;
        d->ep.dsp->core.is_idle = false;
        d->ep.dsp->core.cycle_count = 0;
    }

    mcpx_apu_update_dsp_preference(d);

    fprintf(stderr, "[APU] DSP GP/EP initialized (%s)\n",
            dsp_enabled() ? "DSP56300 emulation"
                          : "passthrough, RECOMP_APU_DSP=0");
}
