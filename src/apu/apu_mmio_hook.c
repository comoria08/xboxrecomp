/*
 * MCPX APU MMIO Hook - VEH instruction decoder for APU register access
 *
 * Reuses the same x86-64 instruction decoder pattern as nv2a_mmio_hook.c
 * but routes reads/writes through the MCPX APU register handlers.
 */

#include "apu.h"
#include <stdio.h>
#include <stdlib.h>   /* getenv: implicit, it returned a truncated int on x64 */
#include <stdbool.h>
#include <stdint.h>

/* Global APU state pointer -- referenced from main.c regardless of which
 * platform's MMIO hook is active, so define it before the #if guard. */
MCPXAPUState *g_apu_state = NULL;

/* The MMIO hook is a Win32-VEH x86-64 instruction decoder. On Linux the
 * equivalent goes through sigaction + ucontext_t (Stage 2 / main.c). For
 * now the whole body is Windows-only so apu_emu links on Debian. */
#if defined(_WIN32)
#include <windows.h>

/* APU MMIO base in Xbox VA space */
#define APU_MMIO_BASE  0xFE800000u
#define APU_MMIO_SIZE  0x00080000u  /* 512KB */

/* (g_apu_state is defined above, outside the Win32 guard) */

/* Statistics */
static int g_apu_mmio_read_count = 0;
static int g_apu_mmio_write_count = 0;
static int g_apu_mmio_decode_fail = 0;

/* ============================================================
 * x86-64 register access helpers (same as NV2A hook)
 * ============================================================ */

static uint64_t *ctx_reg64(PCONTEXT ctx, int reg)
{
    switch (reg & 0xF) {
    case 0:  return (uint64_t*)&ctx->Rax;
    case 1:  return (uint64_t*)&ctx->Rcx;
    case 2:  return (uint64_t*)&ctx->Rdx;
    case 3:  return (uint64_t*)&ctx->Rbx;
    case 4:  return (uint64_t*)&ctx->Rsp;
    case 5:  return (uint64_t*)&ctx->Rbp;
    case 6:  return (uint64_t*)&ctx->Rsi;
    case 7:  return (uint64_t*)&ctx->Rdi;
    case 8:  return (uint64_t*)&ctx->R8;
    case 9:  return (uint64_t*)&ctx->R9;
    case 10: return (uint64_t*)&ctx->R10;
    case 11: return (uint64_t*)&ctx->R11;
    case 12: return (uint64_t*)&ctx->R12;
    case 13: return (uint64_t*)&ctx->R13;
    case 14: return (uint64_t*)&ctx->R14;
    case 15: return (uint64_t*)&ctx->R15;
    default: return NULL;
    }
}

static int decode_modrm_len(const uint8_t *ip, int has_rex_b)
{
    uint8_t modrm = *ip;
    int mod = (modrm >> 6) & 3;
    int rm = (modrm & 7) | (has_rex_b ? 8 : 0);
    int len = 1;

    if (mod == 3) return 1;
    if ((rm & 7) == 4) len += 1; /* SIB */
    if (mod == 0 && (rm & 7) == 5) len += 4; /* disp32 */
    else if (mod == 1) len += 1;
    else if (mod == 2) len += 4;
    return len;
}

/* Flags an emulated instruction leaves, as the CPU would. AF is not kept:
 * nothing compiled from C tests it. */
#define FL_CF 0x0001u
#define FL_PF 0x0004u
#define FL_ZF 0x0040u
#define FL_SF 0x0080u
#define FL_OF 0x0800u

static uint64_t size_mask(int size)
{
    return size == 8 ? ~0ULL : (1ULL << (size * 8)) - 1;
}

static void set_flags_szp(PCONTEXT ctx, uint64_t r, int size)
{
    uint8_t b = (uint8_t)r;
    r &= size_mask(size);
    if (!r) ctx->EFlags |= FL_ZF;
    if ((r >> (size * 8 - 1)) & 1) ctx->EFlags |= FL_SF;
    b ^= b >> 4; b ^= b >> 2; b ^= b >> 1;
    if (!(b & 1)) ctx->EFlags |= FL_PF;
}

static void set_flags_logic(PCONTEXT ctx, uint64_t r, int size)
{
    ctx->EFlags &= ~(FL_CF | FL_PF | FL_ZF | FL_SF | FL_OF);
    set_flags_szp(ctx, r, size);
}

static void set_flags_arith(PCONTEXT ctx, uint64_t a, uint64_t b, int size, int sub)
{
    const uint64_t m = size_mask(size);
    const int top = size * 8 - 1;
    uint64_t r;
    int sa, sb, sr;
    a &= m; b &= m;
    r = (sub ? a - b : a + b) & m;
    sa = (int)((a >> top) & 1); sb = (int)((b >> top) & 1); sr = (int)((r >> top) & 1);
    ctx->EFlags &= ~(FL_CF | FL_PF | FL_ZF | FL_SF | FL_OF);
    if (sub ? a < b : r < a) ctx->EFlags |= FL_CF;
    if (sub ? (sa != sb && sr != sa) : (sa == sb && sr != sa)) ctx->EFlags |= FL_OF;
    set_flags_szp(ctx, r, size);
}

/* The immediate after a ModRM operand: imm8 for the byte forms and 83,
 * imm16 under a 66 prefix, imm32 otherwise (sign-extended for REX.W). */
static uint64_t read_imm(const uint8_t *p, int opcode, int access_size, int *len)
{
    if (opcode == 0x80 || opcode == 0xF6) { *len = 1; return p[0]; }
    if (opcode == 0x83) { *len = 1; return (uint64_t)(int64_t)(int8_t)p[0]; }
    if (access_size == 2) { *len = 2; return *(const uint16_t *)p; }
    *len = 4;
    return (uint64_t)(int64_t)*(const int32_t *)p;
}

/* ============================================================
 * Instruction decoder for APU MMIO access
 * ============================================================ */

static bool apu_decode_and_handle(PCONTEXT ctx, uint32_t mmio_offset, int is_write)
{
    const uint8_t *ip = (const uint8_t *)ctx->Rip;
    if (!g_apu_state) return false;

    int prefix_len = 0;
    int has_66 = 0;
    int rex = 0, has_rex = 0;

    while (1) {
        uint8_t b = ip[prefix_len];
        if (b == 0x66) { has_66 = 1; prefix_len++; }
        else if (b == 0xF2 || b == 0xF3) { prefix_len++; }
        else if (b >= 0x40 && b <= 0x4F) { rex = b; has_rex = 1; prefix_len++; }
        else break;
    }

    int rex_w = has_rex && (rex & 0x08);
    int rex_r = has_rex && (rex & 0x04);
    int rex_b = has_rex && (rex & 0x01);

    const uint8_t *opcode = ip + prefix_len;
    int access_size = 4;
    if (has_66) access_size = 2;
    if (rex_w) access_size = 8;

    /* MOV r/m, r (write: 88/89) */
    if (opcode[0] == 0x89 || opcode[0] == 0x88) {
        if (opcode[0] == 0x88) access_size = 1;
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        int reg = ((opcode[1] >> 3) & 7) | (rex_r ? 8 : 0);
        uint64_t val = *ctx_reg64(ctx, reg);
        mcpx_apu_mmio_write(g_apu_state, mmio_offset, val, access_size);
        ctx->Rip += prefix_len + 1 + modrm_len;
        g_apu_mmio_write_count++;
        return true;
    }

    /* MOV r/m, imm32 (write: C7 /0) */
    if (opcode[0] == 0xC7) {
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        uint32_t imm = *(uint32_t *)(opcode + 1 + modrm_len);
        mcpx_apu_mmio_write(g_apu_state, mmio_offset, imm, access_size);
        ctx->Rip += prefix_len + 1 + modrm_len + 4;
        g_apu_mmio_write_count++;
        return true;
    }

    /* MOV r/m8, imm8 (write: C6 /0) */
    if (opcode[0] == 0xC6) {
        access_size = 1;
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        uint8_t imm = *(opcode + 1 + modrm_len);
        mcpx_apu_mmio_write(g_apu_state, mmio_offset, imm, 1);
        ctx->Rip += prefix_len + 1 + modrm_len + 1;
        g_apu_mmio_write_count++;
        return true;
    }

    /* MOV r, r/m (read: 8A/8B) */
    if (opcode[0] == 0x8B || opcode[0] == 0x8A) {
        if (opcode[0] == 0x8A) access_size = 1;
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        int reg = ((opcode[1] >> 3) & 7) | (rex_r ? 8 : 0);
        uint64_t val = mcpx_apu_mmio_read(g_apu_state, mmio_offset, access_size);
        uint64_t *dst = ctx_reg64(ctx, reg);
        if (access_size == 1) *dst = (*dst & ~0xFFULL) | (val & 0xFF);
        else if (access_size == 2) *dst = (*dst & ~0xFFFFULL) | (val & 0xFFFF);
        else if (access_size == 4) *dst = val & 0xFFFFFFFF;
        else *dst = val;
        ctx->Rip += prefix_len + 1 + modrm_len;
        g_apu_mmio_read_count++;
        return true;
    }

    /* MOVZX r32, r/m8 (0F B6) */
    if (opcode[0] == 0x0F && opcode[1] == 0xB6) {
        access_size = 1;
        int modrm_len = decode_modrm_len(opcode + 2, rex_b);
        int reg = ((opcode[2] >> 3) & 7) | (rex_r ? 8 : 0);
        uint64_t val = mcpx_apu_mmio_read(g_apu_state, mmio_offset, 1);
        *ctx_reg64(ctx, reg) = val & 0xFF;
        ctx->Rip += prefix_len + 2 + modrm_len;
        g_apu_mmio_read_count++;
        return true;
    }

    /* MOVZX r32, r/m16 (0F B7) */
    if (opcode[0] == 0x0F && opcode[1] == 0xB7) {
        access_size = 2;
        int modrm_len = decode_modrm_len(opcode + 2, rex_b);
        int reg = ((opcode[2] >> 3) & 7) | (rex_r ? 8 : 0);
        uint64_t val = mcpx_apu_mmio_read(g_apu_state, mmio_offset, 2);
        *ctx_reg64(ctx, reg) = val & 0xFFFF;
        ctx->Rip += prefix_len + 2 + modrm_len;
        g_apu_mmio_read_count++;
        return true;
    }

    /* TEST r/m, r (84/85) - read */
    if (opcode[0] == 0x85 || opcode[0] == 0x84) {
        if (opcode[0] == 0x84) access_size = 1;
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        int reg = ((opcode[1] >> 3) & 7) | (rex_r ? 8 : 0);
        uint64_t mem_val = mcpx_apu_mmio_read(g_apu_state, mmio_offset, access_size);
        uint64_t reg_val = *ctx_reg64(ctx, reg);
        uint64_t result = mem_val & reg_val;
        ctx->EFlags &= ~(0x0001 | 0x0040 | 0x0080 | 0x0800);
        if (result == 0) ctx->EFlags |= 0x0040;
        if (result & (1ULL << (access_size * 8 - 1))) ctx->EFlags |= 0x0080;
        ctx->Rip += prefix_len + 1 + modrm_len;
        g_apu_mmio_read_count++;
        return true;
    }

    /* CMP r/m, r (38/39) */
    if (opcode[0] == 0x39 || opcode[0] == 0x38) {
        if (opcode[0] == 0x38) access_size = 1;
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        int reg = ((opcode[1] >> 3) & 7) | (rex_r ? 8 : 0);
        uint64_t mem_val = mcpx_apu_mmio_read(g_apu_state, mmio_offset, access_size);
        uint64_t reg_val = *ctx_reg64(ctx, reg);
        if (access_size <= 4) {
            mem_val &= (1ULL << (access_size * 8)) - 1;
            reg_val &= (1ULL << (access_size * 8)) - 1;
        }
        uint64_t result = mem_val - reg_val;
        ctx->EFlags &= ~(0x0001 | 0x0040 | 0x0080 | 0x0800);
        if (result == 0) ctx->EFlags |= 0x0040;
        if (mem_val < reg_val) ctx->EFlags |= 0x0001;
        if (result & (1ULL << (access_size * 8 - 1))) ctx->EFlags |= 0x0080;
        ctx->Rip += prefix_len + 1 + modrm_len;
        g_apu_mmio_read_count++;
        return true;
    }

    /* OR r/m, r (08/09) */
    if (opcode[0] == 0x09 || opcode[0] == 0x08) {
        if (opcode[0] == 0x08) access_size = 1;
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        int reg = ((opcode[1] >> 3) & 7) | (rex_r ? 8 : 0);
        uint64_t mem_val = mcpx_apu_mmio_read(g_apu_state, mmio_offset, access_size);
        uint64_t reg_val = *ctx_reg64(ctx, reg);
        mcpx_apu_mmio_write(g_apu_state, mmio_offset, mem_val | reg_val, access_size);
        ctx->Rip += prefix_len + 1 + modrm_len;
        g_apu_mmio_write_count++;
        return true;
    }

    /* AND r/m, r (20/21) */
    if (opcode[0] == 0x21 || opcode[0] == 0x20) {
        if (opcode[0] == 0x20) access_size = 1;
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        int reg = ((opcode[1] >> 3) & 7) | (rex_r ? 8 : 0);
        uint64_t mem_val = mcpx_apu_mmio_read(g_apu_state, mmio_offset, access_size);
        uint64_t reg_val = *ctx_reg64(ctx, reg);
        mcpx_apu_mmio_write(g_apu_state, mmio_offset, mem_val & reg_val, access_size);
        ctx->Rip += prefix_len + 1 + modrm_len;
        g_apu_mmio_write_count++;
        return true;
    }

    /* The forms below are ones Clang emits where MSVC did not. Measured on
     * Def Jam built with llvm-mingw: "test dword ptr [mem], imm32" on the
     * first APU register the title polls, which killed the game at boot. */

    /* TEST r/m, imm (F6 /0 ib, F7 /0 id) - read */
    if ((opcode[0] == 0xF6 || opcode[0] == 0xF7) && ((opcode[1] >> 3) & 7) == 0) {
        int imm_len;
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        uint64_t imm, mem_val;
        if (opcode[0] == 0xF6) access_size = 1;
        imm = read_imm(opcode + 1 + modrm_len, opcode[0], access_size, &imm_len);
        mem_val = mcpx_apu_mmio_read(g_apu_state, mmio_offset, access_size);
        set_flags_logic(ctx, mem_val & imm, access_size);
        ctx->Rip += prefix_len + 1 + modrm_len + imm_len;
        g_apu_mmio_read_count++;
        return true;
    }

    /* ADD, OR, AND, SUB, XOR, CMP r/m, imm (80 /n ib, 81 /n id, 83 /n ib).
     * CMP only reads; the others write the result back. ADC and SBB would
     * need the incoming carry and are left to the failure below. */
    if ((opcode[0] == 0x80 || opcode[0] == 0x81 || opcode[0] == 0x83) &&
        ((opcode[1] >> 3) & 7) != 2 && ((opcode[1] >> 3) & 7) != 3) {
        int op = (opcode[1] >> 3) & 7;
        int imm_len;
        int modrm_len = decode_modrm_len(opcode + 1, rex_b);
        uint64_t imm, m, r;
        if (opcode[0] == 0x80) access_size = 1;
        imm = read_imm(opcode + 1 + modrm_len, opcode[0], access_size, &imm_len)
            & size_mask(access_size);
        m = mcpx_apu_mmio_read(g_apu_state, mmio_offset, access_size) & size_mask(access_size);
        switch (op) {
        case 0:  r = m + imm; set_flags_arith(ctx, m, imm, access_size, 0); break;
        case 1:  r = m | imm; set_flags_logic(ctx, r, access_size); break;
        case 4:  r = m & imm; set_flags_logic(ctx, r, access_size); break;
        case 6:  r = m ^ imm; set_flags_logic(ctx, r, access_size); break;
        default: r = m - imm; set_flags_arith(ctx, m, imm, access_size, 1); break;  /* 5, 7 */
        }
        if (op != 7) {
            mcpx_apu_mmio_write(g_apu_state, mmio_offset, r & size_mask(access_size), access_size);
            g_apu_mmio_write_count++;
        } else {
            g_apu_mmio_read_count++;
        }
        ctx->Rip += prefix_len + 1 + modrm_len + imm_len;
        return true;
    }

    /* Unrecognized */
    g_apu_mmio_decode_fail++;
    if (g_apu_mmio_decode_fail <= 20) {
        fprintf(stderr, "[APU] MMIO decode fail at RIP=%p offset=0x%X: %02X %02X %02X %02X %02X %02X\n",
                (void*)ctx->Rip, mmio_offset, ip[0], ip[1], ip[2], ip[3], ip[4], ip[5]);
        fflush(stderr);
    }
    return false;
}

/* ============================================================
 * Public API (called from VEH in main.c)
 * ============================================================ */

bool apu_hook_handle_mmio(PCONTEXT ctx, uintptr_t fault_addr,
                          uint32_t fault_xbox_va, int is_write)
{
    uint32_t mmio_offset = fault_xbox_va - APU_MMIO_BASE;
    bool ok = apu_decode_and_handle(ctx, mmio_offset, is_write);

    /* What the title actually asks the APU for. The DSPs are stubbed here, so
     * a title that waits on one waits forever, and the only way to work out
     * what it is waiting for is to see the register traffic that precedes the
     * wait. */
    if (getenv("RECOMP_APU_TRACE")) {
        static unsigned n;
        if (n++ < 400) {
            /* The value as well as the offset: finding which register carries
             * the command-block address means recognising the address when it
             * goes past, and an offset alone never shows it. */
            uint64_t v = g_apu_state
                       ? mcpx_apu_mmio_read(g_apu_state, mmio_offset, 4) : 0;
            fprintf(stderr, "  [APUMMIO] %s 0x%05X = %08X%s\n",
                    is_write ? "write" : "read ", mmio_offset,
                    (uint32_t)v, ok ? "" : "  (decode failed)");
        }
    }
    return ok;
}

#endif /* _WIN32 */
