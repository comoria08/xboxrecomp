/*
 * MCPX DSP emulator - compatibility shim
 *
 * The DSP core in this directory is taken from xemu, which builds it inside
 * QEMU. This header stands in for the handful of QEMU facilities it uses, so
 * the sources themselves stay byte-for-byte comparable with upstream:
 *
 *   qemu/osdep.h    the usual C standard headers
 *   qemu/bswap.h    not used by the paths we compile
 *   qemu/compiler.h QEMU_FALLTHROUGH
 *   container_of    offset-of-member back-cast
 *   trace_*         QEMU's tracepoints, compiled out here
 *
 * Copyright (c) 2026 the xboxrecomp authors
 *
 * This file exists only to host the GPL-licensed DSP core, and is released
 * under the same terms:
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 */

#ifndef XBOXRECOMP_DSP_COMPAT_H
#define XBOXRECOMP_DSP_COMPAT_H

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef container_of
#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - offsetof(type, member)))
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/* MSVC has no ssize_t. */
#if defined(_MSC_VER) && !defined(_SSIZE_T_DEFINED)
#define _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#endif

/* Unaligned little-endian 32-bit access. The DSP keeps program memory as
 * host-order words already, and every target this builds for is little
 * endian, so these are plain copies -- memcpy rather than a cast, because
 * the pointer is not promised to be aligned. */
static inline uint32_t ldl_le_p(const void *p)
{
    uint32_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

static inline void stl_le_p(void *p, uint32_t v)
{
    memcpy(p, &v, sizeof v);
}

#ifndef QEMU_FALLTHROUGH
#if defined(__GNUC__) || defined(__clang__)
#define QEMU_FALLTHROUGH __attribute__((fallthrough))
#else
#define QEMU_FALLTHROUGH do {} while (0)
#endif
#endif

/* QEMU tracepoints. The DSP runs 1504 frames a second; tracing every
 * instruction is not something this build offers, so they compile away. */
#define TRACE_DSP56K_EXECUTE_INSTRUCTION_DISASM 0
static inline bool trace_event_get_state(int id) { (void)id; return false; }
static inline void trace_dsp56k_execute_instruction(bool is_gp, uint32_t pc)
{
    (void)is_gp; (void)pc;
}
static inline void trace_dsp56k_execute_instruction_disasm(const char *text)
{
    (void)text;
}
static inline void trace_dsp_read_peripheral(uint32_t addr, uint32_t value)
{
    (void)addr; (void)value;
}
static inline void trace_dsp_write_peripheral(uint32_t addr, uint32_t value)
{
    (void)addr; (void)value;
}

#endif /* XBOXRECOMP_DSP_COMPAT_H */
