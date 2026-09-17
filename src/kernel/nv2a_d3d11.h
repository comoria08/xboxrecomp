/*
 * nv2a_d3d11.h - GPU fill for the pushbuffer executor.
 *
 * nv2a_pb_exec.c decodes the pushbuffer, runs vertex programs and clips on
 * the CPU; this takes the resulting triangles and draws them with Direct3D 11
 * instead of filling pixels one by one. Per-pixel work was 85% of the NV2A
 * thread in a Def Jam: Fight for NY fight, with the game thread spinning on
 * the GPU the whole time.
 *
 * Everything here runs on the thread that executes the pushbuffer. Opt-in
 * with RECOMP_D3D11; the CPU rasteriser stays the default and the reference.
 */
#ifndef NV2A_D3D11_H
#define NV2A_D3D11_H

#include <stdint.h>

/* One vertex, already in clip space: position (x, y, z, w) with z in 0..1 of
 * the depth range times w, texture coordinate 0 in texels, colour 0..1. */
typedef struct {
    float pos[4];
    float uv[2];
    float col[4];
} NvD3DVertex;

/* The pipeline state a triangle is drawn with, in the title's own (NV2A / GL)
 * enums. Compared bytewise to decide when a batch ends, so fill every field
 * (memset first). */
typedef struct {
    uint32_t depth_enable;              /* test against the depth buffer */
    uint32_t depth_write;
    uint32_t depth_func;                /* GL 0x0200.. */
    uint32_t stencil_enable;
    uint32_t stencil_func, stencil_ref, stencil_rmask, stencil_wmask;
    uint32_t stencil_fail, stencil_zfail, stencil_zpass;
    uint32_t blend_enable;
    uint32_t blend_src, blend_dst, blend_eq, blend_color;
    uint32_t color_mask;                /* NV2A SET_COLOR_MASK layout */
    uint32_t alpha_test, alpha_func, alpha_ref;
    int32_t  texture;                   /* handle from nvd3d_texture, -1 none */
    uint32_t addr_u, addr_v;            /* 1 wrap, anything else clamp */
    uint32_t tex_w, tex_h;
    uint32_t scissor[4];                /* x, y, w, h in surface samples */
} NvD3DState;

/* Non-zero once RECOMP_D3D11 is set and the device came up. */
int  nvd3d_active(void);
int  nvd3d_trace(void);                 /* RECOMP_D3D11_TRACE */

/* The colour surface being drawn into: its guest address and size in samples.
 * A surface is known by address and width; its height only grows. Returns the
 * height of the render target actually bound (what normalised coordinates
 * must be computed against), or 0 if there is none. */
uint32_t nvd3d_target(uint32_t base, uint32_t width, uint32_t height);

/* The texture bound to stage 0. The backend keeps one GPU texture per key and
 * re-uploads it when the content hash changes; decode fills width x height
 * A8R8G8B8 texels, top row first, and is only called when an upload is due.
 * mem_base/mem_bytes say where the texels live, so a texture that is really a
 * render target can be recognised. Returns a handle, or -1. */
int  nvd3d_texture(const uint32_t key[6], uint64_t hash,
                   uint32_t width, uint32_t height,
                   uint32_t mem_base, uint32_t mem_bytes,
                   void (*decode)(uint32_t *dst, uint32_t width, uint32_t height));

void nvd3d_triangle(const NvD3DVertex v[3], const NvD3DState *st);

/* Clear on the current target. what: bit 0 depth, bit 1 stencil, bit 2 colour.
 * The colour clear covers rect (x, y, w, h) in samples; depth and stencil the
 * whole buffer, as the CPU path does. */
void nvd3d_clear(uint32_t what, uint32_t argb, float depth, uint8_t stencil,
                 const uint32_t rect[4]);

/* The title swapped. Show the surface at base, region (x, y, w, h) in samples,
 * squeezed to a display of disp_w x disp_h. */
void nvd3d_present(uint32_t base, const uint32_t rect[4],
                   uint32_t disp_w, uint32_t disp_h);

/* Copy the surface at base back into guest memory (pitch bytes a row), for a
 * screenshot or anything else that reads the framebuffer. */
void nvd3d_readback(uint32_t base, uint8_t *dst, uint32_t pitch, uint32_t rows);

#endif
