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
    /* The register combiners (point 140). ps is a handle from
     * nvd3d_ps_program, or -1 for the fixed texel-times-colour shader that
     * every batch the combiners would draw the same way keeps. Only a
     * translated vertex program carries what they read -- the specular colour
     * and the coordinates of stages 1-3 -- so the CPU path always has -1. */
    int32_t  ps;
    int32_t  stage_tex[3];              /* stages 1-3: handle from nvd3d_texture, -1 none */
    uint32_t stage_wh[3][2];            /* their size in texels */
    uint32_t stage_addr[3];             /* addr_u | addr_v << 4, as addr_u/addr_v */
    uint32_t stage_linear[3];           /* 1: a linear image, coordinates in texels */
    uint32_t factor0[8], factor1[8];    /* combiner constants per stage, ARGB */
    /* 1: the interface (DLSS step 1, point 149: an orthographic projection,
     * w == 1 out of the vertex program, on the shown surface). Drawn into a
     * layer of its own instead of the target, and composited onto it before
     * anything reads the target -- so that an upscaler can later take the
     * scene alone. See ui_layer in nv2a_d3d11.c. 2: orthographic too, but
     * across the whole width (a fade, point 151): the layer is composited
     * first and this is drawn into the target, never squeezed. */
    uint32_t layer;
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

/* -- Vertex programs on the GPU (RECOMP_VP_GPU) ----------------------------
 *
 * The CPU path runs the title's NV2A vertex program in an interpreter, one
 * vertex at a time, and hands finished triangles over. Measured in a fight,
 * that interpreter is 55% of the thread every frame waits on, while the GPU
 * sits idle. Here the program is translated to HLSL once and run as the
 * vertex shader instead: the batch goes over as its raw attribute bytes, the
 * constants and a triangle list of vertex indices. */

/* One attribute stream as uploaded: element i starts at base + i * stride in
 * the data. size 0 means the batch does not supply it; the shader then takes
 * the value in NvD3DProgramDraw.absent, as the CPU path would. */
typedef struct {
    uint32_t base, stride, type, size;
} NvD3DAttr;

typedef struct {
    int             vs;                 /* from nvd3d_vs_program */
    const float   (*consts)[4];         /* the program's 192 constants */
    uint32_t        consts_gen;         /* changes whenever they do: sent only then */
    NvD3DAttr       attr[16];
    float           absent[16][4];
    const void     *data;               /* the attribute bytes, as described */
    uint32_t        data_bytes;
    const uint32_t *index;              /* triangle list of vertex indices */
    uint32_t        nindex;
    float           ax, ay;             /* window to NDC, as d3d_triangle */
    float           uv_scale[2];        /* texels per unit, or 1 */
    int             textured;
    /* Point sprites (point 140): each index is vertex << 2 | corner, six per
     * point, and the shader moves each corner by the size the program wrote
     * in oPts and gives stage 3 its 0..1 sprite coordinates. */
    int             points;
    /* Lines (primitives 2-4, point 140): the index list is pairs, drawn as a
     * D3D11 line list, one sample wide as the NV2A draws them. */
    int             lines;
    /* DLSS step 3 (RECOMP_MOTION): drawn a second time into the motion vector
     * target, through vs_motion (the program run on prev_consts, then on
     * consts), when motion is set. */
    int             motion;
    int             vs_motion;
    const float   (*prev_consts)[4];
    uint64_t        prog_hash;          /* what drew it, for RECOMP_MOTION_PROBE */
    uint32_t        pos_addr, pos_n;
} NvD3DProgramDraw;

/* The vertex shader for one translated program, compiled once and cached by
 * hash. -1 if it does not compile: the caller keeps the CPU path for it. With
 * hlsl NULL it only looks: -2 means not seen yet, so translate and call again. */
int  nvd3d_vs_program(uint64_t hash, const char *hlsl, size_t len);

/* The pixel shader for one register-combiner configuration, translated by the
 * executor and compiled once, cached by hash exactly as nvd3d_vs_program:
 * -2 not seen yet (translate and call again), -1 does not compile. */
int  nvd3d_ps_program(uint64_t hash, const char *hlsl, size_t len);

/* Draw a whole batch through a translated program, after whatever the CPU
 * path queued, under the same state d3d_triangle would have used. */
void nvd3d_draw_program(const NvD3DProgramDraw *d, const NvD3DState *st);

/* The same batch through the same shader, but captured by stream output
 * instead of drawn: one NvD3DVertex per index, which is exactly what the CPU
 * path would have handed to nvd3d_triangle. This is how the translation is
 * checked against the interpreter (RECOMP_VP_GPU=2). Returns the count. */
uint32_t nvd3d_check_program(const NvD3DProgramDraw *d, NvD3DVertex *out, uint32_t max);

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
