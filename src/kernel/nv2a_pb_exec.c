/**
 * Execute the parts of the title's pushbuffer that produce visible pixels.
 *
 * The title builds NV2A commands in guest RAM and advances DMA_PUT; without
 * something consuming them the framebuffer stays whatever it was, which is how
 * a fully booted title renders a black screen. This walks the same command
 * stream nv2a_pb_scan.c surveys and carries out the subset that decides what is
 * on screen: which surface is being drawn into, and clearing it.
 *
 * It also rasterises geometry, but only the part that can be drawn honestly:
 * batches whose attribute 0 is already in screen space, flat-shaded, straight
 * into the same guest framebuffer the clear writes. Titles draw their UI, HUD
 * and 2D overlays that way, so it is the first geometry to appear. Batches that
 * need a vertex program executed are counted and skipped rather than drawn
 * somewhere wrong -- see raster_batch(). Texturing, depth and vertex programs
 * are still a renderer, not a command decoder; the upgrade path is the D3D11
 * translator in src/nv2a/nv2a_pgraph_d3d11.c.
 *
 * Everything this does not handle is counted and ranked by
 * nv2a_pb_exec_report(), so what remains is a list rather than a guess.
 *
 * Enabled with RECOMP_PB_EXEC. RECOMP_RASTER_TEST draws one known triangle
 * after every clear, which separates "the pixel path is broken" from "the title
 * has not given us any vertices". RECOMP_FB_DUMP=<prefix> writes the surface to
 * <prefix>NNN.bmp, so the result can be looked at without a display.
 */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "kernel.h"   /* XBOX_CONTIG_BASE / XBOX_CONTIG_SIZE */
#include "xbox_memory_layout.h"   /* xbox_Nv2aFrameCounterFlip */
/* The swizzle decoder the D3D8 layer already uses -- one implementation of
 * Morton order, not a second one that can disagree with it. */
#include "../d3d/d3d8_swizzle.h"

extern ptrdiff_t xbox_GetMemoryOffset(void);
extern void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch);
extern void xbox_FramebufferWindowLatch(void);
extern void xbox_FramebufferWindowFormat(uint32_t width, uint32_t height,
                                         uint32_t bpp, uint32_t xscale);
extern void xbox_FramebufferWindowStart(void);
extern uint32_t g_xbox_image_lo, g_xbox_image_hi;

/* Would writing this surface land on the title's own image?
 *
 * NV097_SET_SURFACE_COLOR_OFFSET is an offset within the colour DMA object,
 * not a guest virtual address, and this executor has always used it as one.
 * That is harmless while the two happen to agree and catastrophic when they do
 * not: the Xbox Dashboard names surface 0x00088000 at 1280x960x4, so clearing
 * it wrote 4.9 MB of opaque black from 0x00088000 to 0x00538000 -- straight
 * over its own code, its D3D context at 0x000BBFC0 and the register-block
 * pointer at 0x000BE2C4. The symptom was a title that submitted one perfect
 * frame and then spun forever in a pushbuffer-full loop, three layers away,
 * with every D3D global reading 0xFF000000: the clear colour.
 *
 * So refuse, and say so. Getting the address right needs the DMA object base
 * this ignores (NV097_SET_CONTEXT_DMA_COLOR); until that exists, writing
 * nothing is strictly better than writing over the guest, and a title that
 * cannot draw is easier to debug than one that has been overwritten.
 */
static int surface_hits_image(uint32_t base, uint32_t bytes)
{
    if (!g_xbox_image_hi || !bytes)
        return 0;
    return base < g_xbox_image_hi && base + bytes > g_xbox_image_lo;
}

/* Where a DMA-object offset actually lives.
 *
 * NV097_SET_SURFACE_COLOR_OFFSET is an offset inside the colour DMA object,
 * and for a framebuffer that object covers physical memory -- so the offset
 * is a physical address, not a guest VA. Those are the same number in this
 * runtime, which is why treating it as a VA works until it does not: on
 * hardware the image is mapped at VA 0x00010000 from arbitrary physical
 * pages, so a framebuffer at physical 0x84000 does not overlap it. Here it
 * would.
 *
 * The title tells us which it is by where it allocated. Half-Life 2's
 * framebuffer comes from MmAllocateContiguousMemory, which this runtime
 * serves from the window at XBOX_CONTIG_BASE, so physical P is visible at
 * XBOX_CONTIG_BASE + P -- clear of the image, and the same bytes the title's
 * own writes and the framebuffer window reach.
 *
 * So: use the offset as a VA when that is credible, and fall back to the
 * physical mirror exactly when it is not. Titles whose surfaces already sit
 * in ordinary RAM (Wreckless renders to the tiled alias of physical
 * 0x01954000) keep the first path and are unaffected.
 */
static uint32_t dma_resolve(uint32_t offset)
{
    extern uint32_t xbox_ContiguousAllocatedBytes(void);

    /* Did this runtime hand the offset out as contiguous memory? Then the
     * bytes live in the window, and that is not a guess: the arena is a bump
     * allocator from XBOX_CONTIG_BASE, so everything below its high-water
     * mark is memory some MmAllocateContiguousMemory call returned. The
     * title's own writes go through the window, so the executor's must too.
     *
     * Checking this BEFORE the image test is the whole point. The image test
     * only catches an offset that would land on the title's code, and whether
     * it does is an accident of where the image happens to end: Half-Life 2's
     * colour surface is physical 0x00A6C000, which clears the image by 700 KB.
     * So it looked like an ordinary VA, and the executor cleared 1.2 MB of
     * black straight through the guest heap -- which faulted the title three
     * frames later on a pointer that had been overwritten, while the real
     * framebuffer at 0x80A6C000 stayed untouched and the screen stayed black. */
    if (offset < xbox_ContiguousAllocatedBytes())
        return XBOX_CONTIG_BASE + offset;
    if (!surface_hits_image(offset, 1))
        return offset;
    if ((uint64_t)offset < XBOX_CONTIG_SIZE)
        return XBOX_CONTIG_BASE + offset;
    return offset;                         /* nothing better to offer */
}

static int surface_write_refused(uint32_t base, uint32_t bytes, const char *what)
{
    static int said;

    if (!surface_hits_image(base, bytes))
        return 0;
    if (!said) {
        said = 1;
        fprintf(stderr,
                "  [GPU] REFUSING to %s surface 0x%08X..0x%08X: that overlaps "
                "the loaded image (0x%08X..0x%08X).\n"
                "  [GPU]   SET_SURFACE_COLOR_OFFSET is a DMA-object offset, not "
                "a guest VA, and this executor treats it as one. Writing here "
                "would destroy the title's own code and globals.\n",
                what, base, base + bytes, g_xbox_image_lo, g_xbox_image_hi);
        fflush(stderr);
    }
    return 1;
}

/* NV097 methods this executor acts on. */
#define NV097_SET_SURFACE_CLIP_HORIZONTAL 0x0200
#define NV097_SET_SURFACE_CLIP_VERTICAL   0x0204
#define NV097_SET_SURFACE_FORMAT          0x0208
/* Fields of the format parameter. Mirrors src/nv2a/nv2a_regs.h, repeated here
 * because this file keeps its own small method table rather than pulling in
 * the full register header. */
#define NV097_SURFACE_FORMAT_COLOR        0x0000000F
#define NV097_SURFACE_COLOR_X1R5G5B5_Z    0x01
#define NV097_SURFACE_COLOR_X1R5G5B5_O    0x02
#define NV097_SURFACE_COLOR_R5G6B5        0x03
#define NV097_SURFACE_COLOR_X8R8G8B8_Z    0x04
#define NV097_SURFACE_COLOR_X8R8G8B8_O    0x05
#define NV097_SURFACE_COLOR_X1A7R8G8B8_Z  0x06
#define NV097_SURFACE_COLOR_X1A7R8G8B8_O  0x07
#define NV097_SURFACE_COLOR_A8R8G8B8      0x08
#define NV097_SURFACE_COLOR_B8            0x09
#define NV097_SURFACE_COLOR_G8B8          0x0A
#define NV097_SURFACE_FORMAT_AA           0x0000F000
#define NV097_SURFACE_AA_CENTER_1         0
#define NV097_SURFACE_AA_CENTER_CORNER_2  1
#define NV097_SURFACE_AA_SQUARE_OFFSET_4  2
#define NV097_SET_SURFACE_PITCH           0x020C
#define NV097_SET_SURFACE_COLOR_OFFSET    0x0210
#define NV097_SET_BLEND_ENABLE            0x0304
#define NV097_SET_BLEND_FUNC_SFACTOR      0x0344
#define NV097_SET_BLEND_FUNC_DFACTOR      0x0348
#define NV097_SET_BLEND_COLOR             0x034C
#define NV097_SET_BLEND_EQUATION          0x0350
#define NV097_SET_ALPHA_TEST_ENABLE       0x0300
#define NV097_SET_DEPTH_TEST_ENABLE       0x030C
#define NV097_SET_STENCIL_TEST_ENABLE     0x032C
#define NV097_SET_COLOR_MASK              0x0358
#define NV097_SET_STENCIL_MASK            0x0360
#define NV097_SET_STENCIL_FUNC_FUNC       0x0364
#define NV097_SET_STENCIL_FUNC_REF        0x0368
#define NV097_SET_STENCIL_FUNC_MASK       0x036C
#define NV097_SET_STENCIL_OP_FAIL         0x0370
#define NV097_SET_STENCIL_OP_ZFAIL        0x0374
#define NV097_SET_STENCIL_OP_ZPASS        0x0378
#define NV097_SET_DEPTH_FUNC              0x0354
#define NV097_SET_DEPTH_MASK              0x035C
#define NV097_SET_ZSTENCIL_CLEAR_VALUE    0x1D8C
#define NV097_SET_ALPHA_FUNC              0x033C
#define NV097_SET_ALPHA_REF               0x0340
#define NV_BLEND_ZERO                     0x0000
#define NV_BLEND_ONE                      0x0001
#define NV_BLEND_SRC_ALPHA                0x0302
#define NV_BLEND_ONE_MINUS_SRC_ALPHA      0x0303
#define NV_BLEND_EQ_SUBTRACT              0x800A
#define NV_BLEND_EQ_REVERSE_SUBTRACT      0x800B
#define NV097_SET_COLOR_CLEAR_VALUE       0x1D90
#define NV097_CLEAR_SURFACE               0x1D94
#define NV097_SET_VERTEX_DATA_ARRAY_OFFSET 0x1720   /* +i*4, 16 attributes */
#define NV097_SET_VERTEX_DATA_ARRAY_FORMAT 0x1760   /* +i*4 */
#define NV097_SET_BEGIN_END               0x17FC
#define NV097_SET_TEXTURE_OFFSET          0x1B00   /* +i*0x40 */
#define NV097_SET_TEXTURE_FORMAT          0x1B04
#define NV097_SET_TEXTURE_ADDRESS         0x1B08
#define NV097_SET_TEXTURE_CONTROL1        0x1B10
#define NV097_SET_TEXTURE_IMAGE_RECT      0x1B1C
#define NV097_SET_TEXTURE_PALETTE         0x1B20   /* offset | DMA bit 0 */
/* The buffer flip. A title double-buffers by telling the GPU which buffer
 * the CRTC reads and which it draws into, advancing the write index and
 * then stalling until the flip has happened. Ignoring these means the
 * stall never clears: Half-Life 2's loader submits its initialisation,
 * asks for a flip, and waits for it in a loop that makes no kernel calls
 * and burns no dispatch, which reads as a hang with no cause.
 *
 * ponytail: the flip completes the moment it is asked for, because there is
 * no scanout to be in the middle of. That makes every frame land instantly
 * and a title that paces itself on the flip runs as fast as it can draw.
 * Pacing wants the vblank clock in the kernel, not a sleep in here. */
#define NV097_SET_FLIP_READ               0x0120
#define NV097_SET_FLIP_WRITE              0x0124
#define NV097_SET_FLIP_MODULO             0x0128
#define NV097_FLIP_INCREMENT_WRITE        0x012C
#define NV097_FLIP_STALL                  0x0130
#define NV097_ARRAY_ELEMENT16             0x1800
#define NV097_INLINE_ARRAY                0x1818
/* Immediate-mode vertices. SET_VERTEX3F/4F carry the position, and writing
 * its last component completes a vertex using whatever the SET_VERTEX_DATA*
 * registers currently hold for the other attributes. This is how Half-Life
 * 2's Xbox loader and the game's own 2D drawing submit every quad -- neither
 * uses INLINE_ARRAY -- so without these the executor saw SET_BEGIN_END pairs
 * with nothing attached and reported `draws 0` while a million and a half
 * textured quads a minute went past it. */
#define NV097_SET_VERTEX3F                0x1500   /* +0..0x08, 3 floats */
#define NV097_SET_VERTEX4F                0x1518   /* +0..0x0C, 4 floats */
#define NV097_SET_VERTEX_DATA2F_M         0x1880   /* + attr*8,  2 floats */
#define NV097_SET_VERTEX_DATA4F_M         0x1A00   /* + attr*16, 4 floats */
#define NV097_SET_VERTEX_DATA4UB          0x1940   /* + attr*4,  D3DCOLOR */

/* One immediate vertex, as this file packs it for the shared draw path:
 * position float4, diffuse D3DCOLOR, texcoord0 float2. */
#define IMM_VERTEX_DWORDS 7

#define NV097_CLEAR_COLOR_MASK            0xF0   /* R,G,B,A bits */

/* One vertex attribute stream, as the title describes it. Attribute 0 is
 * position; the rest are colours, texture coordinates and so on. */
typedef struct {
    uint32_t offset;      /* guest address of element 0 */
    uint32_t type;        /* NV097 data type nibble */
    uint32_t size;        /* components per element */
    uint32_t stride;      /* bytes between elements */
} VertexAttr;

#define NV_VERTEX_ATTRS 16
#define NV_MAX_INDICES  4096
#define NV_MAX_INLINE   4096            /* dwords of INLINE_ARRAY per batch */

/* Texture stage 0, decoded from what the title programmed.
 *
 * Only stage 0: it is the only one the dashboard configures, and a stage
 * nothing writes to is a stage nothing can be sampled from. The rest arrive
 * as unhandled methods and are counted as such, which is how the next title
 * that needs them will say so. */
typedef struct {
    uint32_t offset;                    /* guest address of texel (0,0)  */
    uint32_t width, height;             /* from IMAGE_RECT               */
    uint32_t pitch;                     /* bytes per row, from CONTROL1  */
    uint32_t color;                     /* NV097 colour-format code      */
    uint32_t addr_u, addr_v;            /* wrap mode per axis            */
    uint32_t palette;                   /* guest address, 256 D3DCOLORs  */
    int      valid;
} Texture;

static struct {
    VertexAttr attr[NV_VERTEX_ATTRS];
    uint32_t   prim;                    /* SET_BEGIN_END parameter, 0 = ended */
    uint16_t   idx[NV_MAX_INDICES];
    uint32_t   idx_count;
    /* INLINE_ARRAY payload: vertices written straight into the pushbuffer
     * instead of into a buffer the title points at. Same vertex format, a
     * different place to read them from. */
    uint32_t   inline_buf[NV_MAX_INLINE];
    uint32_t   inline_count;
    /* Current values of the immediate-mode attributes, and how many complete
     * vertices they have produced in this batch. */
    float      imm_pos[4];
    uint32_t   imm_diffuse;
    float      imm_tex[2];
    uint32_t   imm_count;
    int        inline_active;
    uint32_t   draws, verts, nonzero_draws;
    float      min_x, max_x, min_y, max_y;
    uint32_t color_offset, color_base, pitch, format;
    /* The surface the last batch actually drew into. A double-buffered title
     * has already pointed color_offset at the next buffer and cleared it by
     * the time the flip arrives, so dumping the current one dumps the frame
     * that has not been drawn yet -- which is how a correctly rendered
     * sequence came out as 12 black BMPs. */
    uint32_t drawn_offset;
    uint64_t pixels;
    uint32_t pixel_max;   /* brightest value any pixel write carried */
    uint32_t clip_x, clip_w, clip_y, clip_h;
    /* Alpha blending, as the title asked for it. Without this every textured
     * pass writes its texels opaque, so a UI layer drawn over the scene
     * arrives as a solid rectangle of whatever its transparent texels happen
     * to contain -- Def Jam: Fight for NY's loading logo came out as a white
     * box with the chain inside it. */
    uint32_t blend_enable;
    uint32_t blend_sfactor, blend_dfactor, blend_equation, blend_color;
    uint32_t alpha_test, alpha_func, alpha_ref;
    uint32_t depth_test, depth_func, depth_mask, zclear;
    uint32_t stencil_test, stencil_wmask, stencil_func, stencil_ref, stencil_fmask;
    uint32_t stencil_op_fail, stencil_op_zfail, stencil_op_zpass;
    uint32_t color_mask;
    uint32_t clear_color;
    uint32_t clears, unhandled_total;
    uint32_t flip_read, flip_write, flip_modulo, flips;
    uint32_t tris_drawn, tris_skipped_offscreen, batches_untransformed;
    /* Why a batch came out flat. "Untextured" has two causes that look
     * identical on screen and want opposite fixes: the batch carried no
     * texture coordinates, or it did and the stage was not usable. */
    uint32_t batches_textured, batches_no_uv, batches_no_tex;
    Texture  tex;
} s_gpu;

/* Unhandled methods, ranked. The interesting output is not that something was
 * skipped but which things dominate, because that is the order to implement
 * them in. */
#define PB_EXEC_MAX_UNHANDLED 2048
typedef struct { uint32_t method, count, last_param; } PbUnhandled;
static PbUnhandled s_unhandled[PB_EXEC_MAX_UNHANDLED];
static int s_unhandled_count;

/* Every texture-stage register, as the title last set it.
 * Texturing is not implemented yet; knowing which formats and sizes a title
 * actually programs is what decides which ones are worth implementing. */
#define NV_TEX_FIRST 0x1B00
#define NV_TEX_LAST  0x1BFC
static uint32_t s_tex_reg[(NV_TEX_LAST - NV_TEX_FIRST) / 4 + 1];
static uint8_t  s_tex_set[(NV_TEX_LAST - NV_TEX_FIRST) / 4 + 1];

/* Formats whose dimensions come from the format word and whose coordinates
 * arrive normalised, rather than from a pitch and SET_TEXTURE_IMAGE_RECT with
 * coordinates in texels. Swizzled and block-compressed are both in this group,
 * and every place that used to test only for swizzled needs the pair. */
static int tex_size_from_format(uint32_t fmt)
{
    return d3d8_format_is_swizzled(fmt) || d3d8_format_dxt_block_bytes(fmt);
}

static void record_tex_reg(uint32_t method, uint32_t param)
{
    s_tex_reg[(method - NV_TEX_FIRST) / 4] = param;
    s_tex_set[(method - NV_TEX_FIRST) / 4] = 1;
    /* A pitch is a linear texture's property. A swizzled one has no rows and
     * so no pitch, and requiring one here refused every swizzled texture --
     * which is nearly all of them, since swizzled is the Xbox default. That
     * left the title's own textures unsampled and every textured quad drawn in
     * flat vertex colour. */
    s_gpu.tex.valid = s_gpu.tex.offset && s_gpu.tex.width && s_gpu.tex.height
                   && (tex_size_from_format(s_gpu.tex.color)
                       || s_gpu.tex.pitch);
}

/* Every distinct texture a batch was drawn with, and how many batches used it.
 *
 * The per-draw verbose print shows the first few draws of the first frame,
 * which is enough to see that texturing works at all and not enough to answer
 * "is a font page ever bound". This is the same shape as the unhandled-method
 * table below it: a small set, ranked, printed with the rest of the report. */
#define PB_EXEC_MAX_TEXTURES 64
typedef struct {
    uint32_t offset, color, width, height, batches;
} PbTexUse;
static PbTexUse s_tex_use[PB_EXEC_MAX_TEXTURES];
static int s_tex_use_count;

/* Defined below, next to the sampler it goes through. */
static void dump_texture_bmp(uint32_t seq);

static void note_texture_use(void)
{
    int i;

    if (!s_gpu.tex.valid)
        return;
    for (i = 0; i < s_tex_use_count; i++) {
        if (s_tex_use[i].offset == s_gpu.tex.offset
         && s_tex_use[i].color  == s_gpu.tex.color) {
            s_tex_use[i].batches++;
            /* Again once the title has had time to fill it: a texture is
             * often bound before its contents are uploaded. */
            if (s_tex_use[i].batches == 3000)
                dump_texture_bmp(100 + (uint32_t)i);
            return;
        }
    }
    if (s_tex_use_count < PB_EXEC_MAX_TEXTURES) {
        s_tex_use[s_tex_use_count].offset  = s_gpu.tex.offset;
        s_tex_use[s_tex_use_count].color   = s_gpu.tex.color;
        s_tex_use[s_tex_use_count].width   = s_gpu.tex.width;
        s_tex_use[s_tex_use_count].height  = s_gpu.tex.height;
        s_tex_use[s_tex_use_count].batches = 1;
        s_tex_use_count++;
        dump_texture_bmp((uint32_t)s_tex_use_count - 1);
    }
}

static void note_unhandled(uint32_t method, uint32_t param)
{
    int i;

    s_gpu.unhandled_total++;
    for (i = 0; i < s_unhandled_count; i++) {
        if (s_unhandled[i].method == method) {
            s_unhandled[i].count++;
            s_unhandled[i].last_param = param;
            return;
        }
    }
    if (s_unhandled_count < PB_EXEC_MAX_UNHANDLED) {
        s_unhandled[s_unhandled_count].method = method;
        s_unhandled[s_unhandled_count].count = 1;
        s_unhandled[s_unhandled_count].last_param = param;
        s_unhandled_count++;
    }
}

/* Read attribute `a` of vertex `index` as floats. Only the float and the
 * normalised-byte types appear in practice; anything else returns 0 so a
 * caller sees a degenerate vertex rather than reading past the array. */
static int fetch_attr(const VertexAttr *a, uint32_t index, float out[4])
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    const uint8_t *p;
    uint32_t i;

    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    if (!a->size || !a->stride)
        return 0;
    if (s_gpu.inline_active) {
        /* The batch arrived as INLINE_ARRAY, so `offset` is a byte offset into
         * the buffered payload rather than a guest address -- and 0 is a legal
         * one there, which is why the offset test is on the other side of this
         * branch. */
        size_t at = (size_t)a->offset + (size_t)index * a->stride;
        if (at + 4 > (size_t)s_gpu.inline_count * 4)
            return 0;
        p = (const uint8_t *)s_gpu.inline_buf + at;
    } else {
        if (!a->offset)
            return 0;
        p = mem + a->offset + (size_t)index * a->stride;
    }

    switch (a->type) {
    case 0:                                  /* D3DCOLOR */
        /* A DWORD 0xAARRGGBB, so little-endian bytes are B,G,R,A -- not the
         * component order of every other format here. Returned as R,G,B,A so
         * callers need not know which format the title chose. */
        out[0] = (float)p[2] / 255.0f;
        out[1] = (float)p[1] / 255.0f;
        out[2] = (float)p[0] / 255.0f;
        out[3] = (float)p[3] / 255.0f;
        return 1;
    case 2:                                  /* float */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = ((const float *)p)[i];
        return 1;
    case 4:                                  /* unsigned byte, normalised */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = (float)p[i] / 255.0f;
        return 1;
    /* The short formats. EA's APT sends every shape as whole-pixel short
     * positions (type 5), and returning zero for them put the whole front end
     * on one point. */
    case 1:                                  /* signed short, normalised */
        for (i = 0; i < a->size && i < 4; i++) {
            int16_t v = (int16_t)(p[2 * i] | (p[2 * i + 1] << 8));
            out[i] = v < -32767 ? -1.0f : (float)v / 32767.0f;
        }
        return 1;
    case 5:                                  /* signed short, as is */
        for (i = 0; i < a->size && i < 4; i++)
            out[i] = (float)(int16_t)(p[2 * i] | (p[2 * i + 1] << 8));
        return 1;
    case 6: {                                /* packed 11:11:10, normalised */
        uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
                   | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        int32_t x = (int32_t)(v << 21) >> 21;
        int32_t y = (int32_t)(v << 10) >> 21;
        int32_t z = (int32_t)v >> 22;
        out[0] = (float)x / 1023.0f;
        out[1] = (float)y / 1023.0f;
        out[2] = (float)z / 511.0f;
        return 1;
    }
    default:
        return 0;
    }
}

/* How many samples across make one displayed pixel.
 *
 * NV097_SET_SURFACE_FORMAT's anti-aliasing field says whether the surface is
 * multisampled, and a multisampled surface is physically wider than the clip
 * rectangle: CENTER_CORNER_2 doubles the row, SQUARE_OFFSET_4 doubles both
 * axes. Everything that walks the surface has to know, because the row is not
 * clip_w pixels long. */
static uint32_t surface_aa_x(void)
{
    switch ((s_gpu.format & NV097_SURFACE_FORMAT_AA) >> 12) {
    case NV097_SURFACE_AA_CENTER_CORNER_2: return 2;
    case NV097_SURFACE_AA_SQUARE_OFFSET_4: return 2;
    default:                                                     return 1;
    }
}

static uint32_t surface_aa_y(void)
{
    return ((s_gpu.format & NV097_SURFACE_FORMAT_AA) >> 12)
           == NV097_SURFACE_AA_SQUARE_OFFSET_4 ? 2 : 1;
}

/* Bytes per sample.
 *
 * This used to be pitch/clip_w, on the reasoning that the pitch is always
 * exactly how many bytes a row occupies. The pitch is -- but a row is not
 * clip_w samples wide once the surface is multisampled, and dividing by the
 * clip width then gives a number that is not a pixel size at all. Def Jam:
 * Fight for NY renders 640x480 with 2x multisampling: pitch 5120 over clip
 * width 640 came out as 8, which no branch here handles, so the clear wrote
 * nothing and the framebuffer window painted black for a title that was
 * rendering the whole time.
 *
 * The colour field is the real answer. The pitch heuristic stays as the
 * fallback for a surface whose format method has not arrived yet, since that
 * is the case it was right about. */
static uint32_t surface_bpp(void)
{
    switch (s_gpu.format & NV097_SURFACE_FORMAT_COLOR) {
    case NV097_SURFACE_COLOR_X1R5G5B5_Z:
    case NV097_SURFACE_COLOR_X1R5G5B5_O:
    case NV097_SURFACE_COLOR_R5G6B5:
        return 2;
    case NV097_SURFACE_COLOR_X8R8G8B8_Z:
    case NV097_SURFACE_COLOR_X8R8G8B8_O:
    case NV097_SURFACE_COLOR_X1A7R8G8B8_Z:
    case NV097_SURFACE_COLOR_X1A7R8G8B8_O:
    case NV097_SURFACE_COLOR_A8R8G8B8:
        return 4;
    case NV097_SURFACE_COLOR_B8:
        return 1;
    case NV097_SURFACE_COLOR_G8B8:
        return 2;
    default:
        break;
    }
    if (!s_gpu.clip_w)
        return 0;
    return s_gpu.pitch / s_gpu.clip_w;
}

/* Samples per row and rows in the surface, as opposed to displayed pixels. */
static uint32_t surface_sample_w(void) { return s_gpu.clip_w * surface_aa_x(); }
static uint32_t surface_sample_h(void) { return s_gpu.clip_h * surface_aa_y(); }


/* Write the current surface out as a 24-bit BMP.
 *
 * A framebuffer window needs someone watching it. A file does not, which makes
 * this the only way to check what a title actually rendered on a machine you
 * are not sitting at -- and the only way to put a picture in a bug report.
 *
 * ponytail: bottom-up 24bpp BMP, no palette, no compression. That is the one
 * format every viewer reads and it is 30 lines; PNG would need a dependency.
 */
static void dump_surface_bmp(void)
{
    const char *prefix = getenv("RECOMP_FB_DUMP");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t bpp = surface_bpp();
    static int seq;
    char path[512];
    uint32_t w = s_gpu.clip_w, h = s_gpu.clip_h, y, x;
    uint32_t row_bytes, pad, filesz;
    uint8_t hdr[54];
    FILE *f;

    if (!prefix || !w || !h || (bpp != 2 && bpp != 4) || !s_gpu.color_offset)
        return;

    row_bytes = w * 3;
    pad = (4 - (row_bytes & 3)) & 3;
    filesz = 54 + (row_bytes + pad) * h;

    snprintf(path, sizeof path, "%s%03d.bmp", prefix, seq++);
    f = fopen(path, "wb");
    if (!f)
        return;

    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &filesz, 4);
    hdr[10] = 54;
    hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1;
    hdr[28] = 24;
    fwrite(hdr, 1, sizeof hdr, f);

    /* BMP rows run bottom-up. */
    for (y = h; y-- > 0; ) {
        const uint8_t *row = mem + dma_resolve(s_gpu.drawn_offset
                                                ? s_gpu.drawn_offset
                                                : s_gpu.color_offset)
                           + (size_t)(s_gpu.clip_y + y) * s_gpu.pitch;
        for (x = 0; x < w; x++) {
            uint8_t bgr[3];
            if (bpp == 4) {
                /* Resolve the samples that make up this pixel, the way the
                 * display does. Reading sample x instead would show the left
                 * half of a multisampled frame across the whole picture. */
                const uint32_t *p = (const uint32_t *)row;
                uint32_t n = surface_aa_x(), i, rr = 0, gg = 0, bb = 0;
                for (i = 0; i < n; i++) {
                    uint32_t v = p[(s_gpu.clip_x + x) * n + i];
                    rr += (v >> 16) & 0xFF;
                    gg += (v >>  8) & 0xFF;
                    bb +=  v        & 0xFF;
                }
                bgr[0] = (uint8_t)(bb / n);
                bgr[1] = (uint8_t)(gg / n);
                bgr[2] = (uint8_t)(rr / n);
            } else {
                uint16_t v = ((const uint16_t *)row)[s_gpu.clip_x + x];
                bgr[0] = (uint8_t)(( v        & 0x1F) << 3);
                bgr[1] = (uint8_t)(((v >>  5) & 0x3F) << 2);
                bgr[2] = (uint8_t)(((v >> 11) & 0x1F) << 3);
            }
            fwrite(bgr, 1, 3, f);
        }
        if (pad) {
            static const uint8_t zero[3] = {0, 0, 0};
            fwrite(zero, 1, pad, f);
        }
    }
    fclose(f);
    if (seq == 1)
        fprintf(stderr, "  [GPU] framebuffer dump: %s (%ux%u from 0x%08X %ubpp)\n",
                path, w, h, s_gpu.color_offset, bpp);
}

/* Defined below, next to the rest of the rasteriser; the clear path uses it
 * for RECOMP_RASTER_TEST. */
static void raster_triangle(const float a[2], const float b[2],
                            const float c[2], uint32_t argb,
                            const float uv[3][2]);

/* Depth, kept on the host.
 *
 * The title's zeta surface is not read or written: a float per sample here
 * is simpler and is all the rasteriser needs to draw a 3D scene in the right
 * order. Without it the crowd behind the ring was painted over the fighters.
 * Depth values are the window z the vertex program produces (0..2^24-1). */
static float   *s_zbuf;
static uint32_t s_zbuf_w, s_zbuf_h;
static float    s_tri_z[3];
static int      s_tri_has_z;
/* Per-vertex 1/w and colour for the triangle being rasterised, when the batch
 * went through a vertex program. With them texture coordinates are
 * interpolated perspective-correct and colour is Gouraud-shaded; without
 * (screen-space batches) the old affine, flat path is kept. */
static float    s_tri_invw[3];
static uint32_t s_tri_col[3];
static int      s_tri_persp;
static int      s_depth_on = 1;        /* RECOMP_NO_DEPTH turns it off */
static int      s_batch_flat_z;        /* current batch is a 2D overlay */

static float *zbuf_get(void)
{
    uint32_t w = surface_sample_w() + s_gpu.clip_x;
    uint32_t h = surface_sample_h() + s_gpu.clip_y;
    if (!w || !h)
        return NULL;
    if (w != s_zbuf_w || h != s_zbuf_h) {
        free(s_zbuf);
        s_zbuf = (float *)malloc((size_t)w * h * sizeof(float));
        s_zbuf_w = s_zbuf ? w : 0;
        s_zbuf_h = s_zbuf ? h : 0;
        if (s_zbuf) {
            size_t i;
            for (i = 0; i < (size_t)w * h; i++)
                s_zbuf[i] = 16777215.0f;
        }
    }
    return s_zbuf;
}

/* Coplanar geometry has to compare equal. The hardware interpolates 24-bit
 * fixed point and gets exactly the same value twice; a float interpolation of
 * values around 7 million does not, and EA's APT draws every element of a
 * menu at one depth with LEQUAL -- so overlapping elements lost to each other
 * by a fraction and the menus lost pieces. */
#define DEPTH_EPS 8.0f

static int depth_pass(float z, float stored)
{
    switch (s_gpu.depth_func) {
    case 0x0200: return 0;                                   /* NEVER */
    case 0x0201: return z <  stored - DEPTH_EPS;             /* LESS */
    case 0x0202: return fabsf(z - stored) <= DEPTH_EPS;      /* EQUAL */
    case 0x0203: return z <= stored + DEPTH_EPS;             /* LEQUAL */
    case 0x0204: return z >  stored + DEPTH_EPS;             /* GREATER */
    case 0x0205: return fabsf(z - stored) > DEPTH_EPS;       /* NOTEQUAL */
    case 0x0206: return z >= stored - DEPTH_EPS;             /* GEQUAL */
    default:     return 1;                                   /* ALWAYS */
    }
}

/* Stencil, alongside the depth buffer and sized with it. EA APT masks every
 * clipped element with it: the mask shape marks the stencil, then a
 * full-screen quad pushes depth to the front everywhere outside it, and the
 * content is drawn behind. Without the stencil that quad covered the whole
 * screen and the menus lost their items. */
static uint8_t *s_sbuf;
static uint32_t s_sbuf_n;

static uint8_t *sbuf_get(void)
{
    uint32_t n;
    if (!zbuf_get())
        return NULL;
    n = s_zbuf_w * s_zbuf_h;
    if (n != s_sbuf_n) {
        free(s_sbuf);
        s_sbuf = (uint8_t *)calloc(n, 1);
        s_sbuf_n = s_sbuf ? n : 0;
    }
    return s_sbuf;
}

static int stencil_func_pass(uint8_t stored)
{
    const uint32_t m = s_gpu.stencil_fmask & 0xFF;
    const uint32_t r = s_gpu.stencil_ref & m, v = stored & m;
    switch (s_gpu.stencil_func) {
    case 0x0200: return 0;
    case 0x0201: return r <  v;
    case 0x0202: return r == v;
    case 0x0203: return r <= v;
    case 0x0204: return r >  v;
    case 0x0205: return r != v;
    case 0x0206: return r >= v;
    default:     return 1;
    }
}

static void stencil_apply(uint8_t *sp, uint32_t op)
{
    uint8_t v = *sp, n;
    switch (op) {
    case 0x0000: n = 0; break;                                   /* ZERO */
    case 0x1E01: n = (uint8_t)s_gpu.stencil_ref; break;          /* REPLACE */
    case 0x1E02: n = v == 0xFF ? 0xFF : (uint8_t)(v + 1); break; /* INCR */
    case 0x1E03: n = v == 0 ? 0 : (uint8_t)(v - 1); break;       /* DECR */
    case 0x150A: n = (uint8_t)~v; break;                         /* INVERT */
    case 0x8507: n = (uint8_t)(v + 1); break;                    /* INCR_WRAP */
    case 0x8508: n = (uint8_t)(v - 1); break;                    /* DECR_WRAP */
    default:     return;                                         /* KEEP */
    }
    *sp = (uint8_t)((v & ~s_gpu.stencil_wmask) | (n & s_gpu.stencil_wmask));
}

static int alpha_pass(uint32_t color)
{
    const uint32_t av = color >> 24, ref = s_gpu.alpha_ref;
    switch (s_gpu.alpha_func) {
    case 0x0200: return 0;
    case 0x0201: return av <  ref;
    case 0x0202: return av == ref;
    case 0x0203: return av <= ref;
    case 0x0204: return av >  ref;
    case 0x0205: return av != ref;
    case 0x0206: return av >= ref;
    default:     return 1;
    }
}

/* Alpha test, stencil test, depth test and their writes, in the order the
 * GPU applies them. A batch without a meaningful depth (s_tri_has_z == 0, a
 * 2D overlay) passes the depth test and writes no depth, but still honours
 * the stencil. */
static int fragment_tests(int x, int y, float z, uint32_t color)
{
    float *zb;
    size_t idx;
    int zpass = 1;

    if (s_gpu.alpha_test && !alpha_pass(color))
        return 0;
    if (!s_depth_on)
        return 1;
    zb = zbuf_get();
    if (!zb || (uint32_t)x >= s_zbuf_w || (uint32_t)y >= s_zbuf_h)
        return 1;
    idx = (size_t)y * s_zbuf_w + (size_t)x;
    if (s_tri_has_z && s_gpu.depth_test)
        zpass = depth_pass(z, zb[idx]);
    if (s_gpu.stencil_test) {
        uint8_t *sb = sbuf_get();
        if (sb) {
            if (!stencil_func_pass(sb[idx])) {
                stencil_apply(sb + idx, s_gpu.stencil_op_fail);
                return 0;
            }
            stencil_apply(sb + idx, zpass ? s_gpu.stencil_op_zpass
                                          : s_gpu.stencil_op_zfail);
        }
    }
    if (!zpass)
        return 0;
    if (s_tri_has_z && s_gpu.depth_mask)
        zb[idx] = z;
    return 1;
}

static uint32_t s_vpf_flip = 0xFFFFFFFFu;   /* frame being logged by VPF */

static void clear_surface(uint32_t param)
{
    if (s_gpu.flips == s_vpf_flip)
        fprintf(stderr, "  [VPF] f%u CLEAR 0x%02X clip %ux%u+%u+%u\n", s_gpu.flips, param,
                s_gpu.clip_w, s_gpu.clip_h, s_gpu.clip_x, s_gpu.clip_y);
    if (getenv("RECOMP_ZCLEAR_TRACE")) {
        static unsigned n;
        static uint32_t last_param = 0xFFFFFFFFu, last_val = 0xFFFFFFFFu;
        if (param != last_param || s_gpu.zclear != last_val || (++n % 600) == 0) {
            last_param = param;
            last_val = s_gpu.zclear;
            fprintf(stderr, "  [ZCLEAR] param 0x%02X zclear 0x%08X depth test %u mask %u\n",
                    param, s_gpu.zclear, s_gpu.depth_test, s_gpu.depth_mask);
        }
    }
    if (param & 2) {                       /* NV097_CLEAR_SURFACE_STENCIL */
        uint8_t *sb = sbuf_get();
        if (sb)
            memset(sb, (int)(s_gpu.zclear & 0xFF), s_sbuf_n);
    }
    if (param & 1) {                       /* NV097_CLEAR_SURFACE_Z */
        float *zb = zbuf_get();
        if (zb) {
            float v = (float)(s_gpu.zclear >> 8);
            size_t i, n = (size_t)s_zbuf_w * s_zbuf_h;
            for (i = 0; i < n; i++)
                zb[i] = v;
        }
    }
    uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
    uint32_t bpp = surface_bpp();
    uint32_t y, x, clear_w, clear_h;

    if (!(param & NV097_CLEAR_COLOR_MASK))
        return;                            /* depth/stencil only */
    if (!s_gpu.color_offset || !s_gpu.pitch || !s_gpu.clip_h || bpp == 0)
        return;
    {
        uint32_t base = dma_resolve(s_gpu.color_offset);
        if (surface_write_refused(base,
                                  (s_gpu.clip_y + surface_sample_h())
                                      * s_gpu.pitch,
                                  "clear"))
            return;
        s_gpu.color_base = base;
    }

    /* Sample counts, not pixel counts: on a multisampled surface every
     * displayed pixel is several samples wide, and clearing only clip_w of
     * them leaves the rest of each row holding the previous frame. */
    clear_w = surface_sample_w();
    clear_h = surface_sample_h();

    for (y = 0; y < clear_h; y++) {
        uint8_t *row = mem + s_gpu.color_base
                     + (size_t)(s_gpu.clip_y + y) * s_gpu.pitch;
        if (bpp == 4) {
            uint32_t *p = (uint32_t *)row + s_gpu.clip_x;
            for (x = 0; x < clear_w; x++)
                p[x] = s_gpu.clear_color;
        } else if (bpp == 2) {
            /* The clear value is always given as A8R8G8B8; a 16-bit surface
             * takes the same colour reduced to 5:6:5. */
            uint16_t v = (uint16_t)(((s_gpu.clear_color >> 8) & 0xF800)
                                  | ((s_gpu.clear_color >> 5) & 0x07E0)
                                  | ((s_gpu.clear_color >> 3) & 0x001F));
            uint16_t *p = (uint16_t *)row + s_gpu.clip_x;
            for (x = 0; x < clear_w; x++)
                p[x] = v;
        }
    }
    s_gpu.clears++;
    /* Progress markers, interleaved with everything else in the log. The
     * summary says drawing stopped; only a marker next to the surrounding
     * activity says what the title was doing when it stopped. */
    if ((s_gpu.clears % 100) == 0)
        fprintf(stderr, "  [GPU] clear #%u\n", s_gpu.clears);
    /* Distinct clear colours actually used. "Cleared to black" and "the clear
     * never ran" look identical in the framebuffer, and only one of them is a
     * bug -- so record what was asked for, not just how often. */
    {
        static uint32_t seen[8];
        static int n;
        int i;
        for (i = 0; i < n; i++)
            if (seen[i] == s_gpu.clear_color) break;
        if (i == n && n < 8) {
            seen[n++] = s_gpu.clear_color;
            fprintf(stderr, "  [GPU] clear colour 0x%08X -> surface 0x%08X"
                            " (%ubpp)\n",
                    s_gpu.clear_color, s_gpu.color_offset, surface_bpp());
        }
    }

    /* Prove the pixel path end to end, independent of whether the title has
     * given us any geometry yet.
     *
     * "Nothing on screen" has three very different causes -- the surface
     * address or pitch is wrong, the rasteriser is broken, or the title's
     * vertex buffers are empty -- and they are indistinguishable from a black
     * window. RECOMP_RASTER_TEST draws one known triangle into the surface
     * just cleared, so a visible triangle rules out the first two and leaves
     * only the third. On Wreckless it is the third: attribute 0 decodes
     * correctly (float, size 2, stride 16) and the buffer it points at stays
     * zero.
     *
     * ponytail: bring-up aid, not a feature. It costs one branch per clear. */
    if (getenv("RECOMP_RASTER_TEST")) {
        static int announced;
        /* Every clear, not once: the title clears each frame and double-buffers,
         * so a triangle drawn a single time is erased before anyone sees it. */
        if (s_gpu.clip_w && s_gpu.clip_h) {
            float a[2], b[2], c[2];
            a[0] = s_gpu.clip_w * 0.5f; a[1] = s_gpu.clip_h * 0.15f;
            b[0] = s_gpu.clip_w * 0.85f; b[1] = s_gpu.clip_h * 0.85f;
            c[0] = s_gpu.clip_w * 0.15f; c[1] = s_gpu.clip_h * 0.85f;
            raster_triangle(a, b, c, 0xFFFF00FFu, NULL);  /* magenta: never a clear colour */
            if (announced++ == 0)
            fprintf(stderr, "  [GPU] raster self-test: triangle (%.0f,%.0f)"
                            " (%.0f,%.0f) (%.0f,%.0f) into 0x%08X %ubpp\n",
                    a[0], a[1], b[0], b[1], c[0], c[1],
                    s_gpu.color_offset, surface_bpp());
        }
    }

    /* Show the surface actually being drawn into. A title that double-buffers
     * renders into the back buffer, so following AvSetDisplayMode's address
     * would show the one nothing is writing. */
    /* The window has to read where the pixels actually are, which is the
     * resolved address rather than the DMA-object offset. */
    xbox_FramebufferWindowSet(dma_resolve(s_gpu.color_offset), s_gpu.pitch);
    /* And the geometry, so the window does not have to guess the pixel size
     * from the pitch -- which a multisampled surface makes impossible. */
    xbox_FramebufferWindowFormat(s_gpu.clip_w, s_gpu.clip_h,
                                 surface_bpp(), surface_aa_x());

    /* And open the window, rather than waiting for AvSetDisplayMode to do it.
     *
     * That was the only caller, so a title which draws before setting a display
     * mode -- or never sets one at all -- got no window however much it
     * rendered. The Xbox Dashboard clears a 1280x960 surface at 0x00088000 on
     * its first frame and had not called AvSetDisplayMode by then, so
     * RECOMP_FB_WINDOW=1 was set, the executor knew the address and the pitch,
     * and nothing appeared.
     *
     * Here is the better trigger anyway: this runs when a surface address is
     * known to be real, because a clear just used it. Idempotent and gated on
     * RECOMP_FB_WINDOW, so the cost is one interlocked compare per clear. */
    xbox_FramebufferWindowStart();
}


/* ── Rasteriser ──────────────────────────────────────────────────────────
 *
 * Fills triangles straight into the guest framebuffer, the same memory
 * clear_surface() writes and the framebuffer window already shows. That is the
 * whole reason it is done on the CPU rather than through D3D: nothing new has
 * to be plumbed for the result to be visible.
 *
 * ponytail: flat-shaded, no depth buffer, no texturing, no perspective
 * correction, and only batches whose attribute 0 is already in screen space.
 * A title running a vertex program hands over object-space positions that mean
 * nothing without executing the program, so those batches are counted and
 * skipped rather than drawn somewhere wrong. Upgrade path is the D3D11
 * translator in src/nv2a/nv2a_pgraph_d3d11.c once vertex programs are
 * translated; this exists to get the first geometry on screen for every title,
 * which in practice is UI, HUD and 2D overlays -- all pre-transformed.
 */

/* One texel, in the title's own format.
 *
 * The codes are the NV097 colour field, which is the Xbox D3DFMT_ enum --
 * src/d3d/d3d8_xbox.h is the table, and it is the table to check against
 * rather than recollection: 0x1E is LIN_X8R8G8B8 and not, as this first read
 * it, a byte-reversed BGRA. Getting that one wrong turned an opaque black
 * render target into a screen of pure blue, which is the kind of wrong that
 * looks like content.
 *
 * Only the linear (LIN_) formats are read. A swizzled texture stores its
 * texels in Morton order rather than in rows, so reading one as if it had a
 * pitch does not give a slightly wrong colour, it gives a different image --
 * and inventing that image is exactly what this is not for. An unsupported
 * format samples nothing and the caller keeps the vertex colour, which is
 * visibly wrong rather than quietly wrong.
 *
 * ponytail: nearest texel, no filtering, whatever SET_TEXTURE_FILTER asked
 * for. Bilinear when a title's output actually depends on it.
 */
/* Off the edge of the texture, the way the title asked for.
 *
 * Refusing to sample instead is not neutral: it hands the caller back the
 * vertex colour, so a pass whose coordinates reach the last texel by half a
 * texel gets a bright line down the edge of the screen. The dashboard's
 * resolve does exactly that -- its last column and last row, 1119 pixels of
 * white on a black frame, from a rounding step at the boundary.
 */
static uint32_t wrap_coord(uint32_t c, uint32_t size, uint32_t mode)
{
    if (!size)
        return 0;
    if (mode == 1)                         /* wrap */
        return c % size;
    return c >= size ? size - 1 : c;       /* clamp, and everything else */
}

static uint32_t expand(uint32_t v, uint32_t bits)
{
    return d3d8_expand_channel(v, bits);
}

/* The linear format that decodes the same texels as a swizzled one.
 *
 * Swizzling changes where a texel lives, not what it says: A8R8G8B8 (0x06) and
 * LIN_A8R8G8B8 (0x12) are the same four bytes in the same order. So the whole
 * difference is the address calculation, and one of those lets every format
 * below serve both. Pairs read off the table in d3d8_xbox.h rather than
 * recalled -- the comment above this one is about getting exactly that wrong. */
static uint32_t linear_twin(uint32_t fmt)
{
    switch (fmt) {
    case 0x00: return 0x13;                /* L8        -> LIN_L8        */
    case 0x02: return 0x10;                /* A1R5G5B5  -> LIN_A1R5G5B5  */
    case 0x03: return 0x1C;                /* X1R5G5B5  -> LIN_X1R5G5B5  */
    case 0x04: return 0x1D;                /* A4R4G4B4  -> LIN_A4R4G4B4  */
    case 0x05: return 0x11;                /* R5G6B5    -> LIN_R5G6B5    */
    case 0x06: return 0x12;                /* A8R8G8B8  -> LIN_A8R8G8B8  */
    case 0x07: return 0x1E;                /* X8R8G8B8  -> LIN_X8R8G8B8  */
    case 0x19: return 0x1F;                /* A8        -> LIN_A8        */
    default:   return fmt;                 /* already linear, or unhandled */
    }
}

static int sample_texture(uint32_t u, uint32_t v, uint32_t *argb)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    const uint8_t *p;
    uint32_t fmt;

    if (!s_gpu.tex.valid)
        return 0;
    u = wrap_coord(u, s_gpu.tex.width,  s_gpu.tex.addr_u);
    v = wrap_coord(v, s_gpu.tex.height, s_gpu.tex.addr_v);

    fmt = s_gpu.tex.color;
    if (d3d8_format_dxt_block_bytes(fmt))
        return d3d8_dxt_decode_texel(mem + s_gpu.tex.offset, fmt, u, v,
                                     s_gpu.tex.width, argb);
    if (d3d8_format_is_swizzled(fmt)) {
        /* Morton order: a texel's index is interleaved from x and y instead of
         * v*pitch + u, so index from the base of the image. The switch below
         * casts to each format's own width, which makes that index a texel
         * index for every one of them. */
        fmt = linear_twin(fmt);
        p = mem + s_gpu.tex.offset;
        u = swizzle_offset(u, v, s_gpu.tex.width, s_gpu.tex.height);
    } else {
        p = mem + s_gpu.tex.offset + (size_t)v * s_gpu.tex.pitch;
    }

    switch (fmt) {

    /* 32-bit, alpha-red-green-blue in the dword. */
    case 0x12:                                      /* LIN_A8R8G8B8 */
        *argb = ((const uint32_t *)p)[u];
        return 1;
    case 0x1E:                                      /* LIN_X8R8G8B8 */
        *argb = ((const uint32_t *)p)[u] | 0xFF000000u;
        return 1;

    /* 32-bit, other channel orders. The name gives the byte order from the
     * top of the dword down, so each is a permutation of the same four. */
    case 0x3F: {                                    /* LIN_A8B8G8R8 */
        uint32_t t = ((const uint32_t *)p)[u];
        *argb = (t & 0xFF00FF00u) | ((t & 0xFF) << 16) | ((t >> 16) & 0xFF);
        return 1;
    }
    case 0x40: {                                    /* LIN_B8G8R8A8 */
        uint32_t t = ((const uint32_t *)p)[u];
        *argb = ((t & 0xFFu) << 24)                 /* A, from the bottom */
              | (((t >>  8) & 0xFFu) << 16)         /* R */
              | (((t >> 16) & 0xFFu) <<  8)         /* G */
              |  ((t >> 24) & 0xFFu);               /* B, from the top */
        return 1;
    }
    case 0x41: {                                    /* LIN_R8G8B8A8 */
        uint32_t t = ((const uint32_t *)p)[u];
        *argb = ((t & 0xFFu) << 24) | (t >> 8);
        return 1;
    }

    /* 16-bit. */
    case 0x10: {                                    /* LIN_A1R5G5B5 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = ((t & 0x8000u) ? 0xFF000000u : 0u)
              | (expand((t >> 10) & 0x1F, 5) << 16)
              | (expand((t >>  5) & 0x1F, 5) <<  8)
              |  expand( t        & 0x1F, 5);
        return 1;
    }
    case 0x1C: {                                    /* LIN_X1R5G5B5 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = 0xFF000000u
              | (expand((t >> 10) & 0x1F, 5) << 16)
              | (expand((t >>  5) & 0x1F, 5) <<  8)
              |  expand( t        & 0x1F, 5);
        return 1;
    }
    case 0x11: {                                    /* LIN_R5G6B5 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = 0xFF000000u
              | (expand((t >> 11) & 0x1F, 5) << 16)
              | (expand((t >>  5) & 0x3F, 6) <<  8)
              |  expand( t        & 0x1F, 5);
        return 1;
    }
    case 0x1D: {                                    /* LIN_A4R4G4B4 */
        uint32_t t = ((const uint16_t *)p)[u];
        *argb = (expand((t >> 12) & 0x0F, 4) << 24)
              | (expand((t >>  8) & 0x0F, 4) << 16)
              | (expand((t >>  4) & 0x0F, 4) <<  8)
              |  expand( t        & 0x0F, 4);
        return 1;
    }

    /* Packed 4:2:2 YUV, two bytes a pixel. EA's MAD video player uploads
     * every decoded frame this way; unsampled, the intro movie was a blank
     * screen. 0x24 is CR8YB8CB8YA8 (bytes Y0 Cb Y1 Cr, YUY2), 0x25 is
     * YB8CR8YA8CB8 (bytes Cb Y0 Cr Y1, UYVY). BT.601 studio range. */
    case 0x24:
    case 0x25: {
        const uint8_t *q = p + (size_t)(u & ~1u) * 2;
        int y, cb, cr, c, d, e, r, g, b;
        if (fmt == 0x24) {
            y = (u & 1) ? q[2] : q[0]; cb = q[1]; cr = q[3];
        } else {
            y = (u & 1) ? q[3] : q[1]; cb = q[0]; cr = q[2];
        }
        c = y - 16; d = cb - 128; e = cr - 128;
        r = (298 * c + 409 * e + 128) >> 8;
        g = (298 * c - 100 * d - 208 * e + 128) >> 8;
        b = (298 * c + 516 * d + 128) >> 8;
        r = r < 0 ? 0 : r > 255 ? 255 : r;
        g = g < 0 ? 0 : g > 255 ? 255 : g;
        b = b < 0 ? 0 : b > 255 ? 255 : b;
        *argb = 0xFF000000u | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
        return 1;
    }

    /* 8-bit. */
    case 0x0B:                                      /* P8, swizzled */
        if (!s_gpu.tex.palette)
            return 0;
        *argb = ((const uint32_t *)(mem + s_gpu.tex.palette))[p[u]];
        return 1;
    case 0x13: {                                    /* LIN_L8 */
        uint32_t t = p[u];
        *argb = 0xFF000000u | (t << 16) | (t << 8) | t;
        return 1;
    }
    case 0x1F:                                      /* LIN_A8 */
        *argb = ((uint32_t)p[u] << 24) | 0x00FFFFFFu;
        return 1;

    default:
        return 0;
    }
}

/* Write a bound texture out as a BMP, through the sampler rather than around it.
 *
 * "Which texture is this" is not answerable from an address and a format, and
 * it is the question behind most of the ones that matter -- is that a font
 * page or an icon atlas, did the swizzle decode, is the alpha inverted. Going
 * through sample_texture means the file shows exactly what the rasteriser
 * sees, so a decode bug appears here rather than only as a wrong-looking
 * triangle.
 *
 * ponytail: RGB only, alpha dropped. A glyph page is alpha and would come out
 * black, so alpha is composited onto mid-grey to stay legible; that is a
 * viewing choice, not a decode. One file per distinct texture, first use only.
 */
static void dump_texture_bmp(uint32_t seq)
{
    const char *prefix = getenv("RECOMP_TEX_DUMP");
    uint32_t w = s_gpu.tex.width, h = s_gpu.tex.height, x, y;
    uint32_t row_bytes, pad, filesz;
    uint8_t hdr[54];
    char path[512];
    FILE *f;

    if (!prefix || !w || !h || w > 4096 || h > 4096)
        return;
    row_bytes = w * 3;
    pad = (4 - (row_bytes & 3)) & 3;
    filesz = 54 + (row_bytes + pad) * h;

    snprintf(path, sizeof path, "%s%02u_%08X_fmt%02X.bmp",
             prefix, seq, s_gpu.tex.offset, s_gpu.tex.color);
    f = fopen(path, "wb");
    if (!f)
        return;
    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &filesz, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1; hdr[28] = 24;
    fwrite(hdr, 1, sizeof hdr, f);

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            uint32_t argb = 0, a;
            uint8_t px[3];
            if (!sample_texture(x, h - 1 - y, &argb))
                argb = 0;
            a = (argb >> 24) & 0xFFu;
            /* over mid-grey, so an alpha-only page is visible either way */
            px[0] = (uint8_t)(((argb & 0xFFu) * a + 128u * (255u - a)) / 255u);
            px[1] = (uint8_t)((((argb >> 8) & 0xFFu) * a + 128u * (255u - a)) / 255u);
            px[2] = (uint8_t)((((argb >> 16) & 0xFFu) * a + 128u * (255u - a)) / 255u);
            fwrite(px, 1, 3, f);
        }
        if (pad) {
            static const uint8_t zero[3] = {0, 0, 0};
            fwrite(zero, 1, pad, f);
        }
    }
    fclose(f);
    fprintf(stderr, "  [TEXDUMP] %s (%ux%u fmt 0x%02X)\n",
            path, w, h, s_gpu.tex.color);
    fflush(stderr);
}

/* The surface, resolved once per batch.
 *
 * dma_resolve consults the contiguous arena's high-water mark and
 * surface_hits_image walks the image range; both were being done per pixel
 * -- dma_resolve twice -- which cost more than the rasterisation they
 * guarded. Neither answer can change inside a batch, because the colour
 * offset arrives as a method and a method cannot arrive mid-triangle.
 *
 * This is not a micro-optimisation for its own sake: the loader's video
 * paces on frames actually presented, so the rasteriser's throughput is the
 * playback rate. */
static uint8_t *s_surface;          /* host address of surface row 0 */

static int surface_begin_batch(const uint8_t *mem)
{
    uint32_t base = dma_resolve(s_gpu.color_offset);

    if (surface_hits_image(base, (s_gpu.clip_y + s_gpu.clip_h) * s_gpu.pitch))
        return 0;
    s_surface = (uint8_t *)mem + base;
    return 1;
}

/* x and y are surface samples, not displayed pixels -- see raster_triangle. */
/* One blend factor, 0..255, for the channel at bit `shift` (24 = alpha).
 * The values are the GL enums the NV2A takes. */
static int32_t blend_term(uint32_t f, uint32_t src, uint32_t dst, int shift)
{
    const int32_t sa = (int32_t)(src >> 24), da = (int32_t)(dst >> 24);
    const int32_t sc = (int32_t)((src >> shift) & 0xFF);
    const int32_t dc = (int32_t)((dst >> shift) & 0xFF);
    const int32_t cc = (int32_t)((s_gpu.blend_color >> shift) & 0xFF);
    const int32_t ca = (int32_t)(s_gpu.blend_color >> 24);
    switch (f) {
    case NV_BLEND_ZERO:                     return 0;
    case NV_BLEND_ONE:                      return 255;
    case 0x0300: /* SRC_COLOR */            return sc;
    case 0x0301: /* ONE_MINUS_SRC_COLOR */  return 255 - sc;
    case NV_BLEND_SRC_ALPHA:                return sa;
    case NV_BLEND_ONE_MINUS_SRC_ALPHA:      return 255 - sa;
    case 0x0304: /* DST_ALPHA */            return da;
    case 0x0305: /* ONE_MINUS_DST_ALPHA */  return 255 - da;
    case 0x0306: /* DST_COLOR */            return dc;
    case 0x0307: /* ONE_MINUS_DST_COLOR */  return 255 - dc;
    case 0x0308: /* SRC_ALPHA_SATURATE */
        return shift == 24 ? 255 : (sa < 255 - da ? sa : 255 - da);
    case 0x8001: /* CONSTANT_COLOR */           return cc;
    case 0x8002: /* ONE_MINUS_CONSTANT_COLOR */ return 255 - cc;
    case 0x8003: /* CONSTANT_ALPHA */           return ca;
    case 0x8004: /* ONE_MINUS_CONSTANT_ALPHA */ return 255 - ca;
    default:                                return 255;
    }
}

static int put_pixel(uint8_t *mem, uint32_t bpp, int x, int y, uint32_t argb)
{
    uint8_t *row;

    (void)mem;
    if (x < (int)s_gpu.clip_x || x >= (int)(s_gpu.clip_x + surface_sample_w()))
        return 0;
    if (y < (int)s_gpu.clip_y || y >= (int)(s_gpu.clip_y + surface_sample_h()))
        return 0;
    /* Alpha test, before anything is written: cut-out textures (fences,
     * hair, crowd sprites) rely on it rather than on blending. */
    if (s_gpu.alpha_test) {
        const uint32_t a = argb >> 24, ref = s_gpu.alpha_ref;
        int pass;
        switch (s_gpu.alpha_func) {
        case 0x0200: pass = 0; break;                  /* NEVER */
        case 0x0201: pass = a <  ref; break;
        case 0x0202: pass = a == ref; break;
        case 0x0203: pass = a <= ref; break;
        case 0x0204: pass = a >  ref; break;
        case 0x0205: pass = a != ref; break;
        case 0x0206: pass = a >= ref; break;
        default:     pass = 1; break;                  /* ALWAYS */
        }
        if (!pass)
            return 0;
    }
    s_gpu.pixels++;
    if ((argb & 0x00FFFFFFu) > (s_gpu.pixel_max & 0x00FFFFFFu))
        s_gpu.pixel_max = argb;
    row = s_surface + (size_t)y * s_gpu.pitch;

    /* Source-over, when the title asked for blending.
     *
     * ponytail: the factors themselves (SET_BLEND_FUNC_SFACTOR / DFACTOR) are
     * not decoded, so this assumes SRC_ALPHA / ONE_MINUS_SRC_ALPHA. That is
     * what a UI layer uses, and it is the difference between a logo and a
     * white rectangle; additive and modulate passes stay wrong until the
     * factors are read. An opaque texel takes the plain store below, so an
     * opaque pass costs nothing. */
    if (s_gpu.blend_enable && bpp == 4) {
        const uint32_t sf = s_gpu.blend_sfactor, df = s_gpu.blend_dfactor;
        if (sf == NV_BLEND_SRC_ALPHA && df == NV_BLEND_ONE_MINUS_SRC_ALPHA) {
            /* Source-over: the common case keeps its shortcuts. */
            uint32_t sa = argb >> 24;
            if (sa == 0)
                return 0;                    /* fully transparent */
            if (sa != 0xFF) {
                uint32_t d  = ((const uint32_t *)row)[x];
                uint32_t ia = 255u - sa;
                uint32_t r = (((argb >> 16) & 0xFF) * sa + ((d >> 16) & 0xFF) * ia) / 255u;
                uint32_t g = (((argb >>  8) & 0xFF) * sa + ((d >>  8) & 0xFF) * ia) / 255u;
                uint32_t b = (( argb        & 0xFF) * sa + ( d        & 0xFF) * ia) / 255u;
                ((uint32_t *)row)[x] = 0xFF000000u | (r << 16) | (g << 8) | b;
                return 1;
            }
        } else {
            /* Everything else, per channel, from the factors the title set.
             * Def Jam draws its fight vignette as a white-centred texture in
             * multiply mode; treated as source-over it painted the screen
             * white over the whole arena. */
            const uint32_t d = ((const uint32_t *)row)[x];
            uint32_t out = 0;
            int k;
            for (k = 0; k < 32; k += 8) {
                int32_t sc = (int32_t)((argb >> k) & 0xFF);
                int32_t dc = (int32_t)((d >> k) & 0xFF);
                int32_t v = blend_term(sf, argb, d, k) * sc
                          + blend_term(df, argb, d, k) * dc;
                switch (s_gpu.blend_equation) {
                case NV_BLEND_EQ_SUBTRACT:
                    v = blend_term(sf, argb, d, k) * sc - blend_term(df, argb, d, k) * dc;
                    break;
                case NV_BLEND_EQ_REVERSE_SUBTRACT:
                    v = blend_term(df, argb, d, k) * dc - blend_term(sf, argb, d, k) * sc;
                    break;
                default:
                    break;
                }
                v = (v + 127) / 255;
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                out |= (uint32_t)v << k;
            }
            ((uint32_t *)row)[x] = out;
            return 1;
        }
    }

    if (bpp == 4) {
        ((uint32_t *)row)[x] = argb;
    } else if (bpp == 2) {
        ((uint16_t *)row)[x] = (uint16_t)(((argb >> 8) & 0xF800)
                                        | ((argb >> 5) & 0x07E0)
                                        | ((argb >> 3) & 0x001F));
    }
    return 1;
}

/* Half-space fill. Barycentric edge functions rather than scanline slopes:
 * the same test decides both windings, so a title that emits clockwise
 * triangles does not silently render nothing. */
static void raster_triangle(const float a[2], const float b[2],
                            const float c[2], uint32_t argb,
                            const float uv[3][2])
{
    uint8_t *mem = (uint8_t *)xbox_GetMemoryOffset();
    uint32_t bpp = surface_bpp();
    float area;
    int minx, maxx, miny, maxy, x, y;
    int textured = uv && s_gpu.tex.valid;

    /* Vertices arrive in displayed pixels; the surface is addressed in
     * samples. On a multisampled target the two differ, and writing pixel
     * coordinates straight into a 2x-wide surface puts the whole frame in the
     * left half of it -- which the display resolve then squeezes into the left
     * half of the screen. Scaling here rather than at every call site keeps
     * the barycentric weights unchanged, so texturing is unaffected. */
    const float sx = (float)surface_aa_x(), sy = (float)surface_aa_y();
    const float a2[2] = { a[0] * sx, a[1] * sy };
    const float b2[2] = { b[0] * sx, b[1] * sy };
    const float c2[2] = { c[0] * sx, c[1] * sy };

    a = a2; b = b2; c = c2;

    if (bpp != 4 && bpp != 2)
        return;

    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (area == 0.0f)
        return;                            /* degenerate */

    /* Where this batch writes. The same check the per-pixel path made, made
     * once: a surface address landing on the title's own image is no safer
     * one pixel at a time than 4.9 MB at once. */
    if (!surface_begin_batch(mem))
        return;

    minx = (int)floorf(fminf(a[0], fminf(b[0], c[0])));
    maxx = (int)ceilf (fmaxf(a[0], fmaxf(b[0], c[0])));
    miny = (int)floorf(fminf(a[1], fminf(b[1], c[1])));
    maxy = (int)ceilf (fmaxf(a[1], fmaxf(b[1], c[1])));

    if (minx < (int)s_gpu.clip_x) minx = (int)s_gpu.clip_x;
    if (miny < (int)s_gpu.clip_y) miny = (int)s_gpu.clip_y;
    if (maxx > (int)(s_gpu.clip_x + surface_sample_w()))
        maxx = (int)(s_gpu.clip_x + surface_sample_w());
    if (maxy > (int)(s_gpu.clip_y + surface_sample_h()))
        maxy = (int)(s_gpu.clip_y + surface_sample_h());
    if (minx >= maxx || miny >= maxy) {
        s_gpu.tris_skipped_offscreen++;
        return;
    }

    for (y = miny; y < maxy; y++) {
        for (x = minx; x < maxx; x++) {
            float px = (float)x + 0.5f, py = (float)y + 0.5f;
            float w0 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]);
            float w1 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0]);
            float w2 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0]);
            uint32_t color = argb;
            float z = 0.0f;
            /* Barycentric weights of a, b, c. */
            float ba = w1 / area, bb = w2 / area, bc = w0 / area;
            if (!((w0 >= 0 && w1 >= 0 && w2 >= 0)
               || (w0 <= 0 && w1 <= 0 && w2 <= 0)))
                continue;
            if (s_tri_persp) {
                /* Perspective: weight by 1/w and renormalise. */
                float pa = ba * s_tri_invw[0], pb = bb * s_tri_invw[1], pc = bc * s_tri_invw[2];
                float sum = pa + pb + pc;
                if (sum != 0.0f) {
                    ba = pa / sum; bb = pb / sum; bc = pc / sum;
                }
                {
                    uint32_t k, m = 0;
                    for (k = 0; k < 32; k += 8) {
                        float v = ba * (float)((s_tri_col[0] >> k) & 0xFF)
                                + bb * (float)((s_tri_col[1] >> k) & 0xFF)
                                + bc * (float)((s_tri_col[2] >> k) & 0xFF);
                        int iv = (int)(v + 0.5f);
                        if (iv < 0) iv = 0;
                        if (iv > 255) iv = 255;
                        m |= (uint32_t)iv << k;
                    }
                    color = m;
                    argb = m;
                }
            }
            if (textured) {
                /* Barycentric, straight from the edge functions already
                 * computed: w1 is the area opposite a, w2 opposite b, w0
                 * opposite c, and the three sum to the whole triangle.
                 * No perspective correction. */
                uint32_t texel;
                float su = ba * uv[0][0] + bb * uv[1][0] + bc * uv[2][0];
                float sv = ba * uv[0][1] + bb * uv[1][1] + bc * uv[2][1];
                if (su < 0.0f) su = 0.0f;
                if (sv < 0.0f) sv = 0.0f;
                if (sample_texture((uint32_t)su, (uint32_t)sv, &texel)) {
                    /* Modulate by the vertex colour, the default stage
                     * setup. EA APT draws its text as a white glyph page
                     * tinted by the vertex colour. */
                    if (argb != 0xFFFFFFFFu) {
                        uint32_t k, m = 0;
                        for (k = 0; k < 32; k += 8)
                            m |= ((((texel >> k) & 0xFF) * ((argb >> k) & 0xFF) + 127) / 255) << k;
                        texel = m;
                    }
                    color = texel;
                }
            }
            if (s_tri_has_z)
                z = (float)(((double)w1 * s_tri_z[0] + (double)w2 * s_tri_z[1]
                             + (double)w0 * s_tri_z[2]) / area);
            if (!fragment_tests(x, y, z, color))
                continue;
            if (s_gpu.color_mask & 0x01010101u)
                put_pixel(mem, bpp, x, y, color);
        }
    }
    if (s_tri_has_z && s_gpu.depth_mask && s_tri_z[0] < 1.0f && getenv("RECOMP_ZCLEAR_TRACE")) {
        static uint32_t last_flip = 0xFFFFFFFFu;
        if (last_flip != s_gpu.flips) {
            last_flip = s_gpu.flips;
            fprintf(stderr, "  [ZWRITE0] f%u tri (%.1f,%.1f)-(%.1f,%.1f) tex 0x%08X fmt %02X textured %d test %u func %X argb %08X\n",
                    s_gpu.flips, a[0], a[1], b[0], b[1], s_gpu.tex.offset, s_gpu.tex.color,
                    textured, s_gpu.depth_test, s_gpu.depth_func, argb);
        }
    }
    s_gpu.tris_drawn++;
    s_gpu.drawn_offset = s_gpu.color_offset;
}

/* Attribute 3 is diffuse colour in every NV2A layout that sets one. Absent it,
 * white -- a visible wrong colour beats an invisible correct one during
 * bring-up. */
/* Which attribute carries the colour.
 *
 * Slot 3 is diffuse by convention and titles that follow it are read straight
 * from there. Half-Life 2 does not: its vertex is position, colour, texcoord
 * at stride 24, and the colour arrives in slot 5. So fall back to the format
 * rather than the slot number -- D3DCOLOR is the one attribute type that is
 * only ever a colour, which makes it a stronger signal than the convention. */
static const VertexAttr *color_attr(void)
{
    uint32_t a;

    if (s_gpu.attr[3].offset && s_gpu.attr[3].stride)
        return &s_gpu.attr[3];
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].type == 0 && s_gpu.attr[a].size == 4
                && s_gpu.attr[a].offset && s_gpu.attr[a].stride)
            return &s_gpu.attr[a];
    return &s_gpu.attr[3];
}

/* Attribute 9 is texture coordinate 0 in the NV2A vertex layout, the same way
 * 0 is position and 3 is diffuse -- for a title that follows the convention.
 *
 * Half-Life 2 does not, in either place. Its menu and HUD vertex is position,
 * colour, texcoord at stride 24, with the colour in slot 5 and the texcoords
 * in slot 7, so reading slot 9 found nothing and every batch drew untextured.
 * That is invisible rather than wrong-looking: the menu paints a full-screen
 * quad and then draws its text over it, and with no sampling both come out
 * white, so the screen is blank white and nothing suggests the text was ever
 * drawn.
 *
 * Falling back to the format works because the three attributes of such a
 * vertex are distinguishable: position is float3, colour is D3DCOLOR, and a
 * float2 is a texture coordinate and nothing else.
 *
 * ponytail: takes the first float2 it finds, so a title with two texcoord sets
 * gets stage 0's -- which is what this single-texture rasteriser samples
 * anyway. Multi-texture wants the D3D11 translator, not another heuristic. */
static const VertexAttr *texcoord_attr(void)
{
    uint32_t a;

    if (s_gpu.attr[9].offset && s_gpu.attr[9].stride)
        return &s_gpu.attr[9];
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].type == 2 && s_gpu.attr[a].size == 2
                && s_gpu.attr[a].offset && s_gpu.attr[a].stride)
            return &s_gpu.attr[a];
    return &s_gpu.attr[9];
}

/* Texel coordinates, whichever convention the title used.
 *
 * The two are not interchangeable and the format decides which is in force: a
 * swizzled texture is addressed in [0,1], a linear one in texels. Both are
 * scaled to texels here so that everything downstream -- the barycentric
 * interpolation and the sampler -- works in one unit.
 *
 * This mattered the moment swizzled formats became samplable. Normalised
 * coordinates truncated to a texel index land on texel 0 for any coordinate
 * below 1.0, so a whole quad sampled a single texel and came out flat: the
 * background painted one near-black colour, which looks like a texture that
 * decoded wrong rather than one that was never indexed. */
static int fetch_texcoord(uint32_t index, float out[2])
{
    float t[4];

    if (!fetch_attr(texcoord_attr(), index, t))
        return 0;
    out[0] = t[0];
    out[1] = t[1];
    if (tex_size_from_format(s_gpu.tex.color)) {
        out[0] *= (float)s_gpu.tex.width;
        out[1] *= (float)s_gpu.tex.height;
    }
    return 1;
}

static uint32_t vertex_color(uint32_t index)
{
    float c[4];

    if (!fetch_attr(color_attr(), index, c))
        return 0xFFFFFFFFu;
    return ((uint32_t)(c[3] * 255.0f) << 24)
         | ((uint32_t)(c[0] * 255.0f) << 16)
         | ((uint32_t)(c[1] * 255.0f) <<  8)
         |  (uint32_t)(c[2] * 255.0f);
}

/* An untransformed batch drawn as if it were screen space smears a few pixels
 * into the corner, so the batch has to be classified before it is rasterised.
 *
 * This used to demand that every vertex land inside the surface, which is a
 * different question and the wrong one: geometry that extends past the
 * viewport is ordinary, and clipping it is raster_triangle's job (it clamps
 * its span to the clip rect). The dashboard is exactly the case that exposed
 * it -- a full-screen pass drawn as one oversized triangle, vertices at
 * (-0.5,-0.5), (2*w,-0.5), (-0.5,2*h), all correct and all rejected.
 *
 * What actually separates the two is scale. Object-space positions are model
 * units, a handful either side of the origin; screen-space ones are measured
 * in pixels of a surface hundreds of pixels wide. So: the batch has to be able
 * to touch the surface at all, and it has to be bigger than object space.
 *
 * ponytail: a genuinely tiny screen-space sprite reads as object space and is
 * skipped. It is counted as skipped rather than silently dropped, and the
 * unambiguous answer needs the vertex-program state, which is not tracked yet.
 */
#define OBJECT_SPACE_SPAN 8.0f

static int batch_is_screen_space(void)
{
    float p[4], lo_x, hi_x, lo_y, hi_y;
    uint32_t i;

    if (!s_gpu.clip_w || !s_gpu.clip_h || !s_gpu.idx_count)
        return 0;
    if (!fetch_attr(&s_gpu.attr[0], s_gpu.idx[0], p))
        return 0;
    lo_x = hi_x = p[0];
    lo_y = hi_y = p[1];
    for (i = 1; i < s_gpu.idx_count; i++) {
        if (!fetch_attr(&s_gpu.attr[0], s_gpu.idx[i], p))
            return 0;
        if (p[0] < lo_x) lo_x = p[0];
        if (p[0] > hi_x) hi_x = p[0];
        if (p[1] < lo_y) lo_y = p[1];
        if (p[1] > hi_y) hi_y = p[1];
    }

    /* Entirely off the surface: nothing to draw under either reading. */
    if (hi_x < (float)s_gpu.clip_x
     || lo_x > (float)(s_gpu.clip_x + s_gpu.clip_w)
     || hi_y < (float)s_gpu.clip_y
     || lo_y > (float)(s_gpu.clip_y + s_gpu.clip_h))
        return 0;

    /* Small enough to be model units rather than pixels. */
    if (hi_x - lo_x < OBJECT_SPACE_SPAN && hi_y - lo_y < OBJECT_SPACE_SPAN)
        return 0;

    return 1;
}

/* NV097 primitive types that are triangles under some winding.
 *
 * SET_BEGIN_END takes the D3D primitive type unchanged: 1 points, 2 lines,
 * 3 line loop, 4 line strip, 5 triangles, 6 strip, 7 fan, 8 quads, 9 quad
 * strip, 10 polygon. These were once numbered from 4, one too low, so every
 * strip was drawn as a fan -- a four-vertex sprite then lost a corner, and
 * a longer strip turned into slivers radiating from its first vertex. */
#define NV_PRIM_TRIANGLES      5
#define NV_PRIM_TRIANGLE_STRIP 6
#define NV_PRIM_TRIANGLE_FAN   7
#define NV_PRIM_QUADS          8
#define NV_PRIM_QUAD_STRIP     9
#define NV_PRIM_POLYGON        10

/* How many post-draw captures to keep: enough to see whether the geometry
 * is stable from frame to frame, few enough not to fill a directory. */
#define FB_DUMP_AFTER_DRAW 8
static int s_drawn_dumps;

static void dump_surface_bmp(void);

/* One triangle by vertex index: gather position and, if the batch has one,
 * texture coordinate 0. A vertex whose position cannot be read is not drawn;
 * a batch whose texcoords cannot be read is drawn untextured rather than not
 * at all, so a missing coordinate stream costs the colour and not the shape.
 */
static void raster_indexed(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t argb)
{
    float p[3][4], uv[3][2];
    int textured;

    if (!fetch_attr(&s_gpu.attr[0], i0, p[0])
     || !fetch_attr(&s_gpu.attr[0], i1, p[1])
     || !fetch_attr(&s_gpu.attr[0], i2, p[2]))
        return;

    textured = fetch_texcoord(i0, uv[0])
            && fetch_texcoord(i1, uv[1])
            && fetch_texcoord(i2, uv[2]);

    raster_triangle(p[0], p[1], p[2], argb,
                    textured ? (const float (*)[2])uv : NULL);
}

/* ── Vertex programs ─────────────────────────────────────────────
 *
 * Most titles never hand the GPU screen-space vertices: they upload a vertex
 * program (NV097_SET_TRANSFORM_PROGRAM) and object-space positions, and the
 * GPU runs the program per vertex. Def Jam: Fight for NY draws its whole EA
 * APT front end that way, so without running the program nothing but the
 * loading logo reached the screen.
 *
 * This is a plain CPU interpreter of the NV20 microcode. Each instruction is
 * four dwords, loaded in the order the title writes them; dword 0 is unused.
 * Field positions, as {dword, bit, width}, checked against what the title
 * actually uploads (`mov oPos, v0` and `dp4 oPos.z, v0, c98`):
 *
 *   dword 1   ILU op 25/3, MAC op 21/4, const 13/8, input 9/4,
 *             A neg 8, A swizzle 6,4,2,0
 *   dword 2   A reg 28/4, A mux 26/2, B neg 25, B swizzle 23,21,19,17,
 *             B reg 13/4, B mux 11/2, C neg 10, C swizzle 8,6,4,2,
 *             C reg high 0/2
 *   dword 3   C reg low 30/2, C mux 28/2, MAC mask 24/4, temp 20/4,
 *             ILU mask 16/4, output mask 12/4, output register 3/8,
 *             output from ILU 2, relative addressing 1, last 0
 *
 * Muxes: 1 temp, 2 input, 3 constant. Masks have x in bit 3. The MAC result
 * goes to the temp register, the ILU result to R1, and one of the two to the
 * output register. R12 is oPos.
 *
 * ponytail: one texture coordinate and one colour are carried to the
 * rasteriser, because that is all it samples. */
#define NV097_SET_TRANSFORM_EXECUTION_MODE 0x1E94
#define NV097_SET_TRANSFORM_PROGRAM_LOAD   0x1E9C
#define NV097_SET_TRANSFORM_PROGRAM_START  0x1EA0
#define NV097_SET_TRANSFORM_CONSTANT_LOAD  0x1EA4
#define NV097_SET_TRANSFORM_PROGRAM        0x0B00   /* 32 dwords */
#define NV097_SET_TRANSFORM_CONSTANT       0x0B80   /* 16 dwords */
#define NV097_SET_VIEWPORT_OFFSET          0x0A20   /* 4 floats */
#define NV097_SET_VIEWPORT_SCALE           0x0AF0   /* 4 floats */

#define VP_MAX_INSNS  136
#define VP_CONSTANTS  192
#define VP_OUT_POS    0
#define VP_OUT_D0     3
#define VP_OUT_T0     9

static struct {
    uint32_t prog[VP_MAX_INSNS][4];
    uint32_t load, start, mode;
    float    c[VP_CONSTANTS][4];
    uint32_t c_load;
    uint32_t runs, bad_insns;
} s_vp;

static int vp_active(void)
{
    return (s_vp.mode & 3) == 2;
}

static int vp_method(uint32_t method, uint32_t param)
{
    if (method >= NV097_SET_TRANSFORM_PROGRAM
            && method < NV097_SET_TRANSFORM_PROGRAM + 32 * 4) {
        uint32_t slot = (method - NV097_SET_TRANSFORM_PROGRAM) / 4;
        if (s_vp.load < VP_MAX_INSNS) {
            s_vp.prog[s_vp.load][slot % 4] = param;
            if (slot % 4 == 3)
                s_vp.load++;
        }
        return 1;
    }
    if (method >= NV097_SET_TRANSFORM_CONSTANT
            && method < NV097_SET_TRANSFORM_CONSTANT + 16 * 4) {
        uint32_t slot = (method - NV097_SET_TRANSFORM_CONSTANT) / 4;
        if (s_vp.c_load < VP_CONSTANTS) {
            memcpy(&s_vp.c[s_vp.c_load][slot % 4], &param, 4);
            if (slot % 4 == 3)
                s_vp.c_load++;
        }
        return 1;
    }
    /* The viewport is not a constant upload but lands in the same table:
     * the scale in c58 and the offset in c59, which is where the transform
     * the runtime appends to every program reads them. */
    if (method >= NV097_SET_VIEWPORT_OFFSET && method < NV097_SET_VIEWPORT_OFFSET + 16) {
        memcpy(&s_vp.c[59][(method - NV097_SET_VIEWPORT_OFFSET) / 4], &param, 4);
        return 1;
    }
    if (method >= NV097_SET_VIEWPORT_SCALE && method < NV097_SET_VIEWPORT_SCALE + 16) {
        memcpy(&s_vp.c[58][(method - NV097_SET_VIEWPORT_SCALE) / 4], &param, 4);
        return 1;
    }
    switch (method) {
    case NV097_SET_TRANSFORM_EXECUTION_MODE: s_vp.mode = param;   return 1;
    case NV097_SET_TRANSFORM_PROGRAM_LOAD:   s_vp.load = param;   return 1;
    case NV097_SET_TRANSFORM_PROGRAM_START:  s_vp.start = param;  return 1;
    case NV097_SET_TRANSFORM_CONSTANT_LOAD:  s_vp.c_load = param; return 1;
    default: return 0;
    }
}

#define VP_BITS(w, pos, len) (((w) >> (pos)) & ((1u << (len)) - 1))

typedef struct {
    float r[13][4];          /* R0..R11, R12 = oPos (kept in o[0]) */
    float o[16][4];
    float v[16][4];
    int   a0;
} VpState;

static float *vp_temp(VpState *s, uint32_t i)
{
    return i == 12 ? s->o[VP_OUT_POS] : s->r[i < 12 ? i : 0];
}

static void vp_read(VpState *s, const uint32_t *w, uint32_t mux, uint32_t reg,
                    int neg, uint32_t swz, float out[4])
{
    const float *src;
    static const float zero[4];
    uint32_t k;

    switch (mux) {
    case 1: src = vp_temp(s, reg); break;
    case 2: src = s->v[VP_BITS(w[1], 9, 4)]; break;
    case 3: {
        int idx = (int)VP_BITS(w[1], 13, 8);
        if (VP_BITS(w[3], 1, 1))
            idx += s->a0;
        src = (idx >= 0 && idx < VP_CONSTANTS) ? s_vp.c[idx] : zero;
        break;
    }
    default: src = zero; break;
    }
    for (k = 0; k < 4; k++) {
        float x = src[(swz >> (6 - 2 * k)) & 3];
        out[k] = neg ? -x : x;
    }
}

static void vp_write(float *dst, const float val[4], uint32_t mask)
{
    uint32_t k;
    for (k = 0; k < 4; k++)
        if (mask & (8u >> k))
            dst[k] = val[k];
}

/* Run the current program for one vertex. Returns 0 if it has no end. */
static int vp_run(VpState *s)
{
    uint32_t pc;

    for (pc = s_vp.start; pc < VP_MAX_INSNS; pc++) {
        const uint32_t *w = s_vp.prog[pc];
        uint32_t mac = VP_BITS(w[1], 21, 4), ilu = VP_BITS(w[1], 25, 3);
        float a[4], b[4], c[4], m[4] = {0, 0, 0, 0}, l[4] = {0, 0, 0, 0};
        float d;
        uint32_t k;

        vp_read(s, w, VP_BITS(w[2], 26, 2), VP_BITS(w[2], 28, 4),
                (int)VP_BITS(w[1], 8, 1), VP_BITS(w[1], 0, 8), a);
        vp_read(s, w, VP_BITS(w[2], 11, 2), VP_BITS(w[2], 13, 4),
                (int)VP_BITS(w[2], 25, 1), VP_BITS(w[2], 17, 8), b);
        vp_read(s, w, VP_BITS(w[3], 28, 2),
                (VP_BITS(w[2], 0, 2) << 2) | VP_BITS(w[3], 30, 2),
                (int)VP_BITS(w[2], 10, 1), VP_BITS(w[2], 2, 8), c);

        switch (mac) {
        case 0: break;                                      /* NOP */
        case 1: memcpy(m, a, sizeof m); break;              /* MOV */
        case 2: for (k = 0; k < 4; k++) m[k] = a[k] * b[k]; break;
        case 3: for (k = 0; k < 4; k++) m[k] = a[k] + b[k]; break;
        case 4: for (k = 0; k < 4; k++) m[k] = a[k] * b[k] + c[k]; break;
        case 5:                                             /* DP3 */
            d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            m[0] = m[1] = m[2] = m[3] = d;
            break;
        case 6:                                             /* DPH */
            d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + b[3];
            m[0] = m[1] = m[2] = m[3] = d;
            break;
        case 7:                                             /* DP4 */
            d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
            m[0] = m[1] = m[2] = m[3] = d;
            break;
        case 8:                                             /* DST */
            m[0] = 1.0f; m[1] = a[1] * b[1]; m[2] = a[2]; m[3] = b[3];
            break;
        case 9:  for (k = 0; k < 4; k++) m[k] = a[k] < b[k] ? a[k] : b[k]; break;
        case 10: for (k = 0; k < 4; k++) m[k] = a[k] > b[k] ? a[k] : b[k]; break;
        case 11: for (k = 0; k < 4; k++) m[k] = a[k] <  b[k] ? 1.0f : 0.0f; break;
        case 12: for (k = 0; k < 4; k++) m[k] = a[k] >= b[k] ? 1.0f : 0.0f; break;
        case 13: s->a0 = (int)floorf(a[0] + 0.001f); break; /* ARL */
        default: s_vp.bad_insns++; break;
        }

        switch (ilu) {                                      /* scalar: c.x */
        case 0: break;
        case 1: memcpy(l, c, sizeof l); break;              /* MOV */
        case 2:                                             /* RCP */
            d = c[0] != 0.0f ? 1.0f / c[0] : 3.4e38f;
            l[0] = l[1] = l[2] = l[3] = d;
            break;
        case 3:                                             /* RCC */
            d = c[0] != 0.0f ? 1.0f / c[0] : 1.84467e19f;
            if (d > 0) d = d < 5.42101e-20f ? 5.42101e-20f : d > 1.84467e19f ? 1.84467e19f : d;
            else d = d > -5.42101e-20f ? -5.42101e-20f : d < -1.84467e19f ? -1.84467e19f : d;
            l[0] = l[1] = l[2] = l[3] = d;
            break;
        case 4:                                             /* RSQ */
            d = fabsf(c[0]);
            d = d != 0.0f ? 1.0f / sqrtf(d) : 3.4e38f;
            l[0] = l[1] = l[2] = l[3] = d;
            break;
        case 5:                                             /* EXP */
            l[0] = l[1] = l[2] = l[3] = powf(2.0f, c[0]);
            break;
        case 6:                                             /* LOG */
            d = fabsf(c[0]);
            l[0] = l[1] = l[2] = l[3] = d != 0.0f ? log2f(d) : -3.4e38f;
            break;
        case 7: {                                           /* LIT */
            float nl = c[0] > 0 ? c[0] : 0, ex = c[3];
            if (ex < -127.9961f) ex = -127.9961f;
            if (ex > 127.9961f) ex = 127.9961f;
            l[0] = 1.0f;
            l[1] = nl;
            l[2] = c[0] > 0 ? powf(c[1] > 0 ? c[1] : 0, ex) : 0.0f;
            l[3] = 1.0f;
            break;
        }
        }

        if (mac && mac != 13)
            vp_write(vp_temp(s, VP_BITS(w[3], 20, 4)), m, VP_BITS(w[3], 24, 4));
        if (ilu)
            vp_write(s->r[1], l, VP_BITS(w[3], 16, 4));
        {
            uint32_t orb = VP_BITS(w[3], 3, 8);
            if (orb < 16)
                vp_write(s->o[orb], VP_BITS(w[3], 2, 1) ? l : m,
                         VP_BITS(w[3], 12, 4));
        }
        if (w[3] & 1)
            return 1;
    }
    return 0;
}

/* One vertex after the program: position in displayed pixels, colour, and
 * texture coordinate 0 already scaled to texels the way fetch_texcoord does. */
typedef struct {
    float xy[2];
    float z;
    float w;
    float uv[2];
    uint32_t argb;
    int ok;
} VpVertex;

static uint8_t vp_clamp8(float f)
{
    return f <= 0.0f ? 0 : f >= 1.0f ? 255 : (uint8_t)(f * 255.0f + 0.5f);
}

static void vp_transform(uint32_t index, VpVertex *out)
{
    VpState s;
    uint32_t a;
    float w;

    memset(&s, 0, sizeof s);
    s.o[VP_OUT_D0][3] = 1.0f;
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        fetch_attr(&s_gpu.attr[a], index, s.v[a]);

    out->ok = vp_run(&s);
    s_vp.runs++;
    /* The Xbox D3D runtime ends every program with the viewport transform
     * and the divide -- `rcc r1.x, oPos.w` then `mad oPos.xyz, oPos, r1.x,
     * c59` -- so oPos.xy already are window coordinates. w is kept only for
     * perspective-correct interpolation, which this rasteriser does not do. */
    w = s.o[VP_OUT_POS][3];
    if (!(w == w) || !(s.o[VP_OUT_POS][0] == s.o[VP_OUT_POS][0]))
        out->ok = 0;
    /* Behind the eye. There is no clipper here, and a vertex with w <= 0 has
     * been divided into a mirrored position hundreds of thousands of pixels
     * away: the triangle it belongs to then covered the whole screen with
     * garbage and cost a full-surface scan each. Dropping the triangle loses
     * a sliver at the near plane instead.
     *
     * ponytail: real near-plane clipping needs the pre-divide position. */
    /* Vertices behind the eye are kept: raster_batch_program clips the
     * triangles they belong to against the near plane. */
    out->xy[0] = s.o[VP_OUT_POS][0];
    out->xy[1] = s.o[VP_OUT_POS][1];
    out->z = s.o[VP_OUT_POS][2];
    out->w = w;
    out->argb = ((uint32_t)vp_clamp8(s.o[VP_OUT_D0][3]) << 24)
              | ((uint32_t)vp_clamp8(s.o[VP_OUT_D0][0]) << 16)
              | ((uint32_t)vp_clamp8(s.o[VP_OUT_D0][1]) << 8)
              |  (uint32_t)vp_clamp8(s.o[VP_OUT_D0][2]);
    out->uv[0] = s.o[VP_OUT_T0][0];
    out->uv[1] = s.o[VP_OUT_T0][1];
    if (tex_size_from_format(s_gpu.tex.color)) {
        out->uv[0] *= (float)s_gpu.tex.width;
        out->uv[1] *= (float)s_gpu.tex.height;
    }

    if (getenv("RECOMP_VP_TRACE")) {
        static int shown, listed;
        static uint32_t seen_hash[8];
        uint32_t hash = 2166136261u, pc0;
        int known = 0;
        for (pc0 = s_vp.start; pc0 < VP_MAX_INSNS; pc0++) {
            hash = (hash ^ s_vp.prog[pc0][1] ^ s_vp.prog[pc0][2] ^ s_vp.prog[pc0][3]) * 16777619u;
            if (s_vp.prog[pc0][3] & 1)
                break;
        }
        for (pc0 = 0; pc0 < (uint32_t)listed; pc0++)
            known |= seen_hash[pc0] == hash;
        if (!known && listed < 8) {
            uint32_t pc;
            seen_hash[listed++] = hash;
            shown = 0;
            fprintf(stderr, "  [VP] program from %u:\n", s_vp.start);
            for (pc = 0; pc < NV_VERTEX_ATTRS; pc++)
                if (s_gpu.attr[pc].size)
                    fprintf(stderr, "  [VP]   attr%u type %u size %u stride %u off 0x%08X\n",
                            pc, s_gpu.attr[pc].type, s_gpu.attr[pc].size,
                            s_gpu.attr[pc].stride, s_gpu.attr[pc].offset);
            for (pc = s_vp.start; pc < VP_MAX_INSNS; pc++) {
                const uint32_t *iw = s_vp.prog[pc];
                fprintf(stderr, "  [VP]   %3u: %08X %08X %08X %08X  mac %u ilu %u"
                                " v%u c%u  A m%u r%u  B m%u r%u  -> R%u m%X  O%u m%X %s\n",
                        pc, iw[0], iw[1], iw[2], iw[3],
                        VP_BITS(iw[1], 21, 4), VP_BITS(iw[1], 25, 3),
                        VP_BITS(iw[1], 9, 4), VP_BITS(iw[1], 13, 8),
                        VP_BITS(iw[2], 26, 2), VP_BITS(iw[2], 28, 4),
                        VP_BITS(iw[2], 11, 2), VP_BITS(iw[2], 13, 4),
                        VP_BITS(iw[3], 20, 4), VP_BITS(iw[3], 24, 4),
                        VP_BITS(iw[3], 3, 8), VP_BITS(iw[3], 12, 4),
                        VP_BITS(iw[3], 2, 1) ? "ilu" : "mac");
                if (iw[3] & 1)
                    break;
            }
            for (pc = 0; pc < VP_CONSTANTS; pc++)
                if (s_vp.c[pc][0] || s_vp.c[pc][1] || s_vp.c[pc][2] || s_vp.c[pc][3])
                    fprintf(stderr, "  [VP]   c%-3u = %g %g %g %g\n", pc,
                            s_vp.c[pc][0], s_vp.c[pc][1], s_vp.c[pc][2], s_vp.c[pc][3]);
        }
        if (shown < 6) {
            shown++;
            fprintf(stderr, "  [VP] v%u in (%.2f %.2f %.2f %.2f) -> pos (%.2f %.2f %.2f %.2f)"
                            " d0 %08X t0 (%.3f %.3f) ok %d\n",
                    index, s.v[0][0], s.v[0][1], s.v[0][2], s.v[0][3],
                    s.o[0][0], s.o[0][1], s.o[0][2], s.o[0][3],
                    out->argb, s.o[VP_OUT_T0][0], s.o[VP_OUT_T0][1], out->ok);
        }
    }
}

/* Triangles of a batch, as positions into the index list. The fourth entry is
 * the position whose colour a flat-shaded triangle takes. Returns the count. */
static uint32_t batch_triangles(uint32_t prim, uint32_t n,
                                uint32_t (*out)[4], uint32_t max)
{
    uint32_t i, t = 0;
#define TRI(x, y, z, c) do { if (t < max) { out[t][0] = (x); out[t][1] = (y); \
                             out[t][2] = (z); out[t][3] = (c); t++; } } while (0)
    switch (prim) {
    case 5:                                        /* triangles */
        for (i = 0; i + 2 < n; i += 3) TRI(i, i + 1, i + 2, i);
        break;
    case 6:                                        /* strip */
        for (i = 0; i + 2 < n; i++) TRI(i, i + 1, i + 2, i);
        break;
    case 7: case 10:                               /* fan, polygon */
        for (i = 1; i + 1 < n; i++) TRI(0, i, i + 1, 0);
        break;
    case 8:                                        /* quads */
        for (i = 0; i + 3 < n; i += 4) {
            TRI(i, i + 1, i + 2, i);
            TRI(i, i + 2, i + 3, i);
        }
        break;
    case 9:                                        /* quad strip */
        for (i = 0; i + 3 < n; i += 2) {
            TRI(i, i + 1, i + 3, i);
            TRI(i, i + 3, i + 2, i);
        }
        break;
    default:
        break;
    }
#undef TRI
    return t;
}

static void vp_draw_triangle(const VpVertex *p0, const VpVertex *p1,
                             const VpVertex *p2, uint32_t flat, int textured)
{
    float uv[3][2];

    uv[0][0] = p0->uv[0]; uv[0][1] = p0->uv[1];
    uv[1][0] = p1->uv[0]; uv[1][1] = p1->uv[1];
    uv[2][0] = p2->uv[0]; uv[2][1] = p2->uv[1];
    s_tri_z[0] = p0->z; s_tri_z[1] = p1->z; s_tri_z[2] = p2->z;
    s_tri_has_z = !s_batch_flat_z;
    s_tri_persp = !s_batch_flat_z && p0->w > 0.0f && p1->w > 0.0f && p2->w > 0.0f;
    if (s_tri_persp) {
        s_tri_invw[0] = 1.0f / p0->w;
        s_tri_invw[1] = 1.0f / p1->w;
        s_tri_invw[2] = 1.0f / p2->w;
    }
    s_tri_col[0] = p0->argb; s_tri_col[1] = p1->argb; s_tri_col[2] = p2->argb;
    raster_triangle(p0->xy, p1->xy, p2->xy, flat,
                    textured ? (const float (*)[2])uv : NULL);
    s_tri_has_z = 0;
    s_tri_persp = 0;
}

/* Near-plane clipping.
 *
 * The runtime's program has already divided by w and added the viewport
 * offset (c59), so the homogeneous position is rebuilt as (win - offset) * w,
 * which is linear in clip space. The triangle is clipped against w = NEAR in
 * that space, attributes interpolated there, and the pieces projected back.
 * Before this, triangles crossing the eye plane were dropped whole, and the
 * spectators standing next to the camera came out as scattered shards. */
#define VP_NEAR_W 1e-3f

typedef struct {
    float X, Y, Z, W, u, v;
    float col[4];
} VpClipVert;

static void vp_to_clip(const VpVertex *p, VpClipVert *c)
{
    int k;
    c->W = p->w;
    c->X = (p->xy[0] - s_vp.c[59][0]) * p->w;
    c->Y = (p->xy[1] - s_vp.c[59][1]) * p->w;
    c->Z = (p->z - s_vp.c[59][2]) * p->w;
    c->u = p->uv[0];
    c->v = p->uv[1];
    for (k = 0; k < 4; k++)
        c->col[k] = (float)((p->argb >> (8 * k)) & 0xFF);
}

static void vp_from_clip(const VpClipVert *c, VpVertex *p)
{
    uint32_t argb = 0;
    int k;
    p->w = c->W;
    p->xy[0] = c->X / c->W + s_vp.c[59][0];
    p->xy[1] = c->Y / c->W + s_vp.c[59][1];
    p->z = c->Z / c->W + s_vp.c[59][2];
    p->uv[0] = c->u;
    p->uv[1] = c->v;
    for (k = 0; k < 4; k++) {
        int iv = (int)(c->col[k] + 0.5f);
        if (iv < 0) iv = 0;
        if (iv > 255) iv = 255;
        argb |= (uint32_t)iv << (8 * k);
    }
    p->argb = argb;
    p->ok = 1;
}

static void vp_draw_clipped(const VpVertex *v[3], uint32_t flat, int textured)
{
    VpClipVert in[3], out[4];
    VpVertex r[4];
    int k, j, n = 0, inside = 0;

    if (s_batch_flat_z || !vp_active()) {
        vp_draw_triangle(v[0], v[1], v[2], flat, textured);
        return;
    }
    {
        static int clip_on = -1;
        if (clip_on < 0)
            clip_on = getenv("RECOMP_NEAR_CLIP") != NULL;
        if (!clip_on) {
            /* Default for now: drop anything reaching behind the eye. */
            if (v[0]->w > VP_NEAR_W && v[1]->w > VP_NEAR_W && v[2]->w > VP_NEAR_W)
                vp_draw_triangle(v[0], v[1], v[2], flat, textured);
            return;
        }
    }
    /* The near plane is clip-space z = 0 (D3D convention), where w is the
     * near distance. Clipping at w ~ 0 instead left vertices a thousand
     * screens away, and every such triangle cost a full-surface scan. */
    for (k = 0; k < 3; k++) {
        vp_to_clip(v[k], &in[k]);
        inside += in[k].Z >= 0.0f && in[k].W > VP_NEAR_W;
    }
    if (inside == 3) {
        vp_draw_triangle(v[0], v[1], v[2], flat, textured);
        return;
    }
    if (inside == 0)
        return;

    for (k = 0; k < 3; k++) {
        const VpClipVert *a = &in[k], *b = &in[(k + 1) % 3];
        const int ain = a->Z >= 0.0f, bin = b->Z >= 0.0f;
        if (ain)
            out[n++] = *a;
        if (ain != bin && n < 4) {
            const float t = a->Z / (a->Z - b->Z);
            VpClipVert *m = &out[n++];
            m->X = a->X + (b->X - a->X) * t;
            m->Y = a->Y + (b->Y - a->Y) * t;
            m->Z = 0.0f;
            m->W = a->W + (b->W - a->W) * t;
            m->u = a->u + (b->u - a->u) * t;
            m->v = a->v + (b->v - a->v) * t;
            for (j = 0; j < 4; j++)
                m->col[j] = a->col[j] + (b->col[j] - a->col[j]) * t;
        }
    }
    for (k = 0; k < n; k++) {
        if (out[k].W <= VP_NEAR_W)
            return;                        /* degenerate projection */
        vp_from_clip(&out[k], &r[k]);
    }
    for (k = 1; k + 1 < n; k++)
        vp_draw_triangle(&r[0], &r[k], &r[k + 1], r[0].argb, textured);
    (void)flat;
}

static uint32_t s_vp_tris[NV_MAX_INDICES * 2][4];
static VpVertex s_vp_verts[NV_MAX_INDICES];

static void raster_batch_program(void)
{
    uint32_t i, n = s_gpu.idx_count, t;
    int textured = texcoord_attr()->offset || s_gpu.inline_active;

    if (!textured)
        s_gpu.batches_no_uv++;
    else if (!s_gpu.tex.valid)
        s_gpu.batches_no_tex++;
    else {
        s_gpu.batches_textured++;
        note_texture_use();
    }

    for (i = 0; i < n; i++)
        vp_transform(s_gpu.idx[i], &s_vp_verts[i]);

    /* RECOMP_VP_FRAME=N: every batch of the N-th swapped frame, with where it
     * lands and what it samples. */
    {
        static int want = -1;
        if (want < 0)
            want = getenv("RECOMP_VP_FRAME") ? atoi(getenv("RECOMP_VP_FRAME")) : 0;
        static int every = -1;
        if (every < 0)
            every = getenv("RECOMP_VP_FRAME_EVERY") ? atoi(getenv("RECOMP_VP_FRAME_EVERY")) : 0;
        /* RECOMP_VP_FRAME_AT=S: the first whole frame after S seconds. */
        {
            static ULONGLONG start;
            static int at = -1;
            if (at < 0) {
                at = getenv("RECOMP_VP_FRAME_AT") ? atoi(getenv("RECOMP_VP_FRAME_AT")) : 0;
                start = GetTickCount64();
            }
            if (at > 0 && !want && GetTickCount64() - start >= (ULONGLONG)at * 1000)
                want = (int)s_gpu.flips + 1;
            /* F9, whenever someone playing wants the current scene logged. */
            if ((GetAsyncKeyState(VK_F9) & 1) && s_gpu.flips + 1 != (uint32_t)want) {
                want = (int)s_gpu.flips + 1;
                fprintf(stderr, "  [VPF] F9: logging frame %d\n", want);
            }
        }
        if ((want && s_gpu.flips == (uint32_t)want)
                || (every > 0 && s_gpu.flips % (uint32_t)every == 0)) {
            float lx = 1e30f, hx = -1e30f, ly = 1e30f, hy = -1e30f;
            for (i = 0; i < n; i++) {
                if (s_vp_verts[i].xy[0] < lx) lx = s_vp_verts[i].xy[0];
                if (s_vp_verts[i].xy[0] > hx) hx = s_vp_verts[i].xy[0];
                if (s_vp_verts[i].xy[1] < ly) ly = s_vp_verts[i].xy[1];
                if (s_vp_verts[i].xy[1] > hy) hy = s_vp_verts[i].xy[1];
            }
            s_vpf_flip = s_gpu.flips;
            fprintf(stderr, "  [VPF] f%u depth t%u f%X m%u clear%06X z0 %.1f fmt%08X\n",
                    s_gpu.flips, s_gpu.depth_test, s_gpu.depth_func, s_gpu.depth_mask,
                    s_gpu.zclear >> 8, s_vp_verts[0].z, s_gpu.format);
            fprintf(stderr, "  [VPF] f%u prim %u n %u  x %.1f..%.1f y %.1f..%.1f  d0 %08X"
                            "  tex 0x%08X %ux%u fmt %02X valid %d  uv0 %.2f %.2f"
                            "  blend %u\n",
                    s_gpu.flips, s_gpu.prim, n, lx, hx, ly, hy, s_vp_verts[0].argb,
                    s_gpu.tex.offset, s_gpu.tex.width, s_gpu.tex.height,
                    s_gpu.tex.color, s_gpu.tex.valid,
                    s_vp_verts[0].uv[0], s_vp_verts[0].uv[1], s_gpu.blend_enable);
            {
                static const int rows[] = { 0, 1, 2, 3, 4, 5, 8, 9, 96, 97, 98, 99 };
                uint32_t r;
                float in[4];
                fetch_attr(&s_gpu.attr[0], s_gpu.idx[0], in);
                fprintf(stderr, "  [VPF]   v0 in %.2f %.2f %.2f %.2f\n", in[0], in[1], in[2], in[3]);
                for (r = 0; r < sizeof rows / sizeof rows[0]; r++)
                    fprintf(stderr, "  [VPF]   c%-3d %13g %13g %13g %13g\n", rows[r],
                            s_vp.c[rows[r]][0], s_vp.c[rows[r]][1],
                            s_vp.c[rows[r]][2], s_vp.c[rows[r]][3]);
            }
        }
    }

    /* A batch whose every vertex has z exactly 0 is a 2D overlay passed
     * through in screen space (EA's font renderer, the movie quad). Its
     * depth means nothing, and writing it made every later element of the
     * menus fail LEQUAL against the near plane: the screens went black. */
    {
        uint32_t k;
        s_batch_flat_z = 1;
        for (k = 0; k < n; k++)
            if (s_vp_verts[k].z != 0.0f) {
                s_batch_flat_z = 0;
                break;
            }
    }

    t = batch_triangles(s_gpu.prim, n, s_vp_tris, NV_MAX_INDICES * 2);
    for (i = 0; i < t; i++) {
        const VpVertex *v[3];
        v[0] = &s_vp_verts[s_vp_tris[i][0]];
        v[1] = &s_vp_verts[s_vp_tris[i][1]];
        v[2] = &s_vp_verts[s_vp_tris[i][2]];
        if (!v[0]->ok || !v[1]->ok || !v[2]->ok)
            continue;
        vp_draw_clipped(v, s_vp_verts[s_vp_tris[i][3]].argb, textured);
    }
}

static void raster_batch(void)
{
    uint32_t i;
    uint32_t before = s_gpu.tris_drawn;

    if (s_gpu.idx_count < 3)
        return;
    if (vp_active() && !getenv("RECOMP_VP_OFF")) {
        raster_batch_program();
        if (s_gpu.tris_drawn != before && s_drawn_dumps < FB_DUMP_AFTER_DRAW) {
            s_drawn_dumps++;
            dump_surface_bmp();
        }
        return;
    }
    if (!batch_is_screen_space()) {
        s_gpu.batches_untransformed++;
        return;
    }


    /* Count why, once per batch: the texture stage cannot change inside one. */
    {
        const VertexAttr *tc = texcoord_attr();

        if (!(tc->offset && tc->stride))
            s_gpu.batches_no_uv++;
        else if (!s_gpu.tex.valid)
            s_gpu.batches_no_tex++;
        else {
            s_gpu.batches_textured++;
            note_texture_use();
        }
    }

    switch (s_gpu.prim) {
    case NV_PRIM_TRIANGLES:
        for (i = 0; i + 2 < s_gpu.idx_count; i += 3)
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
        break;
    case NV_PRIM_TRIANGLE_STRIP:
        for (i = 0; i + 2 < s_gpu.idx_count; i++)
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+2],
                           vertex_color(s_gpu.idx[i]));
        break;
    case NV_PRIM_TRIANGLE_FAN:
    case NV_PRIM_POLYGON:                  /* convex, so a fan is exact */
        for (i = 1; i + 1 < s_gpu.idx_count; i++)
            raster_indexed(s_gpu.idx[0], s_gpu.idx[i], s_gpu.idx[i+1],
                           vertex_color(s_gpu.idx[0]));
        break;
    case NV_PRIM_QUADS:
        /* Every four vertices are one quad, split along its first diagonal.
         * Treating the whole batch as one fan -- which is only right for a
         * single quad -- joined every quad to the first vertex: an EA APT
         * screen, which draws all its shapes as one QUADS batch, came out as
         * a few large slivers radiating from one corner. */
        for (i = 0; i + 3 < s_gpu.idx_count; i += 4) {
            uint32_t c = vertex_color(s_gpu.idx[i]);
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+2], c);
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+2], s_gpu.idx[i+3], c);
        }
        break;
    case NV_PRIM_QUAD_STRIP:
        /* Quad k is vertices 2k, 2k+1, 2k+3, 2k+2 in drawing order. */
        for (i = 0; i + 3 < s_gpu.idx_count; i += 2) {
            uint32_t c = vertex_color(s_gpu.idx[i]);
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+1], s_gpu.idx[i+3], c);
            raster_indexed(s_gpu.idx[i], s_gpu.idx[i+3], s_gpu.idx[i+2], c);
        }
        break;
    default:
        break;                             /* points and lines: not yet */
    }

    if (s_gpu.tris_drawn && (s_gpu.tris_drawn % 500) == 0)
        fprintf(stderr, "  [GPU] %u triangles rasterised\n", s_gpu.tris_drawn);

    /* Capture the surface while the geometry is still on it.
     *
     * The periodic report dumps too, but a title clears every frame and draws
     * in only some of them, so a report almost always lands on a surface that
     * was wiped a moment ago -- which reads as "nothing was drawn" when the
     * triangles went down correctly just before it. A few frames that actually
     * contain geometry are worth more than any number of clears. */
    if (s_gpu.tris_drawn != before && s_drawn_dumps < FB_DUMP_AFTER_DRAW) {
        s_drawn_dumps++;
        dump_surface_bmp();
    }
}

/* What a batch actually contains. Before anything can be rasterised, the
 * question is what space attribute 0 arrives in: a title running a vertex
 * program hands over object-space positions that mean nothing without running
 * it, while pre-transformed screen-space coordinates can be drawn directly. */
static void draw_primitive(void)
{
    float v[4];
    uint32_t i;

    if (!s_gpu.prim || !s_gpu.idx_count)
        return;
    s_gpu.draws++;
    if ((s_gpu.draws % 200) == 0)
        fprintf(stderr, "  [GPU] draw #%u\n", s_gpu.draws);
    s_gpu.verts += s_gpu.idx_count;

    /* How many batches carry coordinates at all, and what range they span.
     * A pipeline that decodes perfectly and draws nothing is indistinguishable
     * from one that never ran, unless the vertices themselves are measured. */
    {
        float p[4];
        if (fetch_attr(&s_gpu.attr[0], s_gpu.idx[0], p)) {
            if (p[0] != 0.0f || p[1] != 0.0f || p[2] != 0.0f) {
                s_gpu.nonzero_draws++;
                if (p[0] < s_gpu.min_x) s_gpu.min_x = p[0];
                if (p[0] > s_gpu.max_x) s_gpu.max_x = p[0];
                if (p[1] < s_gpu.min_y) s_gpu.min_y = p[1];
                if (p[1] > s_gpu.max_y) s_gpu.max_y = p[1];
            }
        }
    }

    raster_batch();

    if (getenv("RECOMP_PB_EXEC_VERBOSE")) {
        static int shown;
        if (shown++ < 6) {
            fprintf(stderr, "  [GPU] prim %u, %u indices, pos attr:"
                            " off 0x%08X type %u size %u stride %u\n",
                    s_gpu.prim, s_gpu.idx_count, s_gpu.attr[0].offset,
                    s_gpu.attr[0].type, s_gpu.attr[0].size, s_gpu.attr[0].stride);
            /* The texture stage, for either kind of batch. This used to print
             * only for inline batches, which meant a title drawing through
             * vertex arrays -- Half-Life 2's menu, for one -- showed no
             * texture state at all, and the reason a quad sampled flat was
             * invisible. */
            fprintf(stderr, "  [GPU]   tex: off 0x%08X %ux%u pitch %u"
                            " colour 0x%02X swizzled %d valid %d\n",
                    s_gpu.tex.offset, s_gpu.tex.width, s_gpu.tex.height,
                    s_gpu.tex.pitch, s_gpu.tex.color,
                    d3d8_format_is_swizzled(s_gpu.tex.color), s_gpu.tex.valid);
            {
                uint32_t k;
                for (k = 0; k < s_gpu.idx_count && k < 3; k++) {
                    float t[2];
                    if (fetch_texcoord(s_gpu.idx[k], t))
                        fprintf(stderr, "  [GPU]   uv[%u] = %.3f %.3f\n",
                                k, t[0], t[1]);
                }
            }
            /* An inline batch has no guest buffer to go and look at -- the
             * vertices are the payload -- so print the payload too. */
            if (s_gpu.inline_active) {
                uint32_t k;
                fprintf(stderr, "  [GPU]   inline %u dwords:", s_gpu.inline_count);
                for (k = 0; k < s_gpu.inline_count && k < 16; k++)
                    fprintf(stderr, " %08X", s_gpu.inline_buf[k]);
                fprintf(stderr, "\n");
                for (k = 0; k < s_gpu.idx_count && k < 4; k++) {
                    float t[2];
                    if (fetch_texcoord(s_gpu.idx[k], t))
                        fprintf(stderr, "  [GPU]   uv[%u] = %.3f %.3f\n",
                                k, t[0], t[1]);
                }
                fprintf(stderr, "  [GPU]   tex: off 0x%08X %ux%u pitch %u"
                                " colour 0x%02X valid %d\n",
                        s_gpu.tex.offset, s_gpu.tex.width, s_gpu.tex.height,
                        s_gpu.tex.pitch, s_gpu.tex.color, s_gpu.tex.valid);
            }
            {
                /* Every attribute the batch has, not just position. If the
                 * other streams carry data and position does not, the problem
                 * is one buffer rather than the whole vertex path. */
                const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
                uint32_t a, k;
                for (a = 0; a < NV_VERTEX_ATTRS; a++) {
                    const VertexAttr *at = &s_gpu.attr[a];
                    uint32_t nz = 0;
                    if (!at->offset || !at->size)
                        continue;
                    for (k = 0; k < 64; k++)
                        if (mem[at->offset + k]) nz++;
                    fprintf(stderr, "  [GPU]   attr%-2u off 0x%08X type %u"
                                    " size %u stride %-3u  %u/64 bytes set\n",
                            a, at->offset, at->type, at->size, at->stride, nz);
                }
            }
            {
                /* Raw bytes at the array, in case the values read as zero:
                 * that looks the same whether the offset is wrong or the
                 * buffer genuinely has not been filled yet. */
                const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
                uint32_t k;
                fprintf(stderr, "  [GPU]   bytes @0x%08X:", s_gpu.attr[0].offset);
                for (k = 0; k < 32; k++)
                    fprintf(stderr, " %02X", mem[s_gpu.attr[0].offset + k]);
                fprintf(stderr, "\n");
                fprintf(stderr, "  [GPU]   indices:");
                for (k = 0; k < s_gpu.idx_count && k < 8; k++)
                    fprintf(stderr, " %u", s_gpu.idx[k]);
                fprintf(stderr, "\n");
            }
            for (i = 0; i < s_gpu.idx_count && i < 3; i++) {
                if (fetch_attr(&s_gpu.attr[0], s_gpu.idx[i], v))
                    fprintf(stderr, "  [GPU]   v[%u] = %.3f %.3f %.3f %.3f\n",
                            s_gpu.idx[i], v[0], v[1], v[2], v[3]);
            }
        }
    }
}

/* Draw the vertices the title wrote straight into the pushbuffer.
 *
 * INLINE_ARRAY carries no offsets and no indices: the dwords between BEGIN and
 * END *are* the vertex buffer, packed in attribute order using the same
 * SET_VERTEX_DATA_ARRAY_FORMAT registers an ordinary array would use. So the
 * whole batch is describable as a vertex array whose base happens to be that
 * payload, which means synthesising the layout and handing it to the existing
 * path -- rather than a second copy of the topology and rasterisation code.
 *
 * The title's own attribute table is saved and put back: these offsets and
 * strides are ours, and it has not stopped using its.
 *
 * ponytail: each attribute is padded to a whole dword. That is exact for the
 * float and D3DCOLOR formats every inline batch actually uses; a packed
 * sub-dword attribute would need the unpadded layout.
 */
static void draw_inline_array(void)
{
    VertexAttr saved[NV_VERTEX_ATTRS];
    uint32_t off = 0, a, i, vsize, count;

    memcpy(saved, s_gpu.attr, sizeof saved);

    for (a = 0; a < NV_VERTEX_ATTRS; a++) {
        uint32_t bytes;
        if (!s_gpu.attr[a].size)
            continue;
        switch (s_gpu.attr[a].type) {
        case 0:  bytes = 4;                        break;  /* D3DCOLOR   */
        case 2:  bytes = 4 * s_gpu.attr[a].size;   break;  /* float      */
        case 4:  bytes = s_gpu.attr[a].size;       break;  /* ubyte norm */
        default: bytes = 4 * s_gpu.attr[a].size;   break;
        }
        s_gpu.attr[a].offset = off;
        off += (bytes + 3u) & ~3u;
    }
    vsize = off;
    if (!vsize)
        goto out;

    count = (s_gpu.inline_count * 4) / vsize;
    if (count < 3 || count > NV_MAX_INDICES)
        goto out;
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].size)
            s_gpu.attr[a].stride = vsize;

    for (i = 0; i < count; i++)
        s_gpu.idx[i] = (uint16_t)i;
    s_gpu.idx_count = count;

    s_gpu.inline_active = 1;
    draw_primitive();
    s_gpu.inline_active = 0;

out:
    memcpy(s_gpu.attr, saved, sizeof saved);
    s_gpu.idx_count = 0;
}

/* Draw the vertices SET_VERTEX3F/4F completed.
 *
 * Same trick as draw_inline_array: rather than a second copy of the topology
 * and rasterisation code, describe what was accumulated as an ordinary vertex
 * array and hand it to the existing path. The layout is ours and fixed, so
 * the attribute table is written out here rather than derived from the
 * title's format registers.
 *
 * The title's own table is saved and put back -- it has not stopped using it.
 *
 * ponytail: position, diffuse and texcoord0 only. That is what a 2D quad
 * carries and what this rasteriser samples; a second texcoord set or a normal
 * would need the D3D11 translator, not more slots here.
 */
static void draw_immediate(void)
{
    VertexAttr saved[NV_VERTEX_ATTRS];
    uint32_t i;

    if (s_gpu.imm_count < 3)
        return;

    memcpy(saved, s_gpu.attr, sizeof saved);
    memset(s_gpu.attr, 0, sizeof s_gpu.attr);
    /* Offsets are byte offsets into inline_buf here, not guest addresses --
     * fetch_attr reads them that way while inline_active is set, which is
     * also why 0 is a legal offset for position. */
    s_gpu.attr[0].type = 2; s_gpu.attr[0].size = 4;   /* position float4  */
    s_gpu.attr[0].offset = 0;
    s_gpu.attr[3].type = 0; s_gpu.attr[3].size = 4;   /* diffuse D3DCOLOR */
    s_gpu.attr[3].offset = 16;
    s_gpu.attr[9].type = 2; s_gpu.attr[9].size = 2;   /* texcoord0 float2 */
    s_gpu.attr[9].offset = 20;
    s_gpu.attr[0].stride = s_gpu.attr[3].stride = s_gpu.attr[9].stride =
        IMM_VERTEX_DWORDS * 4;

    for (i = 0; i < s_gpu.imm_count && i < NV_MAX_INDICES; i++)
        s_gpu.idx[i] = (uint16_t)i;
    s_gpu.idx_count = i;

    /* fetch_attr bounds-checks against inline_count dwords. */
    s_gpu.inline_count = s_gpu.imm_count * IMM_VERTEX_DWORDS;
    s_gpu.inline_active = 1;
    draw_primitive();
    s_gpu.inline_active = 0;

    memcpy(s_gpu.attr, saved, sizeof saved);
    s_gpu.idx_count = 0;
    s_gpu.inline_count = 0;
}

/* A vertex is complete: append it in the layout draw_immediate describes. */
static void imm_emit_vertex(void)
{
    uint32_t at = s_gpu.imm_count * IMM_VERTEX_DWORDS;

    if (!s_gpu.prim || at + IMM_VERTEX_DWORDS > NV_MAX_INLINE)
        return;
    memcpy(&s_gpu.inline_buf[at],     s_gpu.imm_pos, 4 * sizeof(float));
    memcpy(&s_gpu.inline_buf[at + 4], &s_gpu.imm_diffuse, sizeof(uint32_t));
    memcpy(&s_gpu.inline_buf[at + 5], s_gpu.imm_tex, 2 * sizeof(float));
    s_gpu.imm_count++;
}

/* The immediate-mode writes. Returns 1 if `method` was one of them.
 *
 * Split out because it is a range test against five separate bases, and that
 * reads better than five more cases in an already long switch.
 */
static int imm_vertex_method(uint32_t method, uint32_t param)
{
    union { uint32_t u; float f; } v;
    v.u = param;

    if (method >= NV097_SET_VERTEX4F && method < NV097_SET_VERTEX4F + 16) {
        uint32_t c = (method - NV097_SET_VERTEX4F) / 4;
        s_gpu.imm_pos[c] = v.f;
        if (c == 3)                      /* w completes the vertex */
            imm_emit_vertex();
        return 1;
    }
    if (method >= NV097_SET_VERTEX3F && method < NV097_SET_VERTEX3F + 12) {
        uint32_t c = (method - NV097_SET_VERTEX3F) / 4;
        s_gpu.imm_pos[c] = v.f;
        if (c == 2) {                    /* z completes it, w is implicitly 1 */
            s_gpu.imm_pos[3] = 1.0f;
            imm_emit_vertex();
        }
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA2F_M
            && method < NV097_SET_VERTEX_DATA2F_M + NV_VERTEX_ATTRS * 8) {
        uint32_t off = method - NV097_SET_VERTEX_DATA2F_M;
        if (off / 8 == 9)                /* attribute 9 is texture coord 0 */
            s_gpu.imm_tex[(off % 8) / 4] = v.f;
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA4F_M
            && method < NV097_SET_VERTEX_DATA4F_M + NV_VERTEX_ATTRS * 16) {
        uint32_t off = method - NV097_SET_VERTEX_DATA4F_M;
        uint32_t attr = off / 16, c = (off % 16) / 4;
        if (attr == 0) {
            s_gpu.imm_pos[c] = v.f;
            if (c == 3)
                imm_emit_vertex();
        } else if (attr == 9 && c < 2) {
            s_gpu.imm_tex[c] = v.f;
        }
        return 1;
    }
    if (method >= NV097_SET_VERTEX_DATA4UB
            && method < NV097_SET_VERTEX_DATA4UB + NV_VERTEX_ATTRS * 4) {
        if ((method - NV097_SET_VERTEX_DATA4UB) / 4 == 3)   /* diffuse */
            s_gpu.imm_diffuse = param;
        return 1;
    }
    return 0;
}
void nv2a_pb_exec_method(uint32_t subch, uint32_t method, uint32_t param)
{
    static int inited;
    if (!inited) {
        inited = 1;
        s_gpu.min_x = s_gpu.min_y = 1e30f;
        /* Until the title says otherwise: source-over, what the rasteriser
         * always assumed before the factors were decoded. */
        s_gpu.blend_sfactor = NV_BLEND_SRC_ALPHA;
        s_gpu.blend_dfactor = NV_BLEND_ONE_MINUS_SRC_ALPHA;
        s_gpu.blend_equation = 0x8006;     /* FUNC_ADD */
        s_depth_on = getenv("RECOMP_NO_DEPTH") == NULL;
        /* Power-on defaults. */
        s_gpu.color_mask = 0x01010101u;
        s_gpu.stencil_wmask = s_gpu.stencil_fmask = 0xFF;
        s_gpu.stencil_func = 0x0207;
        s_gpu.stencil_op_fail = s_gpu.stencil_op_zfail = s_gpu.stencil_op_zpass = 0x1E00;
        s_gpu.depth_mask = 1;
        s_gpu.depth_func = 0x0201;
        s_gpu.max_x = s_gpu.max_y = -1e30f;
    }
    /* Bring-up: the first parameters each surface method carries. A wrong
     * pitch or clip is indistinguishable from a method never arriving unless
     * the values are visible. */
    if (getenv("RECOMP_PB_EXEC_VERBOSE")) {
        static int shown[8];
        int slot = -1;
        switch (method) {
        case NV097_SET_SURFACE_CLIP_HORIZONTAL: slot = 0; break;
        case NV097_SET_SURFACE_CLIP_VERTICAL:   slot = 1; break;
        case NV097_SET_SURFACE_FORMAT:          slot = 2; break;
        case NV097_SET_SURFACE_PITCH:           slot = 3; break;
        case NV097_SET_SURFACE_COLOR_OFFSET:    slot = 4; break;
        case NV097_SET_COLOR_CLEAR_VALUE:       slot = 5; break;
        case NV097_CLEAR_SURFACE:               slot = 6; break;
        default: break;
        }
        if (slot >= 0 && shown[slot]++ < 4)
            fprintf(stderr, "  [GPU] subch %u method 0x%04X param 0x%08X\n",
                    subch, method, param);
    }

    if (subch != 0) {                      /* 3D class lives on subchannel 0 */
        note_unhandled(method, param);
        return;
    }
    switch (method) {
    case NV097_SET_SURFACE_CLIP_HORIZONTAL:
        s_gpu.clip_x = param & 0xFFFF;
        s_gpu.clip_w = (param >> 16) & 0xFFFF;
        break;
    case NV097_SET_SURFACE_CLIP_VERTICAL:
        s_gpu.clip_y = param & 0xFFFF;
        s_gpu.clip_h = (param >> 16) & 0xFFFF;
        break;
    case NV097_SET_SURFACE_FORMAT:
        s_gpu.format = param;
        break;
    case NV097_SET_SURFACE_PITCH:
        s_gpu.pitch = param & 0xFFFF;      /* colour pitch; zeta is the top half */
        break;
    case NV097_SET_SURFACE_COLOR_OFFSET:
        s_gpu.color_offset = param;
        break;
    case NV097_SET_BLEND_ENABLE:
        s_gpu.blend_enable = param;
        break;
    case NV097_SET_BLEND_FUNC_SFACTOR:
        s_gpu.blend_sfactor = param;
        break;
    case NV097_SET_BLEND_FUNC_DFACTOR:
        s_gpu.blend_dfactor = param;
        break;
    case NV097_SET_BLEND_COLOR:
        s_gpu.blend_color = param;
        break;
    case NV097_SET_BLEND_EQUATION:
        s_gpu.blend_equation = param;
        break;
    case NV097_SET_STENCIL_TEST_ENABLE: s_gpu.stencil_test = param; break;
    case NV097_SET_COLOR_MASK:          s_gpu.color_mask = param; break;
    case NV097_SET_STENCIL_MASK:        s_gpu.stencil_wmask = param & 0xFF; break;
    case NV097_SET_STENCIL_FUNC_FUNC:   s_gpu.stencil_func = param; break;
    case NV097_SET_STENCIL_FUNC_REF:    s_gpu.stencil_ref = param & 0xFF; break;
    case NV097_SET_STENCIL_FUNC_MASK:   s_gpu.stencil_fmask = param & 0xFF; break;
    case NV097_SET_STENCIL_OP_FAIL:     s_gpu.stencil_op_fail = param; break;
    case NV097_SET_STENCIL_OP_ZFAIL:    s_gpu.stencil_op_zfail = param; break;
    case NV097_SET_STENCIL_OP_ZPASS:    s_gpu.stencil_op_zpass = param; break;
    case NV097_SET_DEPTH_TEST_ENABLE:
        s_gpu.depth_test = param;
        break;
    case NV097_SET_DEPTH_FUNC:
        s_gpu.depth_func = param;
        break;
    case NV097_SET_DEPTH_MASK:
        s_gpu.depth_mask = param;
        break;
    case NV097_SET_ZSTENCIL_CLEAR_VALUE:
        s_gpu.zclear = param;
        break;
    case NV097_SET_ALPHA_TEST_ENABLE:
        s_gpu.alpha_test = param;
        break;
    case NV097_SET_ALPHA_FUNC:
        s_gpu.alpha_func = param;
        break;
    case NV097_SET_ALPHA_REF:
        s_gpu.alpha_ref = param & 0xFF;
        break;
    case NV097_SET_COLOR_CLEAR_VALUE:
        s_gpu.clear_color = param;
        break;
    case NV097_CLEAR_SURFACE:
        clear_surface(param);
        break;

    case NV097_SET_BEGIN_END:
        if (param) {
            s_gpu.prim = param;
            s_gpu.idx_count = 0;
            s_gpu.inline_count = 0;
            s_gpu.imm_count = 0;
        } else {
            /* Three ways a batch can have arrived, and only one is in use at
             * a time: vertices completed by SET_VERTEX4F, a payload written
             * with INLINE_ARRAY, or indices into the title's own arrays. */
            if (s_gpu.imm_count)
                draw_immediate();
            else if (s_gpu.inline_count)
                draw_inline_array();
            else
                draw_primitive();
            s_gpu.prim = 0;
            s_gpu.inline_count = 0;
            s_gpu.imm_count = 0;
        }
        break;

    case NV097_INLINE_ARRAY:
        /* Vertex data, not a pointer to it. Buffered rather than decoded here
         * because the format is only fully known at END. */
        if (s_gpu.prim && s_gpu.inline_count < NV_MAX_INLINE)
            s_gpu.inline_buf[s_gpu.inline_count++] = param;
        break;

    case NV097_ARRAY_ELEMENT16:
        /* Two 16-bit indices per parameter word. */
        if (s_gpu.prim && s_gpu.idx_count + 2 <= NV_MAX_INDICES) {
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(param & 0xFFFF);
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(param >> 16);
        }
        break;
    case NV097_SET_TEXTURE_OFFSET:
        /* A texture offset is a DMA-object offset, exactly like a surface or a
         * vertex array offset -- physical, and reachable only through the
         * contiguous window when it names contiguous memory. */
        s_gpu.tex.offset = dma_resolve(param);
        record_tex_reg(method, param);
        break;

    case NV097_SET_FLIP_READ:
        s_gpu.flip_read = param;
        return;

    case NV097_SET_FLIP_WRITE:
        s_gpu.flip_write = param;
        return;

    case NV097_SET_FLIP_MODULO:
        s_gpu.flip_modulo = param;
        return;

    case NV097_FLIP_INCREMENT_WRITE:
        s_gpu.flip_write = s_gpu.flip_modulo
                         ? (s_gpu.flip_write + 1) % s_gpu.flip_modulo
                         : s_gpu.flip_write + 1;
        s_gpu.flips++;
        return;

    case NV097_FLIP_STALL:
        /* The stall ends when the buffer being read is the one just finished.
         * There is no scanout here to wait for, so that is now. */
        s_gpu.flip_read = s_gpu.flip_write;
        /* And this is a completed swap, which is what a title's own swap
         * counter counts -- see xbox_Nv2aFrameCounterFlip. */
        xbox_Nv2aFrameCounterFlip();
        xbox_FramebufferWindowLatch();
        if (getenv("RECOMP_PB_EXEC_VERBOSE")) {
            static unsigned n;
            if (n++ < 8) {
                fprintf(stderr, "  [GPU] flip %u: read=%u write=%u\n",
                        s_gpu.flips, s_gpu.flip_read, s_gpu.flip_write);
                fflush(stderr);
            }
        }
        return;

    case NV097_SET_TEXTURE_FORMAT:
        s_gpu.tex.color = (param >> 8) & 0xFF;
        /* A swizzled texture carries its own dimensions here, as log2 in
         * BASE_SIZE_U/V (nv2a_regs.h: 0x00F00000 / 0x0F000000). It has to:
         * SET_TEXTURE_IMAGE_RECT describes a linear image, and a title that
         * only uses swizzled textures never sends one -- this title sends it
         * once and sets a format 3,176 times. Without this the width and
         * height stayed zero and nothing was ever sampled. */
        if (tex_size_from_format((param >> 8) & 0xFF)) {
            s_gpu.tex.width  = 1u << ((param >> 20) & 0xF);
            s_gpu.tex.height = 1u << ((param >> 24) & 0xF);
        }
        record_tex_reg(method, param);
        break;

    case NV097_SET_TEXTURE_ADDRESS:
        /* Four bits per axis. 1 is wrap, 3 is clamp-to-edge; the rest (mirror,
         * border) fall back to clamp, which is wrong at an edge rather than
         * wrong everywhere. */
        s_gpu.tex.addr_u =  param        & 0xF;
        s_gpu.tex.addr_v = (param >>  8) & 0xF;
        record_tex_reg(method, param);
        break;

    case NV097_SET_TEXTURE_CONTROL1:
        /* Pitch lives in the top half. Only meaningful for a linear format; a
         * swizzled texture has no pitch because it has no rows. */
        s_gpu.tex.pitch = param >> 16;
        record_tex_reg(method, param);
        break;

    case NV097_SET_TEXTURE_PALETTE:
        /* The palette of an 8-bit indexed texture. EA's APT keeps every text
         * page in this format, so without it the glyphs sampled as nothing
         * and the text came out as flat vertex colour. */
        s_gpu.tex.palette = dma_resolve(param & ~0x3Fu);
        record_tex_reg(method, param);
        break;

    case NV097_SET_TEXTURE_IMAGE_RECT:
        s_gpu.tex.width  = param >> 16;
        s_gpu.tex.height = param & 0xFFFF;
        record_tex_reg(method, param);
        break;

    default:
        if (method >= NV_TEX_FIRST && method <= NV_TEX_LAST)
            record_tex_reg(method, param);
        if (method >= NV097_SET_VERTEX_DATA_ARRAY_OFFSET
                && method < NV097_SET_VERTEX_DATA_ARRAY_OFFSET + NV_VERTEX_ATTRS * 4) {
            /* Resolved here, once, so every consumer -- the rasteriser's
             * attribute reads and the diagnostics alike -- sees the same
             * address. A vertex array offset is a DMA-object offset exactly
             * like a surface offset: physical, and addressable only through
             * the window when it names contiguous memory. */
            s_gpu.attr[(method - NV097_SET_VERTEX_DATA_ARRAY_OFFSET) / 4].offset =
                dma_resolve(param);
        } else if (method >= NV097_SET_VERTEX_DATA_ARRAY_FORMAT
                && method < NV097_SET_VERTEX_DATA_ARRAY_FORMAT + NV_VERTEX_ATTRS * 4) {
            VertexAttr *a = &s_gpu.attr[(method - NV097_SET_VERTEX_DATA_ARRAY_FORMAT) / 4];
            a->type   =  param        & 0x0F;
            a->size   = (param >> 4)  & 0x0F;
            a->stride = (param >> 8)  & 0xFF;
        } else if (!imm_vertex_method(method, param)
                && !vp_method(method, param)) {
            note_unhandled(method, param);
        }
        break;
    }
}

/* Find where the title actually wrote its quad.
 *
 * The GPU is pointed at a buffer that stays zero, which says the data went
 * somewhere else -- and the only way to find somewhere else is to look for the
 * data. A screen-space quad for a 640x480 target contains 640.0f and 480.0f as
 * floats, which is a distinctive enough pair to search guest RAM for. Whatever
 * address that turns up is where the title's writes are landing, and the
 * difference from the programmed offset is the bug. */
static void find_quad_vertices(void)
{
    const uint32_t W = 0x44200000u;   /* 640.0f */
    const uint32_t H = 0x43F00000u;   /* 480.0f */
    const uint32_t *ram = (const uint32_t *)xbox_GetMemoryOffset();
    uint32_t i, hits = 0;

    fprintf(stderr, "[GPU] searching guest RAM for 640.0f/480.0f pairs...\n");
    for (i = 0x1000 / 4; i < (0x04000000u / 4) - 8 && hits < 12; i++) {
        if (ram[i] != W && ram[i] != H)
            continue;
        /* Both values within a few words of each other: a lone 640.0f is
         * common, the pair much less so. */
        {
            int has_w = 0, has_h = 0;
            uint32_t k;
            for (k = 0; k < 8; k++) {
                if (ram[i + k] == W) has_w = 1;
                if (ram[i + k] == H) has_h = 1;
            }
            if (!has_w || !has_h)
                continue;
        }
        hits++;
        fprintf(stderr, "  [GPU]   0x%08X:", i * 4);
        {
            uint32_t k;
            for (k = 0; k < 8; k++)
                fprintf(stderr, " %08X", ram[i + k]);
        }
        fprintf(stderr, "\n");
        i += 8;
    }
    if (!hits)
        fprintf(stderr, "  [GPU]   none found -- the quad is not in RAM in"
                        " that form\n");
    fflush(stderr);
}

/* Locate NaN-filled transform matrices in guest RAM.
 *
 * A matrix arriving as NaN says the maths went wrong somewhere upstream, and
 * the only way to find where is to find the matrix and watch who writes it.
 * Three consecutive real-indefinite values is a distinctive enough signature:
 * ordinary data does not contain runs of 0xFFC00000. */
static void find_nan_matrices(void)
{
    const uint32_t NAN_NEG = 0xFFC00000u;
    const uint32_t *ram = (const uint32_t *)xbox_GetMemoryOffset();
    uint32_t i, hits = 0;

    fprintf(stderr, "[GPU] searching guest RAM for NaN matrices...\n");
    for (i = 0x1000 / 4; i < (0x04000000u / 4) - 20 && hits < 10; i++) {
        if (ram[i] != NAN_NEG || ram[i + 1] != NAN_NEG || ram[i + 2] != NAN_NEG)
            continue;
        hits++;
        fprintf(stderr, "  [GPU]   0x%08X:", i * 4);
        {
            uint32_t k;
            for (k = 0; k < 16; k++)
                fprintf(stderr, " %08X", ram[i + k]);
        }
        fprintf(stderr, "\n");
        i += 16;
    }
    if (!hits)
        fprintf(stderr, "  [GPU]   none in RAM -- the NaNs are computed into"
                        " registers, not stored\n");
    fflush(stderr);
}

/* Print guest dwords named by RECOMP_PEEK, as hex and as float.
 *
 * Chasing a value backwards means reading it, and a value that is only wrong
 * for one frame in a thousand cannot be caught by stopping. Both
 * interpretations are printed because the question is usually "is this a
 * pointer or a number", and guessing wrong costs a run. */
static void peek_addresses(void)
{
    const char *spec = getenv("RECOMP_PEEK");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    char buf[256];
    char *tok, *ctx = NULL;

    if (!spec)
        return;
    strncpy(buf, spec, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    for (tok = strtok_s(buf, ",", &ctx); tok; tok = strtok_s(NULL, ",", &ctx)) {
        uint32_t va = (uint32_t)strtoul(tok, NULL, 0);
        uint32_t v;
        float f;
        if (va < 0x1000u || va >= 0x04000000u)
            continue;
        v = *(const uint32_t *)(mem + va);
        memcpy(&f, &v, 4);
        fprintf(stderr, "  [PEEK] 0x%08X = %08X  (%g)\n", va, v, f);
    }
    fflush(stderr);
}

/* Walk a pointer chain and print every step.
 *
 * RECOMP_PEEK_CHAIN="0x1315A8,8,0x10,0" starts at that address, and for each
 * offset dereferences the current pointer and adds it. Following a chain by
 * hand costs one run per level; this costs one run for the whole chain, and
 * prints where it goes wrong when a level is null.
 */
static void peek_chain(void)
{
    const char *spec = getenv("RECOMP_PEEK_CHAIN");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    char buf[256], *tok, *ctx = NULL;
    uint32_t cur = 0;
    int step = 0;

    if (!spec)
        return;
    strncpy(buf, spec, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;

    for (tok = strtok_s(buf, ",", &ctx); tok; tok = strtok_s(NULL, ",", &ctx)) {
        uint32_t off = (uint32_t)strtoul(tok, NULL, 0);
        if (step == 0) {
            cur = off;
            fprintf(stderr, "  [CHAIN] start 0x%08X\n", cur);
        } else {
            if (cur < 0x1000u || cur + 4 >= 0x04000000u) {
                fprintf(stderr, "  [CHAIN] step %d: 0x%08X is not a guest"
                                " pointer -- chain ends\n", step, cur);
                return;
            }
            cur = *(const uint32_t *)(mem + cur) + off;
            fprintf(stderr, "  [CHAIN] step %d: deref +0x%X -> 0x%08X\n",
                    step, off, cur);
        }
        step++;
    }
    if (cur >= 0x1000u && cur + 4 < 0x04000000u)
        fprintf(stderr, "  [CHAIN] final value at 0x%08X = 0x%08X\n",
                cur, *(const uint32_t *)(mem + cur));
    fflush(stderr);
}

void nv2a_pb_exec_report(void)
{
    peek_addresses();
    peek_chain();
    if (getenv("RECOMP_FIND_NAN")) {
        /* Every report, not once: the matrix is fine early on and only turns
         * to NaN later, so a single scan at startup finds nothing and says
         * nothing. */
        find_nan_matrices();
    }
    if (getenv("RECOMP_FIND_QUAD")) {
        static int done;
        if (!done) { done = 1; find_quad_vertices(); }
    }
    int i, j;

    fprintf(stderr, "[GPU] surface 0x%08X pitch %u clip %ux%u+%u+%u"
                    " clears %u | %u unhandled methods (%d distinct)\n",
            s_gpu.color_offset, s_gpu.pitch, s_gpu.clip_w, s_gpu.clip_h,
            s_gpu.clip_x, s_gpu.clip_y, s_gpu.clears,
            s_gpu.unhandled_total, s_unhandled_count);
    fprintf(stderr, "[GPU] draws %u (%u with coordinates), %u indices;"
                    " x %.1f..%.1f  y %.1f..%.1f\n",
            s_gpu.draws, s_gpu.nonzero_draws, s_gpu.verts,
            s_gpu.min_x, s_gpu.max_x, s_gpu.min_y, s_gpu.max_y);
    /* One picture per report rather than per clear: a title clears hundreds of
     * times a second and nobody wants that many files. */
    dump_surface_bmp();

    /* Drawn and skipped separately: "nothing appeared" and "every batch needed
     * a vertex program we do not run" look identical on screen, and only one
     * of them means the rasteriser is broken. */
    fprintf(stderr, "[GPU] brightest pixel written 0x%08X\n", s_gpu.pixel_max);
    fprintf(stderr, "[GPU] %llu pixels written; draw surface 0x%08X"
                    " -> 0x%08X, clear surface 0x%08X -> 0x%08X\n",
            (unsigned long long)s_gpu.pixels, s_gpu.drawn_offset,
            dma_resolve(s_gpu.drawn_offset), s_gpu.color_offset,
            dma_resolve(s_gpu.color_offset));
    fprintf(stderr, "[GPU] rasterised %u triangles; %u batches skipped as not"
                    " screen-space, %u triangles fully off-surface\n",
            s_gpu.tris_drawn, s_gpu.batches_untransformed,
            s_gpu.tris_skipped_offscreen);
    fprintf(stderr, "[GPU] vertex programs: mode %u, %u vertices run, %u unknown opcodes\n",
            s_vp.mode, s_vp.runs, s_vp.bad_insns);

    /* And of the batches that did rasterise, how many sampled anything. A menu
     * that draws its background from one texture and its text from another
     * shows both as flat colour if either half is missing, so the split is
     * what says which half. */
    fprintf(stderr, "[GPU] batches: %u textured, %u with no texcoords,"
                    " %u with texcoords but no usable stage\n",
            s_gpu.batches_textured, s_gpu.batches_no_uv, s_gpu.batches_no_tex);
    for (i = 0; i < s_tex_use_count; i++)
        fprintf(stderr, "  [TEXUSE] 0x%08X %ux%u fmt 0x%02X%s: %u batches\n",
                s_tex_use[i].offset, s_tex_use[i].width, s_tex_use[i].height,
                s_tex_use[i].color,
                d3d8_format_dxt_block_bytes(s_tex_use[i].color) ? " dxt"
                    : d3d8_format_is_swizzled(s_tex_use[i].color) ? " swz" : " lin",
                s_tex_use[i].batches);

    if (getenv("RECOMP_TEX_STATE")) {
        uint32_t k;
        for (k = 0; k < sizeof s_tex_set / sizeof s_tex_set[0]; k++)
            if (s_tex_set[k])
                fprintf(stderr, "  [TEX] 0x%04X = 0x%08X\n",
                        (unsigned)(NV_TEX_FIRST + k * 4), s_tex_reg[k]);
    }

    /* Top ten by frequency: selection sort over a small table, once every few
     * seconds, is not worth a better algorithm.
     *
     * RECOMP_PB_UNHANDLED_ALL lists every one instead. Ten is the right
     * default -- the tail is a long list of state registers nobody needs to
     * read -- but when a title stops and the question is which method it
     * stopped on, the answer is as likely to be the one seen twice as the
     * one seen a thousand times, and ten hides it. */
    {
        int shown = getenv("RECOMP_PB_UNHANDLED_ALL") ? s_unhandled_count : 10;
    for (i = 0; i < shown && i < s_unhandled_count; i++) {
        int best = i;
        for (j = i + 1; j < s_unhandled_count; j++)
            if (s_unhandled[j].count > s_unhandled[best].count)
                best = j;
        if (best != i) {
            PbUnhandled t = s_unhandled[i];
            s_unhandled[i] = s_unhandled[best];
            s_unhandled[best] = t;
        }
        {
            /* The value as well as the count. A method nobody decoded is
             * a guess until you see what it carried: screen coordinates,
             * a 0..1 texcoord and a packed colour are told apart at a
             * glance, and that is what says which vertex encoding a title
             * is using. */
            union { uint32_t u; float f; } v;
            v.u = s_unhandled[i].last_param;
            fprintf(stderr, "  [GPU]   0x%04X x%-8u last=0x%08X (%.4f)\n",
                    s_unhandled[i].method, s_unhandled[i].count,
                    v.u, v.f);
        }
    }
    }    fflush(stderr);
}
