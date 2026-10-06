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
#include <stdarg.h>
#include <string.h>
#include "kernel.h"   /* XBOX_CONTIG_BASE / XBOX_CONTIG_SIZE */
#include "xbox_memory_layout.h"   /* xbox_Nv2aFrameCounterFlip */
/* The swizzle decoder the D3D8 layer already uses -- one implementation of
 * Morton order, not a second one that can disagree with it. */
#include "../d3d/d3d8_swizzle.h"
#include "nv2a_d3d11.h"
extern int xbox_FramebufferWindowPixelsWanted(void);

/* Debug switches read on the hot path, once each.
 *
 * getenv takes the CRT lock and walks the whole environment block. Called per
 * vertex, per triangle and per pushbuffer method it was most of the NV2A
 * thread's time -- the profile showed strchr above the rasteriser itself. The
 * environment does not change while the title runs. */
enum {
    ENV_ZCLEAR_TRACE, ENV_VP_TRACE, ENV_VP_OFF, ENV_PB_EXEC_VERBOSE,
    ENV_RASTER_TEST, ENV_CULL_FLIP, ENV_CULL_TRACE,
    ENV_COUNT
};
static const char *const s_env_names[ENV_COUNT] = {
    "RECOMP_ZCLEAR_TRACE", "RECOMP_VP_TRACE", "RECOMP_VP_OFF",
    "RECOMP_PB_EXEC_VERBOSE", "RECOMP_RASTER_TEST", "RECOMP_CULL_FLIP",
    "RECOMP_CULL_TRACE",
};
static signed char s_env_on[ENV_COUNT] = { -1, -1, -1, -1, -1, -1, -1 };

static int env_on(int which)
{
    if (s_env_on[which] < 0)
        s_env_on[which] = getenv(s_env_names[which]) != NULL;
    return s_env_on[which];
}

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
/* Face culling. The title switches this on and off once per batch --
 * 30 361 times in one survey -- and neither path read it, so every
 * model was drawn inside and out. */
#define NV097_SET_CULL_FACE_ENABLE        0x0308
#define NV097_SET_CULL_FACE               0x039C
#define NV097_SET_FRONT_FACE              0x03A0
#define NV_CULL_FACE_FRONT                0x0404
#define NV_CULL_FACE_BACK                 0x0405
#define NV_CULL_FACE_FRONT_AND_BACK       0x0408
#define NV_FRONT_FACE_CW                  0x0900
#define NV_FRONT_FACE_CCW                 0x0901
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
/* What the title asks for as filtering, and how many mip levels its
 * texture carries. Neither was decoded: every texel is taken nearest,
 * from level 0, whatever was asked. */
#define NV097_SET_TEXTURE_FILTER          0x1B14
#define NV097_SET_TEXTURE_IMAGE_RECT      0x1B1C
#define NV097_SET_TEXTURE_PALETTE         0x1B20   /* offset | DMA bit 0 */
/* The fragment pipeline beyond stage 0, captured for the frame log and the
 * inventory (RECOMP_INVENT), not executed. Values copied from
 * src/nv2a/nv2a_regs.h, which this file does not include. */
#define NV097_SET_COMBINER_ALPHA_ICW      0x0260   /* +i*4, 8 stages */
#define NV097_SET_COMBINER_SPECULAR_FOG_CW0 0x0288 /* final combiner, CW1 at +4 */
#define NV097_SET_FOG_MODE                0x029C
#define NV097_SET_FOG_GEN_MODE            0x02A0
#define NV097_SET_FOG_ENABLE              0x02A4
#define NV097_SET_COMBINER_FACTOR0        0x0A60   /* +i*4 */
#define NV097_SET_COMBINER_FACTOR1        0x0A80
#define NV097_SET_COMBINER_ALPHA_OCW      0x0AA0
#define NV097_SET_COMBINER_COLOR_ICW      0x0AC0
#define NV097_SET_COLOR_KEY_COLOR         0x0AE0   /* +i*4, 4 stages */
#define NV097_SET_COMBINER_COLOR_OCW      0x1E40   /* +i*4, 8 stages */
#define NV097_SET_COMBINER_CONTROL        0x1E60
#define NV097_SET_SHADER_STAGE_PROGRAM    0x1E70
#define NV097_SET_SHADER_OTHER_STAGE_INPUT 0x1E78
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
/* The two draw calls this executor never decoded. A batch submitted through
 * either of them was dropped whole and in silence, which on screen is a piece
 * of a model that simply is not there. */
#define NV097_ARRAY_ELEMENT32             0x1808
#define NV097_DRAW_ARRAYS                 0x1810
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
/* Raised from 4096 once DRAW_ARRAYS was honoured: a batch that names runs of
 * vertices rather than listing them reaches further than one that spells every
 * index out, and 28 batches of a fight overran the old cap. An index list that
 * overruns loses the end of a model in silence, which is the defect this cap
 * would otherwise reintroduce -- see the counters in the [GPU] report. */
#define NV_MAX_INDICES  16384
#define NV_MAX_INLINE   16384           /* dwords of INLINE_ARRAY per batch */

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
    uint32_t filter_min, filter_mag;    /* as asked, not as honoured     */
    uint32_t lod_bias, mip_levels;
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
    /* Face culling, as the title asked for it. */
    uint32_t cull_enable, cull_face, front_face;
    /* Culled, and submitted while the bit was on. Without the second
     * number a zero in the first says either "nothing faced away" or
     * "the title never asked", and those want opposite next steps. */
    uint32_t tris_culled, tris_cull_on;
    uint32_t cull_sets, cull_sets_on;
    /* Geometry this executor throws away without saying so. Each of these is
     * a silent hole in a model, and "a few triangles are missing from a
     * fighter" is exactly what they would look like. Counted so that the
     * answer is which one fires, not which one sounds likely. */
    uint32_t idx_dropped, idx_drop_batches, idx_max;
    uint32_t inline_dropped, inline_drop_batches, inline_max;
    uint32_t inline_batches_refused;
    uint32_t tris_skipped_badvert, verts_bad_nan, verts_bad_nolast;
    uint32_t batch_in_frame;
    /* The draw calls that used to be ignored, and how many batches the title
     * opened against how many this executor actually drew. A gap between the
     * two is geometry that never reached the screen. */
    uint32_t draw_arrays_seen, elem32_seen;
    uint32_t begins, batches_drawn, begins_empty;
    uint32_t batches_with_elem32, batches_with_draw_arrays;
    /* Which screen winding faces the camera, measured rather than
     * assumed. On a closed model the near side of every surface is the
     * side that faces us, so averaging depth per winding says which one
     * the title means by "front" -- and that is the one thing the
     * register dump cannot say, because NV2A inherits OpenGL's CW/CCW
     * enums while its window y already points down. */
    double cw_z_sum, ccw_z_sum;
    uint32_t cw_n, ccw_n;
    uint32_t clear_color;
    uint32_t clears, unhandled_total;
    uint32_t flip_read, flip_write, flip_modulo, flips;
    uint32_t tris_drawn, tris_skipped_offscreen, batches_untransformed;
    /* Why a batch came out flat. "Untextured" has two causes that look
     * identical on screen and want opposite fixes: the batch carried no
     * texture coordinates, or it did and the stage was not usable. */
    uint32_t batches_textured, batches_no_uv, batches_no_tex;
    /* Batches drawn with texture stage 0 switched off, and how many of those
     * write colour. Honouring that bit everywhere is a change that reaches
     * every screen, so the cost of announcing it without a number is high:
     * this is that number, over a whole session rather than one logged frame.
     * The ones that write no colour are the depth-only batches of point 81. */
    uint32_t batches_stage_off, batches_stage_off_colour;
    Texture  tex;
} s_gpu;

/* Set the first time a batch overruns one of the fixed buffers above, so that
 * a batch that overruns a thousand times counts as one batch, not a thousand. */
static int s_batch_overflowed;
/* Whether this batch used one of the two draw calls that used to be
 * ignored, so the count is of batches affected, not of methods. */
static int s_batch_odd_route, s_batch_draw_arrays;

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

/* NV097_SET_TEXTURE_CONTROL0 bit 30 turns a texture stage on.
 *
 * Nothing read it before, so a stage the title had switched off went on being
 * sampled with whatever texture was still bound. That is not a small detail:
 * a title draws batches whose only job is to write depth, with the texture
 * disabled and the colour mask closed, and Def Jam builds the stencil of its
 * screen titles exactly that way -- fill the band with a near z, cut the
 * letters into it with a far z, then let the paper through with LEQUAL, which
 * passes only inside the letters. Sampled anyway, that first batch picked a
 * transparent texel, the alpha test dropped the fragment, no depth was
 * written, and the paper came out as a solid slab over the title.
 *
 * Refusing to sample whenever the bit is clear looked too broad at first, and
 * the measure seemed to say so: on the character select screen 47 batches have
 * the stage off, only 5 of them write no colour, and switching the other 42 to
 * flat vertex colour turned them transparent black -- the fighters vanished
 * from their panels. The reading then was that those 42 take their colour from
 * the register combiners, which this rasteriser does not implement.
 *
 * That reading was wrong, and only half the screen could show it. Their vertex
 * colour was transparent black for a quite different reason: the one program
 * that writes it does so with an ADD, whose second source the vertex program
 * interpreter read from the wrong operand slot. With that fixed, all 42 carry
 * an opaque colour -- the bar frames, their dark tracks, the green-to-yellow
 * fills, the blue and red name plates -- and the bit can be honoured wherever
 * it is clear, which is what the hardware does. Sampled anyway, they came out
 * black: a disabled stage still has a texture bound, and these batches point
 * every vertex at texel (0,0) of it.
 *
 * A batch with the stage off AND a closed colour mask exists to write depth
 * and nothing else. For those the alpha test must not run -- it would be
 * judging a colour we cannot compute, and rejecting the fragment would drop
 * the depth with it. That is what hid the screen titles.
 *
 * A stage the title never programmed counts as on, which is what this did
 * before. RECOMP_TEX_ENABLE_BIT=0 restores the old behaviour wholesale. */
static int tex_stage0_enabled(void)
{
    static int honour = -1;
    const uint32_t i = (0x1B0C - NV_TEX_FIRST) / 4;

    if (honour < 0) {
        const char *v = getenv("RECOMP_TEX_ENABLE_BIT");
        honour = !(v && atoi(v) == 0);
    }
    if (!honour || !s_tex_set[i])
        return 1;
    return (s_tex_reg[i] & (1u << 30)) != 0;
}

/* The register combiners, captured and not executed.
 *
 * They are what an NV2A actually computes a fragment's colour with: up to
 * eight general stages, then a final combiner. This rasteriser has one hard
 * wired rule instead -- texel times vertex colour -- so every batch whose
 * combiners say something else comes out wrong, and a batch whose vertex
 * colour is zero comes out invisible. Recording what the title programs is
 * the cheapest way to find out how much of the model it actually uses before
 * deciding to implement any of it. */
static uint32_t s_comb_ctrl, s_comb_fog0, s_comb_fog1;
static uint32_t s_comb_color_icw[8], s_comb_color_ocw[8];
static uint32_t s_comb_alpha_icw[8], s_comb_alpha_ocw[8];
static uint32_t s_comb_factor0[8], s_comb_factor1[8];
/* The rest of the fragment pipeline the title programs and this rasteriser
 * does not run: the texture shader's per-stage modes, the colour keys, fog. */
static uint32_t s_shader_prog, s_shader_other, s_color_key[4];
static uint32_t s_fog_enable, s_fog_mode, s_fog_gen;

static int record_combiner_reg(uint32_t method, uint32_t param)
{
    /* Numbers from nv2a_regs.h. A first version filed SET_COLOR_KEY_COLOR
     * (0x0AE0) as the colour output words and read the combiner control at
     * 0x1294; the real ones, 0x1E40 and 0x1E60, were never captured, so the
     * frame log's "cocw" and "ctrl" meant nothing until point 139. */
    if (method >= NV097_SET_COMBINER_ALPHA_ICW && method < NV097_SET_COMBINER_ALPHA_ICW + 0x20)
        s_comb_alpha_icw[(method - NV097_SET_COMBINER_ALPHA_ICW) / 4] = param;
    else if (method == NV097_SET_COMBINER_SPECULAR_FOG_CW0) s_comb_fog0 = param;
    else if (method == NV097_SET_COMBINER_SPECULAR_FOG_CW0 + 4) s_comb_fog1 = param;
    else if (method >= NV097_SET_COMBINER_FACTOR0 && method < NV097_SET_COMBINER_FACTOR0 + 0x20)
        s_comb_factor0[(method - NV097_SET_COMBINER_FACTOR0) / 4] = param;
    else if (method >= NV097_SET_COMBINER_FACTOR1 && method < NV097_SET_COMBINER_FACTOR1 + 0x20)
        s_comb_factor1[(method - NV097_SET_COMBINER_FACTOR1) / 4] = param;
    else if (method >= NV097_SET_COMBINER_ALPHA_OCW && method < NV097_SET_COMBINER_ALPHA_OCW + 0x20)
        s_comb_alpha_ocw[(method - NV097_SET_COMBINER_ALPHA_OCW) / 4] = param;
    else if (method >= NV097_SET_COMBINER_COLOR_ICW && method < NV097_SET_COMBINER_COLOR_ICW + 0x20)
        s_comb_color_icw[(method - NV097_SET_COMBINER_COLOR_ICW) / 4] = param;
    else if (method >= NV097_SET_COLOR_KEY_COLOR && method < NV097_SET_COLOR_KEY_COLOR + 0x10)
        s_color_key[(method - NV097_SET_COLOR_KEY_COLOR) / 4] = param;
    else if (method >= NV097_SET_COMBINER_COLOR_OCW && method < NV097_SET_COMBINER_COLOR_OCW + 0x20)
        s_comb_color_ocw[(method - NV097_SET_COMBINER_COLOR_OCW) / 4] = param;
    else if (method == NV097_SET_COMBINER_CONTROL) s_comb_ctrl = param;
    else if (method == NV097_SET_SHADER_STAGE_PROGRAM) s_shader_prog = param;
    else if (method == NV097_SET_SHADER_OTHER_STAGE_INPUT) s_shader_other = param;
    else if (method == NV097_SET_FOG_ENABLE) s_fog_enable = param;
    else if (method == NV097_SET_FOG_MODE) s_fog_mode = param;
    else if (method == NV097_SET_FOG_GEN_MODE) s_fog_gen = param;
    else return 0;
    return 1;
}

/* A batch that writes no colour with its texture stage switched off is laying
 * down depth alone. See the note above tex_stage0_enabled. */
static int depth_only_batch(void)
{
    return s_gpu.color_mask == 0 && !tex_stage0_enabled();
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

/* RECOMP_INVENT=1: what each batch asks of the fragment pipeline (point 139).
 *
 * This executor draws triangles and quads, samples stage 0 and multiplies it
 * by the vertex colour. Whatever a title asks beyond that -- points and lines,
 * stages 1 to 3, a texture shader mode, register combiners with more than one
 * stage, colour keys, alpha kill, fog -- comes out wrong or not at all, and
 * nothing said how often. Each distinct request is a signature: its first
 * batch is described once ("new"), and every signature's count is printed
 * every 7200 swaps. Measure what the title uses before implementing any of it.
 *
 * route: 0 indices, 1 inline array, 2 immediate vertices, 3 opened and closed
 * with nothing in between (the begins_empty of the [GPU] report). stages: bit
 * n set when stage n is enabled; a stage 0 never programmed counts as on, as
 * in tex_stage0_enabled. key: CONTROL0 bits 0-2 (colour key mode, alpha kill)
 * of each enabled stage, four bits per stage. comb: a hash of the general
 * stages the control word says are in use, and of the final combiner. */
typedef struct {
    uint32_t prim, route, stages, prog, ctrl, comb, fog, key;
    uint32_t batches, first_flip, tex, tex_w, tex_h, n;
} InventSig;

#define INVENT_MAX 256
static InventSig s_invent[INVENT_MAX];
static uint32_t s_invent_count, s_invent_lost, s_invent_report = 7200;

static uint32_t invent_hash(uint32_t h, uint32_t v)
{
    int k;
    for (k = 0; k < 4; k++, v >>= 8)
        h = (h ^ (v & 0xFF)) * 16777619u;          /* FNV-1a */
    return h;
}

static void invent_batch(uint32_t route, uint32_t n)
{
    static int on = -1;
    InventSig s;
    uint32_t i, st, used;

    if (on < 0)
        on = getenv("RECOMP_INVENT") != NULL;
    if (!on)
        return;

    memset(&s, 0, sizeof s);
    s.prim  = s_gpu.prim;
    s.route = route;
    for (st = 0; st < 4; st++) {
        const uint32_t c0 = (0x1B0C + st * 0x40 - NV_TEX_FIRST) / 4;
        int enabled = s_tex_set[c0] ? (s_tex_reg[c0] & (1u << 30)) != 0 : st == 0;
        if (!enabled)
            continue;
        s.stages |= 1u << st;
        s.key |= (s_tex_set[c0] ? s_tex_reg[c0] & 7u : 0u) << (4 * st);
    }
    s.prog = s_shader_prog;
    s.ctrl = s_comb_ctrl;
    used = s_comb_ctrl & 0xFF;
    if (used < 1) used = 1;
    if (used > 8) used = 8;
    s.comb = 2166136261u;
    for (i = 0; i < used; i++) {
        s.comb = invent_hash(s.comb, s_comb_color_icw[i]);
        s.comb = invent_hash(s.comb, s_comb_color_ocw[i]);
        s.comb = invent_hash(s.comb, s_comb_alpha_icw[i]);
        s.comb = invent_hash(s.comb, s_comb_alpha_ocw[i]);
    }
    s.comb = invent_hash(s.comb, s_comb_fog0);
    s.comb = invent_hash(s.comb, s_comb_fog1);
    s.fog = s_fog_enable ? 0x80000000u | (s_fog_mode & 0xFFFFu) << 8 | (s_fog_gen & 0xFFu) : 0;

    for (i = 0; i < s_invent_count; i++) {
        InventSig *e = &s_invent[i];
        if (e->prim == s.prim && e->route == s.route && e->stages == s.stages
                && e->prog == s.prog && e->ctrl == s.ctrl && e->comb == s.comb
                && e->fog == s.fog && e->key == s.key) {
            e->batches++;
            break;
        }
    }
    if (i == s_invent_count) {
        if (s_invent_count < INVENT_MAX) {
            s.batches = 1;
            s.first_flip = s_gpu.flips;
            s.tex = s_gpu.tex.offset;
            s.tex_w = s_gpu.tex.width;
            s.tex_h = s_gpu.tex.height;
            s.n = n;
            s_invent[s_invent_count++] = s;
            fprintf(stderr, "  [INVENT] new #%u swap %u  prim %u route %u n %u  stages %X key %04X"
                            "  prog %08X other %08X  ctrl %08X comb %08X  fog %08X"
                            "  tex 0x%08X %ux%u\n",
                    s_invent_count - 1, s.first_flip, s.prim, s.route, n, s.stages, s.key,
                    s.prog, s_shader_other, s.ctrl, s.comb, s.fog, s.tex, s.tex_w, s.tex_h);
        } else {
            s_invent_lost++;
        }
    }

    if (s_gpu.flips >= s_invent_report) {
        static char line[INVENT_MAX * 24];
        size_t len = 0;
        s_invent_report = s_gpu.flips + 7200;
        for (i = 0; i < s_invent_count && len + 24 < sizeof line; i++)
            len += (size_t)snprintf(line + len, sizeof line - len, " #%u:%u", i,
                                    s_invent[i].batches);
        line[len] = 0;
        fprintf(stderr, "  [INVENT] swap %u, %u signatures, %u batches over the table, batches per"
                        " signature:%s\n", s_gpu.flips, s_invent_count, s_invent_lost, line);
    }
}

/* Every distinct texture a batch was drawn with, and how many batches used it.
 *
 * The per-draw verbose print shows the first few draws of the first frame,
 * which is enough to see that texturing works at all and not enough to answer
 * "is a font page ever bound". This is the same shape as the unhandled-method
 * table below it: a small set, ranked, printed with the rest of the report. */
/* Enough for a whole session, not just a screen. At 64 the front end filled
 * the table before a fight ever loaded, so the fighters' own atlases were
 * neither counted nor dumped -- and the one texture a session needs to look
 * at is never among the first sixty-four. */
#define PB_EXEC_MAX_TEXTURES 512
typedef struct {
    uint32_t offset, color, width, height, batches;
} PbTexUse;
static PbTexUse s_tex_use[PB_EXEC_MAX_TEXTURES];
static int s_tex_use_count;

/* Defined below, next to the sampler it goes through. */
static void dump_texture_bmp(uint32_t seq);

/* RECOMP_SKIP_TEX=0x8204E500[,0x...]: drop every batch bound to those
 * textures.
 *
 * When one element covers another, the question is what is underneath, and
 * no amount of state logging answers it -- the frame has to be drawn without
 * the thing on top. Lifting the covering batch out for one run says straight
 * away whether what it hides is correct and merely in the wrong order, or
 * missing too. Diagnosis only; it changes what is drawn. */
static int tex_skipped(uint32_t offset)
{
    static uint32_t list[8];
    static int count = -1;

    if (count < 0) {
        const char *spec = getenv("RECOMP_SKIP_TEX");
        count = 0;
        while (spec && *spec && count < (int)(sizeof list / sizeof list[0])) {
            list[count++] = (uint32_t)strtoul(spec, (char **)&spec, 0);
            while (*spec == ',' || *spec == ' ')
                spec++;
        }
        if (count)
            fprintf(stderr, "[NV2A] skipping %d texture(s) on request\n", count);
    }
    for (int i = 0; i < count; i++)
        if (list[i] == offset)
            return 1;
    return 0;
}

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

/* Which filters the title asks for, and how many mip levels it offers with
 * them. One line of totals answers the only question that matters here: is a
 * blurry, aliased floor our sampling, or the texture it was given. */
static struct { uint32_t min, mag, mips, batches; } s_filt[32];
static int s_filt_count;

static void note_filter_use(void)
{
    int i;
    for (i = 0; i < s_filt_count; i++)
        if (s_filt[i].min == s_gpu.tex.filter_min
         && s_filt[i].mag == s_gpu.tex.filter_mag
         && s_filt[i].mips == s_gpu.tex.mip_levels) {
            s_filt[i].batches++;
            return;
        }
    if (s_filt_count < 32) {
        s_filt[s_filt_count].min     = s_gpu.tex.filter_min;
        s_filt[s_filt_count].mag     = s_gpu.tex.filter_mag;
        s_filt[s_filt_count].mips    = s_gpu.tex.mip_levels;
        s_filt[s_filt_count].batches = 1;
        s_filt_count++;
    }
}

static const char *filter_name(uint32_t f)
{
    switch (f) {
    case 1: return "NEAREST";
    case 2: return "LINEAR";
    case 3: return "NEAREST_MIPMAP_NEAREST";
    case 4: return "LINEAR_MIPMAP_NEAREST";
    case 5: return "NEAREST_MIPMAP_LINEAR";
    case 6: return "LINEAR_MIPMAP_LINEAR";
    case 7: return "TENT_LOD0";
    case 8: return "TENT_LOD1";
    case 9: return "TENT_LOD2";
    default: return "?";
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
/* The value an attribute keeps while its array is switched off.
 *
 * SET_VERTEX_DATA4UB sets an attribute's current value, and a vertex whose
 * array for that attribute is disabled (size 0) takes it -- that is how a
 * title draws a whole batch in one colour without putting it in every vertex.
 * Only the inline path used it here, so an ordinary indexed batch with the
 * colour array off handed the vertex program (0,0,0,1) instead: transparent
 * black. Def Jam draws the stat bars of its select screen exactly that way,
 * and their combiners ask for the diffuse colour and nothing else, so they
 * came out invisible -- 113 batches of 129 in that frame have the array off.
 *
 * RECOMP_IMM_ATTR=0 goes back to ignoring it. */
static uint32_t s_imm_attr[NV_VERTEX_ATTRS];
static uint8_t  s_imm_set[NV_VERTEX_ATTRS];

static int imm_attr_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECOMP_IMM_ATTR");
        on = !(v && atoi(v) == 0);
    }
    return on;
}

/* A colour attribute the batch does not supply at all.
 *
 * fetch_attr leaves (0,0,0,1) behind, and for a colour that is transparent
 * black: a vertex program that reads v3 and writes it to oD0 then paints the
 * batch with nothing. White is the value the D3D runtime leaves in those
 * registers, and it is the neutral one for the modulate the combiners ask
 * for -- vertex_color has returned white for an absent colour since bring-up
 * for exactly this reason; the vertex program path never got the same
 * treatment. Only the two colour registers, diffuse and specular: white is
 * meaningless for a position or a texture coordinate. */
static int default_color_attr(uint32_t a, float out[4])
{
    if ((a != 3 && a != 4) || !imm_attr_enabled())
        return 0;
    out[0] = out[1] = out[2] = out[3] = 1.0f;
    return 1;
}

static int imm_attr_value(uint32_t a, float out[4])
{
    uint32_t c;

    if (a >= NV_VERTEX_ATTRS || !imm_attr_enabled())
        return 0;
    if (!s_imm_set[a])
        return default_color_attr(a, out);
    c = s_imm_attr[a];
    out[0] = (float)((c >> 16) & 0xFF) / 255.0f;   /* D3DCOLOR is ARGB */
    out[1] = (float)((c >>  8) & 0xFF) / 255.0f;
    out[2] = (float)( c        & 0xFF) / 255.0f;
    out[3] = (float)((c >> 24) & 0xFF) / 255.0f;
    return 1;
}

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
/* The triangle's three vertices, as the title named them: its position in the
 * batch's index list, and the index found there. A degenerate triangle means
 * two different things depending on these -- three equal indices is a strip
 * stitch the title asked for, three different ones collapsing to a point is
 * ours to explain. */
static uint32_t s_tri_pos[3], s_tri_vidx[3];

/* Raised by the point probe when one of the batch's triangles matches, so the
 * batch can be written out whole once its vertices are still in hand. */
static int s_batch_hit_probe;
static int      s_tri_persp;
/* The homogeneous position before the runtime's divide, when the GPU clips:
 * (X, Y, Z, w) with X = (xwin - c59.x) / rcc(w). Exact even at w = 0, where
 * the window position alone has lost it -- shadow volumes are extruded to
 * exactly there. s_tri_c59 is the viewport offset that goes with it. */
static float    s_tri_clip[3][4];
static float    s_tri_c59[3];
static int      s_tri_has_clip;
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

    if (s_gpu.alpha_test && !depth_only_batch() && !alpha_pass(color))
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

/* GPU fill (RECOMP_D3D11), defined next to the rasteriser. */
static int      d3d_bind_target(void);
static uint32_t s_d3d_show_base;           /* surface the window shows */
/* ...and its geometry when it was cleared: by the flip the title has often
 * already set up the next surface, with another pitch. */
static uint32_t s_d3d_show_rect[4], s_d3d_show_pitch, s_d3d_show_disp[2];

extern float xbox_WidescreenAspect(void);

/* The surface clip in samples: x, y, width, height -- what the GPU scissor and
 * a rectangle clear are given (point 139).
 *
 * SET_SURFACE_CLIP is in displayed pixels, and the render target is in
 * samples: Def Jam draws 640x480 with 2x multisampling across, so its target
 * is 1280 samples wide. The width was converted and the origin was not. That
 * is invisible while the clip starts at 0, which is the whole screen; a
 * sub-window starting at x = 319 was cut at (319 + 2*296)/2 = 455.5 pixels --
 * the trophy shown half on the trophy card, the mugshot with no head.
 *
 * Under RECOMP_WIDESCREEN the cutscenes' letterbox is opened as well: the
 * title restricts them to a full-width band centred in the displayed surface
 * (640x360 from y = 60, a 16:9 picture inside its 4:3 one). With the picture
 * already presented 16:9, the band only shrinks it to 2.37:1, so the clip
 * takes the surface's full height and what the projection puts above and
 * below the band is drawn instead of the black. The projection is not touched.
 * RECOMP_WIDESCREEN_CINE=0 keeps the bands. */
static void clip_in_samples(uint32_t out[4])
{
    static int open_cine = -1;
    uint32_t y = s_gpu.clip_y, h = s_gpu.clip_h;

    if (open_cine < 0) {
        const char *e = getenv("RECOMP_WIDESCREEN_CINE");
        open_cine = xbox_WidescreenAspect() > 0.0f && !(e && *e == '0');
    }
    if (open_cine && y > 0 && s_gpu.clip_x == 0
            && s_gpu.clip_w == s_d3d_show_disp[0]
            && 2 * y + h == s_d3d_show_disp[1]
            && dma_resolve(s_gpu.color_offset) == s_d3d_show_base) {
        h += 2 * y;
        y = 0;
    }
    out[0] = s_gpu.clip_x * surface_aa_x();
    out[1] = y * surface_aa_y();
    out[2] = s_gpu.clip_w * surface_aa_x();
    out[3] = h * surface_aa_y();
}

static void clear_surface(uint32_t param)
{
    if (s_gpu.flips == s_vpf_flip)
        fprintf(stderr, "  [VPF] f%u CLEAR 0x%02X clip %ux%u+%u+%u\n", s_gpu.flips, param,
                s_gpu.clip_w, s_gpu.clip_h, s_gpu.clip_x, s_gpu.clip_y);
    if (env_on(ENV_ZCLEAR_TRACE)) {
        static unsigned n;
        static uint32_t last_param = 0xFFFFFFFFu, last_val = 0xFFFFFFFFu;
        if (param != last_param || s_gpu.zclear != last_val || (++n % 600) == 0) {
            last_param = param;
            last_val = s_gpu.zclear;
            fprintf(stderr, "  [ZCLEAR] param 0x%02X zclear 0x%08X depth test %u mask %u\n",
                    param, s_gpu.zclear, s_gpu.depth_test, s_gpu.depth_mask);
        }
    }
    /* On the GPU: the same clears, on the render target and its depth. */
    if (nvd3d_active() && surface_bpp() == 4 && d3d_bind_target()) {
        uint32_t rect[4];
        uint32_t what = ((param & 1) ? 1u : 0u) | ((param & 2) ? 2u : 0u)
                      | ((param & NV097_CLEAR_COLOR_MASK) ? 4u : 0u);
        clip_in_samples(rect);
        nvd3d_clear(what, s_gpu.clear_color,
                    (float)(s_gpu.zclear >> 8) / 16777215.0f,
                    (uint8_t)(s_gpu.zclear & 0xFF), rect);
        if (param & NV097_CLEAR_COLOR_MASK) {
            if (nvd3d_trace()) {
                static uint32_t last = 0xFFFFFFFFu;
                static int said;
                uint32_t b = dma_resolve(s_gpu.color_offset);
                if (b != last && said++ < 200)
                    fprintf(stderr, "  [D3D11] f%u colour clear 0x%08X pitch %u clip %ux%u fmt 0x%08X\n",
                            s_gpu.flips, b, s_gpu.pitch, s_gpu.clip_w, s_gpu.clip_h, s_gpu.format);
                last = b;
            }
            s_gpu.clears++;
            s_gpu.color_base = dma_resolve(s_gpu.color_offset);
            s_d3d_show_base = s_gpu.color_base;
            memcpy(s_d3d_show_rect, rect, sizeof s_d3d_show_rect);
            s_d3d_show_pitch = s_gpu.pitch;
            s_d3d_show_disp[0] = rect[2] / surface_aa_x();   /* clip_w, clip_h -- */
            s_d3d_show_disp[1] = rect[3] / surface_aa_y();   /* or the opened band */
            xbox_FramebufferWindowSet(s_gpu.color_base, s_gpu.pitch);
            xbox_FramebufferWindowFormat(s_gpu.clip_w, s_gpu.clip_h,
                                         surface_bpp(), surface_aa_x());
            xbox_FramebufferWindowStart();
        }
        return;
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
    if (env_on(ENV_RASTER_TEST)) {
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
static void dump_texture_bmp_to(const char *prefix, uint32_t seq)
{
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

static void dump_texture_bmp(uint32_t seq)
{
    dump_texture_bmp_to(getenv("RECOMP_TEX_DUMP"), seq);
}

/* Texture stage n (0-3) decoded from its registers, the way the method cases
 * decode stage 0 into s_gpu.tex. Stages 1-3 are only recorded as they arrive;
 * this reads them back, so the frame log and the dump below can describe what
 * the title binds there. Returns 0 if the stage was never programmed. */
static int tex_from_stage(uint32_t n, Texture *t)
{
    const uint32_t i = (0x1B00 + n * 0x40 - NV_TEX_FIRST) / 4;
    uint32_t fmt;

    memset(t, 0, sizeof *t);
    if (n > 3 || !s_tex_set[i] || !s_tex_set[i + 1])
        return 0;
    fmt = s_tex_reg[i + 1];                            /* +0x04 FORMAT */
    t->offset = dma_resolve(s_tex_reg[i]);             /* +0x00 OFFSET */
    t->color = (fmt >> 8) & 0xFF;
    t->mip_levels = (fmt >> 16) & 0xF;
    if (tex_size_from_format(t->color)) {
        t->width  = 1u << ((fmt >> 20) & 0xF);
        t->height = 1u << ((fmt >> 24) & 0xF);
    } else {                                           /* +0x1C IMAGE_RECT */
        t->width  = s_tex_reg[i + 7] >> 16;
        t->height = s_tex_reg[i + 7] & 0xFFFF;
    }
    t->addr_u = s_tex_reg[i + 2] & 0xF;                /* +0x08 ADDRESS */
    t->addr_v = (s_tex_reg[i + 2] >> 8) & 0xF;
    t->pitch = s_tex_reg[i + 4] >> 16;                 /* +0x10 CONTROL1 */
    t->lod_bias = s_tex_reg[i + 5] & 0x1FFF;           /* +0x14 FILTER */
    t->filter_min = (s_tex_reg[i + 5] >> 16) & 0xFF;
    t->filter_mag = (s_tex_reg[i + 5] >> 24) & 0xF;
    t->palette = dma_resolve(s_tex_reg[i + 8] & ~0x3Fu); /* +0x20 PALETTE */
    t->valid = t->offset && t->width && t->height
            && (tex_size_from_format(t->color) || t->pitch);
    return 1;
}

/* The fragment pipeline of the batch the frame log is describing (point 140):
 * the texture shader program, every combiner stage the control word enables,
 * the final combiner, and stages 1-3 decoded. With RECOMP_STAGE_DUMP=<prefix>
 * each distinct texture bound to stages 1-3 is also written out once, through
 * the same sampler as stage 0 -- what a second stage holds (a light map, an
 * environment, a detail layer) is a question for the eye before it is one for
 * the combiner code. */
static void vpf_fragment_state(void)
{
    static uint32_t dumped[256];
    static uint32_t ndumped;
    const char *prefix = getenv("RECOMP_STAGE_DUMP");
    uint32_t used = s_comb_ctrl & 0xFF, i, st;

    if (used < 1) used = 1;
    if (used > 8) used = 8;
    fprintf(stderr, "  [VPF]   shader prog %08X other %08X  comb ctrl %08X final %08X %08X\n",
            s_shader_prog, s_shader_other, s_comb_ctrl, s_comb_fog0, s_comb_fog1);
    for (i = 0; i < used; i++)
        fprintf(stderr, "  [VPF]   comb s%u cicw %08X cocw %08X aicw %08X aocw %08X"
                        "  f0 %08X f1 %08X\n",
                i, s_comb_color_icw[i], s_comb_color_ocw[i],
                s_comb_alpha_icw[i], s_comb_alpha_ocw[i],
                s_comb_factor0[i], s_comb_factor1[i]);

    for (st = 1; st < 4; st++) {
        const uint32_t c0 = (0x1B0C + st * 0x40 - NV_TEX_FIRST) / 4;
        Texture t;
        if (!tex_from_stage(st, &t))
            continue;
        fprintf(stderr, "  [VPF]   stage%u offset 0x%08X %ux%u fmt %02X ctrl0 %08X addr %X/%X"
                        " filter %02X/%X mips %u  mode %u\n",
                st, t.offset, t.width, t.height, t.color, s_tex_reg[c0],
                t.addr_u, t.addr_v, t.filter_min, t.filter_mag, t.mip_levels,
                (s_shader_prog >> (5 * st)) & 0x1F);
        if (prefix && t.valid && ((s_shader_prog >> (5 * st)) & 0x1F) && ndumped < 256) {
            uint32_t k;
            for (k = 0; k < ndumped && dumped[k] != t.offset; k++)
                ;
            if (k == ndumped) {
                /* sample_texture reads s_gpu.tex: lend it the stage. The stage
                 * number goes in the name so the files sort by stage. */
                const Texture keep = s_gpu.tex;
                char pre[512];
                dumped[ndumped++] = t.offset;
                snprintf(pre, sizeof pre, "%sst%u_", prefix, st);
                s_gpu.tex = t;
                dump_texture_bmp_to(pre, ndumped);
                s_gpu.tex = keep;
            }
        }
    }
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

/* ── GPU fill (RECOMP_D3D11) ─────────────────────────────────────────────
 *
 * The same triangles, the same state, handed to nv2a_d3d11.c instead of being
 * filled here. Everything the per-pixel code below decides from s_gpu and the
 * s_tri_* globals is translated one for one, so the two paths can be compared
 * frame against frame. */
static uint32_t s_d3d_rt_h;                /* height of the bound target */

static int d3d_bind_target(void)
{
    uint32_t base, w, h;

    if (surface_bpp() != 4 || !s_gpu.color_offset || !s_gpu.pitch || !s_gpu.clip_h)
        return 0;
    base = dma_resolve(s_gpu.color_offset);
    w = s_gpu.pitch / 4;
    h = s_gpu.clip_y + surface_sample_h();
    if (surface_hits_image(base, h * s_gpu.pitch))
        return 0;
    s_d3d_rt_h = nvd3d_target(base, w, h);
    return s_d3d_rt_h != 0;
}

/* Bytes of guest memory the bound texture's texels occupy. */
static uint32_t d3d_tex_bytes(void)
{
    const uint32_t w = s_gpu.tex.width, h = s_gpu.tex.height, f = s_gpu.tex.color;
    const uint32_t bb = d3d8_format_dxt_block_bytes(f);

    if (bb)
        return ((w + 3) / 4) * ((h + 3) / 4) * bb;
    if (d3d8_format_is_swizzled(f)) {
        switch (linear_twin(f)) {
        case 0x12: case 0x1E:           return w * h * 4;
        case 0x0B: case 0x13: case 0x1F: return w * h;
        default:                        return w * h * 2;
        }
    }
    return s_gpu.tex.pitch * h;
}

static uint64_t d3d_hash(const uint8_t *p, size_t n, uint64_t h)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        memcpy(&v, p + i, 8);
        h = (h ^ v) * 0x100000001B3ull;
        h ^= h >> 29;
    }
    for (; i < n; i++)
        h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

static void d3d_decode(uint32_t *dst, uint32_t w, uint32_t h)
{
    uint32_t x, y;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            if (!sample_texture(x, y, &dst[(size_t)y * w + x]))
                dst[(size_t)y * w + x] = 0xFFFFFFFFu;
}

/* The GPU texture for what s_gpu.tex describes, looked at once per frame per
 * binding: a video or a font page rewritten between frames is uploaded again.
 * One remembered binding per stage (point 140): stages 1-3 lend their texture
 * to s_gpu.tex for the call, and sharing stage 0's memory would re-hash both
 * textures on every batch that uses two. */
static int32_t d3d_texture_slot(int slot)
{
    static uint32_t last_key[4][6];
    static uint32_t last_flip[4] = { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
    static int32_t  last[4] = { -1, -1, -1, -1 };
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t key[6], probe, bytes;
    uint64_t h;

    if (!s_gpu.tex.valid)
        return -1;
    key[0] = s_gpu.tex.offset; key[1] = s_gpu.tex.color;
    key[2] = s_gpu.tex.width;  key[3] = s_gpu.tex.height;
    key[4] = s_gpu.tex.palette; key[5] = s_gpu.tex.pitch;
    if (last_flip[slot] == s_gpu.flips && !memcmp(key, last_key[slot], sizeof key))
        return last[slot];
    memcpy(last_key[slot], key, sizeof key);
    last_flip[slot] = s_gpu.flips;
    last[slot] = -1;
    if (!sample_texture(0, 0, &probe)) {
        if (nvd3d_trace()) {
            static int said;
            if (said++ < 100)
                fprintf(stderr, "  [D3D11] texture 0x%08X fmt 0x%02X %ux%u pitch %u refused"
                        " (sampler lacks the format)\n", s_gpu.tex.offset, s_gpu.tex.color,
                        s_gpu.tex.width, s_gpu.tex.height, s_gpu.tex.pitch);
        }
        return -1;                          /* a format the sampler lacks */
    }
    bytes = d3d_tex_bytes();
    if (!bytes || bytes > 64u * 1024 * 1024)
        return -1;
    h = d3d_hash(mem + s_gpu.tex.offset, bytes, 1469598103934665603ull);
    if (s_gpu.tex.color == 0x0B && s_gpu.tex.palette)
        h = d3d_hash(mem + s_gpu.tex.palette, 1024, h);
    last[slot] = nvd3d_texture(key, h, s_gpu.tex.width, s_gpu.tex.height,
                               s_gpu.tex.offset, bytes, d3d_decode);
    return last[slot];
}

static int32_t d3d_texture(void)
{
    return d3d_texture_slot(0);
}

/* Where a vertex lands, from its position before the divide: window = X *
 * rcc(w) + c59, scaled to samples, then to NDC -- written out so that it
 * stays linear in (X, w). clip is X, Y, Z, w. The GPU vertex path computes
 * the same thing in its shader and is checked against this. */
static void d3d_clip_pos(const float clip[4], const float c59[3],
                         float ax, float ay, float pos[4])
{
    const float wc = clip[3];
    pos[0] = clip[0] * ax + (c59[0] * ax - 1.0f) * wc;
    pos[1] = -clip[1] * ay + (1.0f - c59[1] * ay) * wc;
    pos[2] = (clip[2] + c59[2] * wc) / 16777215.0f;
    pos[3] = wc;
}

static void comb_state(NvD3DState *st);   /* the combiners, with the vertex programs */

/* RECOMP_UI_LAYER=1 (DLSS step 1): the interface goes into a layer of its
 * own (NvD3DState.layer). A batch is interface when it is drawn into the shown
 * surface and its first and last vertex come out of the vertex program with
 * w == 1 exactly -- an orthographic projection. Point 149 measured that rule
 * on 15 060 frames of a fight: no perspective batch ever came after the first
 * orthographic one. Set per batch by ui_classify, from draw_primitive. */
static int s_batch_ui;

/* RECOMP_UI_LAYER_AB=<swaps>: the check without a player. Phases of that many
 * swaps turn the layer off and on in turn, with a snapshot in the middle of
 * each (<RECOMP_FB_SNAP>NNN_off.bmp, NNN_on.bmp). Within one run, so the two
 * are the same screen: an "on" no further from its "off" neighbours than they
 * are from each other is the layer adding nothing. Two separate runs drift
 * apart (the attract mode moves on at its own pace) and cannot be compared. */
static uint32_t s_ui_ab, s_ui_ab_phase, s_ui_ab_end, s_ui_ab_snap;
/* F8, under RECOMP_UI_LAYER: the layer off and on again, for the player to
 * check by eye at the same spot that nothing changes. */
static int      s_ui_f8_off;

static int ui_layer_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RECOMP_UI_LAYER");
        const char *ab = getenv("RECOMP_UI_LAYER_AB");
        const char *q = getenv("RECOMP_UI_43");       /* needs the layer (point 151) */
        const char *dl = getenv("RECOMP_DLSS");       /* so does DLSS (point 155) */
        on = (e && atoi(e) != 0) || (q && atoi(q) != 0) || (dl && atoi(dl) != 0);
        if (ab && atoi(ab) > 0) {
            s_ui_ab = (uint32_t)atoi(ab);
            on = 1;
        }
    }
    return on && !s_ui_f8_off && (!s_ui_ab || (s_ui_ab_phase & 1));
}

extern void xbox_FramebufferSnapAsk(const char *path);

static void ui_ab_tick(void)
{
    static int env = -1;
    if (env < 0) {
        const char *e = getenv("RECOMP_UI_LAYER");
        const char *q = getenv("RECOMP_UI_43");
        env = (e && atoi(e) != 0) || (q && atoi(q) != 0);
    }
    if (env && (GetAsyncKeyState(VK_F8) & 1)) {
        s_ui_f8_off = !s_ui_f8_off;
        fprintf(stderr, "  [UI] F8 swap %u : interface layer %s\n", s_gpu.flips,
                s_ui_f8_off ? "OFF" : "ON");
    }
    if (!s_ui_ab)
        return;
    if (s_gpu.flips == s_ui_ab_snap && s_ui_ab_phase) {
        const char *pre = getenv("RECOMP_FB_SNAP");
        char path[512];
        snprintf(path, sizeof path, "%s%03u_%s.bmp", pre ? pre : "ab_", s_ui_ab_phase,
                 (s_ui_ab_phase & 1) ? "on" : "off");
        xbox_FramebufferSnapAsk(path);
    }
    if (s_gpu.flips < s_ui_ab_end)
        return;
    s_ui_ab_phase++;
    s_ui_ab_end = s_gpu.flips + s_ui_ab;
    s_ui_ab_snap = s_gpu.flips + s_ui_ab / 2;
}

/* Everything about a batch that is not its vertices. Constant across a
 * batch, so the GPU vertex path computes it once for the whole draw. program:
 * drawn through a translated vertex program, the only path that carries what
 * the register combiners read (point 140). */
static void d3d_batch_state(NvD3DState *st, int has_z, int textured, int program)
{
    memset(st, 0, sizeof *st);
    st->ps = -1;
    st->stage_tex[0] = st->stage_tex[1] = st->stage_tex[2] = -1;
    st->layer = (uint32_t)s_batch_ui;
    if (s_depth_on) {
        if (has_z) {
            st->depth_enable = s_gpu.depth_test != 0;
            st->depth_write  = s_gpu.depth_mask != 0;
            st->depth_func   = s_gpu.depth_func;
        }
        if (s_gpu.stencil_test) {
            st->stencil_enable = 1;
            st->stencil_func  = s_gpu.stencil_func;
            st->stencil_ref   = s_gpu.stencil_ref & 0xFF;
            st->stencil_rmask = s_gpu.stencil_fmask & 0xFF;
            st->stencil_wmask = s_gpu.stencil_wmask & 0xFF;
            st->stencil_fail  = s_gpu.stencil_op_fail;
            st->stencil_zfail = s_gpu.stencil_op_zfail;
            st->stencil_zpass = s_gpu.stencil_op_zpass;
        }
    }
    if (s_gpu.blend_enable) {
        st->blend_enable = 1;
        st->blend_src    = s_gpu.blend_sfactor;
        st->blend_dst    = s_gpu.blend_dfactor;
        st->blend_eq     = s_gpu.blend_equation;
        st->blend_color  = s_gpu.blend_color;
    }
    st->color_mask = s_gpu.color_mask;
    if (s_gpu.alpha_test && !depth_only_batch()) {
        st->alpha_test = 1;
        st->alpha_func = s_gpu.alpha_func;
        st->alpha_ref  = s_gpu.alpha_ref;
    }
    st->texture = textured ? d3d_texture() : -1;
    if (st->texture >= 0) {
        st->addr_u = s_gpu.tex.addr_u;
        st->addr_v = s_gpu.tex.addr_v;
        st->tex_w  = s_gpu.tex.width;
        st->tex_h  = s_gpu.tex.height;
    }
    clip_in_samples(st->scissor);
    if (program)
        comb_state(st);
}

/* a, b, c are already in surface samples. */
static void d3d_triangle(const float *p[3], uint32_t argb,
                         const float uv[3][2], int textured)
{
    NvD3DVertex v[3];
    NvD3DState st;
    float W, H;
    int k;

    if (!d3d_bind_target())
        return;
    W = (float)(s_gpu.pitch / 4);
    H = (float)s_d3d_rt_h;
    for (k = 0; k < 3; k++) {
        const float wc = s_tri_has_clip ? s_tri_clip[k][3]
                       : s_tri_persp ? 1.0f / s_tri_invw[k] : 1.0f;
        const uint32_t col = s_tri_persp ? s_tri_col[k] : argb;
        float z = s_tri_has_z ? s_tri_z[k] / 16777215.0f : 0.0f;
        /* Clamping a vertex would bend the depth plane; the rasteriser
         * clamps each pixel instead (depth clip is off). A flat batch has
         * no w to undo it, so it is clamped here. */
        if (!s_tri_persp) {
            if (z < 0.0f) z = 0.0f;
            if (z > 1.0f) z = 1.0f;
        }
        if (s_tri_has_clip) {
            /* window = X * rcc(w) + c59, scaled to samples, then to NDC --
             * written out so that it stays linear in (X, w). */
            const float ax = (float)surface_aa_x() * 2.0f / W;
            const float ay = (float)surface_aa_y() * 2.0f / H;
            d3d_clip_pos(s_tri_clip[k], s_tri_c59, ax, ay, v[k].pos);
        } else {
            v[k].pos[0] = (p[k][0] / W * 2.0f - 1.0f) * wc;
            v[k].pos[1] = (1.0f - p[k][1] / H * 2.0f) * wc;
            v[k].pos[2] = z * wc;
        }
        v[k].pos[3] = wc;
        v[k].uv[0] = textured ? uv[k][0] : 0.0f;
        v[k].uv[1] = textured ? uv[k][1] : 0.0f;
        v[k].col[0] = (float)((col >> 16) & 0xFF) / 255.0f;
        v[k].col[1] = (float)((col >>  8) & 0xFF) / 255.0f;
        v[k].col[2] = (float)( col        & 0xFF) / 255.0f;
        v[k].col[3] = (float)((col >> 24) & 0xFF) / 255.0f;
    }

    d3d_batch_state(&st, s_tri_has_z, textured, 0);

    {
        static int trace = -1;
        static unsigned shown;
        if (trace < 0)
            trace = getenv("RECOMP_D3D11_TRACE") != NULL;
        if (trace && shown++ < 12)
            fprintf(stderr, "  [D3D11] tri W %.0f H %.0f  (%.1f,%.1f) (%.1f,%.1f) (%.1f,%.1f)"
                            "  ndc (%.3f,%.3f) w %.3f  uv (%.1f,%.1f) tex %d %ux%u\n",
                    W, H, p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1],
                    v[0].pos[0] / v[0].pos[3], v[0].pos[1] / v[0].pos[3], v[0].pos[3],
                    v[0].uv[0], v[0].uv[1], st.texture, st.tex_w, st.tex_h);
    }
    nvd3d_triangle(v, &st);
    s_gpu.tris_drawn++;
    s_gpu.drawn_offset = s_gpu.color_offset;
}

/* Face culling, decided in window coordinates because that is where both
 * paths agree on a triangle's facing. Neither path read the title's cull
 * state before, so every model was drawn inside and out: the far side of a
 * neck or a collar could win the depth test against the near side and show
 * through it. RECOMP_CULL=0 restores that, RECOMP_CULL_FLIP=1 takes the
 * opposite facing (the sign convention is the one thing here a register dump
 * cannot settle). */
static int tri_culled(const float a[2], const float b[2], const float c[2])
{
    static int on = -1, flip;
    float area;
    int front;

    if (on < 0) {
        const char *e = getenv("RECOMP_CULL");
        on = !(e && atoi(e) == 0);
        flip = env_on(ENV_CULL_FLIP);
    }
    if (!on || !s_gpu.cull_enable)
        return 0;
    if (s_gpu.cull_face == NV_CULL_FACE_FRONT_AND_BACK)
        return 1;
    /* A vertex behind the eye comes back from the position reconstruction
     * mirrored, so its winding means nothing; those go to the clipper. */
    if (s_tri_has_clip && (s_tri_clip[0][3] <= 0.0f || s_tri_clip[1][3] <= 0.0f
                                                    || s_tri_clip[2][3] <= 0.0f))
        return 0;
    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (area == 0.0f)
        return 0;
    /* Window y grows downwards, so a positive signed area is a clockwise
     * triangle on screen. */
    front = ((area > 0.0f) == (s_gpu.front_face == NV_FRONT_FACE_CW)) ^ flip;
    return front ? (s_gpu.cull_face == NV_CULL_FACE_FRONT)
                 : (s_gpu.cull_face == NV_CULL_FACE_BACK);
}

/* One pixel, and every triangle that covers it.
 *
 * A defect the size of a single triangle is invisible to a per-batch log:
 * under Def Jam's stray collar triangle, 89 of the frame's 384 batches cover
 * the pixel and their bounding boxes say nothing at all -- a fighter's mesh
 * spans most of the screen. RECOMP_TRI_AT=x,y prints the triangles instead,
 * with the vertices that produced them, during the frame F9 logs. */
static void tri_at_probe(const float a[2], const float b[2], const float c[2],
                         const float uv[3][2], int textured)
{
#define TRI_AT_POINTS 4
    static int on = -1, npt;
    static float px[TRI_AT_POINTS], py[TRI_AT_POINTS], pr[TRI_AT_POINTS];
    static unsigned shown;
    float lo_x, hi_x, lo_y, hi_y;
    int k, hit = 0;

    if (on < 0) {
        /* "x,y,r" or several such, separated by ';': one frame can only hold
         * one scene, so points for different scenes can be given at once and
         * only the one in view answers. */
        const char *e = getenv("RECOMP_TRI_AT");
        const char *p = e;
        npt = 0;
        while (p && *p && npt < TRI_AT_POINTS) {
            pr[npt] = 0.0f;
            if (sscanf(p, "%f,%f,%f", &px[npt], &py[npt], &pr[npt]) < 2)
                break;
            npt++;
            p = strchr(p, ';');
            if (p) p++;
        }
        on = npt > 0;
    }
    if (on <= 0 || s_gpu.flips != s_vpf_flip || shown >= 600)
        return;

    /* Geometry behind the eye comes back mirrored and hundreds of thousands
     * of pixels wide; its bounding box covers every point on screen and it
     * drowned the first run's output in 400 lines of noise. It is not what
     * anybody is looking at. */
    if (s_tri_has_clip && (s_tri_clip[0][3] <= 0.0f || s_tri_clip[1][3] <= 0.0f
                                                    || s_tri_clip[2][3] <= 0.0f))
        return;
    lo_x = fminf(a[0], fminf(b[0], c[0]));
    hi_x = fmaxf(a[0], fmaxf(b[0], c[0]));
    lo_y = fminf(a[1], fminf(b[1], c[1]));
    hi_y = fmaxf(a[1], fmaxf(b[1], c[1]));
    if (hi_x - lo_x > 400.0f || hi_y - lo_y > 400.0f)
        return;

    for (k = 0; k < npt && !hit; k++) {
        if (pr[k] > 0.0f) {
            hit = !(hi_x < px[k] - pr[k] || lo_x > px[k] + pr[k]
                 || hi_y < py[k] - pr[k] || lo_y > py[k] + pr[k]);
        } else {
            const float w0 = (b[0]-a[0])*(py[k]-a[1]) - (b[1]-a[1])*(px[k]-a[0]);
            const float w1 = (c[0]-b[0])*(py[k]-b[1]) - (c[1]-b[1])*(px[k]-b[0]);
            const float w2 = (a[0]-c[0])*(py[k]-c[1]) - (a[1]-c[1])*(px[k]-c[0]);
            hit = (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0);
        }
    }
    if (!hit)
        return;
    s_batch_hit_probe = 1;
    if (++shown == 600)
        fprintf(stderr, "  [TRIAT] (600 triangles printed, stopping)\n");
#undef TRI_AT_POINTS
    fprintf(stderr, "  [TRIAT] b%-3u (%7.1f,%7.1f) (%7.1f,%7.1f) (%7.1f,%7.1f)"
                    "  z %9.0f %9.0f %9.0f  w %8.3f %8.3f %8.3f"
                    "  uv (%.0f,%.0f) (%.0f,%.0f) (%.0f,%.0f) t%d"
                    "  pos %u,%u,%u  vtx %u,%u,%u"
                    "  d0 %08X %08X %08X  tex 0x%08X\n",
            s_gpu.batch_in_frame,
            a[0], a[1], b[0], b[1], c[0], c[1],
            s_tri_has_z ? s_tri_z[0] : 0.0f,
            s_tri_has_z ? s_tri_z[1] : 0.0f,
            s_tri_has_z ? s_tri_z[2] : 0.0f,
            s_tri_has_clip ? s_tri_clip[0][3] : (s_tri_persp ? 1.0f / s_tri_invw[0] : 1.0f),
            s_tri_has_clip ? s_tri_clip[1][3] : (s_tri_persp ? 1.0f / s_tri_invw[1] : 1.0f),
            s_tri_has_clip ? s_tri_clip[2][3] : (s_tri_persp ? 1.0f / s_tri_invw[2] : 1.0f),
            uv ? uv[0][0] : 0.0f, uv ? uv[0][1] : 0.0f,
            uv ? uv[1][0] : 0.0f, uv ? uv[1][1] : 0.0f,
            uv ? uv[2][0] : 0.0f, uv ? uv[2][1] : 0.0f,
            textured,
            s_tri_pos[0], s_tri_pos[1], s_tri_pos[2],
            s_tri_vidx[0], s_tri_vidx[1], s_tri_vidx[2],
            s_tri_persp ? s_tri_col[0] : 0u,
            s_tri_persp ? s_tri_col[1] : 0u,
            s_tri_persp ? s_tri_col[2] : 0u,
            s_gpu.tex.offset);
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
    int textured = uv && s_gpu.tex.valid && tex_stage0_enabled();

    /* Vertices arrive in displayed pixels; the surface is addressed in
     * samples. On a multisampled target the two differ, and writing pixel
     * coordinates straight into a 2x-wide surface puts the whole frame in the
     * left half of it -- which the display resolve then squeezes into the left
     * half of the screen. Scaling here rather than at every call site keeps
     * the barycentric weights unchanged, so texturing is unaffected. */
    tri_at_probe(a, b, c, uv, textured);

    const float sx = (float)surface_aa_x(), sy = (float)surface_aa_y();
    const float a2[2] = { a[0] * sx, a[1] * sy };
    const float b2[2] = { b[0] * sx, b[1] * sy };
    const float c2[2] = { c[0] * sx, c[1] * sy };

    a = a2; b = b2; c = c2;

    if (bpp != 4 && bpp != 2)
        return;

    if (s_gpu.cull_enable) {
        s_gpu.tris_cull_on++;
        if (s_tri_has_z && env_on(ENV_CULL_TRACE)) {
            const float ar = (b[0] - a[0]) * (c[1] - a[1])
                           - (b[1] - a[1]) * (c[0] - a[0]);
            const double z = ((double)s_tri_z[0] + s_tri_z[1] + s_tri_z[2]) / 3.0;
            if (ar > 0.0f)      { s_gpu.cw_z_sum  += z; s_gpu.cw_n++;  }
            else if (ar < 0.0f) { s_gpu.ccw_z_sum += z; s_gpu.ccw_n++; }
        }
    }
    if (tri_culled(a, b, c)) {
        s_gpu.tris_culled++;
        return;
    }

    if (bpp == 4 && nvd3d_active()) {
        const float *pts[3];
        pts[0] = a; pts[1] = b; pts[2] = c;
        d3d_triangle(pts, argb, uv, textured);
        return;
    }

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
    if (s_tri_has_z && s_gpu.depth_mask && s_tri_z[0] < 1.0f && env_on(ENV_ZCLEAR_TRACE)) {
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

    if (!fetch_attr(color_attr(), index, c)
            && !imm_attr_value(3, c))
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
 * goes to the temp register, the ILU result to R1 when a MAC operation shares
 * the instruction and to the temp register when it does not, and one of the
 * two to the output register. R12 is oPos.
 *
 * The three operand slots are shared by both units, and the MAC unit is a
 * multiply-add, so an opcode reads the slots its A*B + C form needs: MUL and
 * the dot products A and B, MAD all three, MOV and ARL A alone, and ADD --
 * being A*1 + C -- A and C. Every ILU operation reads C.
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
#define VP_OUT_D1     4             /* the specular colour, read by the combiners */
#define VP_OUT_PTS    6             /* a point sprite's size in pixels */

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

/* Bumped whenever the program text or its entry point changes, so the
 * decoded form below is rebuilt only then. */
static uint32_t s_vp_gen = 1;
static uint32_t s_vp_c_gen;              /* bumped by every constant write */

static int vp_method(uint32_t method, uint32_t param)
{
    if (method >= NV097_SET_TRANSFORM_PROGRAM
            && method < NV097_SET_TRANSFORM_PROGRAM + 32 * 4) {
        uint32_t slot = (method - NV097_SET_TRANSFORM_PROGRAM) / 4;
        if (s_vp.load < VP_MAX_INSNS) {
            s_vp.prog[s_vp.load][slot % 4] = param;
            s_vp_gen++;
            if (slot % 4 == 3)
                s_vp.load++;
        }
        return 1;
    }
    if (method >= NV097_SET_TRANSFORM_CONSTANT
            && method < NV097_SET_TRANSFORM_CONSTANT + 16 * 4) {
        uint32_t slot = (method - NV097_SET_TRANSFORM_CONSTANT) / 4;
        if (s_vp.c_load < VP_CONSTANTS) {
            /* Under F9, every low constant the title uploads during the
             * logged frame. A program reading a constant the title never
             * wrote gets zero, and only the upload log distinguishes "never
             * sent" from "sent and lost on our side". */
            if (s_gpu.flips == s_vpf_flip && s_vp.c_load < 16) {
                float f;
                memcpy(&f, &param, 4);
                fprintf(stderr, "  [VPFC]  c%u.%u = %g", s_vp.c_load,
                        slot % 4, f);
                fprintf(stderr, "\n");
            }
            memcpy(&s_vp.c[s_vp.c_load][slot % 4], &param, 4);
            s_vp_c_gen++;
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
        s_vp_c_gen++;
        return 1;
    }
    if (method >= NV097_SET_VIEWPORT_SCALE && method < NV097_SET_VIEWPORT_SCALE + 16) {
        memcpy(&s_vp.c[58][(method - NV097_SET_VIEWPORT_SCALE) / 4], &param, 4);
        s_vp_c_gen++;
        return 1;
    }
    switch (method) {
    case NV097_SET_TRANSFORM_EXECUTION_MODE: s_vp.mode = param;   return 1;
    case NV097_SET_TRANSFORM_PROGRAM_LOAD:   s_vp.load = param;   return 1;
    case NV097_SET_TRANSFORM_PROGRAM_START:  s_vp.start = param; s_vp_gen++; return 1;
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

/* One instruction, decoded once per program rather than once per vertex:
 * the bit fields below were most of the interpreter's time. */
typedef struct {
    uint8_t mac, ilu;
    uint8_t mux[3], reg[3], neg[3], swz[3];
    uint8_t need;            /* bit k: operand k is read */
    uint8_t input, rel, tmp, mac_mask, ilu_mask, orb, out_mask, from_ilu, last;
    uint16_t cidx;
} VpInsn;

static VpInsn   s_vp_code[VP_MAX_INSNS];
static uint32_t s_vp_code_len, s_vp_code_gen;
static uint32_t s_vp_inputs;             /* attributes the program reads */

static void vp_decode(void)
{
    uint32_t pc, n = 0;

    s_vp_inputs = 0;
    for (pc = s_vp.start; pc < VP_MAX_INSNS; pc++) {
        const uint32_t *w = s_vp.prog[pc];
        VpInsn *d = &s_vp_code[n++];
        uint32_t k;

        d->mac = (uint8_t)VP_BITS(w[1], 21, 4);
        d->ilu = (uint8_t)VP_BITS(w[1], 25, 3);
        d->mux[0] = (uint8_t)VP_BITS(w[2], 26, 2);
        d->reg[0] = (uint8_t)VP_BITS(w[2], 28, 4);
        d->neg[0] = (uint8_t)VP_BITS(w[1], 8, 1);
        d->swz[0] = (uint8_t)VP_BITS(w[1], 0, 8);
        d->mux[1] = (uint8_t)VP_BITS(w[2], 11, 2);
        d->reg[1] = (uint8_t)VP_BITS(w[2], 13, 4);
        d->neg[1] = (uint8_t)VP_BITS(w[2], 25, 1);
        d->swz[1] = (uint8_t)VP_BITS(w[2], 17, 8);
        d->mux[2] = (uint8_t)VP_BITS(w[3], 28, 2);
        d->reg[2] = (uint8_t)((VP_BITS(w[2], 0, 2) << 2) | VP_BITS(w[3], 30, 2));
        d->neg[2] = (uint8_t)VP_BITS(w[2], 10, 1);
        d->swz[2] = (uint8_t)VP_BITS(w[2], 2, 8);
        d->input = (uint8_t)VP_BITS(w[1], 9, 4);
        d->cidx = (uint16_t)VP_BITS(w[1], 13, 8);
        d->rel = (uint8_t)VP_BITS(w[3], 1, 1);
        d->tmp = (uint8_t)VP_BITS(w[3], 20, 4);
        d->mac_mask = (uint8_t)VP_BITS(w[3], 24, 4);
        d->ilu_mask = (uint8_t)VP_BITS(w[3], 16, 4);
        d->orb = (uint8_t)VP_BITS(w[3], 3, 8);
        d->out_mask = (uint8_t)VP_BITS(w[3], 12, 4);
        d->from_ilu = (uint8_t)VP_BITS(w[3], 2, 1);
        d->last = (uint8_t)(w[3] & 1);

        /* Which operand slots each operation reads. There are three slots for
         * the whole instruction, and the MAC unit is a multiply-add: every
         * operation is expressed as A*B + C, so which slots an opcode uses is
         * a property of that expression and not of its arity.
         *
         * ADD is the one that surprises: it is A*1 + C, so its second source
         * arrives in slot C, not in slot B. Reading B instead gave the front
         * end's coloured shapes "a temporary plus the vertex position", which
         * clamps to transparent black -- the stat bars of the character select
         * screen, their frames and their fills, all drawn by the one program
         * whose colour instruction is an ADD (point 82). */
        d->need = 0;
        if (d->mac)                                /* A: every MAC op */
            d->need |= 1;
        if (d->mac == 2 || (d->mac >= 4 && d->mac <= 12))
            d->need |= 2;                          /* B: MUL, MAD, DP3..SGE */
        if (d->mac == 3 || d->mac == 4 || d->ilu)
            d->need |= 4;                          /* C: ADD, MAD, every ILU */
        for (k = 0; k < 3; k++)
            if ((d->need & (1u << k)) && d->mux[k] == 2)
                s_vp_inputs |= 1u << d->input;
        if (d->last)
            break;
    }
    s_vp_code_len = n;
    s_vp_code_gen = s_vp_gen;
}

static void vp_read(VpState *s, const VpInsn *d, uint32_t k, float out[4])
{
    const float *src;
    static const float zero[4];
    const uint32_t swz = d->swz[k];

    switch (d->mux[k]) {
    case 1: src = vp_temp(s, d->reg[k]); break;
    case 2: src = s->v[d->input]; break;
    case 3: {
        int idx = (int)d->cidx;
        if (d->rel)
            idx += s->a0;
        src = (idx >= 0 && idx < VP_CONSTANTS) ? s_vp.c[idx] : zero;
        break;
    }
    default: src = zero; break;
    }
    if (swz == 0x1B) {                      /* xyzw, the usual case */
        memcpy(out, src, 4 * sizeof(float));
    } else {
        out[0] = src[(swz >> 6) & 3];
        out[1] = src[(swz >> 4) & 3];
        out[2] = src[(swz >> 2) & 3];
        out[3] = src[ swz       & 3];
    }
    if (d->neg[k]) {
        out[0] = -out[0]; out[1] = -out[1]; out[2] = -out[2]; out[3] = -out[3];
    }
}

static void vp_write(float *dst, const float val[4], uint32_t mask)
{
    if (mask == 0xF) {
        memcpy(dst, val, 4 * sizeof(float));
        return;
    }
    if (mask & 8) dst[0] = val[0];
    if (mask & 4) dst[1] = val[1];
    if (mask & 2) dst[2] = val[2];
    if (mask & 1) dst[3] = val[3];
}

/* Run the current program for one vertex. Returns 0 if it has no end. */
static int vp_run(VpState *s)
{
    uint32_t pc;

    if (s_vp_code_gen != s_vp_gen)
        vp_decode();

    for (pc = 0; pc < s_vp_code_len; pc++) {
        const VpInsn *ins = &s_vp_code[pc];
        const uint32_t mac = ins->mac, ilu = ins->ilu;
        float a[4], b[4], c[4], m[4] = {0, 0, 0, 0}, l[4] = {0, 0, 0, 0};
        float d;
        uint32_t k;

        if (ins->need & 1) vp_read(s, ins, 0, a);
        if (ins->need & 2) vp_read(s, ins, 1, b);
        if (ins->need & 4) vp_read(s, ins, 2, c);

        switch (mac) {
        case 0: break;                                      /* NOP */
        case 1: memcpy(m, a, sizeof m); break;              /* MOV */
        case 2: for (k = 0; k < 4; k++) m[k] = a[k] * b[k]; break;
        case 3: for (k = 0; k < 4; k++) m[k] = a[k] + c[k]; break; /* ADD: A*1+C */
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
            vp_write(vp_temp(s, ins->tmp), m, ins->mac_mask);
        /* The ILU writes R1 only when it is paired with a MAC operation; alone,
         * it writes the instruction's own temporary. The ring ropes of Club 357
         * end on an unpaired "rcp R11.x, R12.w" and read R11.x back: sent to R1,
         * every vertex collapsed onto the viewport centre (point 138). */
        if (ilu)
            vp_write(mac ? s->r[1] : vp_temp(s, ins->tmp), l, ins->ilu_mask);
        if (ins->orb < 16)
            vp_write(s->o[ins->orb], ins->from_ilu ? l : m, ins->out_mask);
        if (ins->last)
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
    float clip[3];           /* X, Y, Z before the divide (see s_tri_clip) */
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
    if (s_vp_code_gen != s_vp_gen)
        vp_decode();
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if ((s_vp_inputs >> a) & 1 || env_on(ENV_VP_TRACE))
            if (!fetch_attr(&s_gpu.attr[a], index, s.v[a]))
                imm_attr_value(a, s.v[a]);

    out->ok = vp_run(&s);
    if (!out->ok)
        s_gpu.verts_bad_nolast++;     /* the program never reached its last insn */
    s_vp.runs++;
    /* The Xbox D3D runtime ends every program with the viewport transform
     * and the divide -- `rcc r1.x, oPos.w` then `mad oPos.xyz, oPos, r1.x,
     * c59` -- so oPos.xy already are window coordinates. w is kept only for
     * perspective-correct interpolation, which this rasteriser does not do. */
    w = s.o[VP_OUT_POS][3];
    if (!(w == w) || !(s.o[VP_OUT_POS][0] == s.o[VP_OUT_POS][0])) {
        if (out->ok)
            s_gpu.verts_bad_nan++;
        out->ok = 0;
    }
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
    /* Undo `mad oPos.xyz, oPos, rcc(w), c59` with the same rcc the program
     * used, rather than multiplying by w: at w = 0 that gives zero. */
    {
        float rc = w != 0.0f ? 1.0f / w : 1.84467e19f;
        if (rc > 0) rc = rc < 5.42101e-20f ? 5.42101e-20f : rc > 1.84467e19f ? 1.84467e19f : rc;
        else rc = rc > -5.42101e-20f ? -5.42101e-20f : rc < -1.84467e19f ? -1.84467e19f : rc;
        out->clip[0] = (out->xy[0] - s_vp.c[59][0]) / rc;
        out->clip[1] = (out->xy[1] - s_vp.c[59][1]) / rc;
        out->clip[2] = (out->z - s_vp.c[59][2]) / rc;
    }
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

    if (env_on(ENV_VP_TRACE)) {
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
        /* Every other triangle of a strip comes out mirrored if its three
         * vertices are taken in order; the hardware swaps the first two so
         * that the whole strip keeps one facing. It made no difference while
         * both windings were filled, and it decides the answer once the
         * title's cull state is honoured. The provoking vertex stays i. */
        for (i = 0; i + 2 < n; i++) {
            if (i & 1u) TRI(i + 1, i, i + 2, i);
            else        TRI(i, i + 1, i + 2, i);
        }
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

/* Set while the GPU path draws: triangles crossing the eye plane go to it
 * whole, and its clipper cuts them where the CPU path could only drop them. */
static int s_gpu_clips;


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
    /* The GPU clips homogeneous triangles itself, so one reaching behind the
     * eye is handed over whole, with its real (negative) w. */
    if (!s_batch_flat_z && s_gpu_clips) {
        const VpVertex *pv[3];
        int k;
        pv[0] = p0; pv[1] = p1; pv[2] = p2;
        for (k = 0; k < 3; k++) {
            s_tri_clip[k][0] = pv[k]->clip[0];
            s_tri_clip[k][1] = pv[k]->clip[1];
            s_tri_clip[k][2] = pv[k]->clip[2];
            s_tri_clip[k][3] = pv[k]->w;
        }
        s_tri_c59[0] = s_vp.c[59][0];
        s_tri_c59[1] = s_vp.c[59][1];
        s_tri_c59[2] = s_vp.c[59][2];
        s_tri_has_clip = 1;
        s_tri_persp = 1;
    }
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
    s_tri_has_clip = 0;
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
    if (nvd3d_active() && surface_bpp() == 4) {
        if (v[0]->w <= 0.0f && v[1]->w <= 0.0f && v[2]->w <= 0.0f)
            return;                        /* behind the eye, or all at infinity */
        s_gpu_clips = 1;
        vp_draw_triangle(v[0], v[1], v[2], flat, textured);
        s_gpu_clips = 0;
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

/* A whole batch, written out so it can be taken apart away from the game.
 *
 * Reproducing a defect means reaching the same scene by hand, which is slow
 * and never frames twice the same. Everything a batch needs is knowable at
 * the moment it draws -- its index list, the raw bytes of every vertex
 * attribute, the vertex program and its constants -- so RECOMP_BATCH_DUMP
 * writes that out and the analysis happens offline, as many times as it takes.
 * Written for the batches the point probe matches, during the frame F9 logs. */
static void dump_batch(const char *why)
{
    static int seq;
    const char *prefix = getenv("RECOMP_BATCH_DUMP");
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    char path[512];
    FILE *f;
    uint32_t i, a, lo = 0xFFFFFFFFu, hi = 0;

    if (!prefix || !s_gpu.idx_count)
        return;
    {   /* La foule a elle aussi cinq flux et se dessine avant les combattants :
         * trente-deux fichiers ne suffisent pas a les atteindre. */
        const char *max = getenv("RECOMP_BATCH_DUMP_MAX");
        if (seq >= (max ? atoi(max) : 32))
            return;
    }
    snprintf(path, sizeof path, "%s_b%03u_%02d.txt", prefix,
             s_gpu.batch_in_frame, seq);
    f = fopen(path, "w");
    if (!f)
        return;
    seq++;

    fprintf(f, "# batch %u of frame %u (%s)\n", s_gpu.batch_in_frame,
            s_gpu.flips, why);
    fprintf(f, "prim %u\nindices %u\n", s_gpu.prim, s_gpu.idx_count);
    fprintf(f, "texture 0x%08X %ux%u fmt 0x%02X\n", s_gpu.tex.offset,
            s_gpu.tex.width, s_gpu.tex.height, s_gpu.tex.color);
    fprintf(f, "state depth_test %u func 0x%X mask %u  alpha %u func 0x%X ref %u"
               "  blend %u  cull %u face 0x%X front 0x%X  inline %u\n",
            s_gpu.depth_test, s_gpu.depth_func, s_gpu.depth_mask,
            s_gpu.alpha_test, s_gpu.alpha_func, s_gpu.alpha_ref,
            s_gpu.blend_enable, s_gpu.cull_enable, s_gpu.cull_face,
            s_gpu.front_face, s_gpu.inline_active);

    fprintf(f, "\n# the index list, in the order the title sent it\nidx");
    for (i = 0; i < s_gpu.idx_count; i++) {
        fprintf(f, " %u", s_gpu.idx[i]);
        if (s_gpu.idx[i] < lo) lo = s_gpu.idx[i];
        if (s_gpu.idx[i] > hi) hi = s_gpu.idx[i];
    }
    fprintf(f, "\nindex range %u..%u\n", lo, hi);

    fprintf(f, "\n# vertex attribute streams\n");
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if (s_gpu.attr[a].size && s_gpu.attr[a].stride)
            fprintf(f, "attr %u offset 0x%08X type %u size %u stride %u\n",
                    a, s_gpu.attr[a].offset, s_gpu.attr[a].type,
                    s_gpu.attr[a].size, s_gpu.attr[a].stride);

    /* The raw bytes behind each vertex, so a decode can be re-done from
     * scratch rather than trusted. */
    fprintf(f, "\n# raw vertex bytes, one line per index used\n");
    if (!s_gpu.inline_active && hi >= lo && hi - lo < 8192) {
        for (i = lo; i <= hi; i++) {
            fprintf(f, "v %u", i);
            for (a = 0; a < NV_VERTEX_ATTRS; a++) {
                const VertexAttr *at = &s_gpu.attr[a];
                uint32_t b;
                if (!at->size || !at->stride || !at->offset)
                    continue;
                fprintf(f, " | a%u", a);
                for (b = 0; b < at->stride && b < 32; b++)
                    fprintf(f, " %02X", mem[at->offset + (size_t)i * at->stride + b]);
            }
            fprintf(f, "\n");
        }
    } else if (s_gpu.inline_active) {
        fprintf(f, "# inline payload, %u dwords\ninline", s_gpu.inline_count);
        for (i = 0; i < s_gpu.inline_count; i++)
            fprintf(f, " %08X", s_gpu.inline_buf[i]);
        fprintf(f, "\n");
    }

    /* And what our own pipeline made of them, to compare against. */
    fprintf(f, "\n# transformed, as this executor produced them\n");
    for (i = 0; i < s_gpu.idx_count; i++) {
        const VpVertex *p = &s_vp_verts[i];
        fprintf(f, "t %u idx %u  xy %.4f %.4f  z %.1f  w %.6f"
                   "  uv %.4f %.4f  d0 %08X ok %d\n",
                i, s_gpu.idx[i], p->xy[0], p->xy[1], p->z, p->w,
                p->uv[0], p->uv[1], p->argb, p->ok);
    }

    fprintf(f, "\n# vertex program from %u, raw instruction words\n", s_vp.start);
    for (i = s_vp.start; i < VP_MAX_INSNS; i++) {
        fprintf(f, "i %u %08X %08X %08X %08X\n", i, s_vp.prog[i][0],
                s_vp.prog[i][1], s_vp.prog[i][2], s_vp.prog[i][3]);
        if (s_vp.prog[i][3] & 1)
            break;
    }
    fprintf(f, "\n# non-zero constants\n");
    for (i = 0; i < VP_CONSTANTS; i++)
        if (s_vp.c[i][0] || s_vp.c[i][1] || s_vp.c[i][2] || s_vp.c[i][3])
            fprintf(f, "c %u %.6f %.6f %.6f %.6f\n", i, s_vp.c[i][0],
                    s_vp.c[i][1], s_vp.c[i][2], s_vp.c[i][3]);
    fclose(f);
    fprintf(stderr, "  [BATCHDUMP] %s\n", path);
}


/* Whether the current frame is the one being logged: RECOMP_VP_FRAME=N,
 * RECOMP_VP_FRAME_EVERY=N, RECOMP_VP_FRAME_AT=S (the first whole frame after S
 * seconds), or F9 whenever someone playing wants the current scene. The
 * keyboard and the clock are looked at once per swapped frame, not once per
 * batch: that poll was 7% of the NV2A thread in the menus and 1.9% in a fight,
 * spent asking whether F9 was down. */
static int vpf_this_frame(void)
{
    static int want = -1, every, at;
    static ULONGLONG start;
    static uint32_t polled = 0xFFFFFFFFu;

    if (want < 0) {
        want  = getenv("RECOMP_VP_FRAME") ? atoi(getenv("RECOMP_VP_FRAME")) : 0;
        every = getenv("RECOMP_VP_FRAME_EVERY") ? atoi(getenv("RECOMP_VP_FRAME_EVERY")) : 0;
        at    = getenv("RECOMP_VP_FRAME_AT") ? atoi(getenv("RECOMP_VP_FRAME_AT")) : 0;
        start = GetTickCount64();
    }
    if (polled != s_gpu.flips) {
        polled = s_gpu.flips;
        if (at > 0 && !want && GetTickCount64() - start >= (ULONGLONG)at * 1000)
            want = (int)s_gpu.flips + 1;
        if ((GetAsyncKeyState(VK_F9) & 1) && s_gpu.flips + 1 != (uint32_t)want) {
            want = (int)s_gpu.flips + 1;
            fprintf(stderr, "  [VPF] F9: logging frame %d\n", want);
        }
    }
    return (want && s_gpu.flips == (uint32_t)want)
        || (every > 0 && s_gpu.flips % (uint32_t)every == 0);
}

/* ── Vertex programs on the GPU (RECOMP_VP_GPU) ─────────────────────── */
/*
 * Measured in a fight: the interpreter above -- vp_run, vp_read, fetch_attr,
 * vp_transform, vp_draw_triangle -- is 55% of the NV2A thread, the thread the
 * game waits on 88% of its time, while the GPU driver does not even show in
 * the profile. The work is per vertex, so no reduction of the resolution could
 * touch it. The program is therefore translated to HLSL, once per program,
 * and run as the vertex shader; the batch goes over whole.
 *
 * The translation is instruction for instruction with vp_run: the same three
 * operand slots with the same meaning (ADD is A*1 + C, point 83), the same
 * write order -- MAC to its temporary, then the ILU (to R1 if paired), then the output --
 * the same clamps on RCC, and R12 as oPos. The end of the shader is vp_transform
 * and d3d_triangle's clip branch, through d3d_clip_pos's formula, so the
 * pixel shader receives exactly what the CPU path would have given it.
 *
 * Only the batches where that is certainly true go this way. Everything else
 * stays on the CPU path, which is the reference and does not change:
 *   - a batch whose first vertex has z = 0, which may be a flat 2D overlay
 *     (those change the depth rules, and they are small);
 *   - culling on (4 writes in 156335 in this title), the logged frame, the
 *     per-pixel probe and the batch dump -- the diagnostics read the vertices;
 *   - a program with no end, a 16-bit surface, a range the upload refuses.
 *
 * This is the default since it was checked in game (point 93): every program
 * met in the menus and in three arenas came back identical to the interpreter
 * within 7e-7, and the same fight went from 49-51 frames a second to the 120
 * of the display. RECOMP_VP_GPU=0 goes back to the interpreter; =2 draws on
 * the CPU as before but also runs the first batches of every program through
 * the shader with stream output, and prints how far the two disagree -- the
 * way to check a program the title has not shown yet.
 */
static int vp_gpu_mode(void)
{
    static int m = -1;
    if (m < 0) {
        const char *e = getenv("RECOMP_VP_GPU");
        m = e ? atoi(e) : 1;
        if (m < 0 || m > 2)
            m = 0;
    }
    return m;
}

static const char s_vp_hlsl_head[] =
"cbuffer VpC : register(b1) { float4 c[192]; };\n"
"cbuffer VpA : register(b2) { uint4 attr[16]; float4 absent[16]; float4 xf0; float4 xf1; };\n"
"ByteAddressBuffer vd : register(t1);\n"
/* The fields after clip are what the register combiners read (point 140): the
 * specular colour and the coordinates of stages 1-3. They come last so that
 * the fixed pixel shader, which declares only the first three, still links. */
"struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 col : COLOR0;\n"
"               float clip : SV_ClipDistance0; float4 col1 : COLOR1;\n"
"               float4 uv1 : TEXCOORD1; float4 uv2 : TEXCOORD2; float4 uv3 : TEXCOORD3; };\n"
"static const float4 Z4 = float4(0.0, 0.0, 0.0, 0.0);\n"
/* A dword at any byte address: the title's bytes and shorts need not be
 * aligned, and a raw buffer only loads aligned dwords. */
"uint rd32(uint a) {\n"
"  uint o = a & 3u; uint w0 = vd.Load(a - o);\n"
"  if (o == 0u) return w0;\n"
"  return (w0 >> (o * 8u)) | (vd.Load(a - o + 4u) << (32u - o * 8u));\n"
"}\n"
"float s16(uint a) { return (float)(asint(rd32(a) << 16) >> 16); }\n"
"float n16(uint a) { float v = s16(a); return v < -32767.0 ? -1.0 : v / 32767.0; }\n"
/* fetch_attr, format for format. */
"float4 vfetch(uint k, uint i) {\n"
"  uint4 d = attr[k];\n"
"  if (d.w == 0u) return absent[k];\n"
"  uint a = d.x + i * d.y;\n"
"  float4 o = float4(0.0, 0.0, 0.0, 1.0);\n"
"  if (d.z == 0u) {\n"
"    uint v = rd32(a);\n"
"    o = float4((v >> 16) & 255u, (v >> 8) & 255u, v & 255u, v >> 24) / 255.0;\n"
"  } else if (d.z == 2u) {\n"
"    o.x = asfloat(rd32(a));\n"
"    if (d.w > 1u) o.y = asfloat(rd32(a + 4u));\n"
"    if (d.w > 2u) o.z = asfloat(rd32(a + 8u));\n"
"    if (d.w > 3u) o.w = asfloat(rd32(a + 12u));\n"
"  } else if (d.z == 4u) {\n"
"    uint v = rd32(a);\n"
"    o.x = (v & 255u) / 255.0;\n"
"    if (d.w > 1u) o.y = ((v >> 8) & 255u) / 255.0;\n"
"    if (d.w > 2u) o.z = ((v >> 16) & 255u) / 255.0;\n"
"    if (d.w > 3u) o.w = (v >> 24) / 255.0;\n"
"  } else if (d.z == 1u) {\n"
"    o.x = n16(a);\n"
"    if (d.w > 1u) o.y = n16(a + 2u);\n"
"    if (d.w > 2u) o.z = n16(a + 4u);\n"
"    if (d.w > 3u) o.w = n16(a + 6u);\n"
"  } else if (d.z == 5u) {\n"
"    o.x = s16(a);\n"
"    if (d.w > 1u) o.y = s16(a + 2u);\n"
"    if (d.w > 2u) o.z = s16(a + 4u);\n"
"    if (d.w > 3u) o.w = s16(a + 6u);\n"
"  } else if (d.z == 6u) {\n"
"    uint v = rd32(a);\n"
"    o.x = (float)(asint(v << 21) >> 21) / 1023.0;\n"
"    o.y = (float)(asint(v << 10) >> 21) / 1023.0;\n"
"    o.z = (float)(asint(v) >> 22) / 511.0;\n"
"  }\n"
"  return o;\n"
"}\n"
"float4 cr(int i) { return (i >= 0 && i < 192) ? c[clamp(i, 0, 191)] : Z4; }\n"
/* The ILU, scalar on C.x, with vp_run's clamps and its answers at zero. */
"float rcp_(float x) { return x != 0.0 ? 1.0 / x : 3.4e38; }\n"
"float rcc_(float x) {\n"
"  float d = x != 0.0 ? 1.0 / x : 1.84467e19;\n"
"  if (d > 0.0) d = d < 5.42101e-20 ? 5.42101e-20 : (d > 1.84467e19 ? 1.84467e19 : d);\n"
"  else d = d > -5.42101e-20 ? -5.42101e-20 : (d < -1.84467e19 ? -1.84467e19 : d);\n"
"  return d;\n"
"}\n"
"float rsq_(float x) { float d = abs(x); return d != 0.0 ? 1.0 / sqrt(d) : 3.4e38; }\n"
"float log_(float x) { float d = abs(x); return d != 0.0 ? log2(d) : -3.4e38; }\n"
"float4 lit_(float4 v) {\n"
"  float nl = v.x > 0.0 ? v.x : 0.0;\n"
"  float ex = clamp(v.w, -127.9961, 127.9961);\n"
"  float b = v.y > 0.0 ? v.y : 0.0;\n"
"  float p = b > 0.0 ? exp2(ex * log2(b)) : (ex > 0.0 ? 0.0 : (ex == 0.0 ? 1.0 : asfloat(0x7F800000u)));\n"
"  return float4(1.0, nl, v.x > 0.0 ? p : 0.0, 1.0);\n"
"}\n";

/* DLSS step 3 (RECOMP_MOTION): the same program run twice in one shader,
 * first on the constants of the previous frame (cp, b4), then on the current
 * ones, so that the vertex knows where it was. cc() picks between the two.
 * Only for the motion vector pass; the ordinary shader is untouched. */
static const char s_vp_hlsl_motion_head[] =
"cbuffer VpP : register(b4) { float4 cp[192]; };\n"
"static bool g_prev = false;\n"
"float4 cc(int i) { return g_prev ? cp[i] : c[i]; }\n"
"float4 ccr(int i) { return (i >= 0 && i < 192) ? cc(clamp(i, 0, 191)) : Z4; }\n"
"struct VSOutM { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 col : COLOR0;\n"
"                float clip : SV_ClipDistance0; float4 col1 : COLOR1;\n"
"                float4 uv1 : TEXCOORD1; float4 uv2 : TEXCOORD2; float4 uv3 : TEXCOORD3;\n"
"                float4 pc : TEXCOORD4; float4 qc : TEXCOORD5; };\n";
static int s_vh_motion;                 /* vh_src: read constants through cc() */

typedef struct { char *p; size_t n, cap; int over; } VpHlsl;

static void vh(VpHlsl *b, const char *fmt, ...)
{
    va_list ap;
    int k;
    if (b->over)
        return;
    va_start(ap, fmt);
    k = vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    va_end(ap);
    if (k < 0 || (size_t)k >= b->cap - b->n)
        b->over = 1;
    else
        b->n += (size_t)k;
}

static const char *vh_mask(uint32_t m, char buf[5])
{
    int i = 0;
    if (m & 8) buf[i++] = 'x';
    if (m & 4) buf[i++] = 'y';
    if (m & 2) buf[i++] = 'z';
    if (m & 1) buf[i++] = 'w';
    buf[i] = 0;
    return buf;
}

static void vh_temp(char *out, size_t cap, uint32_t i)
{
    if (i == 12)
        snprintf(out, cap, "O%u", VP_OUT_POS);          /* R12 is oPos */
    else
        snprintf(out, cap, "R%u", i < 12 ? i : 0u);     /* as vp_temp */
}

static void vh_src(char *out, size_t cap, const VpInsn *d, uint32_t k)
{
    static const char xyzw[] = "xyzw";
    char base[40];

    switch (d->mux[k]) {
    case 1: vh_temp(base, sizeof base, d->reg[k]); break;
    case 2: snprintf(base, sizeof base, "V%u", d->input); break;
    case 3:
        if (d->rel)
            snprintf(base, sizeof base, s_vh_motion ? "ccr(%d + a0)" : "cr(%d + a0)",
                     (int)d->cidx);
        else if (d->cidx < VP_CONSTANTS)
            snprintf(base, sizeof base, s_vh_motion ? "cc(%u)" : "c[%u]", d->cidx);
        else
            snprintf(base, sizeof base, "Z4");
        break;
    default: snprintf(base, sizeof base, "Z4"); break;
    }
    if (d->swz[k] != 0x1B)
        snprintf(out, cap, "%s(%s).%c%c%c%c", d->neg[k] ? "-" : "", base,
                 xyzw[(d->swz[k] >> 6) & 3], xyzw[(d->swz[k] >> 4) & 3],
                 xyzw[(d->swz[k] >> 2) & 3], xyzw[d->swz[k] & 3]);
    else
        snprintf(out, cap, "%s(%s)", d->neg[k] ? "-" : "", base);
}

static const char *vh_mac(uint32_t mac)
{
    switch (mac) {
    case 1:  return "A";                                    /* MOV */
    case 2:  return "A * B";                                /* MUL */
    case 3:  return "A + C";                                /* ADD: A*1 + C */
    case 4:  return "A * B + C";                            /* MAD */
    case 5:  return "dot(A.xyz, B.xyz).xxxx";               /* DP3 */
    case 6:  return "(dot(A.xyz, B.xyz) + B.w).xxxx";       /* DPH */
    case 7:  return "dot(A, B).xxxx";                       /* DP4 */
    case 8:  return "float4(1.0, A.y * B.y, A.z, B.w)";     /* DST */
    case 9:  return "(A < B ? A : B)";                      /* MIN, as vp_run */
    case 10: return "(A > B ? A : B)";                      /* MAX */
    case 11: return "(float4)(A < B)";                      /* SLT */
    case 12: return "(float4)(A >= B)";                     /* SGE */
    default: return "Z4";                                   /* vp_run writes 0 */
    }
}

static const char *vh_ilu(uint32_t ilu)
{
    switch (ilu) {
    case 1:  return "C";                                    /* MOV */
    case 2:  return "rcp_(C.x).xxxx";
    case 3:  return "rcc_(C.x).xxxx";
    case 4:  return "rsq_(C.x).xxxx";
    case 5:  return "exp2(C.x).xxxx";
    case 6:  return "log_(C.x).xxxx";
    case 7:  return "lit_(C)";
    default: return "Z4";
    }
}

/* The program's instructions, in order. */
static void vp_hlsl_body(VpHlsl *b)
{
    uint32_t pc, k;
    char mask[5], dst[16];

    for (pc = 0; pc < s_vp_code_len; pc++) {
        const VpInsn *d = &s_vp_code[pc];
        vh(b, "  { /* i%u mac %u ilu %u */\n    float4 M = Z4, L = Z4;\n", pc, d->mac, d->ilu);
        for (k = 0; k < 3; k++)
            if (d->need & (1u << k)) {
                char src[64];
                vh_src(src, sizeof src, d, k);
                vh(b, "    float4 %c = %s;\n", "ABC"[k], src);
            }
        if (d->mac == 13)
            vh(b, "    a0 = (int)floor(A.x + 0.001);\n");
        else if (d->mac)
            vh(b, "    M = %s;\n", vh_mac(d->mac));
        if (d->ilu)
            vh(b, "    L = %s;\n", vh_ilu(d->ilu));
        if (d->mac && d->mac != 13 && d->mac_mask) {
            vh_temp(dst, sizeof dst, d->tmp);
            vh_mask(d->mac_mask, mask);
            vh(b, "    %s.%s = M.%s;\n", dst, mask, mask);
        }
        if (d->ilu && d->ilu_mask) {
            /* R1 when paired with a MAC operation, the temporary otherwise, as vp_run. */
            if (d->mac)
                snprintf(dst, sizeof dst, "R1");
            else
                vh_temp(dst, sizeof dst, d->tmp);
            vh_mask(d->ilu_mask, mask);
            vh(b, "    %s.%s = L.%s;\n", dst, mask, mask);
        }
        if (d->orb < 16 && d->out_mask) {
            vh_mask(d->out_mask, mask);
            vh(b, "    O%u.%s = %s.%s;\n", d->orb, mask, d->from_ilu ? "L" : "M", mask);
        }
        vh(b, "  }\n");
    }
}

/* The registers, cleared as vp_run starts them. */
static void vp_hlsl_regs(VpHlsl *b, int declare)
{
    uint32_t a;
    for (a = 0; a < 12; a++)
        vh(b, declare ? "  float4 R%u = Z4;\n" : "  R%u = Z4;\n", a);
    for (a = 0; a < 16; a++)
        vh(b, declare ? "  float4 O%u = Z4;\n" : "  O%u = Z4;\n", a);
    vh(b, declare ? "  O%u.w = 1.0;\n  int a0 = 0;\n" : "  O%u.w = 1.0;\n  a0 = 0;\n", VP_OUT_D0);
}

/* The current program as HLSL. 0 if it does not fit. motion: the variant for
 * the motion vector pass (DLSS step 3) -- the program first run on the
 * previous frame's constants, its position kept, then as always. */
static int vp_hlsl_any(VpHlsl *b, int motion)
{
    uint32_t a;

    vh(b, "%s", s_vp_hlsl_head);
    if (motion)
        vh(b, "%s", s_vp_hlsl_motion_head);
    vh(b, motion ? "VSOutM vp_main(uint vid : SV_VertexID) {\n"
                 : "VSOut vp_main(uint vid : SV_VertexID) {\n");
    vp_hlsl_regs(b, 1);
    /* A point sprite's indices carry the corner in their low two bits. */
    vh(b, "  uint vi = xf1.y > 0.5 ? (vid >> 2) : vid;\n");
    for (a = 0; a < NV_VERTEX_ATTRS; a++)
        if ((s_vp_inputs >> a) & 1)
            vh(b, "  float4 V%u = vfetch(%uu, vi);\n", a, a);

    if (motion) {
        s_vh_motion = 1;
        vh(b, "  g_prev = true;\n");
        vp_hlsl_body(b);
        /* Where it was, in clip space, as below but unjittered. */
        vh(b, "  float pw = O%u.w;\n  float prc = rcc_(pw);\n", VP_OUT_POS);
        vh(b, "  float3 PX = (O%u.xyz - c[59].xyz) / prc;\n", VP_OUT_POS);
        /* z: the window depth it had, as o.pos.z below. Under 0 it was in
         * front of the near plane -- clipped then, so no history (point 154). */
        vh(b, "  float4 pc = float4(PX.x * xf0.x + (c[59].x * xf0.x - 1.0) * pw,\n"
              "                     -PX.y * xf0.y + (1.0 - c[59].y * xf0.y) * pw,\n"
              "                     (PX.z + c[59].z * pw) / 16777215.0, pw);\n");
        vp_hlsl_regs(b, 0);
        vh(b, "  g_prev = false;\n");
    }
    vp_hlsl_body(b);
    s_vh_motion = 0;

    /* vp_transform's rcc and clip, then d3d_clip_pos. */
    vh(b, "  float w = O%u.w;\n", VP_OUT_POS);
    vh(b, "  float rc = rcc_(w);\n");
    vh(b, "  float3 X = (O%u.xyz - c[59].xyz) / rc;\n", VP_OUT_POS);
    vh(b, "  VSOut o;\n");
    vh(b, "  o.pos.x = X.x * xf0.x + (c[59].x * xf0.x - 1.0) * w;\n");
    vh(b, "  o.pos.y = -X.y * xf0.y + (1.0 - c[59].y * xf0.y) * w;\n");
    vh(b, "  o.pos.z = (X.z + c[59].z * w) / 16777215.0;\n");
    vh(b, "  o.pos.w = w;\n");
    /* DLSS step 2: the frame's sub-pixel jitter, in NDC (xf1.zw, 0 unless
     * RECOMP_JITTER and only for the 3D of the shown surface). */
    vh(b, "  o.pos.xy += xf1.zw * w;\n");
    vh(b, "  o.uv = xf1.x > 0.5 ? O%u.xy * xf0.zw : float2(0.0, 0.0);\n", VP_OUT_T0);
    vh(b, "  o.col = floor(saturate(O%u) * 255.0 + 0.5) / 255.0;\n", VP_OUT_D0);
    vh(b, "  o.col1 = floor(saturate(O%u) * 255.0 + 0.5) / 255.0;\n", VP_OUT_D1);
    vh(b, "  o.uv1 = O%u; o.uv2 = O%u; o.uv3 = O%u;\n", VP_OUT_T0 + 1, VP_OUT_T0 + 2,
       VP_OUT_T0 + 3);
    /* A point sprite (point 140): the corner moved by half the size the
     * program wrote in oPts, in window pixels as xf0 converts them, and stage
     * 3 given the sprite's own coordinates, as the NV2A does. */
    vh(b, "  if (xf1.y > 0.5) {\n"
          "    uint corner = vid & 3u;\n"
          "    float ps = clamp(O%u.x, 1.0, 256.0);\n"
          "    float2 cs = float2((corner & 1u) ? 0.5 : -0.5, (corner & 2u) ? 0.5 : -0.5);\n"
          "    o.pos.x += cs.x * ps * xf0.x * o.pos.w;\n"
          "    o.pos.y -= cs.y * ps * xf0.y * o.pos.w;\n"
          "    o.uv3 = float4((corner & 1u) ? 1.0 : 0.0, (corner & 2u) ? 1.0 : 0.0, 0.0, 1.0);\n"
          "  }\n", VP_OUT_PTS);
    vh(b, "  o.clip = o.pos.z;\n");
    if (motion) {
        vh(b, "  VSOutM m;\n"
              "  m.pos = o.pos; m.uv = o.uv; m.col = o.col; m.clip = o.clip; m.col1 = o.col1;\n"
              "  m.uv1 = o.uv1; m.uv2 = o.uv2; m.uv3 = o.uv3;\n"
              "  m.pc = pc;\n"
              "  m.qc = float4(o.pos.xy - xf1.zw * w, 0.0, w);   /* where it is, unjittered */\n"
              "  return m;\n}\n");
    } else {
        vh(b, "  return o;\n}\n");
    }
    return !b->over;
}

static int vp_hlsl(VpHlsl *b)
{
    return vp_hlsl_any(b, 0);
}

/* ── Register combiners on the GPU (point 140) ──────────────────────────
 *
 * The NV2A computes a fragment's colour in up to eight general combiner
 * stages and a final combiner, from the texture stages the texture shader
 * samples, the diffuse and specular colours, and constants. The fixed shader
 * does one thing: texel times diffuse. Measured in two arenas, the title asks
 * for four things, and two of them add to that product -- the specular colour
 * (V1, written by 216 vertex programs) and a second texture, light pools,
 * window light and environment reflections (point 140).
 *
 * The configuration is translated to HLSL, compiled once, cached by hash, as
 * the vertex programs are. Only where it differs from the fixed shader: a
 * configuration that passes R0 = texel x diffuse straight through keeps the
 * fixed shader, so what was already right cannot change. Word layouts are the
 * XDK's (PS_COMBINERINPUTS, PS_COMBINEROUTPUTS, PS_FINALCOMBINERINPUTS_*);
 * src/d3d/d3d8_combiners.h disagrees on the output word and on registers 14
 * and 15 and was never used by this title. A texture-shader mode other than
 * none or 2D, or a blue-to-alpha output, keeps the fixed shader and is
 * counted. RECOMP_COMBINERS=0 turns all of it off, to compare.
 *
 * Only the translated vertex programs carry V1 and the stage 1-3 coordinates,
 * so batches on the CPU vertex path -- the frame F9 logs, diagnostics, the
 * few "possibly flat" ones -- are still drawn texel x diffuse. */

static const char s_comb_head[] =
"struct PSIn2 { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 col : COLOR0;\n"
"               float clip : SV_ClipDistance0; float4 col1 : COLOR1;\n"
"               float4 uv1 : TEXCOORD1; float4 uv2 : TEXCOORD2; float4 uv3 : TEXCOORD3; };\n"
"cbuffer Draw : register(b0) { float4 tex; float4 atest; float4 extra; };\n"
"cbuffer Comb : register(b3) { float4 fc0[8]; float4 fc1[8]; float4 tsz[4]; };\n"
"Texture2D t0 : register(t0); Texture2D t1 : register(t1);\n"
"Texture2D t2 : register(t2); Texture2D t3 : register(t3);\n"
"SamplerState s0 : register(s0); SamplerState s1 : register(s1);\n"
"SamplerState s2 : register(s2); SamplerState s3 : register(s3);\n"
"static const float4 Z4 = float4(0.0, 0.0, 0.0, 0.0);\n"
/* Input mappings, PS_INPUTMAPPING_* */
"float4 m0(float4 x) { return max(x, 0.0); }\n"
"float4 m1(float4 x) { return 1.0 - saturate(x); }\n"
"float4 m2(float4 x) { return 2.0 * max(x, 0.0) - 1.0; }\n"
"float4 m3(float4 x) { return -2.0 * max(x, 0.0) + 1.0; }\n"
"float4 m4(float4 x) { return max(x, 0.0) - 0.5; }\n"
"float4 m5(float4 x) { return -max(x, 0.0) + 0.5; }\n"
"float4 m6(float4 x) { return x; }\n"
"float4 m7(float4 x) { return -x; }\n"
/* Stage 0 exactly as the fixed shader samples it: texel coordinates, the Load
 * path for a texture read back out of a render target. */
"float4 sample0(float2 uv) {\n"
"  if (extra.y > 0.5) return t0.Sample(s0, uv / float2(tex.x, tex.y));\n"
"  int w = (int)tex.x, h = (int)tex.y;\n"
"  int u = (int)max(uv.x, 0.0), v = (int)max(uv.y, 0.0);\n"
"  u = (tex.z == 1.0) ? (u % w) : min(u, w - 1);\n"
"  v = (tex.w == 1.0) ? (v % h) : min(v, h - 1);\n"
"  int sc = (int)extra.x;\n"
"  return t0.Load(int3(u * sc, v * sc, 0));\n"
"}\n"
/* Stages 1-3 in 2D mode: the coordinate as the program wrote it, divided by q
 * when it wrote one, normalised when the image is linear (texel units). */
"float4 sampleN(Texture2D t, SamplerState s, float4 q, float4 sz) {\n"
"  if (sz.w < 0.5) return Z4;\n"
"  float2 uv = q.xy / (q.w != 0.0 ? q.w : 1.0);\n"
"  if (sz.z > 0.5) uv /= sz.xy;\n"
"  return t.Sample(s, uv);\n"
"}\n";

static int comb_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RECOMP_COMBINERS");
        on = !(e && *e == '0');
    }
    return on;
}

static uint32_t s_comb_translated, s_comb_unsupported, s_comb_batches;

/* The fixed shader already draws this: one general stage, texture shader on
 * stage 0 at most, and a final combiner that passes R0 through (D = R0, A, B,
 * C zero, alpha from R0.a). */
static int comb_fixed_is_enough(void)
{
    return (s_comb_ctrl & 0xFF) <= 1 && (s_shader_prog >> 5) == 0
        && s_comb_fog0 == 0x0000000Cu && ((s_comb_fog1 >> 8) & 0xFF) == 0x1C;
}

/* A register as an HLSL value. C0 and C1 are the stage's own factors when the
 * control word says each stage has its own, else stage 0's. */
static void ch_reg(char *out, size_t cap, uint32_t reg, uint32_t stage)
{
    const uint32_t k0 = (s_comb_ctrl & (1u << 12)) ? stage : 0;
    const uint32_t k1 = (s_comb_ctrl & (1u << 16)) ? stage : 0;
    static const char *const names[16] = {
        "Z4", NULL, NULL, "FOG", "V0", "V1", "Z4", "Z4",
        "T0", "T1", "T2", "T3", "R0", "R1", "V1R0", "EFP"
    };
    if (reg == 1)      snprintf(out, cap, "fc0[%u]", k0);
    else if (reg == 2) snprintf(out, cap, "fc1[%u]", k1);
    else               snprintf(out, cap, "%s", names[reg & 0xF]);
}

/* One input byte of a general stage: register, channel, mapping. The RGB half
 * reads .rgb or the alpha replicated; the alpha half the alpha or the blue. */
static void ch_input(char *out, size_t cap, uint32_t byte, int alpha_half, uint32_t stage)
{
    char r[24];
    const int a = (byte >> 4) & 1;
    ch_reg(r, sizeof r, byte & 0xF, stage);
    snprintf(out, cap, "m%u(%s%s)", (byte >> 5) & 7, r,
             alpha_half ? (a ? ".aaaa" : ".bbbb") : (a ? ".aaaa" : ""));
}

/* One input of the final combiner: unsigned only, identity or invert. */
static void ch_final_input(char *out, size_t cap, uint32_t byte, int alpha_only)
{
    char r[24];
    const int a = (byte >> 4) & 1;
    ch_reg(r, sizeof r, byte & 0xF, 0);
    snprintf(out, cap, "%s(saturate(%s%s))", ((byte >> 5) & 7) == 1 ? "1.0 - " : "", r,
             alpha_only ? (a ? ".aaaa" : ".bbbb") : (a ? ".aaaa" : ""));
}

/* An output scale/bias, PS_COMBINEROUTPUT_*, then the registers' [-1, 1]. */
static const char *ch_outmap(uint32_t m)
{
    switch (m) {
    case 1:  return "(%s - 0.5)";
    case 2:  return "(%s * 2.0)";
    case 3:  return "((%s - 0.5) * 2.0)";
    case 4:  return "(%s * 4.0)";
    case 6:  return "(%s * 0.5)";
    default: return "(%s)";
    }
}

static void ch_write(VpHlsl *b, uint32_t dst, const char *mask, const char *val)
{
    static const char *const regs[16] = {
        NULL, NULL, NULL, NULL, "V0", "V1", NULL, NULL,
        "T0", "T1", "T2", "T3", "R0", "R1", NULL, NULL
    };
    if (dst < 16 && regs[dst])
        vh(b, "    %s.%s = %s;\n", regs[dst], mask, val);
}

/* The current combiner state as a pixel shader. 0 if it uses something the
 * translation does not do. */
static int comb_hlsl(VpHlsl *b)
{
    const uint32_t stages = (s_comb_ctrl & 0xFF) < 1 ? 1 : ((s_comb_ctrl & 0xFF) > 8 ? 8 : s_comb_ctrl & 0xFF);
    const char *mux = (s_comb_ctrl & (1u << 8)) ? "(R0.a >= 0.5)"
                                                : "(fmod(floor(R0.a * 255.0 + 0.5), 2.0) >= 1.0)";
    uint32_t st, k, m;
    char in[4][48], fmt[48], val[96];

    for (st = 0; st < 4; st++) {
        m = (s_shader_prog >> (5 * st)) & 0x1F;
        if (m > 1)
            return 0;                           /* only none and 2D are done */
    }
    for (st = 0; st < stages; st++)
        if (s_comb_color_ocw[st] & (3u << 18))
            return 0;                           /* blue-to-alpha */

    vh(b, "%s", s_comb_head);
    vh(b, "float4 ps_comb(PSIn2 i) : SV_Target {\n");
    vh(b, "  float4 V0 = i.col, V1 = i.col1, FOG = Z4;\n");
    if (s_shader_prog & 0x1F)
        vh(b, "  float4 T0 = atest.w > 0.5 ? sample0(i.uv) : float4(1.0, 1.0, 1.0, 1.0);\n");
    else
        vh(b, "  float4 T0 = Z4;\n");
    for (st = 1; st < 4; st++) {
        if ((s_shader_prog >> (5 * st)) & 0x1F)
            vh(b, "  float4 T%u = sampleN(t%u, s%u, i.uv%u, tsz[%u]);\n", st, st, st, st, st - 1);
        else
            vh(b, "  float4 T%u = Z4;\n", st);
    }
    vh(b, "  float4 R0 = float4(0.0, 0.0, 0.0, T0.a), R1 = Z4;\n");

    for (st = 0; st < stages; st++) {
        const uint32_t cicw = s_comb_color_icw[st], aicw = s_comb_alpha_icw[st];
        const uint32_t cocw = s_comb_color_ocw[st], aocw = s_comb_alpha_ocw[st];
        vh(b, "  { /* stage %u */\n", st);
        for (k = 0; k < 4; k++) {
            ch_input(in[k], sizeof in[k], (cicw >> (24 - 8 * k)) & 0xFF, 0, st);
            vh(b, "    float4 c%c = %s;\n", "ABCD"[k], in[k]);
        }
        for (k = 0; k < 4; k++) {
            ch_input(in[k], sizeof in[k], (aicw >> (24 - 8 * k)) & 0xFF, 1, st);
            vh(b, "    float4 a%c = %s;\n", "ABCD"[k], in[k]);
        }
        /* Every read above, every write below: a stage sees the registers as
         * they were when it began. */
        vh(b, "    float4 cab = %s;\n", (cocw & (1u << 13)) ? "dot(cA.rgb, cB.rgb).xxxx" : "cA * cB");
        vh(b, "    float4 ccd = %s;\n", (cocw & (1u << 12)) ? "dot(cC.rgb, cD.rgb).xxxx" : "cC * cD");
        vh(b, "    float4 csum = %s;\n", (cocw & (1u << 14)) ? "Z4" : "cab + ccd");
        if (cocw & (1u << 14))
            vh(b, "    csum = %s ? ccd : cab;\n", mux);
        vh(b, "    float4 aab = aA * aB, acd = aC * aD;\n");
        vh(b, "    float4 asum = %s;\n", (aocw & (1u << 14)) ? "Z4" : "aab + acd");
        if (aocw & (1u << 14))
            vh(b, "    asum = %s ? acd : aab;\n", mux);
        snprintf(fmt, sizeof fmt, "%s", ch_outmap((cocw >> 15) & 7));
        vh(b, "    cab = clamp(%s, -1.0, 1.0);\n", (snprintf(val, sizeof val, fmt, "cab"), val));
        vh(b, "    ccd = clamp(%s, -1.0, 1.0);\n", (snprintf(val, sizeof val, fmt, "ccd"), val));
        vh(b, "    csum = clamp(%s, -1.0, 1.0);\n", (snprintf(val, sizeof val, fmt, "csum"), val));
        snprintf(fmt, sizeof fmt, "%s", ch_outmap((aocw >> 15) & 7));
        vh(b, "    aab = clamp(%s, -1.0, 1.0);\n", (snprintf(val, sizeof val, fmt, "aab"), val));
        vh(b, "    acd = clamp(%s, -1.0, 1.0);\n", (snprintf(val, sizeof val, fmt, "acd"), val));
        vh(b, "    asum = clamp(%s, -1.0, 1.0);\n", (snprintf(val, sizeof val, fmt, "asum"), val));
        ch_write(b, (cocw >> 4) & 0xF, "rgb", "cab.rgb");
        ch_write(b, cocw & 0xF, "rgb", "ccd.rgb");
        ch_write(b, (cocw >> 8) & 0xF, "rgb", "csum.rgb");
        ch_write(b, (aocw >> 4) & 0xF, "a", "aab.x");
        ch_write(b, aocw & 0xF, "a", "acd.x");
        ch_write(b, (aocw >> 8) & 0xF, "a", "asum.x");
        vh(b, "  }\n");
    }

    /* The final combiner: rgb = A*B + (1-A)*C + D, alpha = G, with the two
     * registers only it can read -- V1 + R0 and E * F. */
    ch_final_input(in[0], sizeof in[0], (s_comb_fog1 >> 24) & 0xFF, 0);
    ch_final_input(in[1], sizeof in[1], (s_comb_fog1 >> 16) & 0xFF, 0);
    vh(b, "  float4 EFP = %s * %s;\n", in[0], in[1]);
    vh(b, "  float4 V1R0 = %s + %s;\n",
       (s_comb_fog1 & 0x40) ? "(1.0 - V1)" : "V1", (s_comb_fog1 & 0x20) ? "(1.0 - R0)" : "R0");
    if (s_comb_fog1 & 0x80)
        vh(b, "  V1R0 = saturate(V1R0);\n");
    for (k = 0; k < 4; k++) {
        ch_final_input(in[k], sizeof in[k], (s_comb_fog0 >> (24 - 8 * k)) & 0xFF, 0);
        vh(b, "  float4 f%c = %s;\n", "ABCD"[k], in[k]);
    }
    ch_final_input(in[0], sizeof in[0], (s_comb_fog1 >> 8) & 0xFF, 1);
    vh(b, "  float4 c = saturate(float4(fA.rgb * fB.rgb + (1.0 - fA.rgb) * fC.rgb + fD.rgb, %s.x));\n",
       in[0]);
    /* The alpha test, as the fixed shader runs it. */
    vh(b, "  if (atest.x > 0.5) {\n"
          "    float a = floor(c.a * 255.0 + 0.5), r = atest.z;\n"
          "    int f = (int)atest.y;\n"
          "    bool ok = true;\n"
          "    if (f == 0) ok = false;\n"
          "    else if (f == 1) ok = a <  r;\n"
          "    else if (f == 2) ok = a == r;\n"
          "    else if (f == 3) ok = a <= r;\n"
          "    else if (f == 4) ok = a >  r;\n"
          "    else if (f == 5) ok = a != r;\n"
          "    else if (f == 6) ok = a >= r;\n"
          "    if (!ok) discard;\n"
          "  }\n"
          "  return c;\n}\n");
    return !b->over;
}

static uint64_t comb_hash(void)
{
    const uint32_t stages = (s_comb_ctrl & 0xFF) < 1 ? 1 : ((s_comb_ctrl & 0xFF) > 8 ? 8 : s_comb_ctrl & 0xFF);
    uint64_t h = 1469598103934665603ull;
    uint32_t st;
    h = (h ^ s_shader_prog) * 0x100000001B3ull;
    h = (h ^ s_comb_ctrl) * 0x100000001B3ull;
    h = (h ^ s_comb_fog0) * 0x100000001B3ull;
    h = (h ^ s_comb_fog1) * 0x100000001B3ull;
    for (st = 0; st < stages; st++) {
        h = (h ^ s_comb_color_icw[st]) * 0x100000001B3ull;
        h = (h ^ s_comb_color_ocw[st]) * 0x100000001B3ull;
        h = (h ^ s_comb_alpha_icw[st]) * 0x100000001B3ull;
        h = (h ^ s_comb_alpha_ocw[st]) * 0x100000001B3ull;
    }
    return h;
}

/* The combiner part of a program draw's state: which pixel shader, the
 * textures of stages 1-3 and the factors. */
static void comb_state(NvD3DState *st)
{
    static uint64_t last_hash;
    static int      last_ps = -1;
    uint64_t hash;
    int ps, k;

    if (!comb_on() || comb_fixed_is_enough())
        return;
    hash = comb_hash();
    if (hash == last_hash && last_ps >= 0) {
        ps = last_ps;
    } else {
        ps = nvd3d_ps_program(hash, NULL, 0);
        if (ps == -2) {
            static char text[32 * 1024];
            VpHlsl b;
            b.p = text; b.n = 0; b.cap = sizeof text; b.over = 0;
            if (!comb_hlsl(&b)) {
                s_comb_unsupported++;
                ps = nvd3d_ps_program(hash, "", 0);      /* cache the refusal */
            } else {
                /* RECOMP_COMB_DUMP=<dir>: every translation, one file each,
                 * to read what a configuration became. */
                const char *dir = getenv("RECOMP_COMB_DUMP");
                if (dir) {
                    char path[512];
                    FILE *f;
                    snprintf(path, sizeof path, "%s/comb_%016llX.hlsl", dir,
                             (unsigned long long)hash);
                    f = fopen(path, "w");
                    if (f) {
                        fwrite(text, 1, b.n, f);
                        fclose(f);
                    }
                }
                ps = nvd3d_ps_program(hash, text, b.n);
                s_comb_translated++;
                fprintf(stderr, "  [COMB] configuration %016llX: %u stage(s), texture shader %08X,"
                                " final %08X %08X -> %s\n",
                        (unsigned long long)hash, s_comb_ctrl & 0xFF, s_shader_prog,
                        s_comb_fog0, s_comb_fog1, ps >= 0 ? "translated" : "does not compile");
            }
        }
        last_hash = hash;
        last_ps = ps;
    }
    if (ps < 0)
        return;
    st->ps = ps;
    s_comb_batches++;
    for (k = 0; k < 8; k++) {
        st->factor0[k] = s_comb_factor0[k];
        st->factor1[k] = s_comb_factor1[k];
    }
    for (k = 1; k < 4; k++) {
        Texture t;
        if (!((s_shader_prog >> (5 * k)) & 0x1F) || !tex_from_stage((uint32_t)k, &t) || !t.valid)
            continue;
        {
            const Texture keep = s_gpu.tex;
            s_gpu.tex = t;                      /* d3d_texture_slot reads s_gpu.tex */
            st->stage_tex[k - 1] = d3d_texture_slot(k);
            s_gpu.tex = keep;
        }
        st->stage_wh[k - 1][0] = t.width;
        st->stage_wh[k - 1][1] = t.height;
        st->stage_addr[k - 1] = (t.addr_u & 0xF) | ((t.addr_v & 0xF) << 4);
        st->stage_linear[k - 1] = !tex_size_from_format(t.color);
    }
}

/* The program's identity: the words it runs, from its start to its last. */
static uint64_t vp_program_hash(void)
{
    uint64_t h = 1469598103934665603ull;
    uint32_t pc, k;
    for (pc = s_vp.start; pc < VP_MAX_INSNS; pc++) {
        for (k = 1; k < 4; k++)
            h = (h ^ s_vp.prog[pc][k]) * 0x100000001B3ull;
        if (s_vp.prog[pc][3] & 1)
            break;
    }
    return h;
}

/* RECOMP_SEP=1: can the 3D of a frame be told from its interface? (DLSS,
 * step 0.) An upscaler wants the scene alone, then the interface drawn over
 * it at the output size, so the executor would have to know where one ends.
 * Nothing is decided here: each batch gets a signature from what could
 * classify it -- transform mode, vertex program, depth test and write,
 * blending, the clip w of its first vertex (1 for pretransformed 2D, as
 * point 87 supposed) -- and the frame's batches are kept in order with the
 * surface they went to. Every 600 swaps the sequence drawn into the shown
 * surface is printed as runs of signatures, and the other surfaces (render
 * to texture) by count. Over every frame it counts how often a depth-tested
 * batch comes back after the first batch with the depth test off: if never,
 * "the interface is what follows the 3D" is a usable boundary.
 *
 * w1 (DLSS step 1, the projection): 1 when the first and the last vertex
 * both come out of the vertex program with w == 1 exactly, as an orthographic
 * projection gives; 0 when either does not, as a perspective one gives (w is
 * then the depth); 2 not measured (no vertex program, or no vertex). Taken in
 * draw_primitive, where every route ends, with the same vp_transform the GPU
 * path already runs on the first vertex of every batch it sends. */
typedef struct {
    uint64_t prog;
    uint32_t mode, dt, dw, blend, w1, prim;
    uint32_t main_batches, other_batches, first_flip;
    float    w_lo, w_hi;                        /* of the first vertex, over all batches */
} SepSig;

#define SEP_MAX_SIGS   512
#define SEP_MAX_FRAME  8192
static SepSig   s_sep[SEP_MAX_SIGS];
static uint32_t s_sep_count, s_sep_lost;
static uint16_t s_sep_frame_sig[SEP_MAX_FRAME];
static uint32_t s_sep_frame_tgt[SEP_MAX_FRAME];
static uint32_t s_sep_frame_n, s_sep_frame_over;
static uint32_t s_sep_frames, s_sep_frames_back, s_sep_frames_no2d, s_sep_frames_back_w1;
static uint32_t s_sep_next = 60;

/* A fight once stayed stuck loading while the first version of this probe ran
 * the vertex program from the END handler (point 147). Not shown to be the
 * cause: the GPU path runs the same vp_transform on every batch it draws. */
static int sep_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RECOMP_SEP");
        on = e ? atoi(e) : 0;
    }
    return on;
}

/* RECOMP_SEP_CYCLE=<swaps>: is w == 1 the interface? Phases of that many
 * swaps go round three ways of drawing -- everything, everything but the
 * w == 1 batches, only the w == 1 batches -- and a snapshot is asked for in
 * the middle of each, named <RECOMP_FB_SNAP>NNN_ref.bmp, NNN_sans2d.bmp and
 * NNN_sans3d.bmp. Batches whose w was not measured go with the 3D. */
extern void xbox_FramebufferSnapAsk(const char *path);
static uint32_t s_sep_skip_class;               /* 0 none, 1 the w == 1 ones, 2 the others */
static uint32_t s_sep_cycle, s_sep_phase, s_sep_phase_end, s_sep_snap_at;
static uint32_t s_sep_w = 2;                    /* this batch's w1, from sep_w_probe */
static float    s_sep_w0;

/* From draw_primitive: measure w1, and say whether the cycle leaves it out. */
static int sep_w_probe(void)
{
    VpVertex v;
    uint32_t runs;

    if (!sep_on() || !vp_active() || !s_gpu.idx_count)
        return 0;
    runs = s_vp.runs;
    vp_transform(s_gpu.idx[0], &v);
    if (v.ok) {
        s_sep_w0 = v.w;
        s_sep_w = v.w == 1.0f;
        if (s_sep_w && s_gpu.idx_count > 1) {
            vp_transform(s_gpu.idx[s_gpu.idx_count - 1], &v);
            s_sep_w = v.ok && v.w == 1.0f;
        }
    }
    s_vp.runs = runs;                           /* a probe, not vertices drawn */
    return s_sep_skip_class == 1 ? s_sep_w == 1
         : s_sep_skip_class == 2 ? s_sep_w != 1 : 0;
}

/* RECOMP_UI_LAYER: is this batch the interface? See s_batch_ui. */
static void ui_classify(void)
{
    VpVertex v;
    uint32_t runs;

    s_batch_ui = 0;
    if (!ui_layer_on() || !nvd3d_active() || !vp_active() || !s_gpu.idx_count
            || dma_resolve(s_gpu.color_offset) != s_d3d_show_base)
        return;
    runs = s_vp.runs;
    vp_transform(s_gpu.idx[0], &v);
    if (v.ok && v.w == 1.0f) {
        vp_transform(s_gpu.idx[s_gpu.idx_count - 1], &v);
        s_batch_ui = v.ok && v.w == 1.0f;
    }
    /* A few untextured vertices across the whole width: a fade over the
     * picture (black, alpha 00 to FF, at the start and the end of a fight,
     * point 151), not a piece of the interface. Layer 2: the backend
     * composites what the interface drew so far, then draws this into the
     * target itself -- so it is not squeezed by RECOMP_UI_43, and still
     * darkens what came before it. A textured one stays interface: measured,
     * those are the menu's bar, the videos, a banner wider than the screen. */
    if (s_batch_ui && s_gpu.idx_count <= 16 && s_gpu.clip_w && !tex_stage0_enabled()) {
        float lo = 1e30f, hi = -1e30f;
        uint32_t i;
        for (i = 0; i < s_gpu.idx_count; i++) {
            vp_transform(s_gpu.idx[i], &v);
            if (!v.ok)
                break;
            if (v.xy[0] < lo) lo = v.xy[0];
            if (v.xy[0] > hi) hi = v.xy[0];
        }
        if (i == s_gpu.idx_count && lo <= (float)s_gpu.clip_x + 1.0f
                && hi >= (float)(s_gpu.clip_x + s_gpu.clip_w) - 1.0f)
            s_batch_ui = 2;
        if (s_batch_ui == 2) {
            /* Each kind once: what a full-width batch is, to tell a fade
             * from a background. */
            static struct { uint32_t n, tex, argb; } seen[32];
            static uint32_t nseen;
            uint32_t k;
            for (k = 0; k < nseen; k++)
                if (seen[k].n == s_gpu.idx_count && seen[k].tex == s_gpu.tex.offset
                        && seen[k].argb == v.argb)
                    break;
            if (k == nseen && nseen < 32) {
                seen[nseen].n = s_gpu.idx_count;
                seen[nseen].tex = s_gpu.tex.offset;
                seen[nseen].argb = v.argb;
                nseen++;
                fprintf(stderr, "  [UI] full width #%u swap %u: %u vertices x %.1f..%.1f  colour"
                                " %08X  stage 0 %s tex 0x%08X %ux%u  blend %u %X/%X\n",
                        nseen - 1, s_gpu.flips, s_gpu.idx_count, lo, hi, v.argb,
                        tex_stage0_enabled() ? "on" : "off", s_gpu.tex.offset,
                        s_gpu.tex.width, s_gpu.tex.height, s_gpu.blend_enable,
                        s_gpu.blend_sfactor, s_gpu.blend_dfactor);
            }
        }
    }
    s_vp.runs = runs;                           /* not vertices drawn */
}

/* After the batch was drawn (or left out): its signature, and the frame list. */
static int sep_batch(uint32_t route)
{
    SepSig s;
    uint32_t i;

    (void)route;
    if (!sep_on())
        return -1;
    memset(&s, 0, sizeof s);
    s.mode  = s_vp.mode;
    s.prog  = vp_active() ? vp_program_hash() : 0;
    s.dt    = s_gpu.depth_test != 0;
    s.dw    = s_gpu.depth_mask != 0;
    s.blend = s_gpu.blend_enable != 0;
    s.prim  = s_gpu.prim;
    s.w1    = s_sep_w;
    s.w_lo  = s.w_hi = s_sep_w0;
    for (i = 0; i < s_sep_count; i++) {
        SepSig *e = &s_sep[i];
        if (e->prog == s.prog && e->mode == s.mode && e->dt == s.dt && e->dw == s.dw
                && e->blend == s.blend && e->w1 == s.w1 && e->prim == s.prim) {
            if (s.w1 != 2) {
                if (s_sep_w0 < e->w_lo) e->w_lo = s_sep_w0;
                if (s_sep_w0 > e->w_hi) e->w_hi = s_sep_w0;
            }
            break;
        }
    }
    if (i == s_sep_count) {
        if (s_sep_count == SEP_MAX_SIGS) {
            s_sep_lost++;
            return -1;
        }
        s.first_flip = s_gpu.flips;
        s_sep[s_sep_count++] = s;
        fprintf(stderr, "  [SEP] new #%u swap %u  mode %u prog %016llX  depth test %u write %u"
                        "  blend %u  w1 %u (w %g)  prim %u  surface 0x%08X  tex 0x%08X %ux%u\n",
                i, s.first_flip, s.mode, (unsigned long long)s.prog, s.dt, s.dw, s.blend,
                s.w1, s.w1 == 2 ? 0.0 : (double)s_sep_w0, s.prim,
                dma_resolve(s_gpu.color_offset), s_gpu.tex.offset, s_gpu.tex.width,
                s_gpu.tex.height);
    }
    if (s_sep_frame_n < SEP_MAX_FRAME) {
        s_sep_frame_sig[s_sep_frame_n] = (uint16_t)i;
        s_sep_frame_tgt[s_sep_frame_n] = dma_resolve(s_gpu.color_offset);
        s_sep_frame_n++;
    } else {
        s_sep_frame_over++;
    }
    return (int)i;
}

static void sep_cycle(uint32_t shown)
{
    static const char *const name[3] = { "ref", "sans2d", "sans3d" };

    (void)shown;
    if (!s_sep_cycle) {
        const char *e = getenv("RECOMP_SEP_CYCLE");
        s_sep_cycle = e && atoi(e) > 0 ? (uint32_t)atoi(e) : 0xFFFFFFFFu;
        s_sep_phase_end = s_gpu.flips + s_sep_cycle;
    }
    if (s_sep_cycle == 0xFFFFFFFFu)
        return;
    if (s_gpu.flips == s_sep_snap_at) {
        const char *pre = getenv("RECOMP_FB_SNAP");
        char path[512];
        snprintf(path, sizeof path, "%s%03u_%s.bmp", pre ? pre : "sep_", s_sep_phase,
                 name[s_sep_skip_class]);
        xbox_FramebufferSnapAsk(path);
        fprintf(stderr, "  [SEP] cycle phase %u swap %u : %s\n", s_sep_phase, s_gpu.flips, path);
    }
    if (s_gpu.flips < s_sep_phase_end)
        return;
    s_sep_phase++;
    s_sep_phase_end = s_gpu.flips + s_sep_cycle;
    s_sep_snap_at = s_gpu.flips + s_sep_cycle / 2;
    s_sep_skip_class = s_sep_phase % 3;
}

/* At the swap: the shown surface is known now (the last one cleared). */
static void sep_frame(uint32_t shown)
{
    uint32_t i, seen_2d = 0, back = 0, seen_w1 = 0, back_w1 = 0;

    if (!sep_on())
        return;
    sep_cycle(shown);
    for (i = 0; i < s_sep_frame_n; i++) {
        SepSig *e = &s_sep[s_sep_frame_sig[i]];
        if (s_sep_frame_tgt[i] != shown) {
            e->other_batches++;
            continue;
        }
        e->main_batches++;
        if (!e->dt)
            seen_2d = 1;
        else if (seen_2d)
            back = 1;
        if (e->w1 == 1)
            seen_w1 = 1;
        else if (e->w1 == 0 && seen_w1)
            back_w1 = 1;
    }
    s_sep_frames++;
    s_sep_frames_back += back;
    s_sep_frames_back_w1 += back_w1;
    s_sep_frames_no2d += !seen_2d;

    if (s_gpu.flips >= s_sep_next) {
        static char line[8192];
        size_t len = 0;
        uint32_t run = 0, prev = 0xFFFFFFFFu, others = 0;
        s_sep_next = s_gpu.flips + 600;
        for (i = 0; i <= s_sep_frame_n; i++) {
            uint32_t sig = 0xFFFFFFFEu;
            if (i < s_sep_frame_n) {
                if (s_sep_frame_tgt[i] != shown) { others++; continue; }
                sig = s_sep_frame_sig[i];
            }
            if (sig == prev) { run++; continue; }
            if (run && len + 24 < sizeof line) {
                SepSig *e = &s_sep[prev];
                len += (size_t)snprintf(line + len, sizeof line - len, " %s#%u x%u",
                                        e->w1 == 1 ? "o" : e->w1 == 2 ? "?" : "", prev, run);
            }
            prev = sig;
            run = 1;
        }
        line[len] = 0;
        fprintf(stderr, "  [SEP] swap %u shown 0x%08X disp %ux%u  %u batches, %u to other"
                        " surfaces%s  (o = w == 1, ? = w not measured):%s\n",
                s_gpu.flips, shown, s_d3d_show_disp[0], s_d3d_show_disp[1], s_sep_frame_n,
                others, s_sep_frame_over ? " TRUNCATED" : "", line);
        fprintf(stderr, "  [SEP] %u frames: depth-tested batch after the first untested one in"
                        " %u, no untested batch in %u; w!=1 after w==1 in %u; %u signatures,"
                        " %u lost\n",
                s_sep_frames, s_sep_frames_back, s_sep_frames_no2d, s_sep_frames_back_w1,
                s_sep_count, s_sep_lost);
        {
            static char wl[SEP_MAX_SIGS * 40];
            size_t wn = 0;
            for (i = 0; i < s_sep_count && wn + 40 < sizeof wl; i++)
                if (s_sep[i].w1 != 2)
                    wn += (size_t)snprintf(wl + wn, sizeof wl - wn, " #%u:%g..%g", i,
                                           (double)s_sep[i].w_lo, (double)s_sep[i].w_hi);
            wl[wn] = 0;
            fprintf(stderr, "  [SEP] w of the first vertex per signature:%s\n", wl);
        }
    }
    s_sep_frame_n = 0;
    s_sep_frame_over = 0;
}

static uint8_t *s_vpg_data;
static uint32_t s_vpg_data_cap;
static uint32_t s_vpg_index[NV_MAX_INDICES * 2 * 3];
static uint32_t s_vpg_batches, s_vpg_tris;
static uint32_t s_vpg_cpu_flat, s_vpg_cpu_diag, s_vpg_cpu_nolast, s_vpg_cpu_compile,
                s_vpg_cpu_gather, s_vpg_cpu_small, s_vpg_eligible, s_vpg_eligible_tris;
static uint32_t s_vpg_points;                /* point-sprite batches drawn (point 140) */
static uint32_t s_vpg_lines;                 /* line batches drawn (point 140) */

static uint32_t vp_elem_bytes(const VertexAttr *a)
{
    const uint32_t n = a->size < 4 ? a->size : 4;
    switch (a->type) {
    case 0: case 6: return 4;
    case 2:         return 4 * n;
    case 4:         return n;
    case 1: case 5: return 2 * n;
    default:        return 0;
    }
}

static int vp_guest_range_ok(uint64_t lo, uint64_t hi)
{
    if (lo >= hi)
        return 0;
    if (lo >= 0x00010000u && hi <= 0x04000000u)
        return 1;
    if (lo >= XBOX_CONTIG_BASE && hi <= (uint64_t)XBOX_CONTIG_BASE + XBOX_CONTIG_SIZE)
        return 1;
    return 0;
}

/* The attributes the program reads, copied as they are, each from its lowest
 * to its highest index in the batch. 0 if one cannot be taken safely. */
static int vp_gather(NvD3DProgramDraw *d, uint32_t lo, uint32_t hi)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t a, pos = 0;

    for (a = 0; a < NV_VERTEX_ATTRS; a++) {
        const VertexAttr *va = &s_gpu.attr[a];
        NvD3DAttr *o = &d->attr[a];
        uint32_t eb;
        uint64_t first, last, span;
        const uint8_t *src;

        memset(o, 0, sizeof *o);
        d->absent[a][0] = d->absent[a][1] = d->absent[a][2] = 0.0f;
        d->absent[a][3] = 1.0f;
        if (!((s_vp_inputs >> a) & 1))
            continue;
        eb = vp_elem_bytes(va);
        if (!va->size || !va->stride || !eb || (!s_gpu.inline_active && !va->offset)) {
            imm_attr_value(a, d->absent[a]);      /* else (0,0,0,1), as fetch_attr */
            continue;
        }
        first = (uint64_t)va->offset + (uint64_t)lo * va->stride;
        last  = (uint64_t)va->offset + (uint64_t)hi * va->stride + eb;
        span  = last - first;
        if (span > (8u << 20))
            return 0;
        if (s_gpu.inline_active) {
            if (last > (uint64_t)s_gpu.inline_count * 4)
                return 0;
            src = (const uint8_t *)s_gpu.inline_buf + first;
        } else {
            if (!vp_guest_range_ok(first, last))
                return 0;
            src = mem + first;
        }
        if (pos + span + 8 > s_vpg_data_cap) {
            uint32_t cap = (uint32_t)(pos + span + 8) * 2;
            uint8_t *nb = (uint8_t *)realloc(s_vpg_data, cap < (1u << 20) ? (1u << 20) : cap);
            if (!nb)
                return 0;
            s_vpg_data = nb;
            s_vpg_data_cap = cap < (1u << 20) ? (1u << 20) : cap;
        }
        memcpy(s_vpg_data + pos, src, (size_t)span);
        memset(s_vpg_data + pos + span, 0, 8);   /* the shader's dword reads run past */
        o->base = pos - lo * va->stride;          /* modular, as the shader's */
        o->stride = va->stride;
        o->type = va->type;
        o->size = va->size < 4 ? va->size : 4;
        pos = (uint32_t)((pos + span + 8 + 3) & ~(uint64_t)3);
    }
    d->data = s_vpg_data;
    d->data_bytes = pos ? pos : 4;
    if (!s_vpg_data) {
        s_vpg_data = (uint8_t *)calloc(1, 1u << 20);
        if (!s_vpg_data)
            return 0;
        s_vpg_data_cap = 1u << 20;
        d->data = s_vpg_data;
    }
    return 1;
}

/* RECOMP_VP_GPU=2: the batch through the shader, captured, against vp_run. */
#define VPG_CHECK_MAX 8192
static NvD3DVertex s_vpg_seen[VPG_CHECK_MAX];

static void vp_gpu_check(const NvD3DProgramDraw *d, uint64_t hash, uint32_t tris)
{
    static uint64_t prog[128];
    static int      runs[128], nprog;
    int slot, j, bad = 0, compared = 0;
    uint32_t n;
    float e_ndc = 0, e_z = 0, e_w = 0, e_uv = 0, e_col = 0;

    for (slot = 0; slot < nprog; slot++)
        if (prog[slot] == hash)
            break;
    if (slot == nprog) {
        if (nprog >= 128)
            return;
        prog[nprog] = hash;
        runs[nprog++] = 0;
    }
    if (runs[slot]++ >= 3)
        return;
    n = nvd3d_check_program(d, s_vpg_seen, VPG_CHECK_MAX);
    for (j = 0; j < (int)n; j++) {
        VpVertex v;
        float ref[4], clip[4], e, uv[2];
        const NvD3DVertex *g = &s_vpg_seen[j];
        int k;

        vp_transform(d->index[j], &v);
        if (!v.ok)
            continue;
        compared++;
        clip[0] = v.clip[0]; clip[1] = v.clip[1]; clip[2] = v.clip[2]; clip[3] = v.w;
        d3d_clip_pos(clip, s_vp.c[59], d->ax, d->ay, ref);
        if (fabsf(ref[3]) > 1e-3f && fabsf(g->pos[3]) > 1e-3f) {
            for (k = 0; k < 2; k++) {
                e = fabsf(g->pos[k] / g->pos[3] - ref[k] / ref[3]);
                if (e > e_ndc) e_ndc = e;
            }
            e = fabsf(g->pos[2] / g->pos[3] - ref[2] / ref[3]);
            if (e > e_z) e_z = e;
        }
        e = fabsf(g->pos[3] - ref[3]) / (fabsf(ref[3]) > 1.0f ? fabsf(ref[3]) : 1.0f);
        if (e > e_w) e_w = e;
        uv[0] = d->textured ? v.uv[0] : 0.0f;
        uv[1] = d->textured ? v.uv[1] : 0.0f;
        for (k = 0; k < 2; k++) {
            e = fabsf(g->uv[k] - uv[k]);
            if (e > e_uv) e_uv = e;
        }
        {
            const float col[4] = {
                (float)((v.argb >> 16) & 0xFF), (float)((v.argb >> 8) & 0xFF),
                (float)(v.argb & 0xFF), (float)((v.argb >> 24) & 0xFF) };
            for (k = 0; k < 4; k++) {
                e = fabsf(g->col[k] * 255.0f - col[k]);
                if (e > e_col) e_col = e;
            }
        }
        if (e_ndc > 1e-3f || e_uv > 0.02f || e_col > 1.5f)
            bad++;
    }
    fprintf(stderr, "  [VPGPU] check %016llX run %d: %u tris, %d vertices compared,"
                    " max ndc %.2e depth %.2e w %.2e uv %.4f col %.1f/255 -- %s\n",
            (unsigned long long)hash, runs[slot], tris, compared,
            e_ndc, e_z, e_w, e_uv, e_col, bad ? "OUTSIDE TOLERANCE" : "same");
    fflush(stderr);
}

/* Where the batches went, every 600 frames, in both modes: a batch that
 * silently stays on the CPU is exactly what would hide a gap in coverage. */
static void vp_gpu_summary(void)
{
    static uint32_t last_flip;
    if (s_gpu.flips - last_flip < 600)
        return;
    last_flip = s_gpu.flips;
    fprintf(stderr, "  [VPGPU] %u batches (%u triangles) %s by translated programs;"
                    " kept on the CPU: %u small, %u possibly flat, %u diagnostics, %u no end,"
                    " %u not compiled, %u upload refused\n",
            s_vpg_eligible, s_vpg_eligible_tris,
            vp_gpu_mode() == 2 ? "could be drawn" : "drawn",
            s_vpg_cpu_small, s_vpg_cpu_flat, s_vpg_cpu_diag, s_vpg_cpu_nolast,
            s_vpg_cpu_compile, s_vpg_cpu_gather);
    fprintf(stderr, "  [COMB] %u configurations translated, %u refused, %u batches through"
                    " them; %u point-sprite batches, %u line batches\n",
            s_comb_translated, s_comb_unsupported, s_comb_batches, s_vpg_points, s_vpg_lines);
    fflush(stderr);
}

/* ── DLSS step 3: motion vectors (RECOMP_MOTION=1) ──────────────────────
 *
 * The title gives no motion vectors, so they are made: each batch of 3D on
 * the shown surface is recognised from one frame to the next and its
 * vertex-program constants of the previous frame are kept -- the camera, and
 * the bone matrices if the program skins. The motion pass then runs the
 * program on both. A batch is known by its program, the address of its
 * positions, its index count, and its rank among the batches that share those
 * three in the frame (a crowd draws the same mesh several times with other
 * matrices). A batch seen for the first time has no motion: its own current
 * constants stand in for the previous ones. */
#define MOTION_SLOTS 2048
typedef struct {
    uint64_t prog;
    uint32_t addr, n, frame;            /* frame: last drawn in; 0 empty */
    float    c[VP_CONSTANTS][4];        /* its constants then */
} MotionObj;
static MotionObj *s_motion;
static float      s_motion_prev[VP_CONSTANTS][4];
static uint32_t   s_motion_found, s_motion_new, s_motion_full;
/* For RECOMP_MOTION_PROBE: what the last motion_prev decided. */
static uint32_t   s_mp_candidates, s_mp_matched;
static float      s_mp_best;

static int motion_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RECOMP_MOTION");
        const char *dl = getenv("RECOMP_DLSS");       /* DLSS needs them (point 155) */
        on = (e && atoi(e) != 0) || (dl && atoi(dl) != 0);
        if (on) {
            s_motion = (MotionObj *)calloc(MOTION_SLOTS, sizeof *s_motion);
            if (!s_motion)
                on = 0;
        }
    }
    return on;
}

/* The previous constants of this batch (s_motion_prev), and the current ones
 * kept for the next frame. */
static const float (*motion_prev(uint64_t prog))[4]
{
    /* Not keyed by the address of the positions (point 154): the crowd's
     * meshes are written each frame into slots handed out anew, so one address
     * is another spectator a frame later. Among those of the same program and
     * size, the nearest constants -- the place in the world -- decide. */
    const uint32_t addr = 0, n = s_gpu.idx_count;
    const uint32_t frame = s_gpu.flips + 1;     /* never 0 */
    uint32_t h = (uint32_t)(prog ^ (prog >> 32)) * 2654435761u ^ n;
    uint32_t k, free_at = 0xFFFFFFFFu;
    MotionObj *o = NULL;

    /* Among those of its kind drawn in the previous frame and not yet taken
     * in this one, the one whose constants are nearest to these. Measured in
     * a fight (point 154): matched by rank alone, a few points moved by
     * thousands of pixels while the scene stood still -- the crowd draws one
     * mesh many times, in an order that changes from frame to frame. */
    s_mp_candidates = 0;
    {
        float best = 3.4e38f;
        MotionObj *stale = NULL;
        for (k = 0; k < MOTION_SLOTS; k++) {
            MotionObj *e = &s_motion[(h + k) % MOTION_SLOTS];
            if (!e->frame) {
                if (free_at == 0xFFFFFFFFu)
                    free_at = (h + k) % MOTION_SLOTS;
                break;
            }
            if (e->prog == prog && e->addr == addr && e->n == n && e->frame != frame) {
                if (e->frame + 1 == frame) {
                    float dsum = 0.0f;
                    uint32_t i;
                    s_mp_candidates++;
                    const float *a = &e->c[0][0], *b = &s_vp.c[0][0];
                    for (i = 0; i < VP_CONSTANTS * 4 && dsum < best; i++)
                        dsum += fabsf(a[i] - b[i]);
                    if (dsum < best) {
                        best = dsum;
                        o = e;
                    }
                } else if (!stale) {
                    stale = e;                  /* of its kind, but not seen lately */
                }
            }
            if (free_at == 0xFFFFFFFFu && e->frame + 2 < frame)
                free_at = (h + k) % MOTION_SLOTS;   /* gone for a while: reusable */
        }
        if (!o)
            o = stale;
        s_mp_best = best;
    }
    s_mp_matched = o && o->frame + 1 == frame;
    if (o && o->frame + 1 == frame) {
        memcpy(s_motion_prev, o->c, sizeof s_motion_prev);
        s_motion_found++;
    } else {
        memcpy(s_motion_prev, s_vp.c, sizeof s_motion_prev);
        s_motion_new++;
        if (!o) {
            if (free_at == 0xFFFFFFFFu) {
                s_motion_full++;
                return (const float (*)[4])s_motion_prev;
            }
            o = &s_motion[free_at];
            o->prog = prog; o->addr = addr; o->n = n;
        }
    }
    memcpy(o->c, s_vp.c, sizeof o->c);
    o->frame = frame;
    {
        /* RECOMP_MOTION_SELFTEST=1: the previous constants scaled by 1.01, so
         * that every batch "moves" -- the check, without a player, that the
         * motion pass gives something other than zero. Diagnosis only. */
        static int self = -1;
        if (self < 0)
            self = getenv("RECOMP_MOTION_SELFTEST") != NULL;
        if (self) {
            uint32_t i, j;
            for (i = 0; i < VP_CONSTANTS; i++)
                if (i != 59)                            /* the viewport stays */
                    for (j = 0; j < 4; j++)
                        s_motion_prev[i][j] *= 1.01f;
        }
    }
    {
        static uint32_t said_at;
        if (s_gpu.flips >= said_at) {
            said_at = s_gpu.flips + 1200;
            fprintf(stderr, "  [MOTION] batches matched to the previous frame %u, new %u,"
                            " table full %u\n", s_motion_found, s_motion_new, s_motion_full);
        }
    }
    return (const float (*)[4])s_motion_prev;
}

/* Draw the current batch through its translated program. 1 if it was drawn
 * (or, under RECOMP_VP_GPU=2, never: the CPU path then draws it as always). */
static int vp_gpu_batch(int textured_src)
{
    static int diag = -1;
    const int mode = vp_gpu_mode();
    uint32_t n = s_gpu.idx_count, i, t, lo = 0xFFFFFFFFu, hi = 0;
    NvD3DProgramDraw d;
    NvD3DState st;
    VpVertex v0;
    uint64_t hash;
    int vs, textured;
    float W, H;

    if (!mode || !nvd3d_active() || surface_bpp() != 4)
        return 0;
    vp_gpu_summary();
    /* No batch is too small, measured: with the batch data in rings, the
     * menus -- nothing but two-triangle quads -- run at 2118 frames a second
     * with every batch sent through here, against 1160 on the CPU path.
     * (The first version said the opposite, 540 against 1160; that was a
     * whole megabyte renamed per batch, not the GPU.) RECOMP_VP_GPU_MIN=
     * <indices> keeps smaller batches on the CPU, for experiments. */
    {
        static int min_n = -1;
        if (min_n < 0) {
            const char *e = getenv("RECOMP_VP_GPU_MIN");
            min_n = e ? atoi(e) : 3;
            if (min_n < 3) min_n = 3;
        }
        if (mode == 1 && n < (uint32_t)min_n) {
            s_vpg_cpu_small++;
            return 0;
        }
    }
    if (diag < 0)
        diag = getenv("RECOMP_TRI_AT") || getenv("RECOMP_BATCH_DUMP") || env_on(ENV_VP_TRACE);
    if (diag || s_gpu.cull_enable || vpf_this_frame() || tex_skipped(s_gpu.tex.offset)) {
        s_vpg_cpu_diag++;
        return 0;
    }
    if (s_vp_code_gen != s_vp_gen)
        vp_decode();
    if (!s_vp_code_len || !s_vp_code[s_vp_code_len - 1].last) {
        s_vpg_cpu_nolast++;
        return 0;
    }
    vp_transform(s_gpu.idx[0], &v0);
    if (!v0.ok || v0.z == 0.0f) {
        s_vpg_cpu_flat++;
        return 0;
    }

    hash = vp_program_hash();
    vs = nvd3d_vs_program(hash, NULL, 0);
    if (vs == -2) {
        static char text[96 * 1024];
        VpHlsl b;
        b.p = text; b.n = 0; b.cap = sizeof text; b.over = 0;
        if (!vp_hlsl(&b)) {
            s_vpg_cpu_compile++;
            return 0;
        }
        vs = nvd3d_vs_program(hash, text, b.n);
    }
    if (vs < 0) {
        s_vpg_cpu_compile++;
        return 0;
    }

    for (i = 0; i < n; i++) {
        if (s_gpu.idx[i] < lo) lo = s_gpu.idx[i];
        if (s_gpu.idx[i] > hi) hi = s_gpu.idx[i];
    }
    memset(&d, 0, sizeof d);
    if (!vp_gather(&d, lo, hi) || !d3d_bind_target()) {
        s_vpg_cpu_gather++;
        return 0;
    }
    if (s_gpu.prim == 1) {
        /* Points (point 140): the bling on the fighters' jewellery. Two
         * triangles a point, the corner in the index's low bits. Off with
         * RECOMP_POINTS=0, which leaves them undrawn as before. */
        static int points_on = -1;
        if (points_on < 0) {
            const char *e = getenv("RECOMP_POINTS");
            points_on = !(e && *e == '0');
        }
        if (!points_on)
            return 0;
        for (i = 0, t = 0; i < n && t + 2 <= NV_MAX_INDICES * 2; i++, t += 2) {
            const uint32_t v = s_gpu.idx[i] << 2;
            s_vpg_index[3 * t]     = v | 0; s_vpg_index[3 * t + 1] = v | 1;
            s_vpg_index[3 * t + 2] = v | 2; s_vpg_index[3 * t + 3] = v | 2;
            s_vpg_index[3 * t + 4] = v | 1; s_vpg_index[3 * t + 5] = v | 3;
        }
        s_vpg_points++;
    } else if (s_gpu.prim >= 2 && s_gpu.prim <= 4) {
        /* Lines, line loop, line strip (point 140): a menu of the title draws
         * outlines with them, and nothing drew lines at all. As index pairs, t
         * counting segments; the draw is a D3D11 line list. RECOMP_LINES=0
         * leaves them undrawn. */
        static int lines_on = -1;
        uint32_t segs = 0, k;
        if (lines_on < 0) {
            const char *e = getenv("RECOMP_LINES");
            lines_on = !(e && *e == '0');
        }
        if (!lines_on)
            return 0;
        for (k = 0; k + 1 < n && 2 * (segs + 1) <= NV_MAX_INDICES * 6; k += (s_gpu.prim == 2 ? 2 : 1)) {
            s_vpg_index[2 * segs]     = s_gpu.idx[k];
            s_vpg_index[2 * segs + 1] = s_gpu.idx[k + 1];
            segs++;
        }
        if (s_gpu.prim == 3 && n > 2) {                  /* the loop closes */
            s_vpg_index[2 * segs]     = s_gpu.idx[n - 1];
            s_vpg_index[2 * segs + 1] = s_gpu.idx[0];
            segs++;
        }
        t = segs;
        s_vpg_lines++;
    } else {
        t = batch_triangles(s_gpu.prim, n, s_vp_tris, NV_MAX_INDICES * 2);
        for (i = 0; i < t; i++) {
            s_vpg_index[3 * i]     = s_gpu.idx[s_vp_tris[i][0]];
            s_vpg_index[3 * i + 1] = s_gpu.idx[s_vp_tris[i][1]];
            s_vpg_index[3 * i + 2] = s_gpu.idx[s_vp_tris[i][2]];
        }
    }
    W = (float)(s_gpu.pitch / 4);
    H = (float)s_d3d_rt_h;
    textured = textured_src && s_gpu.tex.valid && tex_stage0_enabled();
    d.vs = vs;
    d.consts = (const float (*)[4])s_vp.c;
    d.consts_gen = s_vp_c_gen;
    d.index = s_vpg_index;
    d.nindex = (s_gpu.prim >= 2 && s_gpu.prim <= 4) ? 2 * t : 3 * t;
    d.ax = (float)surface_aa_x() * 2.0f / W;
    d.ay = (float)surface_aa_y() * 2.0f / H;
    d.uv_scale[0] = tex_size_from_format(s_gpu.tex.color) ? (float)s_gpu.tex.width : 1.0f;
    d.uv_scale[1] = tex_size_from_format(s_gpu.tex.color) ? (float)s_gpu.tex.height : 1.0f;
    d.textured = textured;
    d.points = s_gpu.prim == 1;
    d.lines = s_gpu.prim >= 2 && s_gpu.prim <= 4;
    d3d_batch_state(&st, 1, textured, 1);
    s_vpg_eligible++;
    s_vpg_eligible_tris += t;

    if (mode == 2) {
        if (d.points || d.lines)
            return 0;                       /* nothing on the CPU path to check against */
        vp_gpu_check(&d, hash, t);
        return 0;
    }
    /* The motion pass: 3D of the shown surface that writes depth (what is
     * seen through particles is what moves under them), points and lines
     * aside. */
    if (motion_on() && !s_batch_ui && !d.points && !d.lines && st.depth_write
            && dma_resolve(s_gpu.color_offset) == s_d3d_show_base) {
        const uint64_t mh = hash ^ 0x6D6F74696F6E5633ull;     /* "motionV3" */
        int vm = nvd3d_vs_program(mh, NULL, 0);
        if (vm == -2) {
            static char mtext[192 * 1024];
            VpHlsl b;
            b.p = mtext; b.n = 0; b.cap = sizeof mtext; b.over = 0;
            vm = vp_hlsl_any(&b, 1) ? nvd3d_vs_program(mh, mtext, b.n) : -1;
        }
        if (vm >= 0) {
            d.motion = 1;
            d.vs_motion = vm;
            d.prev_consts = motion_prev(hash);
            d.prog_hash = hash;
            d.pos_addr = s_gpu.attr[0].offset;
            d.pos_n = s_gpu.idx_count;
            /* RECOMP_MOTION_PROBE=1: the first vertex run on the CPU with
             * these constants and with the previous ones; a batch whose
             * vertex jumps by more than 60 displayed pixels is described
             * (point 154: about 1% of the points moved by thousands). */
            {
                static int probe = -1;
                static uint32_t said;
                if (probe < 0)
                    probe = getenv("RECOMP_MOTION_PROBE") != NULL;
                if (probe && said < 80) {
                    static float save[VP_CONSTANTS][4];
                    VpVertex cur, prv;
                    const uint32_t runs = s_vp.runs;
                    vp_transform(s_gpu.idx[0], &cur);
                    memcpy(save, s_vp.c, sizeof save);
                    memcpy(s_vp.c, d.prev_consts, sizeof save);
                    vp_transform(s_gpu.idx[0], &prv);
                    memcpy(s_vp.c, save, sizeof save);
                    s_vp.runs = runs;
                    if (cur.ok && prv.ok) {
                        const float dx = prv.xy[0] - cur.xy[0], dy = prv.xy[1] - cur.xy[1];
                        const float len = sqrtf(dx * dx + dy * dy);
                        if (len > 60.0f) {
                            uint32_t ci, ndiff = 0, first = 0xFFFFFFFFu;
                            for (ci = 0; ci < VP_CONSTANTS; ci++)
                                if (memcmp(save[ci], d.prev_consts[ci], 16)) {
                                    if (first == 0xFFFFFFFFu) first = ci;
                                    ndiff++;
                                }
                            said++;
                            fprintf(stderr, "  [MOTION] jump swap %u prog %016llX n %u addr 0x%08X:"
                                            " %.0f px (%.0f,%.0f -> %.0f,%.0f) w %.2f -> %.2f;"
                                            " candidates %u matched %u distance %.1f;"
                                            " constants differing %u from c%u\n",
                                    s_gpu.flips, (unsigned long long)hash, s_gpu.idx_count,
                                    s_gpu.attr[0].offset, len, prv.xy[0], prv.xy[1],
                                    cur.xy[0], cur.xy[1], prv.w, cur.w, s_mp_candidates,
                                    s_mp_matched, s_mp_best, ndiff, first);
                        }
                    }
                }
            }
        }
    }
    nvd3d_draw_program(&d, &st);
    s_gpu.tris_drawn += t;
    s_gpu.drawn_offset = s_gpu.color_offset;
    s_vpg_batches++;
    s_vpg_tris += t;
    return 1;
}

/* RECOMP_BATCH_DUMP_IDX=<n>[,<n>...] : vider les lots qui ont exactement ce
 * nombre d'indices. Un modele connu se reconnait a la longueur de son ruban
 * (la corde du Club 357 : 266 indices, quatre flux, donc invisible au critere
 * des cinq flux), sans viser un point de l'ecran. */
static int dump_idx_wanted(uint32_t n)
{
    static int parsed;
    static uint32_t want[8];
    static int nwant;
    int i;

    if (!parsed) {
        const char *e = getenv("RECOMP_BATCH_DUMP_IDX");
        parsed = 1;
        while (e && *e && nwant < 8) {
            char *end;
            unsigned long v = strtoul(e, &end, 10);
            if (end == e)
                break;
            want[nwant++] = (uint32_t)v;
            e = (*end == ',') ? end + 1 : end;
        }
    }
    for (i = 0; i < nwant; i++)
        if (want[i] == n)
            return 1;
    return 0;
}

static void raster_batch_program(void)
{
    uint32_t i, n = s_gpu.idx_count, t;
    int textured = texcoord_attr()->offset || s_gpu.inline_active;

    /* Numbered so that a triangle printed by the point probe can be traced
     * back to the batch the frame log describes. */
    s_gpu.batch_in_frame++;
    s_gpu.batches_drawn++;
    s_batch_hit_probe = 0;

    if (!textured)
        s_gpu.batches_no_uv++;
    else if (!s_gpu.tex.valid)
        s_gpu.batches_no_tex++;
    else {
        s_gpu.batches_textured++;
        note_texture_use();
    }
    if (!tex_stage0_enabled()) {
        s_gpu.batches_stage_off++;
        if (s_gpu.color_mask)
            s_gpu.batches_stage_off_colour++;
    }

    /* The whole batch on the GPU, when that is certainly the same thing. */
    if (vp_gpu_batch(textured))
        return;

    /* A mesh names each vertex several times; run the program once per
     * index. The stamp says whether an index was seen in this batch. */
    {
        static uint32_t stamp[65536];
        static uint16_t first[65536];
        static uint32_t batch;
        if (++batch == 0) {
            memset(stamp, 0, sizeof stamp);
            batch = 1;
        }
        for (i = 0; i < n; i++) {
            const uint16_t ix = s_gpu.idx[i];
            if (stamp[ix] == batch) {
                s_vp_verts[i] = s_vp_verts[first[ix]];
            } else {
                vp_transform(ix, &s_vp_verts[i]);
                stamp[ix] = batch;
                first[ix] = (uint16_t)i;
            }
        }
    }

    /* RECOMP_VP_FRAME=N: every batch of the N-th swapped frame, with where it
     * lands and what it samples. */
    {
        if (vpf_this_frame()) {
            float lx = 1e30f, hx = -1e30f, ly = 1e30f, hy = -1e30f;
            /* Which part of the texture the batch samples, not just where it
             * lands. A quad showing the wrong picture is either reading the
             * wrong texture or the wrong region of the right one, and only
             * the span of the texture coordinates separates the two. */
            float lu = 1e30f, hu = -1e30f, lv = 1e30f, hv = -1e30f;
            for (i = 0; i < n; i++) {
                if (s_vp_verts[i].xy[0] < lx) lx = s_vp_verts[i].xy[0];
                if (s_vp_verts[i].xy[0] > hx) hx = s_vp_verts[i].xy[0];
                if (s_vp_verts[i].xy[1] < ly) ly = s_vp_verts[i].xy[1];
                if (s_vp_verts[i].xy[1] > hy) hy = s_vp_verts[i].xy[1];
                if (s_vp_verts[i].uv[0] < lu) lu = s_vp_verts[i].uv[0];
                if (s_vp_verts[i].uv[0] > hu) hu = s_vp_verts[i].uv[0];
                if (s_vp_verts[i].uv[1] < lv) lv = s_vp_verts[i].uv[1];
                if (s_vp_verts[i].uv[1] > hv) hv = s_vp_verts[i].uv[1];
            }
            s_vpf_flip = s_gpu.flips;
            /* Which surface the batch lands in. Two batches that cover the
             * same screen rectangle are only fighting over the same pixels
             * if they are drawn into the same target; one of them may be
             * filling a texture the other will later display. */
            fprintf(stderr, "  [VPF] f%u depth t%u f%X m%u clear%06X z0 %.1f fmt%08X"
                            "  dst 0x%08X pitch %u  sten t%u f%X r%02X m%02X\n",
                    s_gpu.flips, s_gpu.depth_test, s_gpu.depth_func, s_gpu.depth_mask,
                    s_gpu.zclear >> 8, s_vp_verts[0].z, s_gpu.format,
                    s_gpu.color_offset, s_gpu.pitch,
                    s_gpu.stencil_test, s_gpu.stencil_func,
                    s_gpu.stencil_ref, s_gpu.stencil_fmask);
            fprintf(stderr, "  [VPF] b%u f%u prim %u n %u  x %.1f..%.1f y %.1f..%.1f  d0 %08X"
                            "  tex 0x%08X %ux%u fmt %02X valid %d  u %.2f..%.2f"
                            " v %.2f..%.2f  blend %u %X/%X eq %X bc %08X  alpha %u %X %02X"
                            "  cmask %08X tc0 %08X  cull %u %X front %X\n",
                    s_gpu.batch_in_frame,
                    s_gpu.flips, s_gpu.prim, n, lx, hx, ly, hy, s_vp_verts[0].argb,
                    s_gpu.tex.offset, s_gpu.tex.width, s_gpu.tex.height,
                    s_gpu.tex.color, s_gpu.tex.valid,
                    lu, hu, lv, hv, s_gpu.blend_enable,
                    s_gpu.blend_sfactor, s_gpu.blend_dfactor,
                    s_gpu.blend_equation, s_gpu.blend_color,
                    s_gpu.alpha_test, s_gpu.alpha_func, s_gpu.alpha_ref,
                    s_gpu.color_mask, s_tex_reg[(0x1B0C - NV_TEX_FIRST) / 4],
                    s_gpu.cull_enable, s_gpu.cull_face, s_gpu.front_face);
            /* The stages past the first. This rasteriser samples stage 0 and
             * nothing else, so a title that combines two stages -- a stencil
             * pattern in one and the matter that fills it in the other -- gets
             * the matter alone, spread over the whole quad. Printing what the
             * other stages hold says whether that is what is happening. */
            vpf_fragment_state();
            {
                /* Every constant the program could be reading, not just the
                 * dozen usually worth printing: a batch that comes out with
                 * no colour and no texture coordinates is reading something,
                 * and the only way to find which slot carries what is to see
                 * them all. Zero rows are skipped, so this stays short. */
                /* The program itself, decoded. Which outputs it writes and
                 * which constants it reads is the only thing that says why a
                 * batch comes out with no colour and no texture coordinates:
                 * a program that never writes oD0 cannot be fixed by feeding
                 * it a better vertex colour. */
                {
                    uint32_t ii;
                    for (ii = 0; ii < s_vp_code_len; ii++) {
                        const VpInsn *d = &s_vp_code[ii];
                        char src[3][24];
                        uint32_t k;
                        /* Name only the slots this opcode reads. A slot an
                         * operation does not use holds whatever the assembler
                         * left there, and printing it as a source is how the
                         * previous reading of ADD looked plausible. */
                        for (k = 0; k < 3; k++) {
                            if (!(d->need & (1u << k))) {
                                snprintf(src[k], sizeof src[k], "-");
                                continue;
                            }
                            switch (d->mux[k]) {
                            case 1:
                                snprintf(src[k], sizeof src[k], "%sR%u.%c%c%c%c",
                                         d->neg[k] ? "-" : "", d->reg[k],
                                         "xyzw"[(d->swz[k] >> 6) & 3],
                                         "xyzw"[(d->swz[k] >> 4) & 3],
                                         "xyzw"[(d->swz[k] >> 2) & 3],
                                         "xyzw"[ d->swz[k]       & 3]);
                                break;
                            case 2:
                                snprintf(src[k], sizeof src[k], "%sv%u",
                                         d->neg[k] ? "-" : "", d->input);
                                break;
                            case 3:
                                snprintf(src[k], sizeof src[k], "%sc%u%s",
                                         d->neg[k] ? "-" : "", d->cidx,
                                         d->rel ? "+a0" : "");
                                break;
                            default:
                                snprintf(src[k], sizeof src[k], "0");
                                break;
                            }
                        }
                        fprintf(stderr, "  [VPF]   i%-2u mac %2u ilu %u out o%-2u"
                                        " mask %X mmask %X(R%u) imask %X"
                                        "  A %-10s B %-10s C %-10s\n",
                                ii, d->mac, d->ilu, d->orb, d->out_mask,
                                d->mac_mask, d->tmp, d->ilu_mask,
                                src[0], src[1], src[2]);
                    }
                }
                {
                    int ci, shown = 0;
                    fprintf(stderr, "  [VPF]   nonzero c:");
                    for (ci = 0; ci < 96; ci++) {
                        if (s_vp.c[ci][0] == 0.0f && s_vp.c[ci][1] == 0.0f
                         && s_vp.c[ci][2] == 0.0f && s_vp.c[ci][3] == 0.0f)
                            continue;
                        if (++shown > 16)
                            break;
                        fprintf(stderr, " c%d(%g %g %g %g)", ci,
                                s_vp.c[ci][0], s_vp.c[ci][1],
                                s_vp.c[ci][2], s_vp.c[ci][3]);
                    }
                    fprintf(stderr, "\n");
                }
                static const int rows[] = { 0, 1, 2, 3, 4, 5, 8, 9, 96, 97, 98, 99 };
                uint32_t r;
                float in[4];
                fetch_attr(&s_gpu.attr[0], s_gpu.idx[0], in);
                fprintf(stderr, "  [VPF]   v0 in %.2f %.2f %.2f %.2f\n", in[0], in[1], in[2], in[3]);
                /* Where the diffuse colour is read from, and what comes back.
                 * A batch drawn in transparent black is either reading the
                 * wrong place or reading it the wrong way, and the stream's
                 * type and stride say which. */
                {
                    const VertexAttr *ca = color_attr();
                    float c[4];
                    int got = fetch_attr(ca, s_gpu.idx[0], c);
                    fprintf(stderr, "  [VPF]   colour off 0x%08X type %u size %u"
                                    " stride %u -> %s %.3f %.3f %.3f %.3f\n",
                            ca->offset, ca->type, ca->size, ca->stride,
                            got ? "read" : "absent,white", c[0], c[1], c[2], c[3]);
                }
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

    if (tex_skipped(s_gpu.tex.offset))
        return;

    t = batch_triangles(s_gpu.prim, n, s_vp_tris, NV_MAX_INDICES * 2);
    for (i = 0; i < t; i++) {
        const VpVertex *v[3];
        uint32_t k;
        v[0] = &s_vp_verts[s_vp_tris[i][0]];
        v[1] = &s_vp_verts[s_vp_tris[i][1]];
        v[2] = &s_vp_verts[s_vp_tris[i][2]];
        if (!v[0]->ok || !v[1]->ok || !v[2]->ok) {
            s_gpu.tris_skipped_badvert++;
            continue;
        }
        for (k = 0; k < 3; k++) {
            s_tri_pos[k]  = s_vp_tris[i][k];
            s_tri_vidx[k] = s_gpu.idx[s_vp_tris[i][k]];
        }
        vp_draw_clipped(v, s_vp_verts[s_vp_tris[i][3]].argb, textured);
    }
    if (s_batch_hit_probe)
        dump_batch("covers a probe point");
    else if (getenv("RECOMP_BATCH_DUMP_SKIN")) {
        /* Les combattants sont les seuls lots a cinq flux (position, UV,
         * couleur, normale, indice de palette d'os) : c'est le critere le plus
         * sur pour attraper leurs matrices sans viser un point de l'ecran. */
        uint32_t a2, flux = 0;
        for (a2 = 0; a2 < NV_VERTEX_ATTRS; a2++)
            if (s_gpu.attr[a2].size && s_gpu.attr[a2].stride)
                flux++;
        if (flux >= 5)
            dump_batch("lot de combattant (cinq flux)");
    }
    else if (dump_idx_wanted(s_gpu.idx_count))
        dump_batch("nombre d'indices demande");
}

static void raster_batch(void)
{
    uint32_t i;
    uint32_t before = s_gpu.tris_drawn;

    if (s_gpu.idx_count < 3)
        return;
    if (vp_active() && !env_on(ENV_VP_OFF)) {
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
        if (!tex_stage0_enabled()) {
            s_gpu.batches_stage_off++;
            if (s_gpu.color_mask)
                s_gpu.batches_stage_off_colour++;
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
/* RECOMP_SEP: the quads of program 55F6628D2D8CC1F8 (point 147: a full-screen
 * one after the 3D darkens the corners -- but the same program also draws the
 * 3D effects, additively). Each new texture, vertex count and blend is
 * described once -- texture address, format, size, the
 * range of the raw positions -- and with RECOMP_SEP_DUMP=<prefix> the texture
 * is written out (<prefix>NN_<address>_fmtFF.bmp). Read only: positions are fetched, the
 * program is not run. */
static void sep_quad_probe(void)
{
    /* Keyed by texture and blend only: the particle systems of the effects
     * atlas (point 147) change their vertex count and extent every frame, and
     * with either in the key they filled the table by themselves, twice. */
    static struct { uint32_t tex, n, src, dst, batches; } seen[128];
    static uint32_t nseen, next_report = 600;
    float p[4], lo[2] = { 1e30f, 1e30f }, hi[2] = { -1e30f, -1e30f };
    uint32_t i;

    if (!sep_on() || s_gpu.prim != 8 || !vp_active()
            || vp_program_hash() != 0x55F6628D2D8CC1F8ull)
        return;
    for (i = 0; i < s_gpu.idx_count; i++) {
        if (!fetch_attr(&s_gpu.attr[0], s_gpu.idx[i], p))
            continue;
        if (p[0] < lo[0]) lo[0] = p[0];
        if (p[1] < lo[1]) lo[1] = p[1];
        if (p[0] > hi[0]) hi[0] = p[0];
        if (p[1] > hi[1]) hi[1] = p[1];
    }
    if (s_gpu.flips >= next_report && nseen) {
        static char line[128 * 32];
        size_t len = 0;
        next_report = s_gpu.flips + 1200;
        for (i = 0; i < nseen && len + 32 < sizeof line; i++)
            len += (size_t)snprintf(line + len, sizeof line - len, " #%u:%u(max %u v)", i,
                                    seen[i].batches, seen[i].n);
        fprintf(stderr, "  [SEP] quads swap %u, batches per entry:%s\n", s_gpu.flips, line);
    }
    for (i = 0; i < nseen; i++)
        if (seen[i].tex == s_gpu.tex.offset
                && seen[i].src == s_gpu.blend_sfactor && seen[i].dst == s_gpu.blend_dfactor) {
            seen[i].batches++;
            if (s_gpu.idx_count > seen[i].n)
                seen[i].n = s_gpu.idx_count;    /* the most vertices seen */
            return;
        }
    if (nseen == 128)
        return;
    seen[nseen].tex = s_gpu.tex.offset;
    seen[nseen].n = s_gpu.idx_count;
    seen[nseen].src = s_gpu.blend_sfactor;
    seen[nseen].dst = s_gpu.blend_dfactor;
    seen[nseen].batches = 1;
    nseen++;
    fprintf(stderr, "  [SEP] quad #%u swap %u  tex 0x%08X fmt 0x%02X %ux%u  %u vertices"
                    "  x %.1f..%.1f  y %.1f..%.1f  blend %u src 0x%X dst 0x%X  surface 0x%08X\n",
            nseen - 1, s_gpu.flips, s_gpu.tex.offset, s_gpu.tex.color, s_gpu.tex.width,
            s_gpu.tex.height, s_gpu.idx_count, lo[0], hi[0], lo[1], hi[1],
            s_gpu.blend_enable, s_gpu.blend_sfactor, s_gpu.blend_dfactor,
            dma_resolve(s_gpu.color_offset));
    {
        const char *pre = getenv("RECOMP_SEP_DUMP");
        if (pre)
            dump_texture_bmp_to(pre, nseen - 1);
    }
}

static void draw_primitive(void)
{
    float v[4];
    uint32_t i;

    if (!s_gpu.prim || !s_gpu.idx_count)
        return;
    sep_quad_probe();
    if (sep_w_probe())
        return;                                 /* RECOMP_SEP_CYCLE leaves it out */
    ui_classify();
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

    if (env_on(ENV_PB_EXEC_VERBOSE)) {
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
    if (count < 3 || count > NV_MAX_INDICES) {
        s_gpu.inline_batches_refused++;
        goto out;
    }
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
        uint32_t a = (method - NV097_SET_VERTEX_DATA4UB) / 4;
        s_imm_attr[a] = param;
        s_imm_set[a] = 1;
        if (a == 3)                                        /* diffuse */
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
        s_gpu.cull_face = NV_CULL_FACE_BACK;
        s_gpu.front_face = NV_FRONT_FACE_CCW;
        s_gpu.max_x = s_gpu.max_y = -1e30f;
    }
    /* Bring-up: the first parameters each surface method carries. A wrong
     * pitch or clip is indistinguishable from a method never arriving unless
     * the values are visible. */
    if (env_on(ENV_PB_EXEC_VERBOSE)) {
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
    case NV097_SET_CULL_FACE_ENABLE:
        /* Counted as well as stored: "no triangle was culled" reads the same
         * whether the title never switched it on or this is the wrong method
         * number, and only the count of non-zero writes tells them apart. */
        s_gpu.cull_sets++;
        if (param)
            s_gpu.cull_sets_on++;
        s_gpu.cull_enable = param;
        break;
    case NV097_SET_CULL_FACE:
        s_gpu.cull_face = param;
        break;
    case NV097_SET_FRONT_FACE:
        s_gpu.front_face = param;
        break;
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
            s_batch_overflowed = 0;
            s_batch_odd_route = s_batch_draw_arrays = 0;
            s_gpu.begins++;
        } else {
            /* Three ways a batch can have arrived, and only one is in use at
             * a time: vertices completed by SET_VERTEX4F, a payload written
             * with INLINE_ARRAY, or indices into the title's own arrays. */
            if (s_gpu.imm_count)
                invent_batch(2, s_gpu.imm_count);
            else if (s_gpu.inline_count)
                invent_batch(1, s_gpu.inline_count);
            else
                invent_batch(s_gpu.idx_count ? 0 : 3, s_gpu.idx_count);
            {
                uint32_t route = s_gpu.imm_count ? 2 : s_gpu.inline_count ? 1
                               : s_gpu.idx_count ? 0 : 3;
                s_sep_w = 2;                    /* set by draw_primitive, if it runs */
                s_batch_ui = 0;                 /* likewise */
            if (s_gpu.imm_count)
                draw_immediate();
            else if (s_gpu.inline_count)
                draw_inline_array();
            else if (s_gpu.idx_count)
                draw_primitive();
            else
                /* Opened, closed, and nothing in between: the title asked for
                 * a batch by a route this executor did not understand. */
                s_gpu.begins_empty++;
                sep_batch(route);
            }
            s_gpu.prim = 0;
            s_gpu.inline_count = 0;
            s_gpu.imm_count = 0;
        }
        break;

    case NV097_INLINE_ARRAY:
        /* Vertex data, not a pointer to it. Buffered rather than decoded here
         * because the format is only fully known at END. */
        if (s_gpu.prim && s_gpu.inline_count < NV_MAX_INLINE) {
            s_gpu.inline_buf[s_gpu.inline_count++] = param;
            if (s_gpu.inline_count > s_gpu.inline_max)
                s_gpu.inline_max = s_gpu.inline_count;
        } else if (s_gpu.prim) {
            if (!s_batch_overflowed) { s_batch_overflowed = 1;
                                       s_gpu.inline_drop_batches++; }
            s_gpu.inline_dropped++;
        }
        break;

    case NV097_ARRAY_ELEMENT32:
        /* One 32-bit index per parameter word. */
        s_gpu.elem32_seen++;
        if (!s_batch_odd_route) { s_batch_odd_route = 1;
                                  s_gpu.batches_with_elem32++; }
        if (s_gpu.prim && s_gpu.idx_count < NV_MAX_INDICES) {
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)param;
            if (s_gpu.idx_count > s_gpu.idx_max)
                s_gpu.idx_max = s_gpu.idx_count;
        } else if (s_gpu.prim) {
            if (!s_batch_overflowed) { s_batch_overflowed = 1;
                                       s_gpu.idx_drop_batches++; }
            s_gpu.idx_dropped++;
        }
        break;

    case NV097_DRAW_ARRAYS: {
        /* Not indices but a run of them: count vertices from start_index.
         * Expressed as a range because that is all the hardware is given. */
        uint32_t start = param & 0x00FFFFFFu;
        uint32_t count = ((param >> 24) & 0xFFu) + 1u;
        uint32_t k;
        s_gpu.draw_arrays_seen++;
        if (!s_batch_draw_arrays) { s_batch_draw_arrays = 1;
                                    s_gpu.batches_with_draw_arrays++; }
        if (!s_gpu.prim)
            break;
        for (k = 0; k < count; k++) {
            if (s_gpu.idx_count >= NV_MAX_INDICES) {
                if (!s_batch_overflowed) { s_batch_overflowed = 1;
                                           s_gpu.idx_drop_batches++; }
                s_gpu.idx_dropped++;
                continue;
            }
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(start + k);
        }
        if (s_gpu.idx_count > s_gpu.idx_max)
            s_gpu.idx_max = s_gpu.idx_count;
        break;
    }

    case NV097_ARRAY_ELEMENT16:
        /* Two 16-bit indices per parameter word. */
        if (s_gpu.prim && s_gpu.idx_count + 2 <= NV_MAX_INDICES) {
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(param & 0xFFFF);
            s_gpu.idx[s_gpu.idx_count++] = (uint16_t)(param >> 16);
            if (s_gpu.idx_count > s_gpu.idx_max)
                s_gpu.idx_max = s_gpu.idx_count;
        } else if (s_gpu.prim) {
            if (!s_batch_overflowed) { s_batch_overflowed = 1;
                                       s_gpu.idx_drop_batches++; }
            s_gpu.idx_dropped += 2;
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
        s_gpu.batch_in_frame = 0;
        /* RECOMP_FPS_LOG=1: frames a second, every five seconds, without a
         * window. The title bar is the only other place this is shown, and
         * comparing two ways of drawing is best done headless, where
         * nothing but the drawing differs between the runs. */
        {
            static int on = -1;
            static ULONGLONG t0;
            static uint32_t f0;
            if (on < 0) {
                on = getenv("RECOMP_FPS_LOG") != NULL;
                t0 = GetTickCount64();
                f0 = s_gpu.flips;
            }
            if (on) {
                ULONGLONG now = GetTickCount64();
                if (now - t0 >= 5000) {
                    fprintf(stderr, "  [FPS] %.1f frames/s" "\n",
                            (double)(s_gpu.flips - f0) * 1000.0 / (double)(now - t0));
                    t0 = now;
                    f0 = s_gpu.flips;
                }
            }
        }
        return;

    case NV097_FLIP_STALL:
        /* The stall ends when the buffer being read is the one just finished.
         * There is no scanout here to wait for, so that is now. */
        s_gpu.flip_read = s_gpu.flip_write;
        /* And this is a completed swap, which is what a title's own swap
         * counter counts -- see xbox_Nv2aFrameCounterFlip. */
        xbox_Nv2aFrameCounterFlip();
        sep_frame(s_d3d_show_base);
        ui_layer_on();                          /* reads RECOMP_UI_LAYER_AB once */
        ui_ab_tick();
        if (nvd3d_active() && s_d3d_show_base) {
            nvd3d_present(s_d3d_show_base, s_d3d_show_rect,
                          s_d3d_show_disp[0], s_d3d_show_disp[1]);
            /* A screenshot wants the pixels in guest memory. */
            if (xbox_FramebufferWindowPixelsWanted())
                nvd3d_readback(s_d3d_show_base,
                               (uint8_t *)xbox_GetMemoryOffset() + s_d3d_show_base,
                               s_d3d_show_pitch,
                               s_d3d_show_rect[1] + s_d3d_show_rect[3]);
        }
        xbox_FramebufferWindowLatch();
        if (env_on(ENV_PB_EXEC_VERBOSE)) {
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
        s_gpu.tex.mip_levels = (param >> 16) & 0xF;
        record_tex_reg(method, param);
        break;

    case NV097_SET_TEXTURE_FILTER:
        /* Recorded rather than honoured, which is the point: the gap between
         * what the title asks and what it gets is the size of the defect. */
        s_gpu.tex.lod_bias   =  param        & 0x1FFF;
        s_gpu.tex.filter_min = (param >> 16) & 0xFF;
        s_gpu.tex.filter_mag = (param >> 24) & 0xF;
        note_filter_use();
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
        record_combiner_reg(method, param);
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
    /* What honouring the cull bit cost. It reaches every model in the game,
     * so the share it removes is the number to weigh a regression against:
     * a few per cent is a title culling its back faces, half of everything
     * is a sign convention taken the wrong way round. */
    fprintf(stderr, "[GPU] face culling: %u culled of %u triangles submitted"
                    " with the bit on (%.1f%%), %u submitted in all;"
                    " enable written %u times, %u of them non-zero;"
                    " last state enable %u face %X front %X\n",
            s_gpu.tris_culled, s_gpu.tris_cull_on,
            s_gpu.tris_cull_on ? 100.0 * s_gpu.tris_culled / s_gpu.tris_cull_on : 0.0,
            s_gpu.tris_culled + s_gpu.tris_drawn,
            s_gpu.cull_sets, s_gpu.cull_sets_on,
            s_gpu.cull_enable, s_gpu.cull_face, s_gpu.front_face);
    if (s_gpu.cw_n || s_gpu.ccw_n)
        fprintf(stderr, "[GPU] facing by depth: clockwise on screen %u tris, mean z %.0f;"
                        " counter-clockwise %u tris, mean z %.0f"
                        " -- the nearer one is the side facing us\n",
                s_gpu.cw_n, s_gpu.cw_n ? s_gpu.cw_z_sum / s_gpu.cw_n : 0.0,
                s_gpu.ccw_n, s_gpu.ccw_n ? s_gpu.ccw_z_sum / s_gpu.ccw_n : 0.0);
    /* Geometry dropped without a word. A model with a few triangles missing
     * and everything else right is what each of these looks like on screen,
     * so they are printed together, always, rather than reasoned about. */
    fprintf(stderr, "[GPU] dropped quietly: %u indices over the %u cap in %u batches"
                    " (most in one batch %u); %u inline dwords over the %u cap"
                    " in %u batches (most %u), %u inline batches refused whole;"
                    " %u triangles on a rejected vertex (%u NaN, %u no last insn)\n",
            s_gpu.idx_dropped, (unsigned)NV_MAX_INDICES, s_gpu.idx_drop_batches,
            s_gpu.idx_max,
            s_gpu.inline_dropped, (unsigned)NV_MAX_INLINE, s_gpu.inline_drop_batches,
            s_gpu.inline_max, s_gpu.inline_batches_refused,
            s_gpu.tris_skipped_badvert, s_gpu.verts_bad_nan, s_gpu.verts_bad_nolast);
    for (i = 0; i < s_filt_count; i++)
        fprintf(stderr, "  [FILTER] min %s mag %s, %u mip level(s): %u batches"
                        " -- sampled %s from level 0\n",
                filter_name(s_filt[i].min), filter_name(s_filt[i].mag),
                s_filt[i].mips, s_filt[i].batches,
                getenv("RECOMP_TEX_FILTER")
                && atoi(getenv("RECOMP_TEX_FILTER")) == 0 ? "nearest" : "bilinear");
    fprintf(stderr, "[GPU] batches opened %u, drawn %u, opened and empty %u;"
                    " ARRAY_ELEMENT32 %u in %u batches,"
                    " DRAW_ARRAYS %u in %u batches\n",
            s_gpu.begins, s_gpu.batches_drawn, s_gpu.begins_empty,
            s_gpu.elem32_seen, s_gpu.batches_with_elem32,
            s_gpu.draw_arrays_seen, s_gpu.batches_with_draw_arrays);
    fprintf(stderr, "[GPU] vertex programs: mode %u, %u vertices run, %u unknown opcodes\n",
            s_vp.mode, s_vp.runs, s_vp.bad_insns);

    /* And of the batches that did rasterise, how many sampled anything. A menu
     * that draws its background from one texture and its text from another
     * shows both as flat colour if either half is missing, so the split is
     * what says which half. */
    fprintf(stderr, "[GPU] batches: %u textured, %u with no texcoords,"
                    " %u with texcoords but no usable stage\n",
            s_gpu.batches_textured, s_gpu.batches_no_uv, s_gpu.batches_no_tex);
    fprintf(stderr, "[GPU] stage 0 switched off in %u batches, %u of them"
                    " writing colour (drawn in vertex colour, unsampled)\n",
            s_gpu.batches_stage_off, s_gpu.batches_stage_off_colour);
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
