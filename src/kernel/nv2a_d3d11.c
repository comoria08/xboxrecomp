/*
 * nv2a_d3d11.c - GPU fill for the pushbuffer executor. See nv2a_d3d11.h.
 *
 * The CPU rasteriser in nv2a_pb_exec.c is the reference for every rule here:
 * nearest texel from stage 0 in texel units, modulated by the vertex colour;
 * alpha test; stencil and depth in GL order; blending from the title's
 * factors; colour mask; scissor to the surface clip. Surfaces become render
 * targets keyed by guest address, and the one the title swaps is presented to
 * the framebuffer window through a DXGI swap chain.
 *
 * ponytail: one texture stage, no filtering, no mipmaps, no fog or combiners
 * -- exactly what the CPU path does, so the two can be compared frame for
 * frame. The guest's own copy of a surface is only written on request
 * (nvd3d_readback).
 */
#if defined(_WIN32)

#define COBJMACROS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <wincodec.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "nv2a_d3d11.h"

extern ptrdiff_t xbox_GetMemoryOffset(void);
extern void *xbox_FramebufferWindowHwnd(void);
extern void xbox_FramebufferWindowExternal(void);
extern float xbox_WidescreenAspect(void);

static int s_state = -1;                 /* -1 unknown, 0 off, 1 on */

/* Internal resolution: every render target is drawn S times larger in each
 * axis (RECOMP_D3D11_SCALE, default 1). Everything the title sees -- surface
 * sizes, clip rectangles, texel coordinates, read-back pixels -- stays in its
 * own units; only the GPU side is multiplied. */
static uint32_t S = 1;
extern void xbox_FramebufferWindowScale(uint32_t scale);
static ID3D11Device         *s_dev;
static ID3D11DeviceContext  *s_ctx;
static ID3D11DeviceContext1 *s_ctx1;
static ID3D11VertexShader   *s_vs, *s_vs_present;
static ID3D11PixelShader    *s_ps, *s_ps_present;
static ID3D11InputLayout    *s_layout;
static ID3D11Buffer         *s_vb, *s_cb, *s_cb_present;
static ID3D11RasterizerState *s_rs;
static ID3D11SamplerState   *s_lin;
/* One per wrap/clamp combination, u in bit 0 and v in bit 1. The title
 * sets the mode per axis and a floor that tiles needs wrap on both. */
static ID3D11SamplerState   *s_smp[4];
static int                   s_filter_on = -1;
static ID3D11BlendState     *s_blend_off;
static ID3D11DepthStencilState *s_ds_off;
static IDXGISwapChain1      *s_swap;
static ID3D11RenderTargetView *s_back_rtv;
static uint32_t              s_back_w, s_back_h;

/* ── Render targets ─────────────────────────────────────────────────── */

#define MAX_TARGETS 16
typedef struct {
    uint32_t base, w, h;
    ID3D11Texture2D        *tex;
    ID3D11RenderTargetView *rtv;
    ID3D11ShaderResourceView *srv;
    ID3D11Texture2D        *copy;         /* for sampling it as a texture */
    ID3D11ShaderResourceView *copy_srv;
    uint32_t                used;
} Target;
static Target   s_rt[MAX_TARGETS];
static int      s_rt_count;
static int      s_cur = -1;              /* current target index */
static uint32_t s_use_clock;

typedef struct {
    uint32_t w, h;
    ID3D11Texture2D        *tex;
    ID3D11DepthStencilView *dsv;
} Depth;
static Depth s_depth[4];
static int   s_depth_count;

/* ── Textures ───────────────────────────────────────────────────────── */

#define MAX_TEXTURES 1024
typedef struct {
    uint32_t key[6];
    uint64_t hash;
    uint32_t w, h, used;                  /* the D3D texture's own size */
    uint32_t decl_w, decl_h;              /* what the title declared: the two differ
                                           * when a pack image stands in */
    int      rt;                          /* >= 0: really render target #rt */
    ID3D11Texture2D          *tex;
    ID3D11ShaderResourceView *srv;
} Tex;
static Tex  s_tex[MAX_TEXTURES];
static int  s_tex_count;

/* ── Batch ──────────────────────────────────────────────────────────── */

#define MAX_VERTS 65535
static NvD3DVertex s_verts[MAX_VERTS];
static uint32_t    s_nverts;
static NvD3DState  s_batch;
static int         s_batch_valid;

/* State objects, cached by the fields that make them. */
#define MAX_STATES 256
typedef struct { uint32_t k[7]; ID3D11BlendState *o; } BlendEntry;
typedef struct { uint32_t k[9]; ID3D11DepthStencilState *o; } DsEntry;
static BlendEntry s_blends[MAX_STATES]; static int s_nblends;
static DsEntry    s_dss[MAX_STATES];    static int s_ndss;

static const char s_hlsl[] =
"struct VSIn { float4 pos : POSITION; float2 uv : TEXCOORD0; float4 col : COLOR0; };\n"
"struct PSIn { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 col : COLOR0; };\n"
"struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 col : COLOR0;\n"
"               float clip : SV_ClipDistance0; };\n"
"VSOut vs_main(VSIn i) {\n"
"  VSOut o; o.pos = i.pos; o.uv = i.uv; o.col = i.col;\n"
"  o.clip = i.pos.z;   /* the near plane: window z >= 0; no far plane */\n"
"  return o;\n"
"}\n"
"cbuffer Draw : register(b0) { float4 tex; float4 atest; float4 extra; };\n"
"Texture2D t0 : register(t0);\n"
"SamplerState smp : register(s0);\n"
"float4 ps_main(PSIn i) : SV_Target {\n"
"  float4 c = i.col;\n"
"  if (atest.w > 0.5) {\n"
"    if (extra.y > 0.5) {\n"
/* Bilinear, which is what the title asks for: LINEAR on 617460 batches of a
 * session against 1 mip level, so the filtering is the whole of what was
 * missing. Coordinates arrive in texels; dividing by the size gives the
 * normalised form the sampler wants, and the sampler carries the wrap or
 * clamp the title set per axis. */
"      c = t0.Sample(smp, i.uv / float2(tex.x, tex.y)) * c;\n"
"    } else {\n"
"      int w = (int)tex.x, h = (int)tex.y;\n"
"      int u = (int)max(i.uv.x, 0.0), v = (int)max(i.uv.y, 0.0);\n"
"      u = (tex.z == 1.0) ? (u % w) : min(u, w - 1);\n"
"      v = (tex.w == 1.0) ? (v % h) : min(v, h - 1);\n"
"      int sc = (int)extra.x;\n"
"      c = t0.Load(int3(u * sc, v * sc, 0)) * c;\n"
"    }\n"
"  }\n"
"  if (atest.x > 0.5) {\n"
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
"  return c;\n"
"}\n"
"struct POut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
"POut vs_present(uint id : SV_VertexID) {\n"
"  POut o; float2 t = float2((id << 1) & 2, id & 2);\n"
"  o.uv = t; o.pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1); return o;\n"
"}\n"
"cbuffer Present : register(b0) { float4 rect; };\n"
"Texture2D src : register(t0);\n"
"SamplerState lin : register(s0);\n"
"float4 ps_present(POut i) : SV_Target {\n"
"  return float4(src.Sample(lin, rect.xy + i.uv * rect.zw).rgb, 1.0);\n"
"}\n";

static ID3DBlob *compile(const char *entry, const char *target)
{
    ID3DBlob *code = NULL, *err = NULL;
    HRESULT hr = D3DCompile(s_hlsl, sizeof s_hlsl - 1, "nv2a_d3d11", NULL, NULL,
                            entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
                            &code, &err);
    if (FAILED(hr)) {
        fprintf(stderr, "  [D3D11] shader %s failed: %s\n", entry,
                err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "?");
        if (err) ID3D10Blob_Release(err);
        return NULL;
    }
    if (err) ID3D10Blob_Release(err);
    return code;
}

static int init(void)
{
    ID3DBlob *vs, *ps, *vsp, *psp;
    HRESULT hr;
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };

    if (getenv("RECOMP_D3D11_DEBUG"))
        flags |= D3D11_CREATE_DEVICE_DEBUG;
    {
        const char *sc = getenv("RECOMP_D3D11_SCALE");
        int v = sc ? atoi(sc) : 1;
        S = (uint32_t)(v < 1 ? 1 : v > 8 ? 8 : v);
        xbox_FramebufferWindowScale(S);
    }
    hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, flags, levels, 2,
                           D3D11_SDK_VERSION, &s_dev, NULL, &s_ctx);
    if (FAILED(hr)) {
        fprintf(stderr, "  [D3D11] no device (0x%08lX), staying on the CPU\n", (unsigned long)hr);
        return 0;
    }
    if (FAILED(ID3D11DeviceContext_QueryInterface(s_ctx, &IID_ID3D11DeviceContext1, &s_ctx1)))
        s_ctx1 = NULL;

    vs = compile("vs_main", "vs_5_0");
    ps = compile("ps_main", "ps_5_0");
    vsp = compile("vs_present", "vs_5_0");
    psp = compile("ps_present", "ps_5_0");
    if (!vs || !ps || !vsp || !psp)
        return 0;
    ID3D11Device_CreateVertexShader(s_dev, ID3D10Blob_GetBufferPointer(vs),
                                    ID3D10Blob_GetBufferSize(vs), NULL, &s_vs);
    ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(ps),
                                   ID3D10Blob_GetBufferSize(ps), NULL, &s_ps);
    ID3D11Device_CreateVertexShader(s_dev, ID3D10Blob_GetBufferPointer(vsp),
                                    ID3D10Blob_GetBufferSize(vsp), NULL, &s_vs_present);
    ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(psp),
                                   ID3D10Blob_GetBufferSize(psp), NULL, &s_ps_present);
    {
        const D3D11_INPUT_ELEMENT_DESC el[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };
        ID3D11Device_CreateInputLayout(s_dev, el, 3, ID3D10Blob_GetBufferPointer(vs),
                                       ID3D10Blob_GetBufferSize(vs), &s_layout);
    }
    ID3D10Blob_Release(vs); ID3D10Blob_Release(ps);
    ID3D10Blob_Release(vsp); ID3D10Blob_Release(psp);

    {
        D3D11_BUFFER_DESC bd;
        memset(&bd, 0, sizeof bd);
        bd.ByteWidth = sizeof(NvD3DVertex) * MAX_VERTS;
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_vb);
        bd.ByteWidth = 48;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_cb);
        ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_cb_present);
    }
    {
        D3D11_RASTERIZER_DESC rd;
        memset(&rd, 0, sizeof rd);
        rd.FillMode = D3D11_FILL_SOLID;
        rd.CullMode = D3D11_CULL_NONE;          /* the CPU path draws both windings */
        rd.DepthClipEnable = FALSE;             /* window z is clamped, not clipped */
        rd.ScissorEnable = TRUE;
        ID3D11Device_CreateRasterizerState(s_dev, &rd, &s_rs);
    }
    {
        D3D11_SAMPLER_DESC sd;
        memset(&sd, 0, sizeof sd);
        sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        ID3D11Device_CreateSamplerState(s_dev, &sd, &s_lin);
    }
    {
        int i;
        for (i = 0; i < 4; i++) {
            D3D11_SAMPLER_DESC sd;
            memset(&sd, 0, sizeof sd);
            sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            sd.AddressU = (i & 1) ? D3D11_TEXTURE_ADDRESS_WRAP
                                  : D3D11_TEXTURE_ADDRESS_CLAMP;
            sd.AddressV = (i & 2) ? D3D11_TEXTURE_ADDRESS_WRAP
                                  : D3D11_TEXTURE_ADDRESS_CLAMP;
            sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            sd.MaxLOD = D3D11_FLOAT32_MAX;
            ID3D11Device_CreateSamplerState(s_dev, &sd, &s_smp[i]);
        }
    }
    {
        D3D11_BLEND_DESC bd;
        D3D11_DEPTH_STENCIL_DESC dd;
        memset(&bd, 0, sizeof bd);
        bd.RenderTarget[0].SrcBlend = bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        bd.RenderTarget[0].DestBlend = bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
        bd.RenderTarget[0].BlendOp = bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        ID3D11Device_CreateBlendState(s_dev, &bd, &s_blend_off);
        memset(&dd, 0, sizeof dd);
        dd.DepthFunc = D3D11_COMPARISON_ALWAYS;
        ID3D11Device_CreateDepthStencilState(s_dev, &dd, &s_ds_off);
    }
    fprintf(stderr, "  [D3D11] device ready: triangles are filled on the GPU, scale %u\n", S);
    return 1;
}

int nvd3d_trace(void)
{
    static int on = -1;
    if (on < 0)
        on = getenv("RECOMP_D3D11_TRACE") != NULL;
    return on;
}

int nvd3d_active(void)
{
    if (s_state < 0) {
        s_state = 0;
        if (getenv("RECOMP_D3D11"))
            s_state = init();
    }
    return s_state;
}

/* ── Translation of the title's enums ───────────────────────────────── */

static D3D11_COMPARISON_FUNC cmp(uint32_t f)
{
    return (f >= 0x0200 && f <= 0x0207) ? (D3D11_COMPARISON_FUNC)(f - 0x0200 + 1)
                                        : D3D11_COMPARISON_ALWAYS;
}

static D3D11_STENCIL_OP stencil_op(uint32_t op)
{
    switch (op) {
    case 0x0000: return D3D11_STENCIL_OP_ZERO;
    case 0x1E01: return D3D11_STENCIL_OP_REPLACE;
    case 0x1E02: return D3D11_STENCIL_OP_INCR_SAT;
    case 0x1E03: return D3D11_STENCIL_OP_DECR_SAT;
    case 0x150A: return D3D11_STENCIL_OP_INVERT;
    case 0x8507: return D3D11_STENCIL_OP_INCR;
    case 0x8508: return D3D11_STENCIL_OP_DECR;
    default:     return D3D11_STENCIL_OP_KEEP;
    }
}

static D3D11_BLEND blend_factor(uint32_t f, int alpha)
{
    switch (f) {
    case 0x0000: return D3D11_BLEND_ZERO;
    case 0x0001: return D3D11_BLEND_ONE;
    case 0x0300: return alpha ? D3D11_BLEND_SRC_ALPHA : D3D11_BLEND_SRC_COLOR;
    case 0x0301: return alpha ? D3D11_BLEND_INV_SRC_ALPHA : D3D11_BLEND_INV_SRC_COLOR;
    case 0x0302: return D3D11_BLEND_SRC_ALPHA;
    case 0x0303: return D3D11_BLEND_INV_SRC_ALPHA;
    case 0x0304: return D3D11_BLEND_DEST_ALPHA;
    case 0x0305: return D3D11_BLEND_INV_DEST_ALPHA;
    case 0x0306: return alpha ? D3D11_BLEND_DEST_ALPHA : D3D11_BLEND_DEST_COLOR;
    case 0x0307: return alpha ? D3D11_BLEND_INV_DEST_ALPHA : D3D11_BLEND_INV_DEST_COLOR;
    case 0x0308: return alpha ? D3D11_BLEND_ONE : D3D11_BLEND_SRC_ALPHA_SAT;
    case 0x8001: case 0x8003: return D3D11_BLEND_BLEND_FACTOR;
    case 0x8002: case 0x8004: return D3D11_BLEND_INV_BLEND_FACTOR;
    default:     return D3D11_BLEND_ONE;
    }
}

static ID3D11BlendState *blend_state(const NvD3DState *st)
{
    uint32_t k[7];
    int i;
    k[0] = st->blend_enable; k[1] = st->blend_src; k[2] = st->blend_dst;
    k[3] = st->blend_eq; k[4] = st->color_mask; k[5] = 0; k[6] = 0;
    if (!st->blend_enable)
        k[1] = k[2] = k[3] = 0;
    for (i = 0; i < s_nblends; i++)
        if (!memcmp(s_blends[i].k, k, sizeof k))
            return s_blends[i].o;
    {
        D3D11_BLEND_DESC bd;
        D3D11_RENDER_TARGET_BLEND_DESC *t = &bd.RenderTarget[0];
        ID3D11BlendState *o = NULL;
        UINT8 mask = 0;
        memset(&bd, 0, sizeof bd);
        t->BlendEnable = st->blend_enable != 0;
        t->SrcBlend  = blend_factor(st->blend_src, 0);
        t->DestBlend = blend_factor(st->blend_dst, 0);
        t->BlendOp   = st->blend_eq == 0x800A ? D3D11_BLEND_OP_SUBTRACT
                     : st->blend_eq == 0x800B ? D3D11_BLEND_OP_REV_SUBTRACT
                     : D3D11_BLEND_OP_ADD;
        if (st->blend_src == 0x0302 && st->blend_dst == 0x0303) {
            /* The CPU path's source-over stores an opaque result. */
            t->SrcBlendAlpha = D3D11_BLEND_ONE;
            t->DestBlendAlpha = D3D11_BLEND_ONE;
        } else {
            t->SrcBlendAlpha  = blend_factor(st->blend_src, 1);
            t->DestBlendAlpha = blend_factor(st->blend_dst, 1);
        }
        t->BlendOpAlpha = t->BlendOp;
        if (!t->BlendEnable) {
            t->SrcBlend = t->SrcBlendAlpha = D3D11_BLEND_ONE;
            t->DestBlend = t->DestBlendAlpha = D3D11_BLEND_ZERO;
            t->BlendOp = t->BlendOpAlpha = D3D11_BLEND_OP_ADD;
        }
        if (st->color_mask & 0x00010000u) mask |= D3D11_COLOR_WRITE_ENABLE_RED;
        if (st->color_mask & 0x00000100u) mask |= D3D11_COLOR_WRITE_ENABLE_GREEN;
        if (st->color_mask & 0x00000001u) mask |= D3D11_COLOR_WRITE_ENABLE_BLUE;
        if (st->color_mask & 0x01000000u) mask |= D3D11_COLOR_WRITE_ENABLE_ALPHA;
        t->RenderTargetWriteMask = mask;
        ID3D11Device_CreateBlendState(s_dev, &bd, &o);
        if (s_nblends < MAX_STATES) {
            memcpy(s_blends[s_nblends].k, k, sizeof k);
            s_blends[s_nblends++].o = o;
        }
        return o;
    }
}

static ID3D11DepthStencilState *ds_state(const NvD3DState *st)
{
    uint32_t k[9];
    int i;
    memset(k, 0, sizeof k);
    k[0] = st->depth_enable | (st->depth_write << 1) | (st->stencil_enable << 2);
    k[1] = st->depth_enable ? st->depth_func : 0;
    if (st->stencil_enable) {
        k[2] = st->stencil_func; k[3] = st->stencil_rmask; k[4] = st->stencil_wmask;
        k[5] = st->stencil_fail; k[6] = st->stencil_zfail; k[7] = st->stencil_zpass;
    }
    for (i = 0; i < s_ndss; i++)
        if (!memcmp(s_dss[i].k, k, sizeof k))
            return s_dss[i].o;
    {
        D3D11_DEPTH_STENCIL_DESC dd;
        ID3D11DepthStencilState *o = NULL;
        memset(&dd, 0, sizeof dd);
        dd.DepthEnable = st->depth_enable || st->depth_write;
        dd.DepthWriteMask = st->depth_write ? D3D11_DEPTH_WRITE_MASK_ALL
                                            : D3D11_DEPTH_WRITE_MASK_ZERO;
        dd.DepthFunc = st->depth_enable ? cmp(st->depth_func) : D3D11_COMPARISON_ALWAYS;
        dd.StencilEnable = st->stencil_enable != 0;
        dd.StencilReadMask = (UINT8)st->stencil_rmask;
        dd.StencilWriteMask = (UINT8)st->stencil_wmask;
        dd.FrontFace.StencilFunc = cmp(st->stencil_func);
        dd.FrontFace.StencilFailOp = stencil_op(st->stencil_fail);
        dd.FrontFace.StencilDepthFailOp = stencil_op(st->stencil_zfail);
        dd.FrontFace.StencilPassOp = stencil_op(st->stencil_zpass);
        dd.BackFace = dd.FrontFace;
        ID3D11Device_CreateDepthStencilState(s_dev, &dd, &o);
        if (s_ndss < MAX_STATES) {
            memcpy(s_dss[s_ndss].k, k, sizeof k);
            s_dss[s_ndss++].o = o;
        }
        return o;
    }
}

/* ── Targets ────────────────────────────────────────────────────────── */

static ID3D11DepthStencilView *depth_for(uint32_t w, uint32_t h)
{
    int i;
    for (i = 0; i < s_depth_count; i++)
        if (s_depth[i].w == w && s_depth[i].h == h)
            return s_depth[i].dsv;
    if (s_depth_count == (int)(sizeof s_depth / sizeof s_depth[0]))
        return s_depth[0].dsv;
    {
        D3D11_TEXTURE2D_DESC td;
        Depth *d = &s_depth[s_depth_count];
        memset(&td, 0, sizeof td);
        td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &d->tex)))
            return NULL;
        ID3D11Device_CreateDepthStencilView(s_dev, (ID3D11Resource *)d->tex, NULL, &d->dsv);
        d->w = w; d->h = h;
        ID3D11DeviceContext_ClearDepthStencilView(s_ctx, d->dsv,
            D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
        s_depth_count++;
        return d->dsv;
    }
}

static void target_release(Target *t)
{
    if (t->copy_srv) ID3D11ShaderResourceView_Release(t->copy_srv);
    if (t->copy)     ID3D11Texture2D_Release(t->copy);
    if (t->srv)      ID3D11ShaderResourceView_Release(t->srv);
    if (t->rtv)      ID3D11RenderTargetView_Release(t->rtv);
    if (t->tex)      ID3D11Texture2D_Release(t->tex);
    memset(t, 0, sizeof *t);
}

static int target_index(uint32_t base, uint32_t w, uint32_t h)
{
    int i, victim = -1;
    for (i = 0; i < s_rt_count; i++)
        if (s_rt[i].base == base && s_rt[i].w == w) {
            if (s_rt[i].h >= h)
                return i;
            /* Taller than before: rebuild in place, from guest memory. */
            victim = i;
            break;
        }
    if (victim >= 0) {
        target_release(&s_rt[victim]);
    } else if (s_rt_count < MAX_TARGETS) {
        victim = s_rt_count++;
    } else {
        victim = 0;
        for (i = 1; i < s_rt_count; i++)
            if (s_rt[i].used < s_rt[victim].used)
                victim = i;
        target_release(&s_rt[victim]);
    }
    {
        Target *t = &s_rt[victim];
        D3D11_TEXTURE2D_DESC td;
        D3D11_SUBRESOURCE_DATA init;
        const uint32_t *src = (const uint32_t *)((uintptr_t)base + xbox_GetMemoryOffset());
        uint32_t *big = NULL;
        HRESULT hr;
        memset(t, 0, sizeof *t);
        memset(&td, 0, sizeof td);
        td.Width = w * S; td.Height = h * S; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        /* Start from what guest memory holds: a title may draw over a surface
         * it filled by hand. Scaled up by repetition. */
        init.pSysMem = src;
        init.SysMemPitch = w * 4;
        init.SysMemSlicePitch = 0;
        if (S > 1) {
            uint32_t x, y;
            big = (uint32_t *)malloc((size_t)td.Width * td.Height * 4);
            if (!big) {
                t->base = 0xFFFFFFFFu;
                return -1;
            }
            for (y = 0; y < td.Height; y++)
                for (x = 0; x < td.Width; x++)
                    big[(size_t)y * td.Width + x] = src[(size_t)(y / S) * w + x / S];
            init.pSysMem = big;
            init.SysMemPitch = td.Width * 4;
        }
        hr = ID3D11Device_CreateTexture2D(s_dev, &td, &init, &t->tex);
        free(big);
        if (FAILED(hr)) {
            /* Leave the slot empty but valid for the search above. */
            t->base = 0xFFFFFFFFu;
            return -1;
        }
        ID3D11Device_CreateRenderTargetView(s_dev, (ID3D11Resource *)t->tex, NULL, &t->rtv);
        ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)t->tex, NULL, &t->srv);
        t->base = base; t->w = w; t->h = h;
        fprintf(stderr, "  [D3D11] render target 0x%08X %ux%u (drawn at %ux%u)\n",
                base, w, h, w * S, h * S);
        return victim;
    }
}

static void flush(void);

uint32_t nvd3d_target(uint32_t base, uint32_t width, uint32_t height)
{
    int i;
    if (!width || !height)
        return 0;
    if (s_cur >= 0 && s_rt[s_cur].base == base && s_rt[s_cur].w == width
            && s_rt[s_cur].h >= height)
        return s_rt[s_cur].h;
    flush();
    i = target_index(base, width, height);
    s_cur = i;
    if (i < 0)
        return 0;
    s_rt[i].used = ++s_use_clock;
    {
        ID3D11DepthStencilView *dsv = depth_for(s_rt[i].w * S, s_rt[i].h * S);
        ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 1, &s_rt[i].rtv, dsv);
    }
    return s_rt[i].h;
}

/* ── Textures ───────────────────────────────────────────────────────── */

static uint32_t *s_decode_buf;
static size_t    s_decode_cap;

/* ── Texture pack, keyed by the texture's fingerprint ───────────────── */
/*
 * RECOMP_TEX_PACK=<dir> lets a file stand in for a texture the title loaded,
 * and it may be LARGER than the original. That works because nothing
 * downstream knows the replacement's size: the pixel shader divides the
 * incoming coordinates by tex_w/tex_h, which is what the TITLE declared
 * through SET_TEXTURE_FORMAT, and samples normalised. Measured in a frame
 * log: a 64x64 texture is addressed u 1.25..63.05, a 256x256 one
 * u 19.93..236.16 -- texels of the declared size. Divided by that size the
 * result is 0..1 whatever the file behind it actually is.
 *
 * Hence the one rule: the replacement must keep the ORIGINAL RATIO. A 2x or
 * 4x image lands exactly on the original; a different ratio would stretch the
 * texture over the same UV rectangle, and is refused rather than silently
 * distorting the model.
 *
 * Files are named by the fingerprint the GPU path already computes over the
 * texture's bytes (and its palette), so a pack does not depend on addresses,
 * load order, or which archive a texture came from. RECOMP_TEX_PACK_DUMP=<dir>
 * writes each texture out under that name, which is how a pack is started:
 * run once, enlarge the files worth enlarging, point RECOMP_TEX_PACK at them.
 *
 * BMP, because it needs no decoder and every editor writes it: 32-bit BGRA
 * out (BITMAPV4HEADER, so the alpha survives the round trip -- a fighter's
 * atlas is alpha-tested and would come back opaque otherwise), 24- or 32-bit
 * in.
 *
 * Reserve: a texture read back out of a render target keeps the Load path and
 * is addressed in whole texels, so it is not packable. Those are the title's
 * own frames, not artwork.
 */
static const char *pack_dir(void)
{
    static const char *dir;
    static int asked;
    if (!asked) { asked = 1; dir = getenv("RECOMP_TEX_PACK"); }
    return dir;
}

static const char *pack_dump_dir(void)
{
    static const char *dir;
    static int asked;
    if (!asked) { asked = 1; dir = getenv("RECOMP_TEX_PACK_DUMP"); }
    return dir;
}

static void pack_path(char *out, size_t n, const char *dir, uint64_t hash,
                      const char *ext)
{
    snprintf(out, n, "%s%s%016llX%s", dir,
             (dir[0] && (dir[strlen(dir) - 1] == '\\' || dir[strlen(dir) - 1] == '/'))
                 ? "" : "\\",
             (unsigned long long)hash, ext);
}

/* Reads 24- or 32-bit uncompressed BMP into fresh BGRA. NULL if absent or
 * unusable; *pw / *ph get the file's size. */
static uint32_t *pack_read(const char *path, uint32_t *pw, uint32_t *ph)
{
    FILE *f = fopen(path, "rb");
    uint8_t fh[14], ih[124];
    uint32_t off, hs, bpp, comp, row, pad, x, y;
    int32_t  w, h, flip;
    uint32_t *out;

    if (!f)
        return NULL;
    if (fread(fh, 1, sizeof fh, f) != sizeof fh || fh[0] != 'B' || fh[1] != 'M')
        { fclose(f); return NULL; }
    memcpy(&off, fh + 10, 4);
    if (fread(ih, 1, 4, f) != 4)
        { fclose(f); return NULL; }
    memcpy(&hs, ih, 4);
    if (hs < 40 || hs > sizeof ih || fread(ih + 4, 1, hs - 4, f) != hs - 4)
        { fclose(f); return NULL; }
    memcpy(&w, ih + 4, 4);
    memcpy(&h, ih + 8, 4);
    memcpy(&bpp, ih + 14, 4);
    bpp &= 0xFFFFu;
    memcpy(&comp, ih + 16, 4);
    flip = h > 0;                       /* positive height: rows bottom-up */
    if (h < 0) h = -h;
    if (w <= 0 || h <= 0 || w > 8192 || h > 8192 || (bpp != 24 && bpp != 32)
            || (comp != 0 && comp != 3))
        { fclose(f); return NULL; }

    row = ((uint32_t)w * (bpp / 8) + 3u) & ~3u;
    pad = row - (uint32_t)w * (bpp / 8);
    out = (uint32_t *)malloc((size_t)w * h * 4);
    if (!out) { fclose(f); return NULL; }
    if (fseek(f, (long)off, SEEK_SET) != 0) { free(out); fclose(f); return NULL; }
    for (y = 0; y < (uint32_t)h; y++) {
        uint32_t *dst = out + (size_t)(flip ? (uint32_t)h - 1 - y : y) * w;
        for (x = 0; x < (uint32_t)w; x++) {
            uint8_t px[4] = { 0, 0, 0, 0xFF };
            if (fread(px, 1, bpp / 8, f) != bpp / 8)
                { free(out); fclose(f); return NULL; }
            if (bpp == 24) px[3] = 0xFF;
            dst[x] = ((uint32_t)px[3] << 24) | ((uint32_t)px[2] << 16)
                   | ((uint32_t)px[1] << 8) | px[0];
        }
        if (pad && fseek(f, (long)pad, SEEK_CUR) != 0)
            { free(out); fclose(f); return NULL; }
    }
    fclose(f);
    *pw = (uint32_t)w; *ph = (uint32_t)h;
    return out;
}

static void pack_write(const char *path, const uint32_t *bgra, uint32_t w, uint32_t h)
{
    FILE *f;
    uint8_t fh[14], ih[108];
    uint32_t off = 14 + 108, filesz = off + w * h * 4, y;
    int32_t  neg = -(int32_t)h;         /* top-down, so no row juggling */
    static const uint32_t masks[4] = { 0x00FF0000u, 0x0000FF00u, 0x000000FFu, 0xFF000000u };

    f = fopen(path, "wb");
    if (!f)
        return;
    memset(fh, 0, sizeof fh); memset(ih, 0, sizeof ih);
    fh[0] = 'B'; fh[1] = 'M';
    memcpy(fh + 2, &filesz, 4);
    memcpy(fh + 10, &off, 4);
    ih[0] = 108;                        /* BITMAPV4HEADER */
    memcpy(ih + 4, &w, 4);
    memcpy(ih + 8, &neg, 4);
    ih[12] = 1; ih[14] = 32;
    ih[16] = 3;                         /* BI_BITFIELDS */
    memcpy(ih + 40, masks, sizeof masks);
    ih[56] = 'B'; ih[57] = 'G'; ih[58] = 'R'; ih[59] = 's';   /* LCS_sRGB */
    fwrite(fh, 1, sizeof fh, f);
    fwrite(ih, 1, sizeof ih, f);
    for (y = 0; y < h; y++)
        fwrite(bgra + (size_t)y * w, 4, w, f);
    fclose(f);
}

/* Anything Windows can decode -- PNG, TIFF, JPEG -- through WIC, which ships
 * with the OS and needs no third-party decoder. That matters because an
 * upscaler writes PNG or TIFF, not BMP: without this a pack would need a
 * conversion pass between the tool and the game. BMP still goes through the
 * reader above, which is ours and cannot lose the alpha we wrote.
 *
 * WIC wants COM, which D3D11 does not require, so it is started here and a
 * thread that already chose a different apartment is left alone. */
static uint32_t *pack_read_wic(const char *path, uint32_t *pw, uint32_t *ph)
{
    static IWICImagingFactory *factory;
    static int tried;
    IWICBitmapDecoder   *dec = NULL;
    IWICBitmapFrameDecode *frame = NULL;
    IWICFormatConverter *conv = NULL;
    uint32_t *out = NULL;
    wchar_t wide[512];
    UINT w = 0, h = 0;

    if (!tried) {
        HRESULT hr;
        tried = 1;
        hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        (void)hr;   /* RPC_E_CHANGED_MODE is fine: someone got here first */
        CoCreateInstance(&CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER,
                         &IID_IWICImagingFactory, (void **)&factory);
    }
    if (!factory)
        return NULL;
    if (!MultiByteToWideChar(CP_ACP, 0, path, -1, wide,
                             sizeof wide / sizeof wide[0]))
        return NULL;
    if (FAILED(IWICImagingFactory_CreateDecoderFromFilename(
            factory, wide, NULL, GENERIC_READ,
            WICDecodeMetadataCacheOnDemand, &dec)))
        return NULL;
    if (SUCCEEDED(IWICBitmapDecoder_GetFrame(dec, 0, &frame))
     && SUCCEEDED(IWICImagingFactory_CreateFormatConverter(factory, &conv))
     && SUCCEEDED(IWICFormatConverter_Initialize(
            conv, (IWICBitmapSource *)frame, &GUID_WICPixelFormat32bppBGRA,
            WICBitmapDitherTypeNone, NULL, 0.0, WICBitmapPaletteTypeCustom))
     && SUCCEEDED(IWICFormatConverter_GetSize(conv, &w, &h))
     && w && h && w <= 8192 && h <= 8192) {
        out = (uint32_t *)malloc((size_t)w * h * 4);
        if (out && FAILED(IWICFormatConverter_CopyPixels(
                conv, NULL, w * 4, (UINT)((size_t)w * h * 4), (BYTE *)out))) {
            free(out);
            out = NULL;
        }
    }
    if (conv)  IWICFormatConverter_Release(conv);
    if (frame) IWICBitmapFrameDecode_Release(frame);
    if (dec)   IWICBitmapDecoder_Release(dec);
    if (out) { *pw = w; *ph = h; }
    return out;
}

/* The pack file for this fingerprint, whatever its extension. */
static uint32_t *pack_open(const char *dir, uint64_t hash, uint32_t *pw, uint32_t *ph)
{
    static const char *ext[] = { ".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg" };
    char path[512];
    size_t i;

    for (i = 0; i < sizeof ext / sizeof ext[0]; i++) {
        uint32_t *pix;
        pack_path(path, sizeof path, dir, hash, ext[i]);
        pix = i == 0 ? pack_read(path, pw, ph) : pack_read_wic(path, pw, ph);
        if (pix)
            return pix;
    }
    return NULL;
}

/* The replacement for this fingerprint, or NULL. Refuses a different ratio. */
static uint32_t *pack_lookup(uint64_t hash, uint32_t width, uint32_t height,
                             uint32_t *pw, uint32_t *ph)
{
    const char *dir = pack_dir();
    uint32_t *pix;

    if (!dir || !*dir)
        return NULL;
    pix = pack_open(dir, hash, pw, ph);
    if (!pix)
        return NULL;
    if ((uint64_t)*pw * height != (uint64_t)*ph * width) {
        fprintf(stderr, "  [TEXPACK] %016llX refused: %ux%u is not the ratio of %ux%u\n",
                (unsigned long long)hash, *pw, *ph, width, height);
        free(pix);
        return NULL;
    }
    {
        static int said;
        if (said++ < 64)
            fprintf(stderr, "  [TEXPACK] %016llX %ux%u -> %ux%u\n",
                    (unsigned long long)hash, width, height, *pw, *ph);
        if (said == 1) {
            const char *f = getenv("RECOMP_TEX_FILTER");
            if (f && atoi(f) == 0)
                fprintf(stderr, "  [TEXPACK] warning: RECOMP_TEX_FILTER=0 reads whole texels"
                        " of the declared size, so an enlarged image is only partly seen\n");
        }
    }
    return pix;
}

int nvd3d_texture(const uint32_t key[6], uint64_t hash, uint32_t width, uint32_t height,
                  uint32_t mem_base, uint32_t mem_bytes,
                  void (*decode)(uint32_t *dst, uint32_t width, uint32_t height))
{
    int i, slot = -1;
    Tex *t;
    uint32_t *pack = NULL, *src;
    uint32_t sw, sh;

    if (!width || !height || width > 4096 || height > 4096)
        return -1;
    s_use_clock++;

    /* A texture that lives inside a render target is that render target:
     * sample what the GPU drew, not the stale guest copy. */
    for (i = 0; i < s_rt_count; i++) {
        const Target *r = &s_rt[i];
        if (mem_base < r->base + r->w * 4 * r->h && mem_base + mem_bytes > r->base) {
            static int said;
            if (said++ < 4 || (nvd3d_trace() && said < 200))
                fprintf(stderr, "  [D3D11] texture 0x%08X fmt 0x%02X %ux%u (%u bytes) is render"
                        " target 0x%08X %ux%u; drawing into 0x%08X\n",
                        mem_base, key[1], width, height, mem_bytes, r->base, r->w, r->h,
                        s_cur >= 0 ? s_rt[s_cur].base : 0);
            return 0x10000 + i;
        }
    }

    for (i = 0; i < s_tex_count; i++)
        if (!memcmp(s_tex[i].key, key, sizeof s_tex[i].key)) {
            slot = i;
            break;
        }
    if (slot >= 0 && s_tex[slot].hash == hash && s_tex[slot].decl_w == width
            && s_tex[slot].decl_h == height) {
        s_tex[slot].used = s_use_clock;
        return slot;
    }
    if (slot < 0) {
        if (s_tex_count < MAX_TEXTURES) {
            slot = s_tex_count++;
        } else {
            slot = 0;
            for (i = 1; i < s_tex_count; i++)
                if (s_tex[i].used < s_tex[slot].used)
                    slot = i;
        }
    }
    t = &s_tex[slot];
    if ((size_t)width * height > s_decode_cap) {
        free(s_decode_buf);
        s_decode_cap = (size_t)width * height;
        s_decode_buf = (uint32_t *)malloc(s_decode_cap * 4);
        if (!s_decode_buf) {
            s_decode_cap = 0;
            return -1;
        }
    }
    decode(s_decode_buf, width, height);

    /* A pack image stands in for what the title loaded, at its own size. The
     * declared size is what the shader divides by, so only the ratio has to
     * match -- see the comment on pack_lookup. */
    pack = pack_lookup(hash, width, height, &sw, &sh);
    if (pack) {
        src = pack;
    } else {
        src = s_decode_buf; sw = width; sh = height;
        if (pack_dump_dir()) {
            char path[512];
            pack_path(path, sizeof path, pack_dump_dir(), hash, ".bmp");
            pack_write(path, s_decode_buf, width, height);
        }
    }

    if (t->tex && t->w == sw && t->h == sh) {
        flush();   /* the batch may sample the old contents */
        ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)t->tex, 0, NULL,
                                              src, sw * 4, 0);
    } else {
        D3D11_TEXTURE2D_DESC td;
        D3D11_SUBRESOURCE_DATA init;
        flush();
        if (t->srv) ID3D11ShaderResourceView_Release(t->srv);
        if (t->tex) ID3D11Texture2D_Release(t->tex);
        t->srv = NULL; t->tex = NULL;
        memset(&td, 0, sizeof td);
        td.Width = sw; td.Height = sh; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        init.pSysMem = src;
        init.SysMemPitch = sw * 4;
        init.SysMemSlicePitch = 0;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, &init, &t->tex))) {
            free(pack);
            return -1;
        }
        ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)t->tex, NULL, &t->srv);
    }
    free(pack);
    memcpy(t->key, key, sizeof t->key);
    t->hash = hash;
    t->w = sw;
    t->h = sh;
    t->decl_w = width;
    t->decl_h = height;
    t->used = s_use_clock;
    return slot;
}

static ID3D11ShaderResourceView *texture_view(int32_t handle)
{
    if (handle >= 0x10000) {
        Target *r = &s_rt[handle - 0x10000];
        /* Never sample the target being drawn into: take a copy. */
        if (!r->copy) {
            D3D11_TEXTURE2D_DESC td;
            ID3D11Texture2D_GetDesc(r->tex, &td);
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &r->copy)))
                return NULL;
            ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)r->copy, NULL,
                                                  &r->copy_srv);
        }
        ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)r->copy,
                                         (ID3D11Resource *)r->tex);
        return r->copy_srv;
    }
    if (handle >= 0 && handle < s_tex_count)
        return s_tex[handle].srv;
    return NULL;
}

/* ── Frame log (RECOMP_D3D11_TRACE) ─────────────────────────────────────
 *
 * Everything drawn in the current frame, one line per batch or clear, kept in
 * memory and printed only when the presented image turns out to be (almost)
 * all white -- the flash the title shows for a frame or two -- plus a few
 * ordinary frames to compare against. */
#define FRAMELOG_LINES 1024
static char     s_framelog[FRAMELOG_LINES][200];
static unsigned s_framelog_n;

static void framelog(const char *fmt, ...)
{
    va_list ap;
    if (!nvd3d_trace() || s_framelog_n >= FRAMELOG_LINES)
        return;
    va_start(ap, fmt);
    vsnprintf(s_framelog[s_framelog_n++], sizeof s_framelog[0], fmt, ap);
    va_end(ap);
}

/* ── Drawing ────────────────────────────────────────────────────────── */

static void flush(void)
{
    const NvD3DState *st = &s_batch;
    D3D11_MAPPED_SUBRESOURCE m;
    UINT stride = sizeof(NvD3DVertex), offset = 0;

    if (!s_nverts || s_cur < 0) {
        s_nverts = 0;
        return;
    }
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_vb, 0,
                                       D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        s_nverts = 0;
        return;
    }
    memcpy(m.pData, s_verts, s_nverts * sizeof(NvD3DVertex));
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_vb, 0);

    {
        float cb[12];
        int textured = st->texture >= 0;
        ID3D11ShaderResourceView *srv = textured ? texture_view(st->texture) : NULL;
        if (!srv)
            textured = 0;
        cb[0] = (float)st->tex_w; cb[1] = (float)st->tex_h;
        cb[2] = (float)st->addr_u; cb[3] = (float)st->addr_v;
        cb[4] = st->alpha_test ? 1.0f : 0.0f;
        cb[5] = (float)(st->alpha_func >= 0x0200 && st->alpha_func <= 0x0207
                        ? st->alpha_func - 0x0200 : 7);
        cb[6] = (float)st->alpha_ref;
        cb[7] = textured ? 1.0f : 0.0f;
        cb[8] = st->texture >= 0x10000 ? (float)S : 1.0f;   /* a render target */
        /* Point sampling stays reachable with RECOMP_TEX_FILTER=0, and a
         * texture read back out of a render target keeps it: its copy is S
         * times larger, and the Load path is what knows that. */
        if (s_filter_on < 0) {
            const char *e = getenv("RECOMP_TEX_FILTER");
            s_filter_on = !(e && atoi(e) == 0);
        }
        cb[9] = (s_filter_on && st->texture < 0x10000) ? 1.0f : 0.0f;
        cb[10] = cb[11] = 0.0f;
        if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb, 0,
                                              D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            memcpy(m.pData, cb, sizeof cb);
            ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb, 0);
        }
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &srv);
        {
            int k = (st->addr_u == 1 ? 1 : 0) | (st->addr_v == 1 ? 2 : 0);
            ID3D11DeviceContext_PSSetSamplers(s_ctx, 0, 1, &s_smp[k]);
        }
    }
    {
        D3D11_VIEWPORT vp;
        D3D11_RECT sc;
        FLOAT bf[4];
        const Target *t = &s_rt[s_cur];
        vp.TopLeftX = 0; vp.TopLeftY = 0;
        vp.Width = (FLOAT)(t->w * S); vp.Height = (FLOAT)(t->h * S);
        vp.MinDepth = 0; vp.MaxDepth = 1;
        sc.left = (LONG)(st->scissor[0] * S);
        sc.top = (LONG)(st->scissor[1] * S);
        sc.right = (LONG)((st->scissor[0] + st->scissor[2]) * S);
        sc.bottom = (LONG)((st->scissor[1] + st->scissor[3]) * S);
        ID3D11DeviceContext_RSSetViewports(s_ctx, 1, &vp);
        ID3D11DeviceContext_RSSetScissorRects(s_ctx, 1, &sc);
        ID3D11DeviceContext_RSSetState(s_ctx, s_rs);
        if ((st->blend_src & ~1u) == 0x8003 || (st->blend_dst & ~1u) == 0x8003) {
            bf[0] = bf[1] = bf[2] = bf[3] = (FLOAT)((st->blend_color >> 24) & 0xFF) / 255.0f;
        } else {
            bf[0] = (FLOAT)((st->blend_color >> 16) & 0xFF) / 255.0f;
            bf[1] = (FLOAT)((st->blend_color >>  8) & 0xFF) / 255.0f;
            bf[2] = (FLOAT)( st->blend_color        & 0xFF) / 255.0f;
            bf[3] = (FLOAT)((st->blend_color >> 24) & 0xFF) / 255.0f;
        }
        ID3D11DeviceContext_OMSetBlendState(s_ctx, blend_state(st), bf, 0xFFFFFFFFu);
        ID3D11DeviceContext_OMSetDepthStencilState(s_ctx, ds_state(st), st->stencil_ref & 0xFF);
    }
    ID3D11DeviceContext_IASetInputLayout(s_ctx, s_layout);
    ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_IASetVertexBuffers(s_ctx, 0, 1, &s_vb, &stride, &offset);
    ID3D11DeviceContext_VSSetShader(s_ctx, s_vs, NULL, 0);
    ID3D11DeviceContext_PSSetShader(s_ctx, s_ps, NULL, 0);
    ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_cb);
    ID3D11DeviceContext_Draw(s_ctx, s_nverts, 0);
    {
        const NvD3DVertex *v = s_verts;
        const uint32_t *k = (st->texture >= 0 && st->texture < s_tex_count)
                          ? s_tex[st->texture].key : NULL;
        framelog("draw %5u v  pos(%.1f %.1f %.4f %.2f) col(%.2f %.2f %.2f %.2f) uv(%.1f %.1f)"
                 " tex %d%s 0x%08X f%02X %ux%u  z%u%u f%X  st%u f%X r%u op%X/%X/%X"
                 "  bl%u %X/%X eq%X c%08X  cm%08X  at%u f%X r%u",
                 s_nverts, v->pos[0], v->pos[1], v->pos[2], v->pos[3],
                 v->col[0], v->col[1], v->col[2], v->col[3], v->uv[0], v->uv[1],
                 st->texture, st->texture >= 0x10000 ? "(rt)" : "",
                 k ? k[0] : 0, k ? k[1] : 0, st->tex_w, st->tex_h,
                 st->depth_enable, st->depth_write, st->depth_func,
                 st->stencil_enable, st->stencil_func, st->stencil_ref,
                 st->stencil_fail, st->stencil_zfail, st->stencil_zpass,
                 st->blend_enable, st->blend_src, st->blend_dst, st->blend_eq,
                 st->blend_color, st->color_mask,
                 st->alpha_test, st->alpha_func, st->alpha_ref);
    }
    s_nverts = 0;
}

void nvd3d_triangle(const NvD3DVertex v[3], const NvD3DState *st)
{
    if (s_cur < 0)
        return;
    if (!s_batch_valid || memcmp(&s_batch, st, sizeof s_batch) != 0) {
        flush();
        s_batch = *st;
        s_batch_valid = 1;
    }
    if (s_nverts + 3 > MAX_VERTS)
        flush();
    memcpy(s_verts + s_nverts, v, 3 * sizeof *v);
    s_nverts += 3;
}

void nvd3d_clear(uint32_t what, uint32_t argb, float depth, uint8_t stencil,
                 const uint32_t rect[4])
{
    const Target *t;
    if (s_cur < 0)
        return;
    flush();
    t = &s_rt[s_cur];
    framelog("clear what %u argb %08X depth %.4f stencil %u rect %u,%u %ux%u",
             what, argb, depth, stencil, rect[0], rect[1], rect[2], rect[3]);
    if (what & 3) {
        ID3D11DepthStencilView *dsv = depth_for(t->w * S, t->h * S);
        UINT f = ((what & 1) ? D3D11_CLEAR_DEPTH : 0) | ((what & 2) ? D3D11_CLEAR_STENCIL : 0);
        if (dsv)
            ID3D11DeviceContext_ClearDepthStencilView(s_ctx, dsv, f, depth, stencil);
    }
    if (what & 4) {
        FLOAT c[4];
        c[0] = (FLOAT)((argb >> 16) & 0xFF) / 255.0f;
        c[1] = (FLOAT)((argb >>  8) & 0xFF) / 255.0f;
        c[2] = (FLOAT)( argb        & 0xFF) / 255.0f;
        c[3] = (FLOAT)((argb >> 24) & 0xFF) / 255.0f;
        if (s_ctx1 && (rect[0] || rect[1] || rect[2] < t->w || rect[3] < t->h)) {
            D3D11_RECT r;
            r.left = (LONG)(rect[0] * S); r.top = (LONG)(rect[1] * S);
            r.right = (LONG)((rect[0] + rect[2]) * S); r.bottom = (LONG)((rect[1] + rect[3]) * S);
            ID3D11DeviceContext1_ClearView(s_ctx1, (ID3D11View *)t->rtv, c, &r, 1);
        } else {
            ID3D11DeviceContext_ClearRenderTargetView(s_ctx, t->rtv, c);
        }
    }
}

static int swap_ready(void)
{
    HWND hwnd = (HWND)xbox_FramebufferWindowHwnd();
    RECT rc;
    uint32_t w, h;

    if (!hwnd)
        return 0;
    GetClientRect(hwnd, &rc);
    w = (uint32_t)(rc.right - rc.left);
    h = (uint32_t)(rc.bottom - rc.top);
    if (!w || !h)
        return 0;
    if (!s_swap) {
        IDXGIDevice *dxdev = NULL;
        IDXGIAdapter *ad = NULL;
        IDXGIFactory2 *fac = NULL;
        DXGI_SWAP_CHAIN_DESC1 sd;
        HRESULT hr;
        ID3D11Device_QueryInterface(s_dev, &IID_IDXGIDevice, &dxdev);
        IDXGIDevice_GetAdapter(dxdev, &ad);
        IDXGIAdapter_GetParent(ad, &IID_IDXGIFactory2, &fac);
        memset(&sd, 0, sizeof sd);
        sd.Width = w; sd.Height = h;
        sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 2;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        hr = IDXGIFactory2_CreateSwapChainForHwnd(fac, (IUnknown *)s_dev, hwnd, &sd,
                                                  NULL, NULL, &s_swap);
        IDXGIFactory2_MakeWindowAssociation(fac, hwnd, DXGI_MWA_NO_ALT_ENTER);
        IDXGIFactory2_Release(fac);
        IDXGIAdapter_Release(ad);
        IDXGIDevice_Release(dxdev);
        if (FAILED(hr)) {
            fprintf(stderr, "  [D3D11] no swap chain (0x%08lX)\n", (unsigned long)hr);
            return 0;
        }
        xbox_FramebufferWindowExternal();
        s_back_w = s_back_h = 0;
    }
    if (w != s_back_w || h != s_back_h || !s_back_rtv) {
        ID3D11Texture2D *bb = NULL;
        if (s_back_rtv) {
            ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 0, NULL, NULL);
            ID3D11RenderTargetView_Release(s_back_rtv);
            s_back_rtv = NULL;
        }
        IDXGISwapChain1_ResizeBuffers(s_swap, 0, w, h, DXGI_FORMAT_UNKNOWN, 0);
        IDXGISwapChain1_GetBuffer(s_swap, 0, &IID_ID3D11Texture2D, &bb);
        ID3D11Device_CreateRenderTargetView(s_dev, (ID3D11Resource *)bb, NULL, &s_back_rtv);
        ID3D11Texture2D_Release(bb);
        s_back_w = w; s_back_h = h;
        s_cur = -1;   /* targets were unbound */
    }
    return 1;
}

/* Five pixels of the presented image, read back: all near white is a flash. */
static void framelog_check(const Target *t)
{
    static ID3D11Texture2D *probe;
    static unsigned frames, dumped_white, dumped_plain;
    D3D11_MAPPED_SUBRESOURCE m;
    unsigned white = 0, k, i;
    static const float at[5][2] = { {0.5f, 0.5f}, {0.3f, 0.4f}, {0.7f, 0.4f},
                                    {0.3f, 0.7f}, {0.7f, 0.7f} };

    frames++;
    if (!probe) {
        D3D11_TEXTURE2D_DESC td;
        memset(&td, 0, sizeof td);
        td.Width = 5; td.Height = 1; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &probe)))
            return;
    }
    for (k = 0; k < 5; k++) {
        D3D11_BOX b;
        b.left = (UINT)(at[k][0] * t->w * S); b.right = b.left + 1;
        b.top = (UINT)(at[k][1] * t->h * S);  b.bottom = b.top + 1;
        b.front = 0; b.back = 1;
        ID3D11DeviceContext_CopySubresourceRegion(s_ctx, (ID3D11Resource *)probe, 0, k, 0, 0,
                                                  (ID3D11Resource *)t->tex, 0, &b);
    }
    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)probe, 0,
                                          D3D11_MAP_READ, 0, &m))) {
        const uint8_t *px = (const uint8_t *)m.pData;
        for (k = 0; k < 5; k++)
            if (px[k * 4] > 0xE0 && px[k * 4 + 1] > 0xE0 && px[k * 4 + 2] > 0xE0)
                white++;
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)probe, 0);
    }
    if ((white >= 4 && dumped_white < 6) || (frames % 1500 == 0 && dumped_plain < 3)) {
        if (white >= 4) dumped_white++; else dumped_plain++;
        fprintf(stderr, "[FRAMELOG] frame %u %s, %u lines\n", frames,
                white >= 4 ? "WHITE" : "normal", s_framelog_n);
        for (i = 0; i < s_framelog_n; i++)
            fprintf(stderr, "[FRAMELOG]   %s\n", s_framelog[i]);
        fflush(stderr);
    }
    s_framelog_n = 0;
}

void nvd3d_present(uint32_t base, const uint32_t rect[4],
                   uint32_t disp_w, uint32_t disp_h)
{
    int i, src = -1, prev = s_cur;

    flush();
    for (i = 0; i < s_rt_count; i++)
        if (s_rt[i].base == base) {
            if (src < 0 || s_rt[i].used > s_rt[src].used)
                src = i;
        }
    if (nvd3d_trace()) {
        static uint32_t last = 0xFFFFFFFFu;
        static int said;
        if (base != last && said++ < 200)
            fprintf(stderr, "  [D3D11] present 0x%08X (rect %u,%u %ux%u)%s\n", base,
                    rect[0], rect[1], rect[2], rect[3], src < 0 ? " -- no such target" : "");
        last = base;
    }
    if (src < 0 || !swap_ready())
        return;
    {
        const Target *t = &s_rt[src];
        D3D11_MAPPED_SUBRESOURCE m;
        D3D11_VIEWPORT vp;
        ID3D11ShaderResourceView *none = NULL;
        float cb[4];
        cb[0] = (float)rect[0] / (float)t->w;
        cb[1] = (float)rect[1] / (float)t->h;
        cb[2] = (float)rect[2] / (float)t->w;
        cb[3] = (float)rect[3] / (float)t->h;
        if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb_present, 0,
                                              D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            memcpy(m.pData, cb, sizeof cb);
            ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb_present, 0);
        }
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &none);
        ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 1, &s_back_rtv, NULL);
        /* The title's picture keeps its aspect (disp_w:disp_h, 4:3 here);
         * whatever the window adds becomes black bars. Under
         * RECOMP_WIDESCREEN the picture is anamorphic instead -- the
         * title was told 16:9 and rendered a squeezed 4:3 buffer, so
         * it is presented at 16:9 and comes back out undistorted. */
        {
            static const FLOAT black[4] = { 0, 0, 0, 1 };
            const float ws = xbox_WidescreenAspect();
            const float want = ws > 0.0f ? ws
                : ((disp_w && disp_h) ? (float)disp_w / (float)disp_h : 4.0f / 3.0f);
            float w = (float)s_back_w, h = (float)s_back_h;
            if (w / h > want) w = h * want; else h = w / want;
            ID3D11DeviceContext_ClearRenderTargetView(s_ctx, s_back_rtv, black);
            vp.TopLeftX = ((float)s_back_w - w) * 0.5f;
            vp.TopLeftY = ((float)s_back_h - h) * 0.5f;
            vp.Width = w; vp.Height = h;
        }
        vp.MinDepth = 0; vp.MaxDepth = 1;
        ID3D11DeviceContext_RSSetViewports(s_ctx, 1, &vp);
        {
            D3D11_RECT sc = { 0, 0, (LONG)s_back_w, (LONG)s_back_h };
            ID3D11DeviceContext_RSSetScissorRects(s_ctx, 1, &sc);
        }
        ID3D11DeviceContext_RSSetState(s_ctx, s_rs);
        ID3D11DeviceContext_OMSetBlendState(s_ctx, s_blend_off, NULL, 0xFFFFFFFFu);
        ID3D11DeviceContext_OMSetDepthStencilState(s_ctx, s_ds_off, 0);
        ID3D11DeviceContext_IASetInputLayout(s_ctx, NULL);
        ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ID3D11DeviceContext_VSSetShader(s_ctx, s_vs_present, NULL, 0);
        ID3D11DeviceContext_PSSetShader(s_ctx, s_ps_present, NULL, 0);
        ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_cb_present);
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &t->srv);
        ID3D11DeviceContext_PSSetSamplers(s_ctx, 0, 1, &s_lin);
        ID3D11DeviceContext_Draw(s_ctx, 3, 0);
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &none);
        IDXGISwapChain1_Present(s_swap, 0, 0);
        if (nvd3d_trace())
            framelog_check(t);
    }
    /* Put the title's target back. */
    s_cur = -1;
    if (prev >= 0) {
        uint32_t b = s_rt[prev].base, w = s_rt[prev].w, h = s_rt[prev].h;
        nvd3d_target(b, w, h);
    }
}

void nvd3d_readback(uint32_t base, uint8_t *dst, uint32_t pitch, uint32_t rows)
{
    int i, src = -1;
    D3D11_TEXTURE2D_DESC td;
    ID3D11Texture2D *stage = NULL;
    D3D11_MAPPED_SUBRESOURCE m;
    uint32_t y;

    flush();
    for (i = 0; i < s_rt_count; i++)
        if (s_rt[i].base == base)
            src = i;
    if (src < 0)
        return;
    ID3D11Texture2D_GetDesc(s_rt[src].tex, &td);
    td.Usage = D3D11_USAGE_STAGING;
    td.BindFlags = 0;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &stage)))
        return;
    ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)stage,
                                     (ID3D11Resource *)s_rt[src].tex);
    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)stage, 0,
                                          D3D11_MAP_READ, 0, &m))) {
        /* Back to the title's size: one sample in S each way. */
        const uint32_t gw = td.Width / S, gh = td.Height / S;
        const uint32_t cols = gw < pitch / 4 ? gw : pitch / 4;
        if (nvd3d_trace())
            fprintf(stderr, "  [D3D11] readback 0x%08X %ux%u (scale %u) rowpitch %u -> pitch %u rows %u\n",
                    base, td.Width, td.Height, S, m.RowPitch, pitch, rows);
        for (y = 0; y < rows && y < gh; y++) {
            const uint32_t *srow = (const uint32_t *)((uint8_t *)m.pData + (size_t)y * S * m.RowPitch);
            uint32_t *drow = (uint32_t *)(dst + (size_t)y * pitch);
            uint32_t x;
            if (S == 1) {
                memcpy(drow, srow, (size_t)cols * 4);
            } else {
                for (x = 0; x < cols; x++)
                    drow[x] = srow[x * S];
            }
        }
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)stage, 0);
    }
    ID3D11Texture2D_Release(stage);
}

#else   /* !_WIN32 */

#include "nv2a_d3d11.h"
int  nvd3d_active(void) { return 0; }
uint32_t nvd3d_target(uint32_t b, uint32_t w, uint32_t h) { (void)b; (void)w; (void)h; return 0; }
int  nvd3d_texture(const uint32_t key[6], uint64_t hash, uint32_t w, uint32_t h,
                   uint32_t mb, uint32_t mn, void (*d)(uint32_t *, uint32_t, uint32_t))
{ (void)key; (void)hash; (void)w; (void)h; (void)mb; (void)mn; (void)d; return -1; }
void nvd3d_triangle(const NvD3DVertex v[3], const NvD3DState *st) { (void)v; (void)st; }
void nvd3d_clear(uint32_t w, uint32_t a, float d, uint8_t s, const uint32_t r[4])
{ (void)w; (void)a; (void)d; (void)s; (void)r; }
void nvd3d_present(uint32_t b, const uint32_t r[4], uint32_t w, uint32_t h)
{ (void)b; (void)r; (void)w; (void)h; }
void nvd3d_readback(uint32_t b, uint8_t *d, uint32_t p, uint32_t r)
{ (void)b; (void)d; (void)p; (void)r; }

#endif
