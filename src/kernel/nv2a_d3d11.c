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
#include "nv2a_dlss.h"

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

static int  tex_mips_on(void);   /* mipmaps and anisotropy: see their definitions */
static UINT tex_aniso(void);
extern void xbox_FramebufferWindowScale(uint32_t scale);
static ID3D11Device         *s_dev;
static ID3D11DeviceContext  *s_ctx;
static ID3D11DeviceContext1 *s_ctx1;
static ID3D11VertexShader   *s_vs, *s_vs_present;
static ID3D11PixelShader    *s_ps, *s_ps_present, *s_ps_ui, *s_ps_depth, *s_ps_mv, *s_ps_mvview,
                            *s_ps_copy;
static ID3D11DepthStencilState *s_ds_mv;          /* LESS_EQUAL, no write */
static ID3D11Buffer         *s_cb_prev;           /* b4: the previous constants */
static uint32_t              s_mv_batches;
static ID3D11BlendState     *s_blend_ui_comp;   /* C + dst * T */
static ID3D11Buffer         *s_cb_ui;
static int                   s_bound_ui;        /* the layer is what is bound */
static uint32_t              s_ui_batches, s_ui_fallback, s_ui_composites;
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
    /* The interface layer (NvD3DState.layer): colour C and the scene's
     * transmittance T, so that the target seen by anyone is scene * T + C.
     * ui_dirty: something was drawn into it since the last composite. */
    ID3D11Texture2D        *ui_tex;
    ID3D11RenderTargetView *ui_rtv;
    ID3D11ShaderResourceView *ui_srv;
    int                     ui_dirty;
    /* DLSS step 3: where each pixel of the 3D was in the previous frame,
     * in uv (R16G16), written by the motion pass. */
    ID3D11Texture2D        *mv_tex;
    ID3D11RenderTargetView *mv_rtv;
    ID3D11ShaderResourceView *mv_srv;
} Target;
static Target   s_rt[MAX_TARGETS];
static int      s_rt_count;
static int      s_cur = -1;              /* current target index */
static uint32_t s_use_clock;

typedef struct {
    uint32_t w, h;
    ID3D11Texture2D        *tex;
    ID3D11DepthStencilView *dsv;
    ID3D11ShaderResourceView *srv;       /* the depth, readable (DLSS step 2) */
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
    int      mipped;                      /* carries a generated mip chain */
    int      rt;                          /* >= 0: really render target #rt */
    uint32_t seen;                        /* present count when last bound */
    uint64_t pack_hash;                   /* the pack image shown whole, 0 none:
                                           * what a face pasted in is pasted onto */
    ID3D11Texture2D          *tex;
    ID3D11ShaderResourceView *srv;
} Tex;
static Tex  s_tex[MAX_TEXTURES];
static int  s_tex_count;
static uint32_t s_present_count;        /* presents so far, for the logs */

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
/* vextra.zw: the frame's jitter in NDC (DLSS step 2), 0 unless RECOMP_JITTER. */
"cbuffer DrawV : register(b0) { float4 vtex; float4 vatest; float4 vextra; };\n"
"VSOut vs_main(VSIn i) {\n"
"  VSOut o; o.pos = i.pos; o.uv = i.uv; o.col = i.col;\n"
"  o.pos.xy += vextra.zw * i.pos.w;\n"
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
/* The resolve. foot.xy is how many source texels one output pixel covers on
 * each axis, foot.zw one source texel in uv. At or under one texel a pixel it
 * is the single bilinear tap this always was, to the bit. Over it -- a target
 * drawn larger than the window, which is what RECOMP_D3D11_SCALE is for -- a
 * single tap reads four texels and drops the rest, so the supersampling that
 * was paid for never reached the screen. The footprint is averaged instead:
 * n taps per axis, n = ceil(foot), each bilinear, so taps sit at most a texel
 * apart and every texel under the footprint counts. In linear light, because
 * averaging gamma-encoded values darkens every edge it resolves. */
"cbuffer Present : register(b0) { float4 rect; float4 foot; };\n"
"Texture2D src : register(t0);\n"
"SamplerState lin : register(s0);\n"
"float4 ps_present(POut i) : SV_Target {\n"
"  float2 uv = rect.xy + i.uv * rect.zw;\n"
"  int2 n = int2(clamp(ceil(foot.xy - 0.001), 1.0, 8.0));\n"
"  if (n.x == 1 && n.y == 1)\n"
"    return float4(src.SampleLevel(lin, uv, 0).rgb, 1.0);\n"
"  float3 acc = 0;\n"
"  [loop] for (int y = 0; y < n.y; y++)\n"
"    [loop] for (int x = 0; x < n.x; x++) {\n"
"      float2 o = ((float2(x, y) + 0.5) / float2(n) - 0.5) * foot.xy * foot.zw;\n"
"      acc += pow(max(src.SampleLevel(lin, uv + o, 0).rgb, 0.0), 2.2);\n"
"    }\n"
"  return float4(pow(acc / (float)(n.x * n.y), 1.0 / 2.2), 1.0);\n"
"}\n"
/* The interface layer onto its target: (C, T) out, blended as C + dst * T
 * (see ui_composite). uik.x 1: pixel for pixel. Below 1 (RECOMP_UI_43): the
 * layer squeezed that much across the middle, so that a picture presented
 * stretched to 16:9 shows the interface at its own 4:3 proportions; outside
 * it, nothing (C 0, T 1). Bilinear on (C, T), which are premultiplied. */
"cbuffer Ui : register(b0) { float4 uik; };\n"
"Texture2D uil : register(t0);\n"
/* DLSS step 4: the upscaler's output back into the target, pixel for pixel. */
"float4 ps_copy(POut i) : SV_Target { return uil.Load(int3(i.pos.xy, 0)); }\n"
/* RECOMP_DEPTH_VIEW, F7: the depth instead of the picture. D24 crowds a
 * perspective scene near 1, so it is spread on a log scale: near is white. */
"Texture2D<float> dep : register(t0);\n"
/* DLSS step 3: the motion pass. Its vertex shader is a translated program's
 * motion variant (VSOutM), which gives where the vertex was (pc) and where it
 * is (qc), both unjittered, in clip space. Written in uv, pointing from the
 * current pixel to the previous one. */
"struct MIn { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 col : COLOR0;\n"
"             float clip : SV_ClipDistance0; float4 col1 : COLOR1;\n"
"             float4 uv1 : TEXCOORD1; float4 uv2 : TEXCOORD2; float4 uv3 : TEXCOORD3;\n"
"             float4 pc : TEXCOORD4; float4 qc : TEXCOORD5; };\n"
"float2 ps_mv(MIn i) : SV_Target {\n"
/* Where it was lay before the near plane (window z < 0, what the clip
 * distance cuts away now), or behind the eye: it was not drawn then, so it has
 * no previous position -- no vector, rather than one divided by almost
 * nothing (point 154: thousands of pixels on geometry near the camera). */
"  if (i.pc.w <= 1e-4 || i.qc.w <= 1e-4 || i.pc.z < 0.0) return float2(0.0, 0.0);\n"
"  float2 p = i.pc.xy / i.pc.w, q = i.qc.xy / i.qc.w;\n"
"  return (p - q) * float2(0.5, -0.5);\n"
"}\n"
/* RECOMP_MOTION, F6: the motion vectors instead of the picture, grey where
 * nothing moves, x in red, y in green, 1/20 of the screen to full colour. */
"Texture2D<float2> mvt : register(t0);\n"
"float4 ps_mvview(POut i) : SV_Target {\n"
"  float2 uv = rect.xy + i.uv * rect.zw;\n"
"  float2 m = mvt.SampleLevel(lin, uv, 0);\n"
"  return float4(saturate(0.5 + m * 10.0), 0.5, 1.0);\n"
"}\n"
"float4 ps_depth(POut i) : SV_Target {\n"
"  float2 uv = rect.xy + i.uv * rect.zw;\n"
"  float d = dep.SampleLevel(lin, uv, 0);\n"
"  float v = saturate(-log2(max(1.0 - d, 1e-7)) / 23.0);\n"
"  return float4(1.0 - v, 1.0 - v, 1.0 - v, 1.0);\n"
"}\n"
"float4 ps_ui(POut i) : SV_Target {\n"
"  if (uik.x >= 1.0) return uil.Load(int3(i.pos.xy, 0));\n"
"  float2 uv = i.pos.xy / uik.yz;\n"
"  uv.x = 0.5 + (uv.x - 0.5) / uik.x;\n"
"  if (uv.x < 0.0 || uv.x > 1.0) return float4(0, 0, 0, 1);\n"
"  return uil.SampleLevel(lin, uv, 0);\n"
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
    nvdlss_preinit();                   /* RECOMP_DLSS: Streamline before the device */
    hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, flags, levels, 2,
                           D3D11_SDK_VERSION, &s_dev, NULL, &s_ctx);
    if (FAILED(hr)) {
        fprintf(stderr, "  [D3D11] no device (0x%08lX), staying on the CPU\n", (unsigned long)hr);
        return 0;
    }
    nvdlss_set_device(s_dev);
    if (FAILED(ID3D11DeviceContext_QueryInterface(s_ctx, &IID_ID3D11DeviceContext1, (void **)&s_ctx1)))
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
        ID3DBlob *psu = compile("ps_ui", "ps_5_0");
        ID3DBlob *psd = compile("ps_depth", "ps_5_0");
        ID3DBlob *psm = compile("ps_mv", "ps_5_0");
        ID3DBlob *psv = compile("ps_mvview", "ps_5_0");
        ID3DBlob *psc = compile("ps_copy", "ps_5_0");
        if (psc) {
            ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(psc),
                                           ID3D10Blob_GetBufferSize(psc), NULL, &s_ps_copy);
            ID3D10Blob_Release(psc);
        }
        if (psm) {
            ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(psm),
                                           ID3D10Blob_GetBufferSize(psm), NULL, &s_ps_mv);
            ID3D10Blob_Release(psm);
        }
        if (psv) {
            ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(psv),
                                           ID3D10Blob_GetBufferSize(psv), NULL, &s_ps_mvview);
            ID3D10Blob_Release(psv);
        }
        if (psu) {
            ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(psu),
                                           ID3D10Blob_GetBufferSize(psu), NULL, &s_ps_ui);
            ID3D10Blob_Release(psu);
        }
        if (psd) {
            ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(psd),
                                           ID3D10Blob_GetBufferSize(psd), NULL, &s_ps_depth);
            ID3D10Blob_Release(psd);
        }
    }
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
        ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_cb_ui);
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
            const UINT aniso = tex_aniso();
            memset(&sd, 0, sizeof sd);
            /* Anisotropic along the axis a grazing floor is squashed on; the
             * mip level comes from the other one. See tex_mips_on. */
            sd.Filter = aniso > 1 ? D3D11_FILTER_ANISOTROPIC
                                  : D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            sd.MaxAnisotropy = aniso;
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
        bd.RenderTarget[0].BlendEnable = TRUE;
        bd.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;          /* C */
        bd.RenderTarget[0].DestBlend = D3D11_BLEND_SRC_ALPHA;   /* + dst * T */
        bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;    /* the target's alpha */
        bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;    /* is left as it is */
        ID3D11Device_CreateBlendState(s_dev, &bd, &s_blend_ui_comp);
        memset(&dd, 0, sizeof dd);
        dd.DepthFunc = D3D11_COMPARISON_ALWAYS;
        ID3D11Device_CreateDepthStencilState(s_dev, &dd, &s_ds_off);
        dd.DepthEnable = TRUE;                          /* the motion pass */
        dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        dd.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
        ID3D11Device_CreateDepthStencilState(s_dev, &dd, &s_ds_mv);
    }
    {
        D3D11_BUFFER_DESC bd;
        memset(&bd, 0, sizeof bd);
        bd.ByteWidth = 192 * 16;
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_cb_prev);
    }
    fprintf(stderr, "  [D3D11] device ready: triangles are filled on the GPU, scale %u\n", S);
    {
        const char *pf = getenv("RECOMP_PRESENT_FILTER");
        UINT an = tex_aniso();
        fprintf(stderr, "  [D3D11] textures: %s, anisotropy %s; resolve: %s\n",
                tex_mips_on() ? "mip chains generated" : "one level (RECOMP_TEX_MIPS=0)",
                an > 1 ? (an >= 16 ? "16x" : an >= 8 ? "8x" : an >= 4 ? "4x" : "2x")
                       : "off",
                (pf && atoi(pf) == 0) ? "single tap (RECOMP_PRESENT_FILTER=0)"
                                      : "footprint average when the target is larger");
    }
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

/* The interface layer holds (C, T) and the target is read as scene * T + C.
 * A blend dst' = src * Fs + dst * Fd keeps that exact when Fd is the same
 * scalar for every channel and Fs does not read the destination: then
 * C' = src * Fs + C * Fd and T' = T * Fd. This gives the factor T is
 * multiplied by (as a D3D11 alpha factor), or 0 when the blend cannot be
 * layered -- a destination read, or a per-channel Fd -- and the batch is
 * then drawn into the target itself, as before. Blending off is Fd = 0. */
static D3D11_BLEND ui_t_factor(const NvD3DState *st)
{
    if (!st->blend_enable)
        return D3D11_BLEND_ZERO;
    switch (st->blend_src) {
    case 0x0304: case 0x0305: case 0x0306: case 0x0307: case 0x0308:
        return (D3D11_BLEND)0;                  /* Fs reads the destination */
    default: break;
    }
    switch (st->blend_dst) {
    case 0x0000: return D3D11_BLEND_ZERO;
    case 0x0001: return D3D11_BLEND_ONE;
    case 0x0302: return D3D11_BLEND_SRC_ALPHA;
    case 0x0303: return D3D11_BLEND_INV_SRC_ALPHA;
    case 0x8003: return D3D11_BLEND_BLEND_FACTOR;      /* constant alpha: bf is */
    case 0x8004: return D3D11_BLEND_INV_BLEND_FACTOR;  /* that alpha everywhere */
    default:     return (D3D11_BLEND)0;
    }
}

static ID3D11BlendState *blend_state(const NvD3DState *st)
{
    uint32_t k[7];
    int i;
    k[0] = st->blend_enable; k[1] = st->blend_src; k[2] = st->blend_dst;
    k[3] = st->blend_eq; k[4] = st->color_mask; k[5] = st->layer != 0; k[6] = 0;
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
        if (st->layer) {
            /* Colour as the title asked, in C; alpha carries T. A batch that
             * writes no colour (depth or stencil only) leaves both alone. */
            t->BlendEnable = TRUE;
            if (!st->blend_enable) {
                t->SrcBlend = D3D11_BLEND_ONE;
                t->DestBlend = D3D11_BLEND_ZERO;
                t->BlendOp = D3D11_BLEND_OP_ADD;
            }
            t->SrcBlendAlpha = D3D11_BLEND_ZERO;
            t->DestBlendAlpha = ui_t_factor(st);
            t->BlendOpAlpha = D3D11_BLEND_OP_ADD;
            mask = (mask & D3D11_COLOR_WRITE_ENABLE_ALL & ~D3D11_COLOR_WRITE_ENABLE_ALPHA)
                 ? (UINT8)(mask | D3D11_COLOR_WRITE_ENABLE_ALPHA) : (UINT8)0;
        }
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
        /* Typeless, so that the same depth can also be read by a shader:
         * an upscaler wants it (DLSS step 2), and RECOMP_DEPTH_VIEW shows it. */
        td.Format = DXGI_FORMAT_R24G8_TYPELESS;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &d->tex)))
            return NULL;
        {
            D3D11_DEPTH_STENCIL_VIEW_DESC dv;
            D3D11_SHADER_RESOURCE_VIEW_DESC sv;
            memset(&dv, 0, sizeof dv);
            dv.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
            dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
            ID3D11Device_CreateDepthStencilView(s_dev, (ID3D11Resource *)d->tex, &dv, &d->dsv);
            memset(&sv, 0, sizeof sv);
            sv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
            sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            sv.Texture2D.MipLevels = 1;
            ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)d->tex, &sv, &d->srv);
        }
        d->w = w; d->h = h;
        ID3D11DeviceContext_ClearDepthStencilView(s_ctx, d->dsv,
            D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
        s_depth_count++;
        return d->dsv;
    }
}

static void target_release(Target *t)
{
    if (t->mv_srv)   ID3D11ShaderResourceView_Release(t->mv_srv);
    if (t->mv_rtv)   ID3D11RenderTargetView_Release(t->mv_rtv);
    if (t->mv_tex)   ID3D11Texture2D_Release(t->mv_tex);
    if (t->ui_srv)   ID3D11ShaderResourceView_Release(t->ui_srv);
    if (t->ui_rtv)   ID3D11RenderTargetView_Release(t->ui_rtv);
    if (t->ui_tex)   ID3D11Texture2D_Release(t->ui_tex);
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
        s_bound_ui = 0;
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
/* ── Mipmaps and anisotropy ──────────────────────────────────────────── */
/*
 * The title asks for LINEAR_MIPMAP_LINEAR minification on 617460 batches a
 * minute and ships ONE mip level everywhere (point 85, measured): so the chain
 * it asks for has to be made here, and making it is honouring the request,
 * not embellishing it. GenerateMips after every upload; the samplers were
 * already MIN_MAG_MIP_LINEAR with an open MaxLOD and the shader already uses
 * Sample, which picks the level from screen derivatives -- they simply never
 * had a level to pick.
 *
 * It is also what makes an enlarged pack texture behave: Sample picks the
 * level from the REAL size, so a 2048 standing in for a declared 512 lands on
 * its 512 level whenever the surface is at 1:1, and shows its extra detail
 * only up close. Without a chain the same image shimmers at distance.
 *
 * RECOMP_TEX_MIPS=0 and RECOMP_TEX_ANISO=0 give back the single level and
 * the plain trilinear sampler, to compare two passes over one scene.
 */
static int tex_mips_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RECOMP_TEX_MIPS");
        on = !(e && atoi(e) == 0);
    }
    return on;
}

/* Anisotropy: 16 unless told otherwise; 0 or 1 means off. */
static UINT tex_aniso(void)
{
    const char *e = getenv("RECOMP_TEX_ANISO");
    int v = e ? atoi(e) : 16;
    return (UINT)(v < 1 ? 1 : v > 16 ? 16 : v);
}

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

/* A fighter taking hits: the title copies its battered face (br01, then
 * br02, from the same .xsh) over the top-right quarter of the atlas in place
 * -- measured on Magic (point 158): the atlas changes fingerprint twice at
 * the same address, the quarter equal to br01 then br02 pixel for pixel, the
 * rest untouched. In the Xbox's swizzled order that quarter is the atlas's
 * second quarter of bytes, laid out exactly as a texture of half the size,
 * and the palette is the atlas's: so its fingerprint is br01's or br02's own.
 * The pack holds the atlas and the faces under those; the battered atlas is
 * built here, the face's pack image over the atlas's. Without the face's,
 * the title's face is enlarged into the quarter, rather than the whole
 * fighter going back to the original. d3d_hash of nv2a_pb_exec.c, the same. */
static uint64_t face_hash(const uint8_t *p, size_t n, uint64_t h)
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

static uint32_t *pack_battered(const uint32_t key[6], uint64_t hash, uint64_t base,
                               uint32_t width, uint32_t height, uint32_t mem_bytes,
                               const uint32_t *decoded, uint32_t *pw, uint32_t *ph)
{
    const uint8_t *mem = (const uint8_t *)(uintptr_t)xbox_GetMemoryOffset();
    uint32_t *atlas, *face = NULL, aw, ah, fw = 0, fh = 0, q, x, y;
    uint64_t fhash;

    if (!base || key[1] != 0x0B || !key[4] || width != height || width < 64
            || mem_bytes != width * height)
        return NULL;
    fhash = face_hash(mem + key[0] + mem_bytes / 4, mem_bytes / 4, 1469598103934665603ull);
    fhash = face_hash(mem + key[4], 1024, fhash);
    atlas = pack_lookup(base, width, height, &aw, &ah);
    if (!atlas)
        return NULL;
    q = aw / 2;
    face = pack_lookup(fhash, width / 2, height / 2, &fw, &fh);
    /* Nearest, from whichever is there: a face at another enlargement than
     * the atlas still lands on its quarter. */
    for (y = 0; y < q; y++)
        for (x = 0; x < q; x++)
            atlas[(size_t)y * aw + q + x] = face
                ? face[(size_t)(y * fh / q) * fw + x * fw / q]
                : decoded[(size_t)(y * height / ah) * width + width / 2 + x * width / aw];
    {
        static int said;
        if (said++ < 64)
            fprintf(stderr, "  [TEXPACK] %016llX : atlas %016llX, face %016llX %s\n",
                    (unsigned long long)hash, (unsigned long long)base,
                    (unsigned long long)fhash, face ? "from the pack" : "of the title, enlarged");
    }
    free(face);
    *pw = aw; *ph = ah;
    return atlas;
}

int nvd3d_texture_seen(uint64_t hash, uint32_t within)
{
    int i;
    for (i = 0; i < s_tex_count; i++)
        if (s_tex[i].hash == hash && s_present_count - s_tex[i].seen <= within)
            return 1;
    return 0;
}

int nvd3d_texture(const uint32_t key[6], uint64_t hash, uint32_t width, uint32_t height,
                  uint32_t mem_base, uint32_t mem_bytes,
                  void (*decode)(uint32_t *dst, uint32_t width, uint32_t height))
{
    int i, slot = -1;
    Tex *t;
    uint32_t *pack = NULL, *src;
    uint32_t sw, sh;
    uint64_t base = 0;                    /* the pack image this binding showed */

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
        s_tex[slot].seen = s_present_count;
        return slot;
    }
    if (slot >= 0)
        base = s_tex[slot].pack_hash;
    /* RECOMP_TEX_NEWLOG=1: each fingerprint the first time it is uploaded,
     * with the present count, in the same log as the pad's [PAD] lines -- to
     * find which textures only one screen uses (the fighter select, for the
     * random pick on Y). */
    {
        static int on = -1;
        static uint64_t seen[8192];
        static uint32_t nseen;
        uint32_t k;
        if (on < 0)
            on = getenv("RECOMP_TEX_NEWLOG") != NULL;
        if (on) {
            for (k = 0; k < nseen && seen[k] != hash; k++)
                ;
            if (k == nseen && nseen < 8192) {
                seen[nseen++] = hash;
                fprintf(stderr, "  [TEXNEW] %016llX %ux%u fmt %02X at 0x%08X, present %u\n",
                        (unsigned long long)hash, width, height, key[1], mem_base,
                        s_present_count);
            }
        }
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
    if (pack)
        base = hash;
    else if ((pack = pack_battered(key, hash, base, width, height, mem_bytes,
                                   s_decode_buf, &sw, &sh)) == NULL)
        base = 0;
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

    if (t->tex && t->w == sw && t->h == sh && t->mipped == tex_mips_on()) {
        flush();   /* the batch may sample the old contents */
        ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)t->tex, 0, NULL,
                                              src, sw * 4, 0);
        if (t->mipped)
            ID3D11DeviceContext_GenerateMips(s_ctx, t->srv);
    } else {
        D3D11_TEXTURE2D_DESC td;
        D3D11_SUBRESOURCE_DATA init;
        flush();
        if (t->srv) ID3D11ShaderResourceView_Release(t->srv);
        if (t->tex) ID3D11Texture2D_Release(t->tex);
        t->srv = NULL; t->tex = NULL;
        const int mips = tex_mips_on();
        memset(&td, 0, sizeof td);
        td.Width = sw; td.Height = sh; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        /* GenerateMips wants the texture bindable as a target too; it costs
         * nothing but the flag. The chain cannot come in as initial data,
         * so level 0 goes in afterwards. */
        td.MipLevels = mips ? 0 : 1;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE | (mips ? D3D11_BIND_RENDER_TARGET : 0);
        td.MiscFlags = mips ? D3D11_RESOURCE_MISC_GENERATE_MIPS : 0;
        init.pSysMem = src;
        init.SysMemPitch = sw * 4;
        init.SysMemSlicePitch = 0;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, mips ? NULL : &init,
                                                &t->tex))) {
            free(pack);
            return -1;
        }
        ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)t->tex, NULL, &t->srv);
        if (mips) {
            ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)t->tex, 0, NULL,
                                                  src, sw * 4, 0);
            ID3D11DeviceContext_GenerateMips(s_ctx, t->srv);
        }
        t->mipped = mips;
    }
    free(pack);
    memcpy(t->key, key, sizeof t->key);
    t->hash = hash;
    t->pack_hash = base;
    t->seen = s_present_count;
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

/* Register-combiner pixel shaders (point 140), one per configuration the
 * executor translated, and their constants in register b3: the factors of
 * each stage, then the size and kind of the textures on stages 1-3. */
#define MAX_PS_PROGRAMS 256
typedef struct { uint64_t hash; ID3D11PixelShader *ps; } PsProgram;
static PsProgram     s_psp[MAX_PS_PROGRAMS];
static int           s_npsp;
static ID3D11Buffer *s_cb_comb;
#define COMB_CB_FLOAT4S (8 + 8 + 4)

int nvd3d_ps_program(uint64_t hash, const char *hlsl, size_t len)
{
    int i;
    ID3DBlob *code = NULL, *err = NULL;
    PsProgram *p;

    for (i = 0; i < s_npsp; i++)
        if (s_psp[i].hash == hash)
            return s_psp[i].ps ? i : -1;
    if (!hlsl)
        return -2;                      /* unknown: the caller translates */
    if (s_npsp >= MAX_PS_PROGRAMS || !s_dev)
        return -1;
    p = &s_psp[s_npsp++];
    memset(p, 0, sizeof *p);
    p->hash = hash;
    if (!len)
        return -1;                      /* refused by the translation: cached as such */
    if (FAILED(D3DCompile(hlsl, len, "nv2a_comb", NULL, NULL, "ps_comb", "ps_5_0",
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err))) {
        static int said;
        if (said++ < 8)
            fprintf(stderr, "  [COMB] configuration %016llX does not compile, drawn texel x colour: %s\n",
                    (unsigned long long)hash,
                    err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "?");
        if (err) ID3D10Blob_Release(err);
        return -1;                      /* the failure is cached: no retry */
    }
    if (err) ID3D10Blob_Release(err);
    if (FAILED(ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(code),
                                              ID3D10Blob_GetBufferSize(code), NULL, &p->ps))) {
        ID3D10Blob_Release(code);
        return -1;
    }
    ID3D10Blob_Release(code);
    return (int)(p - s_psp);
}

/* The combiners' own constants and textures, for a batch that has a
 * translated configuration. Stage 0 keeps register t0/s0 and the Draw
 * constants, exactly as the fixed shader reads them. */
static void bind_combiners(const NvD3DState *st)
{
    D3D11_MAPPED_SUBRESOURCE m;
    float cb[COMB_CB_FLOAT4S * 4];
    int k;

    if (!s_cb_comb) {
        D3D11_BUFFER_DESC bd;
        memset(&bd, 0, sizeof bd);
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bd.ByteWidth = sizeof cb;
        if (FAILED(ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_cb_comb)))
            return;
    }
    memset(cb, 0, sizeof cb);
    for (k = 0; k < 8; k++) {
        const uint32_t f0 = st->factor0[k], f1 = st->factor1[k];
        cb[k * 4 + 0] = (float)((f0 >> 16) & 0xFF) / 255.0f;
        cb[k * 4 + 1] = (float)((f0 >>  8) & 0xFF) / 255.0f;
        cb[k * 4 + 2] = (float)( f0        & 0xFF) / 255.0f;
        cb[k * 4 + 3] = (float)((f0 >> 24) & 0xFF) / 255.0f;
        cb[32 + k * 4 + 0] = (float)((f1 >> 16) & 0xFF) / 255.0f;
        cb[32 + k * 4 + 1] = (float)((f1 >>  8) & 0xFF) / 255.0f;
        cb[32 + k * 4 + 2] = (float)( f1        & 0xFF) / 255.0f;
        cb[32 + k * 4 + 3] = (float)((f1 >> 24) & 0xFF) / 255.0f;
    }
    for (k = 0; k < 3; k++) {
        cb[64 + k * 4 + 0] = (float)(st->stage_wh[k][0] ? st->stage_wh[k][0] : 1);
        cb[64 + k * 4 + 1] = (float)(st->stage_wh[k][1] ? st->stage_wh[k][1] : 1);
        cb[64 + k * 4 + 2] = st->stage_linear[k] ? 1.0f : 0.0f;
        cb[64 + k * 4 + 3] = st->stage_tex[k] >= 0 ? 1.0f : 0.0f;
    }
    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb_comb, 0,
                                          D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        memcpy(m.pData, cb, sizeof cb);
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb_comb, 0);
    }
    for (k = 0; k < 3; k++) {
        ID3D11ShaderResourceView *srv = st->stage_tex[k] >= 0 ? texture_view(st->stage_tex[k]) : NULL;
        const uint32_t au = st->stage_addr[k] & 0xF, av = (st->stage_addr[k] >> 4) & 0xF;
        const int s = (au == 1 ? 1 : 0) | (av == 1 ? 2 : 0);
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 1 + k, 1, &srv);
        ID3D11DeviceContext_PSSetSamplers(s_ctx, 1 + k, 1, &s_smp[s]);
    }
    ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 3, 1, &s_cb_comb);
}

/* ── The interface layer (DLSS step 1) ───────────────────────────────────
 *
 * Batches the executor marks as interface (NvD3DState.layer) go into a layer
 * of their target's size, with the target's own depth buffer -- so depth and
 * stencil tests come out exactly as before -- holding (C, T): what they drew,
 * and how much of the scene still shows through. Anything that reads the
 * target composites the layer first (ui_composite): the present, a readback,
 * the target sampled as a texture, a colour clear. What it reads is then
 * scene * T + C, which is the sequential result for every blend ui_t_factor
 * accepts, since the interface comes after all the 3D (point 149). Half-float,
 * so C and T do not add a rounding of their own to the title's 8 bits. */
static int ui_layer_ready(Target *t)
{
    D3D11_TEXTURE2D_DESC td;
    static const FLOAT clear[4] = { 0, 0, 0, 1 };

    if (t->ui_rtv)
        return 1;
    memset(&td, 0, sizeof td);
    td.Width = t->w * S; td.Height = t->h * S; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &t->ui_tex)))
        return 0;
    ID3D11Device_CreateRenderTargetView(s_dev, (ID3D11Resource *)t->ui_tex, NULL, &t->ui_rtv);
    ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)t->ui_tex, NULL, &t->ui_srv);
    if (!t->ui_rtv || !t->ui_srv)
        return 0;
    ID3D11DeviceContext_ClearRenderTargetView(s_ctx, t->ui_rtv, clear);
    fprintf(stderr, "  [D3D11] interface layer for 0x%08X (%ux%u)\n", t->base, td.Width, td.Height);
    return 1;
}

/* ── DLSS step 2: the jitter ─────────────────────────────────────────────
 *
 * RECOMP_JITTER=1: the 3D of the shown surface moves by a sub-pixel offset
 * that changes every frame, the Halton (2, 3) sequence over 16 frames, as an
 * upscaler wants. Never the interface (layer 1): it is drawn after the
 * upscale and must stay still. In pixels of the target as drawn (w * S),
 * given to the vertex shaders in NDC. RECOMP_JITTER_SCALE=<n> multiplies it,
 * so that it can be seen with the eye: the 3D shakes, the interface does not. */
/* The scene's depth, kept for the upscaler. Measured in a fight (point 152):
 * at the present the depth buffer is entirely cleared -- the title clears it
 * after the 3D, before the interface. So it is copied when a depth clear
 * comes after 3D was drawn on the shown surface, or at the present if none
 * did. */
static ID3D11Texture2D          *s_sdepth;
static ID3D11ShaderResourceView *s_sdepth_srv;
static uint32_t                  s_sdepth_w, s_sdepth_h;
static int                       s_scene_drawn;     /* 3D on the shown surface since */
static uint32_t                  s_sdepth_at_clear, s_sdepth_at_present, s_sdepth_this_frame;
static uint32_t                  s_sdepth_hist[4];  /* frames by copies: 0, 1, 2, more */

static void scene_depth_capture(const Target *t)
{
    int k;
    for (k = 0; k < s_depth_count; k++)
        if (s_depth[k].w == t->w * S && s_depth[k].h == t->h * S)
            break;
    if (k == s_depth_count)
        return;
    if (!s_sdepth || s_sdepth_w != s_depth[k].w || s_sdepth_h != s_depth[k].h) {
        D3D11_TEXTURE2D_DESC td;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv;
        if (s_sdepth_srv) ID3D11ShaderResourceView_Release(s_sdepth_srv);
        if (s_sdepth) ID3D11Texture2D_Release(s_sdepth);
        s_sdepth = NULL; s_sdepth_srv = NULL;
        ID3D11Texture2D_GetDesc(s_depth[k].tex, &td);
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &s_sdepth)))
            return;
        memset(&sv, 0, sizeof sv);
        sv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sv.Texture2D.MipLevels = 1;
        ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)s_sdepth, &sv, &s_sdepth_srv);
        s_sdepth_w = s_depth[k].w; s_sdepth_h = s_depth[k].h;
    }
    ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)s_sdepth,
                                     (ID3D11Resource *)s_depth[k].tex);
    s_scene_drawn = 0;
    s_sdepth_this_frame++;
}

static float    s_jit[2];               /* this batch's, in NDC */
static float    s_jit_px[2];            /* this frame's, in target pixels */
static uint32_t s_jit_frame, s_main_base;

static float halton(uint32_t i, uint32_t b)
{
    float f = 1.0f, r = 0.0f;
    while (i) {
        f /= (float)b;
        r += f * (float)(i % b);
        i /= b;
    }
    return r;
}

static float jitter_scale(void)
{
    static float on = -1.0f;
    if (on < 0.0f) {
        const char *e = getenv("RECOMP_JITTER");
        const char *s = getenv("RECOMP_JITTER_SCALE");
        const char *d = getenv("RECOMP_DLSS");          /* DLSS needs it (point 155) */
        on = ((e && atoi(e) != 0) || (d && atoi(d) != 0)) ? (s ? (float)atof(s) : 1.0f) : 0.0f;
        if (on > 0.0f)
            fprintf(stderr, "  [D3D11] jitter on: Halton (2,3), 16 frames, scale %g\n", on);
    }
    return on;
}

/* At each present: the next offset of the sequence. */
static void jitter_next(void)
{
    const uint32_t i = (s_jit_frame++ % 16u) + 1u;
    s_jit_px[0] = (halton(i, 2) - 0.5f) * jitter_scale();
    s_jit_px[1] = (halton(i, 3) - 0.5f) * jitter_scale();
}

static void jitter_for(const Target *t, const NvD3DState *st)
{
    s_jit[0] = s_jit[1] = 0.0f;
    if (jitter_scale() <= 0.0f || st->layer == 1 || t->base != s_main_base || !t->w || !t->h)
        return;
    s_jit[0] =  2.0f * s_jit_px[0] / (float)(t->w * S);
    s_jit[1] = -2.0f * s_jit_px[1] / (float)(t->h * S);
}

/* Bind the target or its layer, whichever this batch goes to. */
static void bind_target_for(Target *t, int ui)
{
    ID3D11DepthStencilView *dsv;
    if (ui == s_bound_ui)
        return;
    dsv = depth_for(t->w * S, t->h * S);
    ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 1, ui ? &t->ui_rtv : &t->rtv, dsv);
    s_bound_ui = ui;
}

/* RECOMP_UI_43=1: how much the layer is squeezed across, so that the
 * interface keeps its 4:3 proportions in a picture presented wider (point
 * 151). 1 when the picture is not widescreen or the option is off. */
static float ui_squeeze(void)
{
    static int on = -1;
    float ws;
    if (on < 0) {
        const char *e = getenv("RECOMP_UI_43");
        on = e && atoi(e) != 0;
    }
    ws = xbox_WidescreenAspect();
    return (on && ws > 4.0f / 3.0f) ? (4.0f / 3.0f) / ws : 1.0f;
}

/* The layer onto its target, then emptied. Leaves the target bound. */
static void ui_composite(Target *t)
{
    static const FLOAT clear[4] = { 0, 0, 0, 1 };
    ID3D11ShaderResourceView *none = NULL;
    D3D11_MAPPED_SUBRESOURCE m;
    D3D11_VIEWPORT vp;
    D3D11_RECT sc;

    if (!t->ui_dirty || !s_ps_ui)
        return;
    if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb_ui, 0,
                                          D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        float cb[4];
        cb[0] = ui_squeeze();
        cb[1] = (float)(t->w * S);
        cb[2] = (float)(t->h * S);
        cb[3] = 0.0f;
        memcpy(m.pData, cb, sizeof cb);
        ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb_ui, 0);
    }
    ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 1, &t->rtv, NULL);
    s_bound_ui = -1;                            /* bound without its depth */
    vp.TopLeftX = 0; vp.TopLeftY = 0;
    vp.Width = (FLOAT)(t->w * S); vp.Height = (FLOAT)(t->h * S);
    vp.MinDepth = 0; vp.MaxDepth = 1;
    sc.left = 0; sc.top = 0; sc.right = (LONG)(t->w * S); sc.bottom = (LONG)(t->h * S);
    ID3D11DeviceContext_RSSetViewports(s_ctx, 1, &vp);
    ID3D11DeviceContext_RSSetScissorRects(s_ctx, 1, &sc);
    ID3D11DeviceContext_RSSetState(s_ctx, s_rs);
    ID3D11DeviceContext_OMSetBlendState(s_ctx, s_blend_ui_comp, NULL, 0xFFFFFFFFu);
    ID3D11DeviceContext_OMSetDepthStencilState(s_ctx, s_ds_off, 0);
    ID3D11DeviceContext_IASetInputLayout(s_ctx, NULL);
    ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_VSSetShader(s_ctx, s_vs_present, NULL, 0);
    ID3D11DeviceContext_PSSetShader(s_ctx, s_ps_ui, NULL, 0);
    ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_cb_ui);
    ID3D11DeviceContext_PSSetSamplers(s_ctx, 0, 1, &s_lin);
    ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &t->ui_srv);
    ID3D11DeviceContext_Draw(s_ctx, 3, 0);
    ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &none);
    ID3D11DeviceContext_ClearRenderTargetView(s_ctx, t->ui_rtv, clear);
    t->ui_dirty = 0;
    s_ui_composites++;
    if (s_cur >= 0 && &s_rt[s_cur] == t)
        bind_target_for(t, 0);
}

/* Everything a batch needs besides its vertices: the pixel shader constants,
 * the texture and its sampler, viewport, scissor, blend and depth. Shared by
 * the CPU path's flush and by a translated program's draw, so the two can
 * never disagree on what a state means. */
static void bind_state(const NvD3DState *st_in)
{
    D3D11_MAPPED_SUBRESOURCE m;
    NvD3DState layered;
    const NvD3DState *st = st_in;

    /* A target this batch samples is read: its layer goes onto it first.
     * Done before anything is bound, since compositing binds its own. */
    {
        int k;
        if (st_in->texture >= 0x10000)
            ui_composite(&s_rt[st_in->texture - 0x10000]);
        for (k = 0; k < 3; k++)
            if (st_in->stage_tex[k] >= 0x10000)
                ui_composite(&s_rt[st_in->stage_tex[k] - 0x10000]);
    }
    /* The target or its interface layer. A blend the layer cannot hold
     * exactly is drawn into the target, as before, and counted. */
    {
        Target *t = &s_rt[s_cur];
        int ui = 0;
        if (st_in->layer) {
            /* Every blend the interface uses, said once: the layer is exact
             * only for the ones ui_t_factor accepts. */
            static uint32_t seen[64][5];
            static int nseen;
            int k;
            for (k = 0; k < nseen; k++)
                if (seen[k][0] == st_in->blend_enable && seen[k][1] == st_in->blend_src
                        && seen[k][2] == st_in->blend_dst && seen[k][3] == st_in->blend_eq
                        && seen[k][4] == st_in->color_mask)
                    break;
            if (k == nseen && nseen < 64) {
                seen[nseen][0] = st_in->blend_enable; seen[nseen][1] = st_in->blend_src;
                seen[nseen][2] = st_in->blend_dst; seen[nseen][3] = st_in->blend_eq;
                seen[nseen][4] = st_in->color_mask;
                nseen++;
                fprintf(stderr, "  [D3D11] interface blend #%d: on %u src 0x%X dst 0x%X eq 0x%X"
                                " mask %08X  alpha test %u  %s\n", nseen - 1,
                        st_in->blend_enable, st_in->blend_src, st_in->blend_dst,
                        st_in->blend_eq, st_in->color_mask, st_in->alpha_test,
                        (ui_t_factor(st_in) != 0 || !st_in->blend_enable)
                            ? "layered" : "LEFT IN THE SCENE");
            }
            if (st_in->layer == 1 && (ui_t_factor(st_in) != 0 || !st_in->blend_enable)) {
                ui = ui_layer_ready(t);
            } else {
                /* A full-width fade (layer 2), or a blend the layer cannot
                 * hold: what the interface drew so far goes onto the target
                 * first, then this is drawn there -- the order the title
                 * drew them in, so the result is exact again. */
                if (st_in->layer == 1)
                    s_ui_fallback++;
                ui_composite(t);
            }
            if (!ui) {
                layered = *st_in;
                layered.layer = 0;
                st = &layered;
            } else {
                s_ui_batches++;
                t->ui_dirty = 1;
            }
        }
        if (!st_in->layer && t->base == s_main_base)
            s_scene_drawn = 1;                  /* 3D: its depth is worth keeping */
        bind_target_for(t, ui);
    }
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
        jitter_for(&s_rt[s_cur], st);
        cb[10] = s_jit[0];
        cb[11] = s_jit[1];
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
}

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

    bind_state(st);
    ID3D11DeviceContext_IASetInputLayout(s_ctx, s_layout);
    ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_IASetVertexBuffers(s_ctx, 0, 1, &s_vb, &stride, &offset);
    ID3D11DeviceContext_VSSetShader(s_ctx, s_vs, NULL, 0);
    ID3D11DeviceContext_VSSetConstantBuffers(s_ctx, 0, 1, &s_cb);   /* the jitter */
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

/* ── Vertex programs on the GPU ─────────────────────────────────────── */
/*
 * See nv2a_d3d11.h. A translated program is an ordinary vs_5_0 shader whose
 * output struct is the one the pixel shader already takes, so the rest of the
 * pipeline -- sampling, alpha test, blend, depth -- is untouched. It reads
 * the batch from a raw buffer (the title's vertex formats are not all DXGI
 * formats: the 11:11:10 normals are not) indexed by SV_VertexID, which under
 * DrawIndexed with no vertex buffers is the index itself.
 */
#define MAX_VS_PROGRAMS 256
typedef struct {
    uint64_t hash;
    ID3D11VertexShader   *vs;
    ID3D11GeometryShader *so;           /* the same shader, stream output only */
    ID3DBlob             *code;
} VsProgram;
static VsProgram s_vsp[MAX_VS_PROGRAMS];
static int       s_nvsp;

static ID3D11Buffer             *s_vp_cbc, *s_vp_cba, *s_vp_ib, *s_vp_data;
static ID3D11ShaderResourceView *s_vp_data_srv;
static uint32_t                  s_vp_data_cap, s_vp_ib_cap;
static ID3D11Buffer             *s_so_buf, *s_so_stage;
static uint32_t                  s_so_cap;
/* The batch data and indices go into rings, mapped NO_OVERWRITE and
 * discarded only on wrap. The first version discarded a 1 MB buffer for
 * every batch, two triangles or two thousand: the driver renamed a megabyte
 * each time, and the menus fell from 1160 frames a second to 540. A raw view
 * of a dynamic buffer may only be mapped NO_OVERWRITE where the driver says
 * so (D3D11.1); elsewhere the data ring falls back to a discard per batch. */
static uint32_t s_vp_data_pos, s_vp_ib_pos, s_vp_consts_gen = 0xFFFFFFFFu;
static int      s_vp_srv_nooverwrite = -1;

static int ring_upload(ID3D11Buffer *b, uint32_t cap, uint32_t *pos, int nooverwrite,
                       const void *src, uint32_t n, uint32_t *at)
{
    D3D11_MAPPED_SUBRESOURCE m;
    D3D11_MAP how = D3D11_MAP_WRITE_NO_OVERWRITE;
    uint32_t p = (*pos + 15u) & ~15u;
    if (!nooverwrite || p + n > cap) {
        p = 0;
        how = D3D11_MAP_WRITE_DISCARD;
    }
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)b, 0, how, 0, &m)))
        return 0;
    memcpy((uint8_t *)m.pData + p, src, n);
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)b, 0);
    *at = p;
    *pos = p + n;
    return 1;
}

int nvd3d_vs_program(uint64_t hash, const char *hlsl, size_t len)
{
    int i;
    ID3DBlob *code = NULL, *err = NULL;
    VsProgram *p;

    for (i = 0; i < s_nvsp; i++)
        if (s_vsp[i].hash == hash)
            return s_vsp[i].vs ? i : -1;
    if (!hlsl)
        return -2;                      /* unknown: the caller translates */
    if (s_nvsp >= MAX_VS_PROGRAMS || !s_dev)
        return -1;
    p = &s_vsp[s_nvsp++];
    memset(p, 0, sizeof *p);
    p->hash = hash;
    if (FAILED(D3DCompile(hlsl, len, "nv2a_vp", NULL, NULL, "vp_main", "vs_5_0",
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err))) {
        static int said;
        if (said++ < 8)
            fprintf(stderr, "  [VPGPU] program %016llX does not compile, kept on the CPU: %s\n",
                    (unsigned long long)hash,
                    err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "?");
        if (err) ID3D10Blob_Release(err);
        return -1;                      /* the failure is cached: no retry */
    }
    if (err) ID3D10Blob_Release(err);
    if (FAILED(ID3D11Device_CreateVertexShader(s_dev, ID3D10Blob_GetBufferPointer(code),
                                               ID3D10Blob_GetBufferSize(code), NULL, &p->vs))) {
        ID3D10Blob_Release(code);
        return -1;
    }
    p->code = code;
    return (int)(p - s_vsp);
}

static int vp_upload(ID3D11Buffer *b, const void *src, size_t n)
{
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)b, 0,
                                       D3D11_MAP_WRITE_DISCARD, 0, &m)))
        return 0;
    memcpy(m.pData, src, n);
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)b, 0);
    return 1;
}

/* Buffers, constants and shader: everything common to drawing a batch and
 * capturing it. 0 if something could not be made. */
static int vp_setup(const NvD3DProgramDraw *d, UINT *first_index)
{
    D3D11_BUFFER_DESC bd;
    uint32_t want = (d->data_bytes + 15u) & ~15u;
    uint32_t data_at = 0, ib_at = 0;

    if (d->vs < 0 || d->vs >= s_nvsp || !s_vsp[d->vs].vs)
        return 0;
    if (s_vp_srv_nooverwrite < 0) {
        D3D11_FEATURE_DATA_D3D11_OPTIONS o;
        memset(&o, 0, sizeof o);
        s_vp_srv_nooverwrite =
            SUCCEEDED(ID3D11Device_CheckFeatureSupport(s_dev, D3D11_FEATURE_D3D11_OPTIONS,
                                                      &o, sizeof o))
            && o.MapNoOverwriteOnDynamicBufferSRV;
        fprintf(stderr, "  [VPGPU] batch data ring: %s\n",
                s_vp_srv_nooverwrite ? "mapped without overwrite"
                                     : "discarded per batch (driver lacks D3D11.1 option)");
    }
    if (!s_vp_cbc) {
        memset(&bd, 0, sizeof bd);
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bd.ByteWidth = 192 * 16;
        if (FAILED(ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_vp_cbc)))
            return 0;
        s_vp_consts_gen = 0xFFFFFFFFu;
        bd.ByteWidth = 16 * 16 + 16 * 16 + 2 * 16;
        if (FAILED(ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_vp_cba)))
            return 0;
    }
    if (want < 16) want = 16;
    if (want > s_vp_data_cap) {
        D3D11_SHADER_RESOURCE_VIEW_DESC sd;
        if (s_vp_data_srv) ID3D11ShaderResourceView_Release(s_vp_data_srv);
        if (s_vp_data) ID3D11Buffer_Release(s_vp_data);
        s_vp_data_srv = NULL; s_vp_data = NULL; s_vp_data_cap = 0;
        memset(&bd, 0, sizeof bd);
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        bd.ByteWidth = want < (8u << 20) ? (8u << 20) : want * 2;
        if (FAILED(ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_vp_data)))
            return 0;
        memset(&sd, 0, sizeof sd);
        sd.Format = DXGI_FORMAT_R32_TYPELESS;
        sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        sd.BufferEx.NumElements = bd.ByteWidth / 4;
        sd.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        if (FAILED(ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)s_vp_data,
                                                          &sd, &s_vp_data_srv)))
            return 0;
        s_vp_data_cap = bd.ByteWidth;
        s_vp_data_pos = 0;
    }
    if (d->nindex * 4u > s_vp_ib_cap) {
        if (s_vp_ib) ID3D11Buffer_Release(s_vp_ib);
        s_vp_ib = NULL; s_vp_ib_cap = 0;
        memset(&bd, 0, sizeof bd);
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bd.ByteWidth = d->nindex * 4u < (2u << 20) ? (2u << 20) : d->nindex * 8u;
        if (FAILED(ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_vp_ib)))
            return 0;
        s_vp_ib_cap = bd.ByteWidth;
        s_vp_ib_pos = 0;
    }

    if (!ring_upload(s_vp_data, s_vp_data_cap, &s_vp_data_pos, s_vp_srv_nooverwrite,
                     d->data, d->data_bytes, &data_at)
     || !ring_upload(s_vp_ib, s_vp_ib_cap, &s_vp_ib_pos, 1,
                     d->index, d->nindex * 4u, &ib_at))
        return 0;
    /* The constants only when the title changed them: most batches in a row
     * share them, and 3 KB per batch was most of what was left. */
    if (d->consts_gen != s_vp_consts_gen) {
        if (!vp_upload(s_vp_cbc, d->consts, 192 * 16))
            return 0;
        s_vp_consts_gen = d->consts_gen;
    }
    *first_index = ib_at / 4u;
    {
        uint8_t cba[16 * 16 + 16 * 16 + 2 * 16];
        float xf[8];
        {
            NvD3DAttr at[16];
            int k;
            memcpy(at, d->attr, sizeof at);
            for (k = 0; k < 16; k++)
                if (at[k].size)
                    at[k].base += data_at;        /* modular, as the shader's */
            memcpy(cba, at, 16 * 16);
        }
        memcpy(cba + 256, d->absent, 16 * 16);
        xf[0] = d->ax; xf[1] = d->ay; xf[2] = d->uv_scale[0]; xf[3] = d->uv_scale[1];
        xf[4] = d->textured ? 1.0f : 0.0f;
        xf[5] = d->points ? 1.0f : 0.0f;     /* xf1.y: indices carry a sprite corner */
        xf[6] = s_jit[0];                     /* xf1.zw: the jitter, set by bind_state */
        xf[7] = s_jit[1];
        memcpy(cba + 512, xf, sizeof xf);
        if (!vp_upload(s_vp_cba, cba, sizeof cba))
            return 0;
    }
    {
        ID3D11Buffer *cbs[2];
        cbs[0] = s_vp_cbc; cbs[1] = s_vp_cba;
        ID3D11DeviceContext_IASetInputLayout(s_ctx, NULL);
        ID3D11DeviceContext_IASetIndexBuffer(s_ctx, s_vp_ib, DXGI_FORMAT_R32_UINT, 0);
        ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ID3D11DeviceContext_VSSetShader(s_ctx, s_vsp[d->vs].vs, NULL, 0);
        ID3D11DeviceContext_VSSetConstantBuffers(s_ctx, 1, 2, cbs);
        ID3D11DeviceContext_VSSetShaderResources(s_ctx, 1, 1, &s_vp_data_srv);
    }
    return 1;
}

static void vp_unbind(void)
{
    ID3D11ShaderResourceView *none = NULL;
    ID3D11DeviceContext_VSSetShaderResources(s_ctx, 1, 1, &none);
}

/* ── DLSS step 3: the motion pass ───────────────────────────────────────
 *
 * The batch just drawn, once more, into the motion vector target: same
 * indices, same jitter, its motion variant as vertex shader (the program run
 * on the previous frame's constants, then on these), ps_mv, the target's own
 * depth tested LESS_EQUAL without writing -- so only what the colour pass left
 * in front gets a vector. Cleared to 0 (no motion) with the target's colour. */
static int mv_ready(Target *t)
{
    D3D11_TEXTURE2D_DESC td;
    static const FLOAT zero[4] = { 0, 0, 0, 0 };
    if (t->mv_rtv)
        return 1;
    memset(&td, 0, sizeof td);
    td.Width = t->w * S; td.Height = t->h * S; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R16G16_FLOAT;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &t->mv_tex)))
        return 0;
    ID3D11Device_CreateRenderTargetView(s_dev, (ID3D11Resource *)t->mv_tex, NULL, &t->mv_rtv);
    ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)t->mv_tex, NULL, &t->mv_srv);
    if (!t->mv_rtv || !t->mv_srv)
        return 0;
    ID3D11DeviceContext_ClearRenderTargetView(s_ctx, t->mv_rtv, zero);
    fprintf(stderr, "  [D3D11] motion vectors for 0x%08X (%ux%u)\n", t->base, td.Width, td.Height);
    return 1;
}

static void motion_pass(Target *t, const NvD3DProgramDraw *d, UINT first)
{
    D3D11_MAPPED_SUBRESOURCE m;
    ID3D11DepthStencilView *dsv;

    if (!s_ps_mv || d->vs_motion < 0 || d->vs_motion >= s_nvsp || !s_vsp[d->vs_motion].vs
            || !d->prev_consts || !mv_ready(t))
        return;
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb_prev, 0,
                                       D3D11_MAP_WRITE_DISCARD, 0, &m)))
        return;
    memcpy(m.pData, d->prev_consts, 192 * 16);
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb_prev, 0);
    dsv = depth_for(t->w * S, t->h * S);
    ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 1, &t->mv_rtv, dsv);
    s_bound_ui = -1;                            /* the next batch binds again */
    ID3D11DeviceContext_OMSetBlendState(s_ctx, s_blend_off, NULL, 0xFFFFFFFFu);
    ID3D11DeviceContext_OMSetDepthStencilState(s_ctx, s_ds_mv, 0);
    ID3D11DeviceContext_VSSetShader(s_ctx, s_vsp[d->vs_motion].vs, NULL, 0);
    ID3D11DeviceContext_VSSetConstantBuffers(s_ctx, 4, 1, &s_cb_prev);
    ID3D11DeviceContext_PSSetShader(s_ctx, s_ps_mv, NULL, 0);
    ID3D11DeviceContext_DrawIndexed(s_ctx, d->nindex, first, 0);
    s_mv_batches++;
}

void nvd3d_draw_program(const NvD3DProgramDraw *d, const NvD3DState *st)
{
    UINT first = 0;

    if (s_cur < 0 || !d->nindex)
        return;
    flush();                            /* what the CPU path queued goes first */
    bind_state(st);
    if (!vp_setup(d, &first))
        return;
    if (st->ps >= 0 && st->ps < s_npsp && s_psp[st->ps].ps) {
        bind_combiners(st);
        ID3D11DeviceContext_PSSetShader(s_ctx, s_psp[st->ps].ps, NULL, 0);
    } else {
        ID3D11DeviceContext_PSSetShader(s_ctx, s_ps, NULL, 0);
    }
    ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_cb);
    if (d->lines)
        ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
    ID3D11DeviceContext_DrawIndexed(s_ctx, d->nindex, first, 0);
    if (d->lines)
        ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (d->motion && !st->layer)
        motion_pass(&s_rt[s_cur], d, first);
    vp_unbind();
    framelog("prog %5u i  vs %d  tex %d %ux%u  z%u%u f%X  bl%u %X/%X  cm%08X  at%u f%X r%u",
             d->nindex, d->vs, st->texture, st->tex_w, st->tex_h,
             st->depth_enable, st->depth_write, st->depth_func,
             st->blend_enable, st->blend_src, st->blend_dst, st->color_mask,
             st->alpha_test, st->alpha_func, st->alpha_ref);
}

uint32_t nvd3d_check_program(const NvD3DProgramDraw *d, NvD3DVertex *out, uint32_t max)
{
    VsProgram *p;
    uint32_t n = d->nindex < max ? d->nindex : max;
    uint32_t bytes = n * (uint32_t)sizeof(NvD3DVertex);
    D3D11_MAPPED_SUBRESOURCE m;
    UINT off = 0, first = 0;
    ID3D11Buffer *none = NULL;

    if (s_cur < 0 || !n || d->vs < 0 || d->vs >= s_nvsp)
        return 0;
    p = &s_vsp[d->vs];
    flush();
    if (!p->so) {
        /* The vertex shader's own output, streamed out without a geometry
         * stage: the declaration is exactly NvD3DVertex, so what comes back is
         * comparable field for field with the CPU path's vertices. */
        static const D3D11_SO_DECLARATION_ENTRY decl[3] = {
            { 0, "SV_Position", 0, 0, 4, 0 },
            { 0, "TEXCOORD",    0, 0, 2, 0 },
            { 0, "COLOR",       0, 0, 4, 0 },
        };
        UINT stride = sizeof(NvD3DVertex);
        if (FAILED(ID3D11Device_CreateGeometryShaderWithStreamOutput(
                s_dev, ID3D10Blob_GetBufferPointer(p->code), ID3D10Blob_GetBufferSize(p->code),
                decl, 3, &stride, 1, D3D11_SO_NO_RASTERIZED_STREAM, NULL, &p->so)))
            return 0;
    }
    if (bytes > s_so_cap) {
        D3D11_BUFFER_DESC bd;
        if (s_so_buf) ID3D11Buffer_Release(s_so_buf);
        if (s_so_stage) ID3D11Buffer_Release(s_so_stage);
        s_so_buf = s_so_stage = NULL; s_so_cap = 0;
        memset(&bd, 0, sizeof bd);
        bd.ByteWidth = bytes * 2;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_STREAM_OUTPUT;
        if (FAILED(ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_so_buf)))
            return 0;
        bd.Usage = D3D11_USAGE_STAGING;
        bd.BindFlags = 0;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_so_stage)))
            return 0;
        s_so_cap = bytes * 2;
    }
    {
        NvD3DProgramDraw dn = *d;
        dn.nindex = n;
        if (!vp_setup(&dn, &first))
            return 0;
    }
    ID3D11DeviceContext_GSSetShader(s_ctx, p->so, NULL, 0);
    ID3D11DeviceContext_SOSetTargets(s_ctx, 1, &s_so_buf, &off);
    ID3D11DeviceContext_DrawIndexed(s_ctx, n, first, 0);
    ID3D11DeviceContext_SOSetTargets(s_ctx, 1, &none, &off);
    ID3D11DeviceContext_GSSetShader(s_ctx, NULL, NULL, 0);
    vp_unbind();
    {
        D3D11_BOX box = { 0, 0, 0, bytes, 1, 1 };
        ID3D11DeviceContext_CopySubresourceRegion(s_ctx, (ID3D11Resource *)s_so_stage, 0, 0, 0, 0,
                                                  (ID3D11Resource *)s_so_buf, 0, &box);
    }
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_so_stage, 0,
                                       D3D11_MAP_READ, 0, &m)))
        return 0;
    memcpy(out, m.pData, bytes);
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_so_stage, 0);
    return n;
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
    if ((what & 1) && s_scene_drawn && t->base == s_main_base) {
        scene_depth_capture(t);                 /* the 3D's depth, before it goes */
        s_sdepth_at_clear++;
    }
    if (what & 3) {
        ID3D11DepthStencilView *dsv = depth_for(t->w * S, t->h * S);
        UINT f = ((what & 1) ? D3D11_CLEAR_DEPTH : 0) | ((what & 2) ? D3D11_CLEAR_STENCIL : 0);
        if (dsv)
            ID3D11DeviceContext_ClearDepthStencilView(s_ctx, dsv, f, depth, stencil);
    }
    if (what & 4) {
        FLOAT c[4];
        /* What the clear covers loses its interface too: composited, then
         * cleared, is the same as cleared. */
        ui_composite(&s_rt[s_cur]);
        if (t->mv_rtv) {                        /* and its motion: none yet */
            static const FLOAT zero[4] = { 0, 0, 0, 0 };
            ID3D11DeviceContext_ClearRenderTargetView(s_ctx, t->mv_rtv, zero);
        }
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
        ID3D11Device_QueryInterface(s_dev, &IID_IDXGIDevice, (void **)&dxdev);
        IDXGIDevice_GetAdapter(dxdev, &ad);
        IDXGIAdapter_GetParent(ad, &IID_IDXGIFactory2, (void **)&fac);
        /* DLSS (manual hooking): the swap chain from Streamline's factory
         * proxy, so that each Present goes through Streamline. */
        if (nvdlss_ready() && nvdlss_upgrade_factory((void **)&fac))
            fprintf(stderr, "  [D3D11] swap chain through Streamline\n");
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
        IDXGISwapChain1_GetBuffer(s_swap, 0, &IID_ID3D11Texture2D, (void **)&bb);
        ID3D11Device_CreateRenderTargetView(s_dev, (ID3D11Resource *)bb, NULL, &s_back_rtv);
        ID3D11Texture2D_Release(bb);
        s_back_w = w; s_back_h = h;
        s_cur = -1;   /* targets were unbound */
        s_bound_ui = -1;
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

/* ── DLSS step 4: DLAA through Streamline (RECOMP_DLSS=1) ───────────────
 *
 * At the present, on the shown surface, when this frame drew 3D: the scene
 * (the target before the interface layer goes on it), its depth copy, its
 * motion vectors and the jitter go to Streamline, and what comes back --
 * the same size, anti-aliased -- is copied over the target. The interface is
 * then composited as always, never through the upscaler. F5 turns it off and
 * on, to compare at the same spot. RECOMP_DLSS_JITTER_SIGN=-1 flips the
 * jitter given to DLSS, should the measured convention turn out the other. */
static ID3D11Texture2D          *s_dlss_out;
static ID3D11ShaderResourceView *s_dlss_out_srv;
static uint32_t                  s_dlss_w, s_dlss_h, s_dlss_ok, s_dlss_fail;
static int                       s_dlss_live = 1, s_dlss_reset = 1;

static void dlss_frame(Target *t, int scene)
{
    const uint32_t w = t->w * S, h = t->h * S;
    ID3D11ShaderResourceView *none = NULL;
    float sign = 1.0f;

    if (!nvdlss_ready())
        return;
    if (GetAsyncKeyState(VK_F5) & 1) {
        s_dlss_live = !s_dlss_live;
        s_dlss_reset = 1;
        fprintf(stderr, "  [DLSS] F5: %s\n", s_dlss_live ? "on" : "off");
    }
    if (!s_dlss_live || !scene || !s_sdepth || !t->mv_tex
            || s_sdepth_w != w || s_sdepth_h != h) {
        s_dlss_reset = 1;                       /* no history across a gap */
        return;
    }
    if (!s_dlss_out || s_dlss_w != w || s_dlss_h != h) {
        D3D11_TEXTURE2D_DESC td;
        if (s_dlss_out_srv) ID3D11ShaderResourceView_Release(s_dlss_out_srv);
        if (s_dlss_out) ID3D11Texture2D_Release(s_dlss_out);
        s_dlss_out = NULL; s_dlss_out_srv = NULL;
        memset(&td, 0, sizeof td);
        td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;     /* a typed UAV everywhere */
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &s_dlss_out)))
            return;
        ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)s_dlss_out, NULL,
                                              &s_dlss_out_srv);
        s_dlss_w = w; s_dlss_h = h;
        s_dlss_reset = 1;
        fprintf(stderr, "  [DLSS] DLAA at %ux%u\n", w, h);
    }
    {
        const char *e = getenv("RECOMP_DLSS_JITTER_SIGN");
        if (e && atoi(e) < 0)
            sign = -1.0f;
    }
    /* Nothing of ours bound while Streamline works; it leaves the state to us. */
    ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 0, NULL, NULL);
    ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &none);
    s_bound_ui = -1;
    if (!nvdlss_evaluate(s_ctx, t->tex, s_sdepth, t->mv_tex, s_dlss_out, w, h,
                         sign * s_jit_px[0], sign * s_jit_px[1], s_dlss_reset)) {
        s_dlss_fail++;
        s_dlss_reset = 1;
        return;
    }
    s_dlss_reset = 0;
    s_dlss_ok++;
    /* How much DLSS changed, measured: every 600 frames, the scene before and
     * after on a 64 x 48 grid -- the mean difference per channel, and how many
     * points moved by more than 8 levels (out of 255). */
    if (s_dlss_ok % 600 == 1) {
        static ID3D11Texture2D *sa, *sb;
        static uint32_t sw, sh;
        D3D11_MAPPED_SUBRESOURCE ma, mb;
        if (!sa || sw != w || sh != h) {
            D3D11_TEXTURE2D_DESC td;
            if (sa) ID3D11Texture2D_Release(sa);
            if (sb) ID3D11Texture2D_Release(sb);
            sa = sb = NULL;
            ID3D11Texture2D_GetDesc(t->tex, &td);
            td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0;
            td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
            ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &sa);
            ID3D11Texture2D_GetDesc(s_dlss_out, &td);
            td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0;
            td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
            ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &sb);
            sw = w; sh = h;
        }
        if (sa && sb) {
            ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)sa, (ID3D11Resource *)t->tex);
            ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)sb, (ID3D11Resource *)s_dlss_out);
            if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)sa, 0, D3D11_MAP_READ, 0, &ma))) {
                if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)sb, 0, D3D11_MAP_READ, 0, &mb))) {
                    uint32_t x, y, n = 0, big = 0;
                    double sum = 0.0;
                    for (y = 0; y < 48; y++)
                        for (x = 0; x < 64; x++) {
                            const uint32_t px = (x * 2 + 1) * w / 128, py = (y * 2 + 1) * h / 96;
                            const uint8_t *a = (const uint8_t *)ma.pData + (size_t)py * ma.RowPitch + px * 4;
                            const uint8_t *b = (const uint8_t *)mb.pData + (size_t)py * mb.RowPitch + px * 4;
                            /* BGRA in, RGBA out */
                            const int d0 = abs((int)a[2] - (int)b[0]), d1 = abs((int)a[1] - (int)b[1]),
                                      d2 = abs((int)a[0] - (int)b[2]);
                            const int dm = d0 > d1 ? (d0 > d2 ? d0 : d2) : (d1 > d2 ? d1 : d2);
                            sum += (d0 + d1 + d2) / 3.0;
                            if (dm > 8) big++;
                            n++;
                        }
                    fprintf(stderr, "  [DLSS] change: mean %.2f levels per channel, %u of %u points"
                                    " by more than 8\n", sum / n, big, n);
                    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)sb, 0);
                }
                ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)sa, 0);
            }
        }
    }
    {
        D3D11_VIEWPORT vp;
        D3D11_RECT sc;
        vp.TopLeftX = 0; vp.TopLeftY = 0; vp.Width = (FLOAT)w; vp.Height = (FLOAT)h;
        vp.MinDepth = 0; vp.MaxDepth = 1;
        sc.left = 0; sc.top = 0; sc.right = (LONG)w; sc.bottom = (LONG)h;
        ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 1, &t->rtv, NULL);
        ID3D11DeviceContext_RSSetViewports(s_ctx, 1, &vp);
        ID3D11DeviceContext_RSSetScissorRects(s_ctx, 1, &sc);
        ID3D11DeviceContext_RSSetState(s_ctx, s_rs);
        ID3D11DeviceContext_OMSetBlendState(s_ctx, s_blend_off, NULL, 0xFFFFFFFFu);
        ID3D11DeviceContext_OMSetDepthStencilState(s_ctx, s_ds_off, 0);
        ID3D11DeviceContext_IASetInputLayout(s_ctx, NULL);
        ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ID3D11DeviceContext_VSSetShader(s_ctx, s_vs_present, NULL, 0);
        ID3D11DeviceContext_PSSetShader(s_ctx, s_ps_copy, NULL, 0);
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &s_dlss_out_srv);
        ID3D11DeviceContext_Draw(s_ctx, 3, 0);
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &none);
    }
    if ((s_dlss_ok + s_dlss_fail) % 1200 == 1)
        fprintf(stderr, "  [DLSS] frames %u through DLAA, %u refused; %s\n", s_dlss_ok,
                s_dlss_fail, nvdlss_status());
}

static int cmp_u32(const void *a, const void *b)
{
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}

/* RECOMP_DEPTH_VIEW: the distribution of a depth texture's 24-bit values. */
static void depth_stats(ID3D11Texture2D *tex, uint32_t w, uint32_t h)
{
    static ID3D11Texture2D *stage;
    static uint32_t sw, sh;
    D3D11_MAPPED_SUBRESOURCE m;
    uint32_t v[64 * 48], n = 0, x, y, cleared = 0;
    struct { uint32_t w, h; ID3D11Texture2D *tex; } dd, *d = &dd;

    dd.w = w; dd.h = h; dd.tex = tex;
    if (!stage || sw != d->w || sh != d->h) {
        D3D11_TEXTURE2D_DESC td;
        if (stage) ID3D11Texture2D_Release(stage);
        stage = NULL;
        ID3D11Texture2D_GetDesc(d->tex, &td);
        td.Usage = D3D11_USAGE_STAGING;
        td.BindFlags = 0;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &stage)))
            return;
        sw = d->w; sh = d->h;
    }
    ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)stage, (ID3D11Resource *)d->tex);
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)stage, 0, D3D11_MAP_READ, 0, &m)))
        return;
    for (y = 0; y < 48; y++)
        for (x = 0; x < 64; x++) {
            const uint32_t *row = (const uint32_t *)((const uint8_t *)m.pData
                                  + (size_t)((y * 2 + 1) * d->h / 96) * m.RowPitch);
            uint32_t z = row[(x * 2 + 1) * d->w / 128] & 0xFFFFFFu;
            if (z == 0xFFFFFFu) cleared++;
            v[n++] = z;
        }
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)stage, 0);
    qsort(v, n, sizeof v[0], cmp_u32);
    fprintf(stderr, "  [DEPTH] %ux%u: min %06X  p10 %06X  p50 %06X  p90 %06X  max %06X"
                    "  cleared (FFFFFF) %u of %u\n", d->w, d->h, v[0], v[n / 10], v[n / 2],
            v[n * 9 / 10], v[n - 1], cleared, n);
}

static float half_to_float(uint16_t h)
{
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t f;
    float out;
    if (e == 0)
        return (s ? -1.0f : 1.0f) * (float)m / 16777216.0f;     /* subnormal */
    f = s | ((e + 112u) << 23) | (m << 13);
    if (e == 31)
        f = s | 0x7F800000u | (m << 13);
    memcpy(&out, &f, 4);
    return out;
}

/* RECOMP_MOTION: what the motion vectors hold, on a 64 x 48 grid, in pixels
 * of the target: how many move, and how far. */
static void mv_stats(const Target *t)
{
    static ID3D11Texture2D *stage;
    static uint32_t sw, sh;
    D3D11_MAPPED_SUBRESOURCE m;
    float v[64 * 48];
    uint32_t n = 0, x, y, moving = 0, wild = 0;
    const uint32_t w = t->w * S, h = t->h * S;

    if (!stage || sw != w || sh != h) {
        D3D11_TEXTURE2D_DESC td;
        if (stage) ID3D11Texture2D_Release(stage);
        stage = NULL;
        ID3D11Texture2D_GetDesc(t->mv_tex, &td);
        td.Usage = D3D11_USAGE_STAGING;
        td.BindFlags = 0;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &stage)))
            return;
        sw = w; sh = h;
    }
    ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)stage, (ID3D11Resource *)t->mv_tex);
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)stage, 0, D3D11_MAP_READ, 0, &m)))
        return;
    for (y = 0; y < 48; y++)
        for (x = 0; x < 64; x++) {
            const uint16_t *row = (const uint16_t *)((const uint8_t *)m.pData
                                  + (size_t)((y * 2 + 1) * h / 96) * m.RowPitch);
            const uint32_t at = (x * 2 + 1) * w / 128;
            const float mx = half_to_float(row[at * 2]) * (float)w;
            const float my = half_to_float(row[at * 2 + 1]) * (float)h;
            const float len = sqrtf(mx * mx + my * my);
            if (len > 0.01f) moving++;
            if (len > 500.0f) wild++;
            v[n++] = len;
        }
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)stage, 0);
    {
        int i, j;                                       /* small: insertion sort */
        for (i = 1; i < (int)n; i++) {
            float k = v[i];
            for (j = i - 1; j >= 0 && v[j] > k; j--)
                v[j + 1] = v[j];
            v[j + 1] = k;
        }
    }
    fprintf(stderr, "  [MOTION] %ux%u: %u of %u points move, %u by more than 500 px; length in"
                    " pixels p50 %.2f  p90 %.2f  max %.2f; motion batches %u\n", w, h, moving,
            n, wild, v[n / 2], v[n * 9 / 10], v[n - 1], s_mv_batches);
}

void nvd3d_present(uint32_t base, const uint32_t rect[4],
                   uint32_t disp_w, uint32_t disp_h)
{
    int i, src = -1, prev = s_cur;

    s_present_count++;
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
    if (s_scene_drawn && s_rt[src].base == s_main_base) {
        scene_depth_capture(&s_rt[src]);        /* no depth clear after the 3D */
        s_sdepth_at_present++;
    }
    /* DLSS on the scene alone, then the interface over what it gives back. */
    dlss_frame(&s_rt[src], s_sdepth_this_frame > 0 && s_rt[src].base == s_main_base);
    ui_composite(&s_rt[src]);
    s_sdepth_hist[s_sdepth_this_frame < 3 ? s_sdepth_this_frame : 3]++;
    s_sdepth_this_frame = 0;
    s_main_base = base;                 /* the surface the jitter applies to */
    if (jitter_scale() > 0.0f)
        jitter_next();
    {
        static uint32_t said_at;
        if (s_ui_batches && s_ui_composites >= said_at) {
            said_at = s_ui_composites + 7200;
            fprintf(stderr, "  [D3D11] interface layer: %u batches, %u composites, %u left in the"
                            " scene (a blend that reads the destination)\n",
                    s_ui_batches, s_ui_composites, s_ui_fallback);
        }
    }
    {
        const Target *t = &s_rt[src];
        D3D11_MAPPED_SUBRESOURCE m;
        D3D11_VIEWPORT vp;
        ID3D11ShaderResourceView *none = NULL;
        float cb[8];
        cb[0] = (float)rect[0] / (float)t->w;
        cb[1] = (float)rect[1] / (float)t->h;
        cb[2] = (float)rect[2] / (float)t->w;
        cb[3] = (float)rect[3] / (float)t->h;
        ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &none);
        ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 1, &s_back_rtv, NULL);
        s_bound_ui = -1;
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
        /* Source texels per output pixel, and one texel in uv: see ps_present.
         * RECOMP_PRESENT_FILTER=0 keeps the single tap, to compare two passes
         * over one scene rather than remember one. */
        {
            static int filter = -1;
            if (filter < 0) {
                const char *e = getenv("RECOMP_PRESENT_FILTER");
                filter = !(e && atoi(e) == 0);
            }
            cb[4] = filter ? (float)(rect[2] * S) / vp.Width  : 1.0f;
            cb[5] = filter ? (float)(rect[3] * S) / vp.Height : 1.0f;
            cb[6] = 1.0f / (float)(t->w * S);
            cb[7] = 1.0f / (float)(t->h * S);
        }
        if (SUCCEEDED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_cb_present, 0,
                                              D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            memcpy(m.pData, cb, sizeof cb);
            ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_cb_present, 0);
        }
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
        {
            /* RECOMP_DEPTH_VIEW=1: F7 shows the target's depth instead. */
            static int allow = -1, show;
            ID3D11ShaderResourceView *dsrv = NULL;
            int k;
            if (allow < 0) {
                const char *e = getenv("RECOMP_DEPTH_VIEW");
                allow = e != NULL;
                show = e && atoi(e) == 2;               /* 2: from the start */
            }
            if (allow && (GetAsyncKeyState(VK_F7) & 1)) {
                show = !show;
                fprintf(stderr, "  [D3D11] F7: %s\n", show ? "depth" : "picture");
            }
            (void)k;
            if (show && s_sdepth_w == t->w * S && s_sdepth_h == t->h * S)
                dsrv = s_sdepth_srv;
            /* What the scene's depth holds, measured rather than guessed:
             * every 300 frames, a 64 x 48 grid of it read back, sorted. */
            if (allow) {
                static uint32_t frames;
                if (++frames % 300 == 0 && s_sdepth) {
                    depth_stats(s_sdepth, s_sdepth_w, s_sdepth_h);
                    fprintf(stderr, "  [DEPTH] copies: %u at a depth clear, %u at the present;"
                                    " frames with 0/1/2/more: %u/%u/%u/%u\n",
                            s_sdepth_at_clear, s_sdepth_at_present, s_sdepth_hist[0],
                            s_sdepth_hist[1], s_sdepth_hist[2], s_sdepth_hist[3]);
                }
            }
            {
                /* RECOMP_MOTION=1: F6 shows the motion vectors instead, and
                 * every 300 frames they are measured. */
                static int mallow = -1, mshow;
                static uint32_t mframes;
                if (mallow < 0) {
                    const char *e = getenv("RECOMP_MOTION");
                    mallow = e && atoi(e) != 0;
                }
                if (mallow && (GetAsyncKeyState(VK_F6) & 1)) {
                    mshow = !mshow;
                    fprintf(stderr, "  [D3D11] F6: %s\n", mshow ? "motion vectors" : "picture");
                }
                if (mallow && t->mv_srv && ++mframes % 300 == 0)
                    mv_stats(t);
                if (mshow && t->mv_srv && s_ps_mvview) {
                    ID3D11DeviceContext_PSSetShader(s_ctx, s_ps_mvview, NULL, 0);
                    ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_cb_present);
                    ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1, &t->mv_srv);
                    dsrv = NULL;
                    goto shaders_set;
                }
            }
            ID3D11DeviceContext_PSSetShader(s_ctx, dsrv && s_ps_depth ? s_ps_depth : s_ps_present,
                                            NULL, 0);
            ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_cb_present);
            ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 1,
                                                     dsrv && s_ps_depth ? &dsrv : &t->srv);
        shaders_set:;
        }
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
    ui_composite(&s_rt[src]);
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
int  nvd3d_texture_seen(uint64_t hash, uint32_t within) { (void)hash; (void)within; return 0; }
int  nvd3d_texture(const uint32_t key[6], uint64_t hash, uint32_t w, uint32_t h,
                   uint32_t mb, uint32_t mn, void (*d)(uint32_t *, uint32_t, uint32_t))
{ (void)key; (void)hash; (void)w; (void)h; (void)mb; (void)mn; (void)d; return -1; }
void nvd3d_triangle(const NvD3DVertex v[3], const NvD3DState *st) { (void)v; (void)st; }
int  nvd3d_vs_program(uint64_t hash, const char *hlsl, size_t len)
{ (void)hash; (void)hlsl; (void)len; return -1; }
int  nvd3d_ps_program(uint64_t hash, const char *hlsl, size_t len)
{ (void)hash; (void)hlsl; (void)len; return -1; }
void nvd3d_draw_program(const NvD3DProgramDraw *d, const NvD3DState *st) { (void)d; (void)st; }
uint32_t nvd3d_check_program(const NvD3DProgramDraw *d, NvD3DVertex *out, uint32_t max)
{ (void)d; (void)out; (void)max; return 0; }
void nvd3d_clear(uint32_t w, uint32_t a, float d, uint8_t s, const uint32_t r[4])
{ (void)w; (void)a; (void)d; (void)s; (void)r; }
void nvd3d_present(uint32_t b, const uint32_t r[4], uint32_t w, uint32_t h)
{ (void)b; (void)r; (void)w; (void)h; }
void nvd3d_readback(uint32_t b, uint8_t *d, uint32_t p, uint32_t r)
{ (void)b; (void)d; (void)p; (void)r; }

#endif
