/*
 * nv2a_dlss.cpp - DLSS through NVIDIA Streamline (DLSS step 4). See
 * nv2a_dlss.h for what it does and why it is shaped this way.
 *
 * Built only when CMake is given RECOMP_STREAMLINE_DIR: the Streamline headers
 * (MIT) come from there, and sl.interposer.dll with the DLSS plugin is loaded
 * at run time from RECOMP_STREAMLINE_BIN, or from the SDK's bin/x64 the build
 * was configured with. Nothing is linked: the functions are taken from the
 * loaded DLL, so any compiler that builds the rest builds this.
 */
#include <windows.h>
#include <stdarg.h>
#include <d3d11.h>
#include <dxgi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sl.h"
#include "sl_dlss.h"
#include "sl_security.h"

#include "nv2a_dlss.h"

#ifndef RECOMP_STREAMLINE_BIN_DEFAULT
#define RECOMP_STREAMLINE_BIN_DEFAULT L""
#endif

namespace {

HMODULE s_mod;
int     s_up;                   /* slInit done */
int     s_ready;                /* device set and DLSS loaded on it */
char    s_status[256] = "off (RECOMP_DLSS not set)";
wchar_t s_bin[MAX_PATH];

PFun_slInit             *p_slInit;
PFun_slShutdown         *p_slShutdown;
PFun_slSetD3DDevice     *p_slSetD3DDevice;
PFun_slUpgradeInterface *p_slUpgradeInterface;
PFun_slIsFeatureLoaded  *p_slIsFeatureLoaded;
PFun_slGetFeatureFunction *p_slGetFeatureFunction;
PFun_slGetNewFrameToken *p_slGetNewFrameToken;
PFun_slSetTagForFrame   *p_slSetTagForFrame;
PFun_slSetConstants     *p_slSetConstants;
PFun_slEvaluateFeature  *p_slEvaluateFeature;
PFun_slDLSSSetOptions   *p_slDLSSSetOptions;

uint32_t s_opt_w, s_opt_h, s_frame;

void say(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s_status, sizeof s_status, fmt, ap);
    va_end(ap);
    fprintf(stderr, "  [DLSS] %s\n", s_status);
    fflush(stderr);
}

void log_cb(sl::LogType type, const char *msg)
{
    /* Streamline's own messages: warnings and errors always, the rest
     * under RECOMP_DLSS_LOG. */
    static int all = -1;
    if (all < 0)
        all = getenv("RECOMP_DLSS_LOG") != NULL;
    if (type != sl::LogType::eInfo || all)
        fprintf(stderr, "  [SL] %s%s", msg, (msg[0] && msg[strlen(msg) - 1] == '\n') ? "" : "\n");
}

template <typename T> bool take(T *&p, const char *name)
{
    p = reinterpret_cast<T *>(GetProcAddress(s_mod, name));
    return p != nullptr;
}

} // namespace

extern "C" int nvdlss_preinit(void)
{
    const char *e = getenv("RECOMP_DLSS");
    wchar_t dll[MAX_PATH + 32];

    if (!e || atoi(e) == 0)
        return 0;
    if (s_up)
        return 1;
    {
        const char *b = getenv("RECOMP_STREAMLINE_BIN");
        if (b && *b)
            MultiByteToWideChar(CP_UTF8, 0, b, -1, s_bin, MAX_PATH);
        else
            wcsncpy(s_bin, RECOMP_STREAMLINE_BIN_DEFAULT, MAX_PATH - 1);
    }
    if (!s_bin[0]) {
        say("no Streamline folder (RECOMP_STREAMLINE_BIN)");
        return 0;
    }
    _snwprintf(dll, MAX_PATH + 31, L"%s\\sl.interposer.dll", s_bin);
    dll[MAX_PATH + 31] = 0;
    /* The full path, as the guide asks: NVIDIA's signature checked before the
     * DLL is loaded at all. */
    if (!sl::security::verifyEmbeddedSignature(dll)) {
        say("sl.interposer.dll missing or not signed by NVIDIA in %ls", s_bin);
        return 0;
    }
    s_mod = LoadLibraryW(dll);
    if (!s_mod) {
        say("sl.interposer.dll did not load (error %lu)", GetLastError());
        return 0;
    }
    if (!take(p_slInit, "slInit") || !take(p_slShutdown, "slShutdown")
            || !take(p_slSetD3DDevice, "slSetD3DDevice")
            || !take(p_slUpgradeInterface, "slUpgradeInterface")
            || !take(p_slIsFeatureLoaded, "slIsFeatureLoaded")
            || !take(p_slGetFeatureFunction, "slGetFeatureFunction")
            || !take(p_slGetNewFrameToken, "slGetNewFrameToken")
            || !take(p_slSetTagForFrame, "slSetTagForFrame")
            || !take(p_slSetConstants, "slSetConstants")
            || !take(p_slEvaluateFeature, "slEvaluateFeature")) {
        say("sl.interposer.dll lacks an expected export");
        return 0;
    }
    {
        static const sl::Feature features[] = { sl::kFeatureDLSS };
        static const wchar_t *paths[1];
        sl::Preferences pref{};
        paths[0] = s_bin;
        pref.showConsole = false;
        pref.logLevel = getenv("RECOMP_DLSS_LOG") ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
        pref.pathsToPlugins = paths;
        pref.numPathsToPlugins = 1;
        pref.logMessageCallback = log_cb;
        /* Manual hooking: our device, Streamline only on the swap chain.
         * No over-the-air updates: the plugin is the one shipped next to it. */
        pref.flags = sl::PreferenceFlags::eDisableCLStateTracking
                   | sl::PreferenceFlags::eUseManualHooking
                   | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
        pref.featuresToLoad = features;
        pref.numFeaturesToLoad = 1;
        pref.engine = sl::EngineType::eCustom;
        pref.engineVersion = "xboxrecomp";
        pref.projectId = "6b1f3c2e-8d4a-4f5b-9c7e-2a1d0e3f4b5c";
        pref.renderAPI = sl::RenderAPI::eD3D11;
        sl::Result r = p_slInit(pref, sl::kSDKVersion);
        if (r != sl::Result::eOk) {
            say("slInit failed (result %d)", (int)r);
            return 0;
        }
    }
    s_up = 1;
    say("Streamline loaded from %ls", s_bin);
    return 1;
}

extern "C" int nvdlss_set_device(void *device)
{
    bool loaded = false;
    if (!s_up)
        return 0;
    sl::Result r = p_slSetD3DDevice(device);
    if (r != sl::Result::eOk) {
        say("slSetD3DDevice failed (result %d)", (int)r);
        return 0;
    }
    if (p_slIsFeatureLoaded(sl::kFeatureDLSS, loaded) != sl::Result::eOk || !loaded) {
        say("DLSS is not available on this GPU or driver (see the [SL] lines)");
        return 0;
    }
    if (p_slGetFeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions",
                               (void *&)p_slDLSSSetOptions) != sl::Result::eOk) {
        say("slDLSSSetOptions not found");
        return 0;
    }
    s_ready = 1;
    say("DLSS ready (DLAA)");
    return 1;
}

extern "C" int nvdlss_upgrade_factory(void **factory)
{
    if (!s_up || !factory || !*factory)
        return 0;
    if (p_slUpgradeInterface(factory) != sl::Result::eOk) {
        say("slUpgradeInterface on the DXGI factory failed");
        return 0;
    }
    return 1;
}

extern "C" int nvdlss_ready(void)
{
    return s_ready;
}

extern "C" int nvdlss_evaluate(void *context, void *color, void *depth, void *mv, void *out,
                               uint32_t w, uint32_t h, float jx, float jy, int reset)
{
    sl::FrameToken *token = nullptr;
    const sl::ViewportHandle viewport(0);
    sl::Result r;

    if (!s_ready)
        return 0;
    if (s_opt_w != w || s_opt_h != h) {
        sl::DLSSOptions o{};
        o.mode = sl::DLSSMode::eDLAA;
        o.outputWidth = w;
        o.outputHeight = h;
        o.colorBuffersHDR = sl::Boolean::eFalse;          /* the title's 8-bit picture */
        o.useAutoExposure = sl::Boolean::eFalse;
        if ((r = p_slDLSSSetOptions(viewport, o)) != sl::Result::eOk) {
            say("slDLSSSetOptions failed (result %d)", (int)r);
            s_ready = 0;
            return 0;
        }
        s_opt_w = w; s_opt_h = h;
        reset = 1;
    }
    {
        uint32_t index = s_frame++;
        if ((r = p_slGetNewFrameToken(token, &index)) != sl::Result::eOk || !token)
            return 0;
    }
    {
        /* Matrices are not used by DLSS super resolution when the motion
         * vectors carry the camera's motion; identity, as they must be set. */
        sl::Constants c{};
        for (int i = 0; i < 4; i++) {
            c.cameraViewToClip.row[i] = sl::float4(i == 0, i == 1, i == 2, i == 3);
            c.clipToCameraView.row[i] = c.cameraViewToClip.row[i];
            c.clipToLensClip.row[i] = c.cameraViewToClip.row[i];
            c.clipToPrevClip.row[i] = c.cameraViewToClip.row[i];
            c.prevClipToClip.row[i] = c.cameraViewToClip.row[i];
        }
        c.jitterOffset = sl::float2(jx, jy);
        c.mvecScale = sl::float2(1.0f, 1.0f);          /* already in uv */
        c.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
        c.cameraPos = sl::float3(0.0f, 0.0f, 0.0f);
        c.cameraUp = sl::float3(0.0f, 1.0f, 0.0f);
        c.cameraRight = sl::float3(1.0f, 0.0f, 0.0f);
        c.cameraFwd = sl::float3(0.0f, 0.0f, 1.0f);
        /* Not read by super resolution when the vectors carry the camera, but
         * validated: plausible values rather than "invalid" (point 155). */
        c.cameraNear = 0.1f;
        c.cameraFar = 1000.0f;
        c.cameraFOV = 1.0f;
        c.cameraAspectRatio = (float)w / (float)h;
        c.depthInverted = sl::Boolean::eFalse;          /* D24, near is low */
        c.cameraMotionIncluded = sl::Boolean::eTrue;
        c.motionVectors3D = sl::Boolean::eFalse;
        c.motionVectorsJittered = sl::Boolean::eFalse;  /* made unjittered, point 153 */
        c.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        if ((r = p_slSetConstants(c, *token, viewport)) != sl::Result::eOk) {
            say("slSetConstants failed (result %d)", (int)r);
            return 0;
        }
    }
    {
        sl::Resource rin(sl::ResourceType::eTex2d, color);
        sl::Resource rout(sl::ResourceType::eTex2d, out);
        sl::Resource rdepth(sl::ResourceType::eTex2d, depth);
        sl::Resource rmv(sl::ResourceType::eTex2d, mv);
        sl::Extent full{};
        full.width = w; full.height = h;
        /* The depth copy is typeless: say how to read it. */
        rdepth.nativeFormat = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        sl::ResourceTag tags[] = {
            sl::ResourceTag(&rin, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &full),
            sl::ResourceTag(&rout, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &full),
            sl::ResourceTag(&rdepth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &full),
            sl::ResourceTag(&rmv, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate, &full),
        };
        if ((r = p_slSetTagForFrame(*token, viewport, tags, 4, (sl::CommandBuffer *)context))
                != sl::Result::eOk) {
            say("slSetTagForFrame failed (result %d)", (int)r);
            return 0;
        }
    }
    {
        const sl::BaseStructure *inputs[] = { &viewport };
        if ((r = p_slEvaluateFeature(sl::kFeatureDLSS, *token, inputs, 1,
                                     (sl::CommandBuffer *)context)) != sl::Result::eOk) {
            static int said;
            if (said++ < 5)
                say("slEvaluateFeature failed (result %d)", (int)r);
            return 0;
        }
    }
    return 1;
}

extern "C" const char *nvdlss_status(void)
{
    return s_status;
}

extern "C" void nvdlss_shutdown(void)
{
    if (s_up && p_slShutdown)
        p_slShutdown();
    s_up = s_ready = 0;
}
