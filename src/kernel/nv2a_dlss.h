/*
 * nv2a_dlss.h - DLSS through NVIDIA Streamline, for the D3D11 backend.
 *
 * DLSS step 4. The inputs are made in nv2a_d3d11.c and nv2a_pb_exec.c: the
 * scene without the interface (layer, point 150), its depth (copied when the
 * title clears it, point 152), motion vectors (points 153-154) and the jitter
 * (point 152). This only hands them to Streamline and gets the picture back.
 *
 * Streamline is loaded at run time (sl.interposer.dll, its signature checked
 * first) and integrated by "manual hooking": the device is ours, made as
 * always, and only the DXGI factory is upgraded to Streamline's proxy so that
 * the swap chain's Present goes through it, as the manual hooking guide
 * requires. Everything here is C; the Streamline headers are C++ and stay in
 * nv2a_dlss.cpp, which is only built when CMake is given RECOMP_STREAMLINE_DIR.
 * Without it nv2a_dlss_stub.c answers "not available" to everything.
 *
 * Opt-in with RECOMP_DLSS=1 (DLAA: the picture comes out at the size it went
 * in, anti-aliased). Off, nothing of Streamline is loaded.
 */
#ifndef NV2A_DLSS_H
#define NV2A_DLSS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Before the D3D11 device: load and initialise Streamline. 1 if it is up. */
int  nvdlss_preinit(void);

/* After the device: tell Streamline which one. 1 if DLSS can be used on it. */
int  nvdlss_set_device(void *d3d11_device);

/* Replace *factory (an IDXGIFactory*) with Streamline's proxy, before the
 * swap chain is made from it. 1 if it was replaced. */
int  nvdlss_upgrade_factory(void **factory);

/* 1 once DLSS can be evaluated. */
int  nvdlss_ready(void);

/* DLAA on one frame, on the immediate context. color: the scene (read),
 * depth: its depth, mv: motion vectors in uv (previous minus current), out:
 * a texture of the same size with UAV access, written. jitter in pixels of
 * that size. reset: the previous frame is no history (first frame, a cut).
 * 1 if out was written. The caller restores the context's state. */
int  nvdlss_evaluate(void *context, void *color, void *depth, void *mv, void *out,
                     uint32_t width, uint32_t height, float jitter_x, float jitter_y,
                     int reset);

/* What happened, for the log and the launcher: a short line. */
const char *nvdlss_status(void);

void nvdlss_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
