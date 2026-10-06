/*
 * nv2a_dlss_stub.c - DLSS when the toolkit is built without Streamline
 * (RECOMP_STREAMLINE_DIR not given to CMake). See nv2a_dlss.h.
 */
#include "nv2a_dlss.h"

int  nvdlss_built(void) { return 0; }
int  nvdlss_preinit(void) { return 0; }
int  nvdlss_set_device(void *d) { (void)d; return 0; }
int  nvdlss_upgrade_factory(void **f) { (void)f; return 0; }
int  nvdlss_ready(void) { return 0; }
int  nvdlss_evaluate(void *c, void *col, void *dep, void *mv, void *out, uint32_t w, uint32_t h,
                     float jx, float jy, int reset)
{
    (void)c; (void)col; (void)dep; (void)mv; (void)out; (void)w; (void)h;
    (void)jx; (void)jy; (void)reset;
    return 0;
}
const char *nvdlss_status(void) { return "built without Streamline (RECOMP_STREAMLINE_DIR)"; }
void nvdlss_shutdown(void) {}
