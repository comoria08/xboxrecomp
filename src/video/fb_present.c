/**
 * Show the guest framebuffer in a window.
 *
 * The title renders into its own framebuffer in guest RAM and tells the kernel
 * where it is through AvSetDisplayMode; on hardware the CRTC scans that memory
 * out. Nothing here scans anything out, so however much of the GPU is
 * implemented, none of it is observable. This is the other half: a window that
 * reads that memory and puts it on screen.
 *
 * Deliberately plain GDI rather than the D3D8 layer. The point is to display
 * whatever the guest actually wrote, so the fewer stages between guest memory
 * and the screen the better -- and it must keep working while the D3D8 layer
 * is busy with something else, such as the FMV player's own window.
 *
 * Off unless RECOMP_FB_WINDOW is set.
 */
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern ptrdiff_t xbox_GetMemoryOffset(void);

static volatile LONG s_fb_running;
static uint32_t      s_fb_va, s_fb_pitch, s_fb_width = 640, s_fb_height = 480;
static uint32_t     *s_rgb;           /* converted 32-bit copy for GDI */

/* Pixel size and multisample factor, when the caller knows them.
 *
 * Inferring the pixel size as pitch/width only holds for a surface whose row
 * is exactly one pixel per column. It is not true of a multisampled one:
 * Def Jam: Fight for NY renders 640x480 with 2x horizontal multisampling, so
 * its rows are 1280 samples of 4 bytes and pitch/width came out as 8 -- a size
 * the converter has no branch for, so it filled the window with black while
 * the title was rendering perfectly well. Zero means "not told", and the old
 * inference is used. */
static uint32_t      s_fb_bpp;        /* bytes per sample */
static uint32_t      s_fb_xscale = 1; /* samples per displayed pixel, across */

void xbox_FramebufferWindowFormat(uint32_t width, uint32_t height,
                                  uint32_t bpp, uint32_t xscale)
{
    if (width)  s_fb_width  = width;
    if (height) s_fb_height = height;
    s_fb_bpp    = bpp;
    s_fb_xscale = xscale ? xscale : 1;
}

void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch)
{
    /* RECOMP_FB_VA pins the window to one guest address instead of following
     * whichever surface is being drawn into. A black window cannot distinguish
     * "the read path is broken" from "the title rendered black", and pointing
     * it at memory known to have content settles that. */
    const char *pin = getenv("RECOMP_FB_VA");

    s_fb_va = pin ? (uint32_t)strtoul(pin, NULL, 0) : fb_va;
    if (pitch)
        s_fb_pitch = pitch;
}

/* The last completed frame.
 *
 * Reading the surface live shows whatever the title is in the middle of
 * drawing: a title that clears and redraws one buffer every frame shows up
 * half painted, with the triangle currently being filled cutting diagonally
 * across the picture. The executor latches a copy when the title swaps, and
 * the window shows that instead once one exists. */
static SRWLOCK       s_latch_lock = SRWLOCK_INIT;
static uint8_t      *s_latch;
static size_t        s_latch_size;
static volatile LONG s_latched;

/* What the title bar reports: frames the title swapped, and -- if the project
 * says what one is -- game logic updates, both counted since the last second. */
static volatile LONG s_flips;
static volatile LONG s_ticks;
static volatile LONG s_ticks_known;

void xbox_FramebufferWindowCountTick(void)
{
    InterlockedIncrement(&s_ticks);
    if (!s_ticks_known)
        InterlockedExchange(&s_ticks_known, 1);
}

/* Someone else presents (the D3D11 fill path draws on the GPU and shows the
 * result through its own swap chain). The window then only pumps messages,
 * and the guest framebuffer is only current when a screenshot asks for it:
 * the window requests pixels, the presenter reads them back into guest memory
 * and latches, and the window writes the file once that latch has happened. */
static HWND          s_hwnd;
static volatile LONG s_external;
static volatile LONG s_want_pixels, s_serving, s_served;

/* How many times the title's size the window opens at (the GPU path draws
 * at that internal resolution). Call before the window opens. */
static uint32_t s_win_scale = 1;

void xbox_FramebufferWindowScale(uint32_t scale)
{
    s_win_scale = scale ? scale : 1;
}

/* RECOMP_WIDESCREEN=1: the title's picture fills a 16:9 frame instead of its
 * own 4:3 one. That is the console's own widescreen -- the Xbox always scans
 * out a 4:3 buffer, and a title told the TV is widescreen renders an
 * anamorphic picture the set stretches back -- so the stretch belongs at
 * presentation and the title keeps rendering the size it always did.
 *
 * It only means anything paired with a projection told the same thing; on
 * its own it is a stretched 4:3 picture. 0 when off, so a caller can fall
 * back to the picture's own aspect. */
float xbox_WidescreenAspect(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RECOMP_WIDESCREEN");
        on = (e && *e && *e != '0') ? 1 : 0;
    }
    return on ? 16.0f / 9.0f : 0.0f;
}

/* F11: borderless full screen on the window's monitor, and back. */
static void fb_toggle_fullscreen(HWND hwnd)
{
    static int full;
    static WINDOWPLACEMENT wp = { sizeof(WINDOWPLACEMENT) };
    DWORD style = (DWORD)GetWindowLongPtrA(hwnd, GWL_STYLE);

    if (!full) {
        MONITORINFO mi = { sizeof(mi) };
        if (GetWindowPlacement(hwnd, &wp)
                && GetMonitorInfoA(MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY), &mi)) {
            SetWindowLongPtrA(hwnd, GWL_STYLE, (LONG_PTR)(style & ~WS_OVERLAPPEDWINDOW));
            SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                         mi.rcMonitor.right - mi.rcMonitor.left,
                         mi.rcMonitor.bottom - mi.rcMonitor.top,
                         SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
            full = 1;
        }
    } else {
        SetWindowLongPtrA(hwnd, GWL_STYLE, (LONG_PTR)(style | WS_OVERLAPPEDWINDOW));
        SetWindowPlacement(hwnd, &wp);
        SetWindowPos(hwnd, NULL, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER
                     | SWP_FRAMECHANGED);
        full = 0;
    }
}

void *xbox_FramebufferWindowHwnd(void)
{
    return s_hwnd;
}

void xbox_FramebufferWindowExternal(void)
{
    InterlockedExchange(&s_external, 1);
}

int xbox_FramebufferWindowPixelsWanted(void)
{
    if (InterlockedExchange(&s_want_pixels, 0)) {
        InterlockedExchange(&s_serving, 1);
        return 1;
    }
    return 0;
}

void xbox_FramebufferWindowLatch(void)
{
    size_t bytes;

    InterlockedIncrement(&s_flips);
    if (s_external && !InterlockedExchange(&s_serving, 0))
        return;
    if (!s_fb_va || !s_fb_pitch || !InterlockedCompareExchange(&s_fb_running, 1, 1))
        return;
    bytes = (size_t)s_fb_pitch * s_fb_height;
    AcquireSRWLockExclusive(&s_latch_lock);
    if (s_latch_size != bytes) {
        free(s_latch);
        s_latch = (uint8_t *)malloc(bytes);
        s_latch_size = s_latch ? bytes : 0;
    }
    if (s_latch) {
        memcpy(s_latch, (const uint8_t *)((uintptr_t)s_fb_va + xbox_GetMemoryOffset()),
               bytes);
        InterlockedExchange(&s_latched, 1);
    }
    ReleaseSRWLockExclusive(&s_latch_lock);
    if (s_external)
        InterlockedIncrement(&s_served);
}

static LRESULT CALLBACK fb_wndproc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    if (m == WM_SIZE) {
        static int said;
        if (said++ < 20)
            fprintf(stderr, "  [FBWIN] client %ux%u (%s)\n",
                    (unsigned)LOWORD(l), (unsigned)HIWORD(l),
                    w == SIZE_MAXIMIZED ? "maximized" : w == SIZE_MINIMIZED ? "minimized"
                    : "restored");
    }
    if (m == WM_CLOSE || m == WM_DESTROY) {
        /* Closing the window ends the run. The window belongs to the runtime
         * rather than to the title, so merely hiding it left the title going
         * and the next frame opened another one. exit() rather than
         * ExitProcess: the atexit handlers (the indirect-call dump) still
         * get to run. */
        InterlockedExchange(&s_fb_running, 0);
        fprintf(stderr, "  [FBWIN] window closed, exiting\n");
        fflush(stderr);
        exit(0);
    }
    return DefWindowProcA(h, m, w, l);
}

/* The display formats an Xbox front buffer is actually set to. The pitch says
 * how wide a row is in bytes, so pitch/width gives the pixel size; the exact
 * component layout only matters for 16-bit, where 5:6:5 and 1:5:5:5 differ. */
static void fb_convert(const uint8_t *src, uint32_t bpp)
{
    uint32_t x, y;

    for (y = 0; y < s_fb_height; y++) {
        const uint8_t *row = src + (size_t)y * s_fb_pitch;
        uint32_t *dst = s_rgb + (size_t)y * s_fb_width;

        if (bpp == 4 && s_fb_xscale > 1) {
            /* Resolve the multisamples the way the GPU's own resolve does:
             * average the group that makes up one displayed pixel. Taking the
             * first sample instead would alias every edge the title paid for
             * the extra samples to smooth. */
            const uint32_t *p = (const uint32_t *)row;
            uint32_t n = s_fb_xscale, i;
            for (x = 0; x < s_fb_width; x++) {
                uint32_t r = 0, g = 0, b = 0;
                for (i = 0; i < n; i++) {
                    uint32_t v = p[x * n + i];
                    r += (v >> 16) & 0xFF;
                    g += (v >>  8) & 0xFF;
                    b +=  v        & 0xFF;
                }
                dst[x] = ((r / n) << 16) | ((g / n) << 8) | (b / n);
            }
        } else if (bpp == 4) {
            memcpy(dst, row, (size_t)s_fb_width * 4);
        } else if (bpp == 2) {
            const uint16_t *p = (const uint16_t *)row;
            for (x = 0; x < s_fb_width; x++) {
                uint16_t v = p[x];
                uint32_t r = (uint32_t)((v >> 11) & 0x1F) * 255u / 31u;
                uint32_t g = (uint32_t)((v >>  5) & 0x3F) * 255u / 63u;
                uint32_t b = (uint32_t)( v        & 0x1F) * 255u / 31u;
                dst[x] = (r << 16) | (g << 8) | b;
            }
        } else {
            memset(dst, 0, (size_t)s_fb_width * 4);
        }
    }
}

/* Write what the window is currently showing to a 24-bit BMP.
 *
 * A black window is ambiguous: it means either that the read path is wrong or
 * that the title really did render black. Dumping the same converted pixels
 * the window draws settles which, and does it without a screenshot. */
int xbox_FramebufferDumpBmp(const char *path)
{
    FILE *f;
    uint32_t row = ((s_fb_width * 3u) + 3u) & ~3u;
    uint32_t img = row * s_fb_height, total = 54u + img, y, x;
    uint8_t hdr[54], *line;

    if (!s_rgb || !s_fb_va)
        return -1;
    f = fopen(path, "wb");
    if (!f)
        return -1;
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &total, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &s_fb_width, 4);
    memcpy(hdr + 22, &s_fb_height, 4);
    hdr[26] = 1; hdr[28] = 24;
    memcpy(hdr + 34, &img, 4);
    fwrite(hdr, 1, sizeof(hdr), f);

    line = (uint8_t *)calloc(1, row);
    for (y = 0; y < s_fb_height; y++) {
        const uint32_t *src = s_rgb + (size_t)(s_fb_height - 1 - y) * s_fb_width;
        for (x = 0; x < s_fb_width; x++) {
            line[x * 3 + 0] = (uint8_t)(src[x] & 0xFF);
            line[x * 3 + 1] = (uint8_t)((src[x] >> 8) & 0xFF);
            line[x * 3 + 2] = (uint8_t)((src[x] >> 16) & 0xFF);
        }
        fwrite(line, 1, row, f);
    }
    free(line);
    fclose(f);
    fprintf(stderr, "  [FBWIN] wrote %s (%ux%u from 0x%08X)\n",
            path, s_fb_width, s_fb_height, s_fb_va);
    return 0;
}

/* A name the project chose, which wins over the certificate's: some titles
 * carry only an abbreviation there (Def Jam: Fight for NY says "DJFFNY"). */
static wchar_t s_fb_name[64];

void xbox_FramebufferWindowSetName(const char *utf8)
{
    if (!utf8 || !MultiByteToWideChar(CP_UTF8, 0, utf8, -1, s_fb_name,
                                      (int)(sizeof s_fb_name / sizeof s_fb_name[0])))
        s_fb_name[0] = 0;
}

/* The title's name, from the certificate of the XBE the loader mapped at
 * 0x00010000: header +0x118 holds the certificate's address, and the
 * certificate carries the name at +0x0C, 40 UTF-16 characters. Empty if the
 * header does not look like one. */
static void fb_title_name(wchar_t *out, size_t cap)
{
    const uint8_t *hdr = (const uint8_t *)((uintptr_t)0x00010000u + xbox_GetMemoryOffset());
    uint32_t hdr_size, cert;
    size_t i;

    out[0] = 0;
    if (memcmp(hdr, "XBEH", 4) != 0)
        return;
    memcpy(&hdr_size, hdr + 0x108, 4);
    memcpy(&cert, hdr + 0x118, 4);
    if (hdr_size > 0x10000 || cert < 0x00010000u || cert + 0x0C + 80 > 0x00010000u + hdr_size)
        return;
    for (i = 0; i + 1 < cap && i < 40; i++) {
        uint16_t c;
        memcpy(&c, hdr + (cert - 0x00010000u) + 0x0C + i * 2, 2);
        if (!c)
            break;
        out[i] = (wchar_t)c;
    }
    out[i] = 0;
}

/* "<name> - <n> FPS - <n> updates/s", refreshed once a second. */
static void fb_update_title(HWND hwnd, const wchar_t *name)
{
    static ULONGLONG last;
    ULONGLONG now = GetTickCount64();
    wchar_t text[160];
    LONG flips, ticks;
    double secs;

    if (!last) {
        last = now;
        return;
    }
    if (now - last < 1000)
        return;
    secs  = (double)(now - last) / 1000.0;
    last  = now;
    flips = InterlockedExchange(&s_flips, 0);
    ticks = InterlockedExchange(&s_ticks, 0);
    if (s_ticks_known)
        swprintf(text, 160, L"%ls - %.0f FPS - %.0f updates/s",
                 name, flips / secs, ticks / secs);
    else
        swprintf(text, 160, L"%ls - %.0f FPS", name, flips / secs);
    SetWindowTextW(hwnd, text);
}

static DWORD WINAPI fb_thread(LPVOID unused)
{
    HWND hwnd;
    HDC hdc;
    BITMAPINFO bi;
    RECT r;
    wchar_t name[64];
    int win_x = CW_USEDEFAULT, win_y = CW_USEDEFAULT;

    (void)unused;
    if (s_fb_name[0])
        wcscpy(name, s_fb_name);
    else
        fb_title_name(name, sizeof name / sizeof name[0]);
    if (!name[0])
        wcscpy(name, L"Xbox Recomp");

    {
        WNDCLASSA wc;
        memset(&wc, 0, sizeof(wc));
        wc.lpfnWndProc   = fb_wndproc;
        wc.hInstance     = GetModuleHandleA(NULL);
        wc.hCursor       = LoadCursorA(NULL, IDC_ARROW);
        wc.lpszClassName = "XboxRecompFramebuffer";
        RegisterClassA(&wc);
    }
    {
        /* The title's size times the internal scale, shrunk (keeping the
         * aspect) to what fits the desktop work area, frame included. */
        RECT work;
        LONG cw = (LONG)(s_fb_width * s_win_scale), ch = (LONG)(s_fb_height * s_win_scale);
        const float ws = xbox_WidescreenAspect();
        RECT frame = { 0, 0, 0, 0 };
        /* In widescreen the picture is stretched to 16:9, so the window opens
         * at 16:9 as well: sized from the title's width it would still be a
         * 4:3 picture with bars, which is what the option exists to remove. */
        if (ws > 0.0f)
            cw = (LONG)(ch * ws + 0.5f);
        AdjustWindowRect(&frame, WS_OVERLAPPEDWINDOW, FALSE);
        if (SystemParametersInfoA(SPI_GETWORKAREA, 0, &work, 0)) {
            LONG maxw = (work.right - work.left) - (frame.right - frame.left);
            LONG maxh = (work.bottom - work.top) - (frame.bottom - frame.top);
            if (cw > maxw) { ch = ch * maxw / cw; cw = maxw; }
            if (ch > maxh) { cw = cw * maxh / ch; ch = maxh; }
            win_x = work.left + ((work.right - work.left) - (cw + frame.right - frame.left)) / 2;
            win_y = work.top;
        }
        r.left = 0; r.top = 0; r.right = cw; r.bottom = ch;
    }
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    hwnd = CreateWindowExA(0, "XboxRecompFramebuffer", "Xbox Recomp",
                           WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                           win_x, win_y,
                           r.right - r.left, r.bottom - r.top,
                           NULL, NULL, GetModuleHandleA(NULL), NULL);
    if (!hwnd) {
        InterlockedExchange(&s_fb_running, 0);
        return 0;
    }
    hdc = GetDC(hwnd);
    SetWindowTextW(hwnd, name);
    s_hwnd = hwnd;

    memset(&bi, 0, sizeof(bi));
    bi.bmiHeader.biSize        = sizeof(bi.bmiHeader);
    bi.bmiHeader.biWidth       = (LONG)s_fb_width;
    bi.bmiHeader.biHeight      = -(LONG)s_fb_height;   /* top-down */
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    s_rgb = (uint32_t *)calloc((size_t)s_fb_width * s_fb_height, 4);

    fprintf(stderr, "  [FBWIN] framebuffer window open (%ux%u)\n",
            s_fb_width, s_fb_height);

    while (InterlockedCompareExchange(&s_fb_running, 1, 1)) {
        MSG msg;
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        fb_update_title(hwnd, name);
        if ((GetAsyncKeyState(VK_F11) & 1) && GetForegroundWindow() == hwnd)
            fb_toggle_fullscreen(hwnd);
        if (s_fb_va && s_fb_pitch && s_rgb && !s_external) {
            const uint8_t *src =
                (const uint8_t *)((uintptr_t)s_fb_va + xbox_GetMemoryOffset());
            uint32_t bpp = s_fb_bpp ? s_fb_bpp : s_fb_pitch / s_fb_width;
            if (InterlockedCompareExchange(&s_latched, 1, 1)) {
                AcquireSRWLockShared(&s_latch_lock);
                if (s_latch && s_latch_size == (size_t)s_fb_pitch * s_fb_height)
                    fb_convert(s_latch, bpp);
                ReleaseSRWLockShared(&s_latch_lock);
            } else {
                fb_convert(src, bpp);
            }
            StretchDIBits(hdc, 0, 0, (int)s_fb_width, (int)s_fb_height,
                          0, 0, (int)s_fb_width, (int)s_fb_height,
                          s_rgb, &bi, DIB_RGB_COLORS, SRCCOPY);
        }
        {
            /* One dump, a few seconds in, so the title has had time to render
             * something rather than catching the first blank frame. */
            const char *dump = getenv("RECOMP_FB_DUMP");
            const char *snap = getenv("RECOMP_FB_SNAP");
            const char *every = getenv("RECOMP_FB_SNAP_EVERY");
            static int frames;
            static char pending[512];
            static LONG served_at;
            char path[512];
            path[0] = 0;
            ++frames;
            if (dump && frames == 600 && !s_external)
                xbox_FramebufferDumpBmp(dump);
            /* RECOMP_FB_SNAP=<prefix> with RECOMP_FB_SNAP_EVERY=N: the window
             * as <prefix>NNN.bmp every N frames, to follow a title past its
             * first screen. Separate from RECOMP_FB_DUMP, which also makes the
             * rasteriser write a file per frame and slows the title down. */
            /* F10: a snapshot on demand, numbered separately. */
            if ((GetAsyncKeyState(VK_F10) & 1) && GetForegroundWindow() == hwnd) {
                static int f10;
                snprintf(path, sizeof path, "%sF10_%03d.bmp",
                         snap ? snap : "capture_", ++f10);
            } else if (snap && every && atoi(every) > 0 && frames % atoi(every) == 0) {
                snprintf(path, sizeof path, "%s%03d.bmp", snap,
                         frames / atoi(every));
            }
            if (path[0]) {
                if (!s_external) {
                    xbox_FramebufferDumpBmp(path);
                } else if (!pending[0]) {
                    /* Ask the presenter for pixels; written once served. */
                    strcpy(pending, path);
                    served_at = s_served;
                    InterlockedExchange(&s_want_pixels, 1);
                }
            }
            if (pending[0] && s_served != served_at && s_fb_va && s_fb_pitch && s_rgb) {
                uint32_t bpp = s_fb_bpp ? s_fb_bpp : s_fb_pitch / s_fb_width;
                AcquireSRWLockShared(&s_latch_lock);
                if (s_latch && s_latch_size == (size_t)s_fb_pitch * s_fb_height)
                    fb_convert(s_latch, bpp);
                ReleaseSRWLockShared(&s_latch_lock);
                xbox_FramebufferDumpBmp(pending);
                pending[0] = 0;
            }
        }
        Sleep(16);
    }

    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);
    free(s_rgb);
    s_rgb = NULL;
    return 0;
}

void xbox_FramebufferWindowStart(void)
{
    HANDLE th;

    if (!getenv("RECOMP_FB_WINDOW"))
        return;
    if (InterlockedCompareExchange(&s_fb_running, 1, 0) != 0)
        return;
    /* Real pixels. Without this, on a display scaled to 150% Windows gives
     * the window a smaller logical size and stretches its image, so a
     * 1920x1440 internal resolution reached the screen blurred. Fails
     * harmlessly if the process already chose. */
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    th = CreateThread(NULL, 0, fb_thread, NULL, 0, NULL);
    if (th)
        CloseHandle(th);
    else
        InterlockedExchange(&s_fb_running, 0);
}

#else
void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch) { (void)fb_va; (void)pitch; }
void xbox_FramebufferWindowFormat(uint32_t width, uint32_t height,
                                  uint32_t bpp, uint32_t xscale)
{ (void)width; (void)height; (void)bpp; (void)xscale; }
void xbox_FramebufferWindowStart(void) {}
void xbox_FramebufferWindowLatch(void) {}
void xbox_FramebufferWindowCountTick(void) {}
void xbox_FramebufferWindowSetName(const char *utf8) { (void)utf8; }
void *xbox_FramebufferWindowHwnd(void) { return 0; }
void xbox_FramebufferWindowScale(uint32_t scale) { (void)scale; }
void xbox_FramebufferWindowExternal(void) {}
int xbox_FramebufferWindowPixelsWanted(void) { return 0; }
#endif
