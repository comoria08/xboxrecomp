/* dsound_hle_sortie.c - la sortie du son HLE : une file entre le mixeur, qui
 * suit le temps emule, et XAudio2, qui suit la carte son.
 *
 * Les deux horloges derivent l'une de l'autre (quelques dizaines de ppm, plus
 * les a-coups du fil de pompe). Le titre ne doit voir que la premiere, donc
 * c'est la sortie qui s'adapte : la cadence de la voix XAudio2 est corrigee
 * d'au plus 0,5 % selon le remplissage de la file (SetFrequencyRatio, que
 * XAudio2 reechantillonne lui-meme). Une file qui deborde (sortie arretee)
 * perd ses trames les plus vieilles ; une file vide joue du silence, compte.
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsound_hle.h"

#define COBJMACROS
#include <xaudio2.h>
#if defined(_MSC_VER)
#pragma comment(lib, "xaudio2.lib")
#pragma comment(lib, "ole32.lib")
#endif

#define TAUX         48000
#define FILE_MAX     (TAUX / 2)      /* 500 ms */
#define FILE_CIBLE   960             /* 20 ms attendues dans la file */
#define MORCEAU      480             /* 10 ms par tampon XAudio2 */
#define NB_TAMPONS_XA 16
#define EN_VOL       4               /* tampons soumis a la carte, vises */

static int16_t           g_file[FILE_MAX][2];
static uint32_t          g_tete, g_queue, g_plein;   /* ecriture, lecture, nombre */
static CRITICAL_SECTION  g_fcs;
static volatile LONG     g_pret;
static uint64_t          g_perdues, g_silences;

void dsh_sortie_ecrire(const int16_t *trames, int n)
{
    int k;
    if (!g_pret) return;
    EnterCriticalSection(&g_fcs);
    for (k = 0; k < n; k++) {
        if (g_plein == FILE_MAX) {           /* la sortie ne suit plus */
            g_queue = (g_queue + 1) % FILE_MAX;
            g_plein--;
            g_perdues++;
        }
        g_file[g_tete][0] = trames[2 * k];
        g_file[g_tete][1] = trames[2 * k + 1];
        g_tete = (g_tete + 1) % FILE_MAX;
        g_plein++;
    }
    LeaveCriticalSection(&g_fcs);
}

/* Prend jusqu'a n trames ; rend le nombre pris. */
static int file_lire(int16_t (*dst)[2], int n)
{
    int k;
    EnterCriticalSection(&g_fcs);
    if ((uint32_t)n > g_plein) n = (int)g_plein;
    for (k = 0; k < n; k++) {
        dst[k][0] = g_file[g_queue][0];
        dst[k][1] = g_file[g_queue][1];
        g_queue = (g_queue + 1) % FILE_MAX;
    }
    g_plein -= (uint32_t)n;
    LeaveCriticalSection(&g_fcs);
    return n;
}

static uint32_t file_remplie(void)
{
    uint32_t n;
    EnterCriticalSection(&g_fcs);
    n = g_plein;
    LeaveCriticalSection(&g_fcs);
    return n;
}

static DWORD WINAPI fil_sortie(LPVOID inutile)
{
    static int16_t tampons[NB_TAMPONS_XA][MORCEAU][2];
    IXAudio2 *xa = NULL;
    IXAudio2MasteringVoice *maitre = NULL;
    IXAudio2SourceVoice *voix = NULL;
    WAVEFORMATEX wfx;
    HRESULT hr;
    int prochain = 0, demarre = 0;
    float rapport = 1.0f, moyenne = FILE_CIBLE;
    ULONGLONG trace = 0;
    (void)inutile;

    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    (void)hr;
    hr = XAudio2Create(&xa, 0, XAUDIO2_DEFAULT_PROCESSOR);
    if (FAILED(hr) || !xa) {
        fprintf(stderr, "[SON] XAudio2Create a echoue (0x%08lX) : pas de son\n", (unsigned long)hr);
        return 0;
    }
    hr = IXAudio2_CreateMasteringVoice(xa, &maitre, 2, TAUX, 0, NULL, NULL, 0);
    if (FAILED(hr)) {
        fprintf(stderr, "[SON] voix maitresse refusee (0x%08lX) : pas de son\n", (unsigned long)hr);
        return 0;
    }
    memset(&wfx, 0, sizeof wfx);
    wfx.wFormatTag = WAVE_FORMAT_PCM;
    wfx.nChannels = 2;
    wfx.nSamplesPerSec = TAUX;
    wfx.wBitsPerSample = 16;
    wfx.nBlockAlign = 4;
    wfx.nAvgBytesPerSec = TAUX * 4;
    hr = IXAudio2_CreateSourceVoice(xa, &voix, &wfx, 0, XAUDIO2_DEFAULT_FREQ_RATIO, NULL, NULL, NULL);
    if (FAILED(hr)) {
        fprintf(stderr, "[SON] voix source refusee (0x%08lX) : pas de son\n", (unsigned long)hr);
        return 0;
    }
    IXAudio2SourceVoice_Start(voix, 0, XAUDIO2_COMMIT_NOW);
    fprintf(stderr, "[SON] XAudio2 : 48 kHz stereo, tampons de %d trames, %d en vol\n",
            MORCEAU, EN_VOL);
    fflush(stderr);

    for (;;) {
        XAUDIO2_VOICE_STATE etat;
        uint32_t f = file_remplie();

        if (!demarre) {                       /* attendre d'avoir de quoi tenir */
            if (f < FILE_CIBLE + MORCEAU * EN_VOL) { Sleep(1); continue; }
            demarre = 1;
        }

        IXAudio2SourceVoice_GetState(voix, &etat, XAUDIO2_VOICE_NOSAMPLESPLAYED);
        while (etat.BuffersQueued < EN_VOL) {
            XAUDIO2_BUFFER b;
            int n;
            if (file_remplie() < MORCEAU && etat.BuffersQueued >= 2)
                break;                     /* encore de la marge : attendre le mixeur */
            n = file_lire(tampons[prochain], MORCEAU);
            if (n < MORCEAU) {             /* la carte allait manquer : silence */
                memset(tampons[prochain][n], 0, (size_t)(MORCEAU - n) * 4);
                g_silences += (uint64_t)(MORCEAU - n);
            }
            memset(&b, 0, sizeof b);
            b.AudioBytes = MORCEAU * 4;
            b.pAudioData = (const BYTE *)tampons[prochain];
            if (FAILED(IXAudio2SourceVoice_SubmitSourceBuffer(voix, &b, NULL))) break;
            prochain = (prochain + 1) % NB_TAMPONS_XA;
            etat.BuffersQueued++;
        }

        /* La derive : une file plus pleine que la cible veut une carte un peu
         * plus rapide, et inversement. Moyenne lente pour ne pas faire
         * chevroter le son. */
        moyenne += (file_remplie() - moyenne) * 0.01f;
        {
            float e = (moyenne - FILE_CIBLE) / FILE_CIBLE;
            float r;
            if (e > 1.0f) e = 1.0f;
            if (e < -1.0f) e = -1.0f;
            r = 1.0f + 0.005f * e;
            if (r - rapport > 0.0001f || rapport - r > 0.0001f) {
                rapport = r;
                IXAudio2SourceVoice_SetFrequencyRatio(voix, rapport, XAUDIO2_COMMIT_NOW);
            }
        }

        if (getenv("RECOMP_SON_HLE_TRACE") && GetTickCount64() >= trace) {
            trace = GetTickCount64() + 1000;
            fprintf(stderr, "[SON] sortie : file %u trames (moyenne %.0f), cadence x%.4f,"
                    " %llu trames de silence, %llu perdues\n", file_remplie(), moyenne,
                    rapport, (unsigned long long)g_silences, (unsigned long long)g_perdues);
        }
        Sleep(2);
    }
    return 0;
}

int dsh_sortie_init(void)
{
    InitializeCriticalSection(&g_fcs);
    InterlockedExchange(&g_pret, 1);
    CloseHandle(CreateThread(NULL, 0, fil_sortie, NULL, 0, NULL));
    return 1;
}
