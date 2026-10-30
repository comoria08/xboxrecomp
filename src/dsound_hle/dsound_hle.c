/* dsound_hle.c - DirectSound de l'XDK 5849 en haut niveau. Voir dsound_hle.h.
 *
 * Ecrit pour Def Jam: Fight for NY (defjam_recomp, GUIDE_FR.md points 166 et
 * suivants), a partir de mesures sur ce titre, pas d'une autre implementation.
 * Ce que le moteur reproduit de la chaine d'origine (DirectSound programmant
 * le VP de l'APU), mesure registre par registre :
 *
 *  - volume d'une mixbin : DirectSound ecrit au VP (headroom - lVolume) * 64 /
 *    100, en 1/64 de dB d'attenuation, tronque ; 4095 coupe. Le headroom d'un
 *    tampon vaut 600 (6 dB) : le tampon principal arrive au VP a 384 ;
 *  - hauteur : 4096 * log2(f / 48000) au VP ;
 *  - filtre (SetFilter, mode 1) : le mode passe tel quel, FC0 et FC1 sont les
 *    deux premiers coefficients ;
 *  - curseurs : la lecture publiee a 64 trames d'avance sur la position reelle
 *    (rustine du VP contre le gresillement, point 78), l'ecriture vaut la
 *    lecture plus max(32 trames, 1 trame) (FUN_0026415B, champ +0x14 = 12
 *    octets pour le tampon principal) ;
 *  - ADPCM Xbox : IMA, blocs de 36 octets par canal, 64 trames par bloc
 *    (l'echantillon d'en-tete, puis les quartets) ;
 *  - repli 5.1 vers stereo : avant + 0,707 x (centre + LFE + arriere), le
 *    repli du passage direct de l'APU (apu_dsp.c, RECOMP_APU_DOWNMIX) ;
 *  - la mixbin 11 (envoi vers la reverberation du DSP) n'est pas rendue :
 *    le DSP ne tournait pas non plus.
 *
 * Deux ecarts voulus avec la chaine d'origine, a juger a l'oreille :
 *  - une voix non bouclee qui a fini rend GetStatus = 0, comme sur la console
 *    (avant, 1 pour toujours : l'APU emule ne prevenait jamais DirectSound) ;
 *  - la hauteur est appliquee (le VP emule lisait toutes les voix a 48 kHz).
 *    RECOMP_SON_HLE_48K=1 rend l'ancien comportement, pour comparer.
 *
 * Fils : les appels du titre (fil son d'EA, fil principal, chargement) et un
 * fil de pompe prennent le meme verrou ; chaque appel commence par rendre le
 * son jusqu'a l'instant emule courant, puis agit. La sortie (XAudio2) est dans
 * dsound_hle_sortie.c et ne voit que des trames deja rendues.
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dsound_hle.h"
#include "kernel/xbox_memory_layout.h"   /* RECOMP_TLS */

extern RECOMP_TLS uint32_t g_eax, g_esp;
extern ptrdiff_t g_xbox_mem_offset;

/* ── Acces invite ─────────────────────────────────────────────────────────── */

#define VA_PTR(va)  ((uint8_t *)((uintptr_t)(va) + g_xbox_mem_offset))
#define M32(va)     (*(volatile uint32_t *)VA_PTR(va))
#define M16(va)     (*(volatile uint16_t *)VA_PTR(va))
#define ARG(k)      M32(g_esp + 4u * (k))
/* stdcall : rendre hr et depiler l'adresse de retour et n arguments. */
#define RETOUR(hr, n)  do { g_eax = (uint32_t)(hr); g_esp += 4u + 4u * (n); } while (0)

/* La memoire du titre : la RAM et sa fenetre contigue en 0x80000000. */
static int lisible(uint32_t va, uint32_t n)
{
    if (va >= 0x00010000u && (uint64_t)va + n <= 0x04000000u) return 1;
    if (va >= 0x80000000u && (uint64_t)va + n <= 0x84000000u) return 1;
    return 0;
}

#define DS_OK              0x00000000u
#define DSERR_INVALIDCALL  0x88780032u
#define DSERR_OUTOFMEMORY  0x8007000Eu

#define DSBPLAY_LOOPING        1u
#define DSBPAUSE_RESUME        0u
#define DSBSTOPEX_ENVELOPE     1u
#define DSBSTOPEX_RELEASEWAVEFORM 2u
#define DSBSTATUS_PLAYING      1u
#define DSBSTATUS_PAUSED       2u
#define DSBSTATUS_LOOPING      4u

/* ── Etat ─────────────────────────────────────────────────────────────────── */

#define TAUX         48000
#define NB_TAMPONS   512
#define NB_PAIRES    8
#define BLOC         256             /* trames rendues d'un coup, au plus */
#define AVANCE_LECTURE  64           /* trames, voir l'en-tete */
#define AVANCE_ECRITURE 32

/* Jetons rendus au titre : hors de toute memoire invitee, pour qu'un titre
 * qui les dereferencerait faute tout de suite au lieu de lire n'importe quoi. */
#define JETON_DS        0xD5000000u
#define JETON_TAMPON0   0xD5100000u
#define JETON_PAS       0x40u

typedef struct {
    int      utilise;
    uint32_t ref;
    /* format */
    int      canaux, bits, adpcm;
    uint32_t taux;                   /* nSamplesPerSec */
    uint32_t align;                  /* PCM : octets par trame ; ADPCM : 36 x canaux par bloc */
    /* donnees (SetBufferData), en trames */
    uint32_t va, octets, trames;
    uint32_t jeu_debut, jeu_fin;
    uint32_t boucle_debut, boucle_fin;
    double   pos;
    int      etat;                   /* 0 arrete, 1 joue, 2 en pause */
    int      boucle;
    uint32_t freq;                   /* 0 : le taux du format */
    int32_t  headroom;               /* centiemes de dB */
    /* routage : paire p -> canal p % canaux */
    int      npaires;
    uint32_t bin[NB_PAIRES];
    int32_t  vol[NB_PAIRES];
    float    gain[NB_PAIRES];
    /* filtre passe-bas (mode 1) */
    int      filtre;
    float    f, q, qn;
    float    bas[6], bande[6];
    /* bloc ADPCM decode */
    int32_t  bloc_idx;
    int16_t  bloc[65 * 6];
} tampon_t;

static tampon_t          g_t[NB_TAMPONS];
static CRITICAL_SECTION  g_cs;
static uint32_t          g_ds_ref;
static uint8_t           g_mbhr[32];        /* SetMixBinHeadroom */
static uint64_t          g_rendu;           /* trames rendues depuis le depart */
static LARGE_INTEGER     g_qpc0, g_qpcf;
static int               g_48k;             /* RECOMP_SON_HLE_48K */
static int               g_trace;           /* RECOMP_SON_HLE_TRACE */
static volatile LONG     g_acces_apu;

void dsound_hle_acces_apu(uint32_t va)
{
    if (InterlockedIncrement(&g_acces_apu) <= 5) {
        fprintf(stderr, "[SON] acces a la fenetre de l'APU (0x%08X) : quelque chose"
                " appelle encore le DirectSound d'origine\n", va);
        fflush(stderr);
    }
}

/* ── Le temps emule ───────────────────────────────────────────────────────
 *
 * Le seul endroit qui dit quelle heure il est pour le son. Aujourd'hui, la
 * meme horloge que KeQueryPerformanceCounter du noyau (kernel_hal.c) ; pour un
 * mode en ligne, c'est ici qu'on la tirera du numero de tick. */
static uint64_t trames_emulees(void)
{
    LARGE_INTEGER t;
    uint64_t d;
    QueryPerformanceCounter(&t);
    d = (uint64_t)(t.QuadPart - g_qpc0.QuadPart);
    return (d / (uint64_t)g_qpcf.QuadPart) * TAUX
         + (d % (uint64_t)g_qpcf.QuadPart) * TAUX / (uint64_t)g_qpcf.QuadPart;
}

/* ── Enregistrement (RECOMP_APU_WAV, meme fichier que l'ancien chemin) ───── */

static FILE    *g_wav;
static uint32_t g_wav_octets;

static void wav_fermer(void)
{
    uint32_t riff = 36 + g_wav_octets, data = g_wav_octets;
    if (!g_wav) return;
    fseek(g_wav, 4, SEEK_SET);  fwrite(&riff, 4, 1, g_wav);
    fseek(g_wav, 40, SEEK_SET); fwrite(&data, 4, 1, g_wav);
    fclose(g_wav);
    g_wav = NULL;
}

static void wav_ouvrir(void)
{
    static const uint8_t entete[44] = {
        'R','I','F','F', 0,0,0,0, 'W','A','V','E',
        'f','m','t',' ', 16,0,0,0, 1,0, 2,0,
        0x80,0xBB,0,0, 0x00,0xEE,0x02,0, 4,0, 16,0,
        'd','a','t','a', 0,0,0,0
    };
    const char *chemin = getenv("RECOMP_APU_WAV");
    if (!chemin || !*chemin) return;
    g_wav = fopen(chemin, "wb");
    if (!g_wav) return;
    fwrite(entete, sizeof entete, 1, g_wav);
    atexit(wav_fermer);
    fprintf(stderr, "[SON] enregistrement dans %s\n", chemin);
}

/* ── Volumes ──────────────────────────────────────────────────────────────── */

static float gain_vp(int32_t headroom, int32_t vol, uint32_t bin)
{
    int32_t vp = (int32_t)(((int64_t)headroom - vol) * 64 / 100);
    float g;
    if (vp < 0) vp = 0;
    if (vp >= 4095) return 0.0f;
    g = powf(10.0f, (float)vp / (64.0f * -20.0f));
    if (bin < 32) g /= (float)(1u << g_mbhr[bin]);
    return g;
}

static void calculer_gains(tampon_t *t)
{
    int p;
    for (p = 0; p < t->npaires; p++)
        t->gain[p] = gain_vp(t->headroom, t->vol[p], t->bin[p]);
}

/* ── ADPCM Xbox (IMA) ─────────────────────────────────────────────────────── */

static const int16_t k_pas[89] = {
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31,
    34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658,
    724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
    3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
};
static const int8_t k_index[16] = { -1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8 };

/* Un bloc : par canal 4 octets d'en-tete (echantillon, index, 0), puis par
 * groupes de 4 octets (8 quartets, poids faible d'abord) canal par canal.
 * Rend 65 trames ; le VP n'en lit que les 64 premieres. */
static void adpcm_bloc(const uint8_t *in, int canaux, int16_t *out)
{
    int32_t pred[6];
    int idx[6], c, g, o, q;

    for (c = 0; c < canaux; c++) {
        pred[c] = (int16_t)(in[0] | (in[1] << 8));
        idx[c] = in[2] > 88 ? 88 : in[2];
        out[c] = (int16_t)pred[c];
        in += 4;
    }
    for (g = 0; g < 8; g++)
        for (c = 0; c < canaux; c++)
            for (o = 0; o < 4; o++, in++)
                for (q = 0; q < 2; q++) {
                    int n = q ? (*in >> 4) : (*in & 15);
                    int pas = k_pas[idx[c]];
                    int d = pas >> 3;
                    if (n & 4) d += pas;
                    if (n & 2) d += pas >> 1;
                    if (n & 1) d += pas >> 2;
                    pred[c] += (n & 8) ? -d : d;
                    if (pred[c] > 32767) pred[c] = 32767;
                    if (pred[c] < -32768) pred[c] = -32768;
                    idx[c] += k_index[n];
                    if (idx[c] < 0) idx[c] = 0;
                    if (idx[c] > 88) idx[c] = 88;
                    out[(1 + g * 8 + o * 2 + q) * canaux + c] = (int16_t)pred[c];
                }
}

/* ── Lecture et mixage ────────────────────────────────────────────────────── */

static void lire_trame(tampon_t *t, uint32_t i, float *s)
{
    int c;
    if (i >= t->trames) {
        for (c = 0; c < t->canaux; c++) s[c] = 0.0f;
        return;
    }
    if (t->adpcm) {
        int32_t b = (int32_t)(i / 64);
        if (b != t->bloc_idx) {
            adpcm_bloc(VA_PTR(t->va + (uint32_t)b * t->align), t->canaux, t->bloc);
            t->bloc_idx = b;
        }
        for (c = 0; c < t->canaux; c++)
            s[c] = t->bloc[(i % 64) * t->canaux + c] / 32768.0f;
    } else if (t->bits == 16) {
        const uint8_t *p = VA_PTR(t->va + i * t->align);
        for (c = 0; c < t->canaux; c++) {
            int16_t v;
            memcpy(&v, p + 2 * c, 2);
            s[c] = v / 32768.0f;
        }
    } else {
        const uint8_t *p = VA_PTR(t->va + i * t->align);
        for (c = 0; c < t->canaux; c++)
            s[c] = ((int)p[c] - 128) / 128.0f;
    }
}

static void mixer_tampon(tampon_t *t, float bins[6][BLOC], int n)
{
    double pas = g_48k ? 1.0 : (double)(t->freq ? t->freq : t->taux) / TAUX;
    float s0[6], s1[6];
    int k, c, p;

    t->bloc_idx = -1;          /* le titre a pu reecrire ses donnees */
    for (k = 0; k < n; k++) {
        uint32_t fin = t->boucle ? t->boucle_fin : t->jeu_fin;
        uint32_t i;
        double fr;

        if (t->pos >= fin) {
            if (t->boucle && t->boucle_fin > t->boucle_debut) {
                while (t->pos >= t->boucle_fin)
                    t->pos -= t->boucle_fin - t->boucle_debut;
            } else {
                t->pos = fin;
                t->etat = 0;   /* fini : GetStatus rend 0 */
                return;
            }
        }
        i = (uint32_t)t->pos;
        fr = t->pos - i;
        lire_trame(t, i, s0);
        if (fr > 0.0) {
            uint32_t j = i + 1;
            if (j >= fin) j = t->boucle ? t->boucle_debut : i;
            lire_trame(t, j, s1);
            for (c = 0; c < t->canaux; c++)
                s0[c] += (s1[c] - s0[c]) * (float)fr;
        }
        if (t->filtre) {
            /* Variables d'etat de Chamberlin, passe-bas, un pas par trame,
             * coefficients comme le VP les tire de FC0/FC1. */
            for (c = 0; c < t->canaux; c++) {
                float h, y;
                t->bande[c] -= t->bande[c] * t->bande[c] * t->bande[c] * 0.001f;
                h = s0[c] * t->qn - t->bas[c] - t->q * t->bande[c];
                t->bande[c] += t->f * h;
                t->bas[c] += t->f * t->bande[c];
                y = t->bas[c];
                s0[c] = y > 1.0f ? 1.0f : y < -1.0f ? -1.0f : y;
            }
        }
        for (p = 0; p < t->npaires; p++)
            if (t->bin[p] < 6 && t->gain[p] > 0.0f)
                bins[t->bin[p]][k] += t->gain[p] * s0[p % t->canaux];
        t->pos += pas;
    }
}

static void rendre(int n)
{
    static float bins[6][BLOC];
    static int16_t sortie[BLOC][2];
    const float r3 = 0.70710678f;
    int i, k;

    memset(bins, 0, sizeof bins);
    for (i = 0; i < NB_TAMPONS; i++) {
        tampon_t *t = &g_t[i];
        if (t->utilise && t->etat == 1 && t->trames)
            mixer_tampon(t, bins, n);
    }
    for (k = 0; k < n; k++) {
        float g = bins[0][k] + r3 * (bins[2][k] + bins[3][k] + bins[4][k]);
        float d = bins[1][k] + r3 * (bins[2][k] + bins[3][k] + bins[5][k]);
        if (g > 1.0f) g = 1.0f; else if (g < -1.0f) g = -1.0f;
        if (d > 1.0f) d = 1.0f; else if (d < -1.0f) d = -1.0f;
        sortie[k][0] = (int16_t)(g * 32767.0f);
        sortie[k][1] = (int16_t)(d * 32767.0f);
    }
    if (g_wav) {
        fwrite(sortie, 4, (size_t)n, g_wav);
        g_wav_octets += (uint32_t)n * 4;
    }
    dsh_sortie_ecrire(&sortie[0][0], n);
    g_rendu += (uint64_t)n;
}

/* Rend jusqu'a l'instant emule. Verrou tenu. */
static void avancer(void)
{
    uint64_t cible = trames_emulees();
    while (g_rendu < cible) {
        uint64_t n = cible - g_rendu;
        rendre(n > BLOC ? BLOC : (int)n);
    }
}

/* ── Demarrage, pompe, trace ──────────────────────────────────────────────── */

static DWORD WINAPI fil_pompe(LPVOID inutile)
{
    uint64_t prochain = 0;
    (void)inutile;
    for (;;) {
        Sleep(2);
        EnterCriticalSection(&g_cs);
        avancer();
        if (g_trace && g_rendu >= prochain) {
            int i, jouent = 0, pause = 0;
            prochain = g_rendu + TAUX;
            for (i = 0; i < NB_TAMPONS; i++) {
                jouent += g_t[i].utilise && g_t[i].etat == 1;
                pause += g_t[i].utilise && g_t[i].etat == 2;
            }
            fprintf(stderr, "[SON] t=%.1f s : %d voix jouent, %d en pause\n",
                    (double)g_rendu / TAUX, jouent, pause);
        }
        LeaveCriticalSection(&g_cs);
    }
    return 0;
}

static BOOL CALLBACK demarrer_une_fois(PINIT_ONCE o, PVOID p, PVOID *c)
{
    const char *s;
    (void)o; (void)p; (void)c;
    InitializeCriticalSection(&g_cs);
    s = getenv("RECOMP_SON_HLE_48K");
    g_48k = s && *s && atoi(s) != 0;
    g_trace = getenv("RECOMP_SON_HLE_TRACE") != NULL;
    QueryPerformanceFrequency(&g_qpcf);
    QueryPerformanceCounter(&g_qpc0);
    wav_ouvrir();
    dsh_sortie_init();
    CloseHandle(CreateThread(NULL, 0, fil_pompe, NULL, 0, NULL));
    fprintf(stderr, "[SON] DirectSound en haut niveau (sans APU), 48 kHz, repli 5.1%s\n",
            g_48k ? ", voix lues a 48 kHz (ancien comportement)" : "");
    fflush(stderr);
    return TRUE;
}

/* Prend le verrou apres avoir demarre le moteur si besoin, et rend le son
 * jusqu'a maintenant : tout appel agit a l'instant emule ou il arrive. */
static void entrer(void)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, demarrer_une_fois, NULL, NULL);
    EnterCriticalSection(&g_cs);
    avancer();
}

static void sortir(void)
{
    LeaveCriticalSection(&g_cs);
}

static tampon_t *tampon(uint32_t jeton)
{
    uint32_t i;
    if (jeton < JETON_TAMPON0 || (jeton - JETON_TAMPON0) % JETON_PAS) return NULL;
    i = (jeton - JETON_TAMPON0) / JETON_PAS;
    if (i >= NB_TAMPONS || !g_t[i].utilise) return NULL;
    return &g_t[i];
}

/* Un appel sur un jeton inconnu : dit une fois, rend DSERR_INVALIDCALL. */
static void jeton_inconnu(const char *f, uint32_t jeton)
{
    static int dit;
    if (dit++ < 5) {
        fprintf(stderr, "[SON] %s sur un tampon inconnu (0x%08X)\n", f, jeton);
        fflush(stderr);
    }
}

#define TAMPON_OU_ERREUR(nom, jeton, nargs)                         \
    tampon_t *t;                                                    \
    entrer();                                                       \
    t = tampon(jeton);                                              \
    if (!t) { sortir(); jeton_inconnu(nom, jeton);                  \
              RETOUR(DSERR_INVALIDCALL, nargs); return; }

/* ── Conversions octets <-> trames ────────────────────────────────────────── */

static uint32_t octets_vers_trames(const tampon_t *t, uint32_t o)
{
    return t->adpcm ? (o / t->align) * 64 : o / t->align;
}

static uint32_t trames_vers_octets(const tampon_t *t, uint32_t f)
{
    return t->adpcm ? (f / 64) * t->align : f * t->align;
}

/* ── IDirectSound ─────────────────────────────────────────────────────────── */

void dsound_hle_DirectSoundCreate(void)
{
    uint32_t pp = ARG(2);
    entrer();
    g_ds_ref++;
    sortir();
    if (lisible(pp, 4)) M32(pp) = JETON_DS;
    RETOUR(DS_OK, 3);
}

void dsound_hle_IDirectSound_Release(void)
{
    uint32_t r;
    entrer();
    r = g_ds_ref ? --g_ds_ref : 0;
    sortir();
    RETOUR(r, 1);
}

void dsound_hle_IDirectSound_SetMixBinHeadroom(void)
{
    uint32_t bin = ARG(2), h = ARG(3);
    entrer();
    if (bin < 32) {
        int i;
        g_mbhr[bin] = (uint8_t)(h > 7 ? 7 : h);
        for (i = 0; i < NB_TAMPONS; i++)
            if (g_t[i].utilise) calculer_gains(&g_t[i]);
    }
    sortir();
    RETOUR(DS_OK, 3);
}

/* L'image d'effets est pour le DSP, qui ne tourne pas (pas plus qu'avant). */
void dsound_hle_IDirectSound_DownloadEffectsImage(void)
{
    uint32_t pp = ARG(5);
    if (lisible(pp, 4)) M32(pp) = 0;
    RETOUR(DS_OK, 5);
}

/* La reverberation I3DL2 est un programme du DSP : acceptee, non rendue. */
void dsound_hle_IDirectSound_SetI3DL2Listener(void)
{
    RETOUR(DS_OK, 3);
}

/* DSBUFFERDESC : dwSize, dwFlags, dwBufferBytes, lpwfxFormat, lpMixBins,
 * dwInputMixBin. WAVEFORMATEX : wFormatTag, nChannels, nSamplesPerSec,
 * nAvgBytesPerSec, nBlockAlign, wBitsPerSample, cbSize. */
static uint32_t creer(uint32_t desc, uint32_t pp)
{
    uint32_t fx, mb, tag, ch, bits, i;
    tampon_t *t = NULL;

    if (!lisible(desc, 24) || !lisible(pp, 4)) return DSERR_INVALIDCALL;
    fx = M32(desc + 12);
    mb = M32(desc + 16);
    if (!lisible(fx, 16)) return DSERR_INVALIDCALL;
    tag = M16(fx); ch = M16(fx + 2); bits = M16(fx + 14);
    if (ch < 1 || ch > 6 || (!(tag == 1 && (bits == 8 || bits == 16)) && !(tag == 0x69 && bits == 4))) {
        static int dit;
        if (dit++ < 5)
            fprintf(stderr, "[SON] format non gere : tag %04X, %u canaux, %u bits\n", tag, ch, bits);
        return DSERR_INVALIDCALL;
    }
    if (M32(desc + 8)) {
        static int dit;
        if (!dit++)
            fprintf(stderr, "[SON] tampon cree avec %u octets a lui : non gere (le titre"
                    " passe d'habitude par SetBufferData)\n", M32(desc + 8));
    }
    for (i = 0; i < NB_TAMPONS && g_t[i].utilise; i++) { }
    if (i == NB_TAMPONS) return DSERR_OUTOFMEMORY;
    t = &g_t[i];
    memset(t, 0, sizeof *t);
    t->utilise = 1;
    t->ref = 1;
    t->canaux = (int)ch;
    t->bits = (int)bits;
    t->adpcm = tag == 0x69;
    t->taux = M32(fx + 4);
    t->align = t->adpcm ? 36u * ch : ch * (bits / 8);
    t->headroom = 600;
    t->bloc_idx = -1;
    if (mb && lisible(mb, 8)) {
        uint32_t n = M32(mb), paires = M32(mb + 4), p;
        for (p = 0; p < n && p < NB_PAIRES && lisible(paires + 8 * p, 8); p++) {
            t->bin[p] = M32(paires + 8 * p);
            t->vol[p] = (int32_t)M32(paires + 8 * p + 4);
        }
        t->npaires = (int)p;
    } else {
        /* Par defaut : mono et stereo a l'avant, 6 canaux dans l'ordre. */
        int p, n = ch == 1 ? 2 : (int)ch;
        for (p = 0; p < n; p++) { t->bin[p] = (uint32_t)p; t->vol[p] = 0; }
        t->npaires = n;
    }
    calculer_gains(t);
    M32(pp) = JETON_TAMPON0 + i * JETON_PAS;
    return DS_OK;
}

void dsound_hle_IDirectSound_CreateSoundBuffer(void)
{
    uint32_t desc = ARG(2), pp = ARG(3), hr;
    entrer();
    hr = creer(desc, pp);
    sortir();
    RETOUR(hr, 4);
}

void dsound_hle_DirectSoundCreateBuffer(void)
{
    uint32_t desc = ARG(1), pp = ARG(2), hr;
    entrer();
    hr = creer(desc, pp);
    sortir();
    RETOUR(hr, 2);
}

/* ── IDirectSoundBuffer ───────────────────────────────────────────────────── */

void dsound_hle_IDirectSoundBuffer_Release(void)
{
    uint32_t r;
    TAMPON_OU_ERREUR("Release", ARG(1), 1);
    r = --t->ref;
    if (!r) t->utilise = 0;
    sortir();
    RETOUR(r, 1);
}

/* DSMIXBINS : dwMixBinCount, lpMixBinVolumePairs { dwMixBin, lVolume }. */
void dsound_hle_IDirectSoundBuffer_SetMixBins(void)
{
    uint32_t mb = ARG(2);
    TAMPON_OU_ERREUR("SetMixBins", ARG(1), 2);
    if (lisible(mb, 8)) {
        uint32_t n = M32(mb), paires = M32(mb + 4), p;
        for (p = 0; p < n && p < NB_PAIRES && lisible(paires + 8 * p, 8); p++) {
            t->bin[p] = M32(paires + 8 * p);
            t->vol[p] = (int32_t)M32(paires + 8 * p + 4);
        }
        t->npaires = (int)p;
        calculer_gains(t);
    }
    sortir();
    RETOUR(DS_OK, 2);
}

/* Change le volume des mixbins nommees, parmi celles du routage. */
void dsound_hle_IDirectSoundBuffer_SetMixBinVolumes(void)
{
    uint32_t mb = ARG(2);
    TAMPON_OU_ERREUR("SetMixBinVolumes", ARG(1), 2);
    if (lisible(mb, 8)) {
        uint32_t n = M32(mb), paires = M32(mb + 4), k;
        int p;
        for (k = 0; k < n && k < 32 && lisible(paires + 8 * k, 8); k++) {
            uint32_t bin = M32(paires + 8 * k);
            for (p = 0; p < t->npaires; p++)
                if (t->bin[p] == bin) t->vol[p] = (int32_t)M32(paires + 8 * k + 4);
        }
        calculer_gains(t);
    }
    sortir();
    RETOUR(DS_OK, 2);
}

/* DSENVELOPEDESC : dwEG, dwMode, dwDelay, dwAttack, dwHold, dwDecay,
 * dwRelease, dwSustain, lPitchScale, lFilterCutOff. Le titre ne pose qu'une
 * enveloppe d'amplitude immediate et pleine (sustain 0xFF), soit un gain de
 * 1 : c'est ce qui est rendu. Une autre est signalee, pas rendue. */
void dsound_hle_IDirectSoundBuffer_SetEG(void)
{
    uint32_t e = ARG(2);
    TAMPON_OU_ERREUR("SetEG", ARG(1), 2);
    if (lisible(e, 40)) {
        int simple = M32(e + 8) == 0 && M32(e + 12) == 0 && M32(e + 16) == 0
                  && M32(e + 20) == 0 && M32(e + 28) == 0xFF;
        static int dit;
        if (!simple && dit++ < 3)
            fprintf(stderr, "[SON] SetEG non rendue : EG %u mode %u delai %u attaque %u"
                    " maintien %u declin %u relache %u soutien %u\n", M32(e), M32(e + 4),
                    M32(e + 8), M32(e + 12), M32(e + 16), M32(e + 20), M32(e + 24), M32(e + 28));
    }
    sortir();
    RETOUR(DS_OK, 2);
}

/* DSFILTERDESC : dwMode, dwQCoefficient, adwCoefficients[4]. Le VP recoit le
 * mode, FC0 = coefficient 0 et FC1 = coefficient 1 (mesure : mode 1, 0,
 * 0xFFFF -> FMODE 1, FCA = 0xFFFF0000) et en tire f = 2^(FC0/4096) et
 * q = FC1/0x8000, bornes. Une voix mono filtre sur le bit 0 du mode, une voix
 * a plusieurs canaux sur le mode 1 seul. */
void dsound_hle_IDirectSoundBuffer_SetFilter(void)
{
    uint32_t d = ARG(2);
    TAMPON_OU_ERREUR("SetFilter", ARG(1), 2);
    if (lisible(d, 24)) {
        uint32_t mode = M32(d);
        int16_t fc0 = (int16_t)(M32(d + 8) & 0xFFFF);
        uint16_t fc1 = (uint16_t)(M32(d + 12) & 0xFFFF);
        float f = powf(2.0f, fc0 / 4096.0f), q = fc1 / 32768.0f;
        int c;
        t->filtre = t->canaux == 1 ? (mode & 1) != 0 : mode == 1;
        t->f = f < 0.003906f ? 0.003906f : f > 1.0f ? 1.0f : f;
        t->q = q < 0.079407f ? 0.079407f : q > 1.0f ? 1.0f : q;
        t->qn = sqrtf(t->q / 2.0f + 0.01f);
        for (c = 0; c < 6; c++) t->bas[c] = t->bande[c] = 0.0f;
    }
    sortir();
    RETOUR(DS_OK, 2);
}

void dsound_hle_IDirectSoundBuffer_SetFrequency(void)
{
    uint32_t hz = ARG(2);
    TAMPON_OU_ERREUR("SetFrequency", ARG(1), 2);
    t->freq = hz;
    sortir();
    RETOUR(DS_OK, 2);
}

/* Les donnees restent dans la memoire du titre, lues a chaque rendu. La
 * region de lecture et la boucle reprennent tout le tampon ; la voix s'arrete
 * et repart du debut. */
void dsound_hle_IDirectSoundBuffer_SetBufferData(void)
{
    uint32_t va = ARG(2), n = ARG(3);
    TAMPON_OU_ERREUR("SetBufferData", ARG(1), 3);
    if (va && n && !lisible(va, n)) {
        sortir();
        fprintf(stderr, "[SON] SetBufferData hors memoire : 0x%08X, %u octets\n", va, n);
        RETOUR(DSERR_INVALIDCALL, 3);
        return;
    }
    t->etat = 0;
    t->va = va;
    t->octets = va ? n : 0;
    t->trames = octets_vers_trames(t, t->octets);
    t->jeu_debut = t->boucle_debut = 0;
    t->jeu_fin = t->boucle_fin = t->trames;
    t->pos = 0.0;
    t->bloc_idx = -1;
    sortir();
    RETOUR(DS_OK, 3);
}

/* Longueur 0 : jusqu'a la fin de la region de lecture (DirectSound fait de
 * meme, FUN_00257E89) ; une boucle qui en deborde est refusee. */
void dsound_hle_IDirectSoundBuffer_SetLoopRegion(void)
{
    uint32_t d = ARG(2), n = ARG(3), fd, ff;
    TAMPON_OU_ERREUR("SetLoopRegion", ARG(1), 3);
    fd = octets_vers_trames(t, d);
    ff = n ? fd + octets_vers_trames(t, n) : t->jeu_fin;
    if (ff <= t->jeu_fin && fd < ff) {
        t->boucle_debut = fd;
        t->boucle_fin = ff;
    }
    sortir();
    RETOUR(DS_OK, 3);
}

void dsound_hle_IDirectSoundBuffer_SetCurrentPosition(void)
{
    uint32_t o = ARG(2);
    TAMPON_OU_ERREUR("SetCurrentPosition", ARG(1), 2);
    t->pos = (double)octets_vers_trames(t, o);
    if (t->pos > t->trames) t->pos = t->trames;
    t->bloc_idx = -1;
    sortir();
    RETOUR(DS_OK, 2);
}

void dsound_hle_IDirectSoundBuffer_Play(void)
{
    uint32_t flags = ARG(4);
    TAMPON_OU_ERREUR("Play", ARG(1), 4);
    t->boucle = (flags & DSBPLAY_LOOPING) != 0;
    if (t->etat == 0) {
        int c;
        for (c = 0; c < 6; c++) t->bas[c] = t->bande[c] = 0.0f;
    }
    t->etat = 1;
    sortir();
    RETOUR(DS_OK, 4);
}

void dsound_hle_IDirectSoundBuffer_Pause(void)
{
    uint32_t pause = ARG(2);
    TAMPON_OU_ERREUR("Pause", ARG(1), 2);
    if (pause == DSBPAUSE_RESUME) {
        if (t->etat == 2) t->etat = 1;
    } else if (t->etat == 1) {
        t->etat = 2;
    }
    sortir();
    RETOUR(DS_OK, 2);
}

/* L'enveloppe du titre a une relache nulle : l'arret par l'enveloppe est
 * immediat. RELEASEWAVEFORM laisse finir l'echantillon hors de sa boucle. */
void dsound_hle_IDirectSoundBuffer_StopEx(void)
{
    uint32_t flags = ARG(4);
    TAMPON_OU_ERREUR("StopEx", ARG(1), 4);
    if ((flags & DSBSTOPEX_RELEASEWAVEFORM) && t->etat == 1)
        t->boucle = 0;
    else
        t->etat = 0;
    sortir();
    RETOUR(DS_OK, 4);
}

void dsound_hle_IDirectSoundBuffer_GetStatus(void)
{
    uint32_t ps = ARG(2), st = 0;
    TAMPON_OU_ERREUR("GetStatus", ARG(1), 2);
    if (t->etat == 1) st = DSBSTATUS_PLAYING;
    else if (t->etat == 2) st = DSBSTATUS_PAUSED;
    if (st && t->boucle) st |= DSBSTATUS_LOOPING;
    sortir();
    if (lisible(ps, 4)) M32(ps) = st;
    RETOUR(DS_OK, 2);
}

/* Les curseurs de l'ancien chemin : lecture = position + 64 trames, ecriture
 * = lecture + max(32 trames, 1 trame), ramenes dans la boucle (ou dans le
 * tampon), en octets. */
void dsound_hle_IDirectSoundBuffer_GetCurrentPosition(void)
{
    uint32_t pj = ARG(2), pe = ARG(3), jeu, ecr, debut, fin;
    TAMPON_OU_ERREUR("GetCurrentPosition", ARG(1), 3);
    debut = t->boucle ? t->boucle_debut : 0;
    fin = t->boucle ? t->boucle_fin : t->trames;
    jeu = (uint32_t)t->pos;
    if (t->etat != 0 && fin > debut) {
        jeu += AVANCE_LECTURE;
        while (jeu >= fin) jeu -= fin - debut;
    }
    ecr = jeu + AVANCE_ECRITURE;
    if (fin > debut)
        while (ecr >= fin) ecr -= fin - debut;
    sortir();
    if (lisible(pj, 4)) M32(pj) = trames_vers_octets(t, jeu);
    if (lisible(pe, 4)) M32(pe) = trames_vers_octets(t, ecr);
    RETOUR(DS_OK, 3);
}
