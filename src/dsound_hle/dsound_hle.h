/* dsound_hle.h - DirectSound de l'XDK en haut niveau, sans emulation de l'APU.
 *
 * Le titre appelle DirectSound ; au lieu d'executer la bibliotheque liee dans
 * le XBE (qui programme la puce son), ses points d'entree sont remplaces par
 * ceux-ci. Les voix vivent cote hote, le titre ne recoit que des jetons, et
 * un mixeur a nous rend la sortie par XAudio2.
 *
 * Regle de conception : tout ce que le titre relit (GetStatus,
 * GetCurrentPosition, codes de retour) avance avec le temps emule
 * (dsound_hle_maintenant), jamais avec la carte son. La sortie hote rattrape
 * l'ecart par un leger changement de cadence (XAudio2, SetFrequencyRatio).
 *
 * Les fonctions dsound_hle_<Nom> ont la convention de l'XDK 5849 (stdcall) :
 * elles lisent leurs arguments sur la pile invitee, posent g_eax et depilent,
 * comme le corps qu'elles remplacent. Le jeu les branche sur ses adresses
 * (voir recomp_manual.c de defjam_recomp). Sans APU : construire le toolkit
 * avec XBOXRECOMP_APU=OFF.
 */
#ifndef DSOUND_HLE_H
#define DSOUND_HLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Un acces a la fenetre de registres de l'APU : il ne devrait jamais y en
 * avoir, rien ne l'emule plus. Compte et signale. */
void dsound_hle_acces_apu(uint32_t va);

/* Points d'entree, XDK 5849. */
void dsound_hle_DirectSoundCreate(void);                    /* (guid, ppDS, unk)         ret 0xC  */
void dsound_hle_DirectSoundCreateBuffer(void);              /* (desc, ppBuf)              ret 8    */
void dsound_hle_IDirectSound_Release(void);                 /* (ds)                       ret 4    */
void dsound_hle_IDirectSound_CreateSoundBuffer(void);       /* (ds, desc, ppBuf, unk)     ret 0x10 */
void dsound_hle_IDirectSound_SetMixBinHeadroom(void);       /* (ds, mixbin, headroom)     ret 0xC  */
void dsound_hle_IDirectSound_DownloadEffectsImage(void);    /* (ds, img, n, loc, ppDesc)  ret 0x14 */
void dsound_hle_IDirectSound_SetI3DL2Listener(void);        /* (ds, listener, apply)      ret 0xC  */
void dsound_hle_IDirectSoundBuffer_Release(void);           /* (buf)                      ret 4    */
void dsound_hle_IDirectSoundBuffer_SetMixBins(void);        /* (buf, mixbins)             ret 8    */
void dsound_hle_IDirectSoundBuffer_SetMixBinVolumes(void);  /* (buf, mixbins)             ret 8    */
void dsound_hle_IDirectSoundBuffer_SetEG(void);             /* (buf, env)                 ret 8    */
void dsound_hle_IDirectSoundBuffer_SetFilter(void);         /* (buf, filtre)              ret 8    */
void dsound_hle_IDirectSoundBuffer_SetFrequency(void);      /* (buf, hz)                  ret 8    */
void dsound_hle_IDirectSoundBuffer_SetBufferData(void);     /* (buf, va, octets)          ret 0xC  */
void dsound_hle_IDirectSoundBuffer_SetLoopRegion(void);     /* (buf, debut, longueur)     ret 0xC  */
void dsound_hle_IDirectSoundBuffer_SetCurrentPosition(void);/* (buf, octet)               ret 8    */
void dsound_hle_IDirectSoundBuffer_Play(void);              /* (buf, r1, r2, flags)       ret 0x10 */
void dsound_hle_IDirectSoundBuffer_Pause(void);             /* (buf, pause)               ret 8    */
void dsound_hle_IDirectSoundBuffer_StopEx(void);            /* (buf, rt lo, rt hi, flags) ret 0x10 */
void dsound_hle_IDirectSoundBuffer_GetStatus(void);         /* (buf, pStatus)             ret 8    */
void dsound_hle_IDirectSoundBuffer_GetCurrentPosition(void);/* (buf, pJeu, pEcriture)     ret 0xC  */

/* ── Interne : la sortie (dsound_hle_sortie.c) ── */
int  dsh_sortie_init(void);
/* Pose n trames stereo 16 bits dans la file de sortie (fil du mixeur). */
void dsh_sortie_ecrire(const int16_t *trames, int n);

#ifdef __cplusplus
}
#endif

#endif
