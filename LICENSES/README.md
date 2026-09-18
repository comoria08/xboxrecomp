# Third-party licence texts

`GPL-2.0.txt` is the verbatim GNU General Public License v2, from

    https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt

It applies to the MCPX DSP core in `src/apu/dsp/`, which descends from
Hatari's DSP M56001 emulation. Unlike the LGPL files below, this one reaches
the whole executable: anything linked against that directory may be
distributed only under the GPL, source included. The top-level NOTICE spells
out what that covers.

`LGPL-2.1.txt` is the verbatim GNU Lesser General Public License v2.1, from

    https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt

It applies to the files listed under "LGPL-2.1-or-later" in the top-level
NOTICE — the MCPX APU sources and `src/nv2a/nv2a_regs.h`, extracted from xemu
and remaining the copyright of espes, Jannik Vogel and Matt Borgerson.

Shipping the verbatim text alongside those files is an LGPL requirement, not
a courtesy.
