"""fxam, and the status word it leaves for fnstsw.

MSVC's floating-point intrinsics (_CIatan2 and friends) classify their
arguments with fxam before choosing a code path. Lifted as a comment, fxam left
the previous compare's bits in the status word, the dispatcher took a
special-value path, and atan2 came back wrong -- in Def Jam: Fight for NY every
tilted element of the front end was turned by about a radian.
"""
import unittest

from .disasm import BasicBlock, Instruction, Operand
from .lifter import Lifter, lift_basic_block


def _lift(*insns):
    lifted, _ = lift_basic_block(Lifter(), BasicBlock(start=0, instructions=list(insns)))
    return "\n".join(lifted)


class FxamTest(unittest.TestCase):
    def test_fxam_classifies_st0(self):
        fxam = Instruction(0, 2, "fxam", "", "d9e5")
        fxam.operands = []
        generated = _lift(fxam)
        self.assertIn("/* fxam */", generated)
        self.assertIn("g_fp_cmp = _c;", generated)
        self.assertIn("0x0400", generated)          # normal -> C2
        self.assertNotIn("FPU: fxam", generated)

    def test_fnstsw_reports_the_class_bits(self):
        fnstsw = Instruction(2, 2, "fnstsw", "ax", "dfe0")
        fnstsw.operands = [Operand(type="reg", reg="ax")]
        generated = _lift(fnstsw)
        self.assertIn("g_fp_cmp >= 0x100 ? ((unsigned)g_fp_cmp & 0x4700u)", generated)

    def test_ficom_compares_against_the_integer(self):
        ficom = Instruction(0, 2, "ficom", "word ptr [edi]", "de17")
        ficom.operands = [Operand(type="mem", mem_base="edi", mem_size=2)]
        generated = _lift(ficom)
        self.assertIn("(double)(int16_t)MEM16(", generated)
        self.assertIn("RECOMP_FCMP(fp_top()", generated)


if __name__ == "__main__":
    unittest.main()
