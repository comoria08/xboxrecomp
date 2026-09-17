"""Result-based flag conditions read a snapshot, not the operand again.

MSVC ends EA's raster row loop (Def Jam: Fight for NY, 0x001DDC11) with

    dec  dword ptr [esp+0x18]
    pop  ecx
    jne  loop

and re-reading [esp+0x18] at the jne, after the pop moved esp, tested the wrong
stack slot: the loop never ended and wrote far past its line.
"""
import unittest

from .disasm import BasicBlock, Instruction, Operand
from .lifter import Lifter, lift_basic_block


def _dec_pop_jne():
    dec = Instruction(0, 4, "dec", "dword ptr [esp + 0x18]", "ff4c2418")
    dec.operands = [Operand(type="mem", mem_base="esp", mem_disp=0x18, mem_size=4)]
    pop = Instruction(4, 1, "pop", "ecx", "59")
    pop.operands = [Operand(type="reg", reg="ecx")]
    jne = Instruction(5, 2, "jne", "0x40", "7500")
    jne.operands = [Operand(type="imm", imm=0x40)]
    jne.jump_target = 0x40
    return [dec, pop, jne]


class ResultSnapshotTest(unittest.TestCase):
    def test_dec_then_pop_branches_on_the_snapshot(self):
        lifted, _ = lift_basic_block(
            Lifter(), BasicBlock(start=0, instructions=_dec_pop_jne()))
        generated = "\n".join(lifted)
        self.assertIn("_fr = (uint32_t)(int32_t)(int32_t)(MEM32(esp + 0x18));",
                      generated)
        jcc = [l for l in lifted if "jne" in l][0]
        self.assertIn("_fr != 0", jcc)
        self.assertNotIn("MEM32", jcc)

    def test_snapshot_follows_the_write(self):
        lifted, _ = lift_basic_block(
            Lifter(), BasicBlock(start=0, instructions=_dec_pop_jne()))
        write = next(i for i, l in enumerate(lifted) if l.startswith("MEM32(esp + 0x18) ="))
        snap = next(i for i, l in enumerate(lifted) if l.startswith("_fr ="))
        self.assertLess(write, snap)

    def test_register_source_is_captured_before_the_write(self):
        # sub eax, [eax+4] -- the source address depends on the destination.
        sub = Instruction(0, 3, "sub", "eax, dword ptr [eax + 4]", "2b4004")
        sub.operands = [Operand(type="reg", reg="eax"),
                        Operand(type="mem", mem_base="eax", mem_disp=4, mem_size=4)]
        jb = Instruction(3, 2, "jl", "0x40", "7c00")
        jb.operands = [Operand(type="imm", imm=0x40)]
        jb.jump_target = 0x40
        lifted, _ = lift_basic_block(
            Lifter(), BasicBlock(start=0, instructions=[sub, jb]))
        pre = next(i for i, l in enumerate(lifted) if l.startswith("_frb ="))
        write = next(i for i, l in enumerate(lifted) if l.startswith("eax ="))
        self.assertLess(pre, write)


if __name__ == "__main__":
    unittest.main()
