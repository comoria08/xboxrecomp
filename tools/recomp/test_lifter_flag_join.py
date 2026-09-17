"""A branch whose predecessors set the flags differently.

MSVC inlines every refcount release as two paths meeting at one jne:

    mov ecx, [esi]            |   ...
    dec ecx                   |   dec eax
    mov eax, ecx              |   mov [esi], eax
    mov [esi], ecx            |   jmp join
    cmp eax, ebx              |
  join:
    jne keep

The predecessors disagree on what set the flags, and the lifter used to lower
the jne to _flags, which nothing assigns -- so the branch never went and every
release freed its object. In Def Jam: Fight for NY that destroyed each APT
movie request inside the call that created it.
"""
import unittest

from .disasm import BasicBlock, Instruction, Operand
from .lifter import JOIN_FLAGS, Lifter, lift_basic_block


EAX = Operand(type="reg", reg="eax")
EBX = Operand(type="reg", reg="ebx")


def _jne(target=0x100):
    insn = Instruction(0x30, 2, "jne", hex(target), "7500")
    insn.operands = [Operand(type="imm", imm=target)]
    insn.jump_target = target
    return insn


class FlagJoinTest(unittest.TestCase):
    def _lift(self, sources):
        lifter = Lifter()
        lifted, out = lift_basic_block(
            lifter, BasicBlock(start=0x30, instructions=[_jne()]),
            flag_state=(JOIN_FLAGS, sources))
        return lifter, "\n".join(lifted), out

    def test_each_predecessor_keeps_its_own_condition(self):
        lifter, generated, _ = self._lift([
            (0x10, ("dec", [EAX])),
            (0x20, ("cmp", [EAX, EBX])),
        ])
        self.assertIn(
            "(_pfrom == 0x00000010u ? (eax != 0) : CMP_NE(_fa, _fb))",
            generated)
        self.assertNotIn("_flags", generated)
        self.assertEqual(lifter.pfrom_preds, {0x10, 0x20})

    def test_matching_conditions_need_no_selector(self):
        lifter, generated, _ = self._lift([
            (0x10, ("dec", [EAX])),
            (0x20, ("sub", [EAX, EBX])),
        ])
        self.assertIn("if ((eax != 0)", generated)
        self.assertNotIn("_pfrom", generated)
        self.assertEqual(lifter.pfrom_preds, set())

    def test_setcc_at_a_join_reads_each_predecessor(self):
        # APT's string compare ends `cmp ecx, esi` on one path and a test on
        # another, both falling into `sete bl`; the setcc used to read _flags.
        sete = Instruction(0x30, 3, "sete", "bl", "0f94c3")
        sete.operands = [Operand(type="reg", reg="bl")]
        lifter = Lifter()
        lifted, _ = lift_basic_block(
            lifter, BasicBlock(start=0x30, instructions=[sete]),
            flag_state=(JOIN_FLAGS, [(0x10, ("dec", [EAX])),
                                     (0x20, ("cmp", [EAX, EBX]))]))
        generated = "\n".join(lifted)
        self.assertIn("_pfrom == 0x00000010u ? (eax == 0) : CMP_EQ(_fa, _fb)",
                      generated)
        self.assertNotIn("_flags", generated)

    def test_join_state_does_not_leave_the_block(self):
        # The block may itself record _pfrom for a later join, which would
        # overwrite the value this one was selected by.
        _, _, out = self._lift([
            (0x10, ("dec", [EAX])),
            (0x20, ("cmp", [EAX, EBX])),
        ])
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
