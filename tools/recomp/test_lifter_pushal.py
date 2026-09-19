"""pushal and popal, which hand-written assembly uses to save everything.

Lifted as a comment, these dropped two things at once and neither was visible
at the fault site. The callee's saved registers never came back, so its caller
carried on with ebx, esi and edi belonging to the callee; and esp never moved
by the 32 bytes the instruction reserves, so every [esp+N] and every push/pop
inside the function addressed the wrong slot.

Def Jam: Fight for NY's audio decode worker sub_0012E560 is exactly that shape
-- push ebp / mov ebp,esp / pushal ... popal / leave / ret -- and is also the
function that returns four samples short of what it is asked for, which is what
the crackle workaround papers over. A build with -DRECOMP_ABI_CHECK named it
directly: "sub_0012E560: ebx esi edi".

The other half of the contract is that a popal is only honoured when the same
function contains a pushal. Def Jam's listing holds 7 pushal against 123 popal:
0x61 is an ordinary byte and a sweep that drifts into data decodes it as popal
all day. Acting on an unmatched one is strictly worse than ignoring it -- it
pops 32 bytes nobody pushed.
"""
import unittest

from .disasm import BasicBlock, Instruction, Operand
from .lifter import Lifter, lift_basic_block


def _lift(*insns, balanced=True):
    lifter = Lifter()
    lifter.func_pushal_balanced = balanced
    lifted, _ = lift_basic_block(lifter, BasicBlock(start=0, instructions=list(insns)))
    return "\n".join(lifted)


def _plain(mnemonic, encoding):
    insn = Instruction(0, 1, mnemonic, "", encoding)
    insn.operands = []
    return insn


class PushalTest(unittest.TestCase):
    def test_pushal_saves_all_eight_and_the_original_esp(self):
        generated = _lift(_plain("pushal", "60"))
        self.assertNotIn("TODO", generated)
        for reg in ("eax", "ecx", "edx", "ebx", "ebp", "esi", "edi"):
            self.assertIn(f"PUSH32(esp, {reg});", generated)
        # The esp slot holds the value from before the instruction, not the
        # value after four pushes have already moved it.
        self.assertIn("uint32_t _sp = esp;", generated)
        self.assertIn("PUSH32(esp, _sp);", generated)

    def test_popal_restores_all_eight_and_discards_the_esp_slot(self):
        generated = _lift(_plain("popal", "61"))
        self.assertNotIn("TODO", generated)
        for reg in ("eax", "ecx", "edx", "ebx", "ebp", "esi", "edi"):
            self.assertIn(f"POP32(esp, {reg});", generated)
        # POPAD pops the esp slot and throws it away rather than loading it.
        # Loading it would undo the seven pops that follow.
        self.assertIn("POP32(esp, _sp);", generated)
        self.assertNotIn("POP32(esp, esp);", generated)

    def test_the_pair_is_stack_neutral(self):
        """Eight pushes then eight pops, so a caller's frame is untouched."""
        generated = _lift(_plain("pushal", "60"), _plain("popal", "61"))
        self.assertEqual(generated.count("PUSH32(esp,"), 8)
        self.assertEqual(generated.count("POP32(esp,"), 8)

    def test_an_unmatched_popal_is_left_alone(self):
        """No matching pushal: almost certainly a stray 0x61."""
        generated = _lift(_plain("popal", "61"), balanced=False)
        self.assertNotIn("POP32(esp,", generated)
        self.assertIn("unmatched", generated)

    def test_an_unmatched_pushal_is_left_alone(self):
        """The mirror case: a lone pushal would leave esp 32 bytes low."""
        generated = _lift(_plain("pushal", "60"), balanced=False)
        self.assertNotIn("PUSH32(esp,", generated)
        self.assertIn("unmatched", generated)

    def test_the_intel_spellings_lift_too(self):
        for mnemonic, encoding in (("pushad", "60"), ("popad", "61")):
            with self.subTest(mnemonic=mnemonic):
                self.assertNotIn("TODO", _lift(_plain(mnemonic, encoding)))


if __name__ == "__main__":
    unittest.main()
