"""An immediate that lands inside an instruction is not a function start.

x86 decodes from anywhere, so a misaligned address probes as a perfectly good
function body and the pass accepts it. The phantom itself is harmless; what it
costs is the real function it truncates, which then has no ret at all and falls
out of the bottom of its own translated body -- leaving esp and every preserved
register wherever they happened to be.

Def Jam: Fight for NY, Story mode. 0x00046500 is the third byte of
`mov eax,[esi+4]` at 0x000464FE and probed as a 169-byte body ending in a ret.
It truncated sub_00046450, which lost its epilogue; eight nested frames then
unwound wrong behind it and the Flash VM died on the first thing that
dereferenced esi.

inside_a_function cannot see this, because the function being truncated is
found by a later pass and does not exist yet when this one runs. Instruction
boundaries do exist. The test belongs to this pass alone: its evidence is a
bare immediate, whereas a call target or a vtable seed means something really
reaches the address, and there it is the preceding decode that is wrong.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.disasm.functions import FunctionDetector  # noqa: E402


class _Insn:
    def __init__(self, addr, size, imm_ref=None):
        self.address = addr
        self.size = size
        self.end_address = addr + size
        self.imm_ref = imm_ref


class _Section:
    def __init__(self, va, size):
        self.virtual_addr = va
        self.virtual_size = size


class _Func:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _Engine:
    def __init__(self, insns):
        self.instructions = {i.address: i for i in insns}

    def probes_as_returning_body(self, addr):
        return True          # the misaligned decode does look like a body

    def probes_as_vcall_thunk(self, addr):
        return False

    def decode_at(self, addr):
        return True


def _detector(insns, functions):
    det = FunctionDetector.__new__(FunctionDetector)
    det.engine = _Engine(insns)
    det.functions = {f.start: f for f in functions}
    det._candidates = {}
    det.added = []
    det._add_candidate = lambda addr, conf, why: det.added.append((addr, why))
    return det


SECTIONS = [_Section(0x00010000, 0x00400000)]


class ImmRefAlignmentTest(unittest.TestCase):
    def test_an_immediate_inside_an_instruction_is_rejected(self):
        """0x00046500, the third byte of a three-byte mov at 0x000464FE."""
        insns = [
            _Insn(0x000464FE, 3),                       # mov eax,[esi+4]
            _Insn(0x00020000, 5, imm_ref=0x00046500),   # someone loads it
        ]
        det = _detector(insns, functions=[])
        self.assertFalse(det._pass_imm_ref_targets(SECTIONS))
        self.assertEqual(det.added, [])

    def test_an_immediate_on_a_boundary_is_still_accepted(self):
        """The pass must keep doing its job: 0x00046501 is one byte later."""
        insns = [
            _Insn(0x000464FE, 3),
            _Insn(0x00020000, 5, imm_ref=0x00046501),
        ]
        det = _detector(insns, functions=[])
        self.assertTrue(det._pass_imm_ref_targets(SECTIONS))
        self.assertIn((0x00046501, "imm_ref_target"), det.added)

    def test_an_immediate_at_an_undecoded_address_is_accepted(self):
        """The whole point of the pass: bodies nothing has reached yet."""
        insns = [_Insn(0x00020000, 5, imm_ref=0x00090000)]
        det = _detector(insns, functions=[])
        self.assertTrue(det._pass_imm_ref_targets(SECTIONS))
        self.assertIn((0x00090000, "imm_ref_target"), det.added)

    def test_an_immediate_at_the_start_of_an_instruction_is_not_split(self):
        """An address equal to a decoded start is on a boundary, not inside."""
        insns = [
            _Insn(0x00046500, 4),
            _Insn(0x00020000, 5, imm_ref=0x00046500),
        ]
        det = _detector(insns, functions=[])
        self.assertTrue(det._pass_imm_ref_targets(SECTIONS))
        self.assertIn((0x00046500, "imm_ref_target"), det.added)


if __name__ == "__main__":
    unittest.main()
