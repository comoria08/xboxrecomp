"""
Self-check for guest return addresses on the simulated stack.

Run: py -3 tools/recomp/test_call_retaddr.py

Regression guard for two coupled bugs:

1. `call` pushed a literal 0 instead of the address of the following
   instruction. `ret` never reads the slot, so this was invisible until guest
   code did -- __SEH_prolog finds its scope table through [esp], _alloca probes
   walk back to it, and "mov eax, [esp]" is the standard CRT idiom for "who
   called me". All of them saw 0.

2. _fixup_icall_esp_save walked backwards from an ICALL looking for the run of
   argument pushes, and tested `startswith('PUSH32(esp,')` before it tested for
   a direct call. A direct call is emitted as "PUSH32(esp, <retva>); name();"
   on one line, so it matched as an argument push and the scan ran straight
   through it -- putting the _icall_esp save before a call that had already
   returned, so a failing ICALL rewound g_esp too far.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.recomp.lifter import Lifter  # noqa: E402
from tools.recomp.translator import _fixup_icall_esp_save  # noqa: E402


class _Op:
    def __init__(self, text, type="reg", reg=None, size=4, mem_size=None, imm=0):
        self.text = text
        self.type = type
        self.reg = reg
        self.size = size
        self.mem_size = mem_size
        self.imm = imm


class _Insn:
    def __init__(self, mnemonic, operands, address, size, call_target=None):
        self.mnemonic = mnemonic
        self.operands = operands
        self.address = address
        self.size = size
        self.call_target = call_target

    @property
    def end_address(self):
        return self.address + self.size


def test_direct_call_pushes_following_address():
    # call rel32 at 0x00120000, 5 bytes long -> return address is 0x00120005.
    insn = _Insn("call", [_Op("0x1a2b3c", type="imm", imm=0x1A2B3C)],
                 address=0x00120000, size=5, call_target=0x001A2B3C)
    out = "\n".join(Lifter().lift_instruction(insn))
    assert "PUSH32(esp, 0x00120005u);" in out, out
    assert "PUSH32(esp, 0);" not in out, out
    # still a real call to the target
    assert "0x001A2B3C" in out, out


def test_indirect_call_pushes_following_address():
    # call eax at 0x00130000, 2 bytes long -> return address is 0x00130002.
    insn = _Insn("call", [_Op("eax", type="reg", reg="eax")],
                 address=0x00130000, size=2, call_target=None)
    out = "\n".join(Lifter().lift_instruction(insn))
    assert "PUSH32(esp, 0x00130002u);" in out, out
    assert "RECOMP_ICALL_SAFE(" in out, out


def test_retaddr_is_the_instruction_after_the_call():
    # A 6-byte call must not report a 5-byte return address.
    for size in (2, 5, 6, 7):
        insn = _Insn("call", [_Op("0x200000", type="imm", imm=0x200000)],
                     address=0x00110000, size=size, call_target=0x00200000)
        out = "\n".join(Lifter().lift_instruction(insn))
        assert f"PUSH32(esp, 0x{0x00110000 + size:08X}u);" in out, (size, out)


def test_ret_still_only_discards_the_slot():
    # The return address is data, never control flow: ret must not try to use it.
    lifter = Lifter()
    lifter.func_start = 0x00140000  # not an SEH helper, so no g_seh_ebp bridge
    out = "\n".join(lifter._lift_ret(_Insn("ret", [], 0x00140010, 1), []))
    assert "esp += 4;" in out, out
    assert "return;" in out, out
    # ret N adds the stdcall cleanup on top of the 4-byte return slot
    imm = _Op("8", type="imm", imm=8)
    out = "\n".join(lifter._lift_ret(_Insn("ret", [imm], 0x00140010, 3), [imm]))
    assert "esp += 12;" in out, out


def test_icall_save_stops_at_a_completed_direct_call():
    # Two arg pushes, a direct call, then an ICALL with one arg of its own.
    # The save must land on the ICALL's own arg push, not before the direct call.
    lines = [
        "    PUSH32(esp, ecx);",
        "    PUSH32(esp, 0x00120005u); sub_001A2B3C(); /* call 0x001A2B3C */",
        "    PUSH32(esp, eax);",
        "    PUSH32(esp, 0x00120010u); RECOMP_ICALL_SAFE(edx, _icall_esp); /* indirect call */",
    ]
    out = _fixup_icall_esp_save(lines)
    save_idx = next(i for i, l in enumerate(out) if "_icall_esp = g_esp" in l)
    call_idx = next(i for i, l in enumerate(out) if "/* call 0x" in l)
    assert save_idx > call_idx, (
        "save must be after the completed direct call:\n" + "\n".join(out))
    # and it must still cover the ICALL's own argument
    arg_idx = next(i for i, l in enumerate(out) if l.strip() == "PUSH32(esp, eax);")
    assert save_idx < arg_idx, "\n".join(out)


def test_icall_save_stops_at_a_callee_saved_register_save():
    # sub_005A1700's shape: the function saves ebx and esi mid-body, then makes
    # an indirect call taking its arguments in ecx/edx only. Two epilogues, so
    # each register is popped twice against a single push -- the counts must
    # compare as pops >= pushes, not equality.
    # an indirect call that takes its arguments in ecx/edx only. Absorbing
    # those saves into the "argument run" made the failure path rewind g_esp
    # past them, and the epilogue popped edi off the return address.
    lines = [
        "    PUSH32(esp, ebx);",
        "    ebx = MEM32(esp + 0x14);",
        "    PUSH32(esp, esi);",
        "    edx = esp + 0x14;",
        "    ecx = ebx;",
        "    PUSH32(esp, 0x005A173Fu); RECOMP_ICALL_SAFE(eax, _icall_esp); /* indirect call */",
        "    POP32(esp, esi);",
        "    POP32(esp, ebx);",
        "    POP32(esp, edi);",
        "loc_005A178C: ;",
        "    POP32(esp, esi);",
        "    POP32(esp, ebx);",
        "    POP32(esp, edi);",
    ]
    out = _fixup_icall_esp_save(lines)
    save_idx = next(i for i, l in enumerate(out) if "_icall_esp = g_esp" in l)
    icall_idx = next(i for i, l in enumerate(out) if "RECOMP_ICALL_SAFE" in l)
    push_esi = next(i for i, l in enumerate(out) if l.strip() == "PUSH32(esp, esi);")
    assert save_idx > push_esi, (
        "the save must not rewind over the function's own register saves:\n"
        + "\n".join(out))
    assert save_idx == icall_idx - 1, "\n".join(out)


def test_icall_save_still_covers_a_pushed_esi_argument():
    # Same register, but the function never pops it -- so it is an argument
    # and the run must still absorb it.
    lines = [
        "    PUSH32(esp, esi);",
        "    PUSH32(esp, 0x00120010u); RECOMP_ICALL_SAFE(edx, _icall_esp); /* indirect call */",
    ]
    out = _fixup_icall_esp_save(lines)
    save_idx = next(i for i, l in enumerate(out) if "_icall_esp = g_esp" in l)
    arg = next(i for i, l in enumerate(out) if l.strip() == "PUSH32(esp, esi);")
    assert save_idx < arg, "\n".join(out)


def test_icall_save_absorbs_a_saved_register_pushed_again_as_an_argument():
    # sub_00135265's shape: edi is saved in the prologue AND pushed again as
    # an argument to the virtual call, but popped only once. The extra push is
    # an argument, so the run must absorb it -- leaving it behind shifts the
    # epilogue and comes back with esi and edi swapped.
    lines = [
        "    PUSH32(esp, ecx);",
        "    PUSH32(esp, esi);",
        "    PUSH32(esp, edi);",
        "    eax = MEM32(esi);",
        "    PUSH32(esp, edi);",
        "    ecx = esi;",
        "    PUSH32(esp, 0x00135299u); RECOMP_ICALL_SAFE(MEM32(eax + 0x68), _icall_esp); /* indirect call */",
        "    POP32(esp, edi);",
        "    POP32(esp, esi);",
        "    POP32(esp, ecx);",
    ]
    out = _fixup_icall_esp_save(lines)
    save_idx = next(i for i, l in enumerate(out) if "_icall_esp = g_esp" in l)
    arg_push = max(i for i, l in enumerate(out) if l.strip() == "PUSH32(esp, edi);")
    assert save_idx < arg_push, (
        "the argument push must be inside the rewind window:\n" + "\n".join(out))
    # ...but not the prologue saves, which the epilogue still needs
    prologue = next(i for i, l in enumerate(out) if l.strip() == "PUSH32(esp, esi);")
    assert save_idx > prologue, "\n".join(out)


def test_icall_save_still_covers_a_plain_arg_run():
    lines = [
        "    PUSH32(esp, ecx);",
        "    PUSH32(esp, eax);",
        "    PUSH32(esp, 0x00120010u); RECOMP_ICALL_SAFE(edx, _icall_esp); /* indirect call */",
    ]
    out = _fixup_icall_esp_save(lines)
    save_idx = next(i for i, l in enumerate(out) if "_icall_esp = g_esp" in l)
    first_arg = next(i for i, l in enumerate(out) if l.strip() == "PUSH32(esp, ecx);")
    assert save_idx < first_arg, "\n".join(out)


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("  ok  %s" % fn.__name__)
    print("%d checks passed" % len(fns))



def test_icall_failure_leaves_cdecl_arguments_to_the_caller():
    # sub_001B54D0's shape (point 94): one argument, the call through a state
    # table, and the caller's own `add esp, 4`. When the target is unknown, the
    # failure path must leave the argument where a real cdecl callee would have
    # -- the caller removes it -- or esp ends 4 bytes high and the epilogue
    # pops esi from the wrong slot.
    lines = [
        "    ecx = MEM32(eax + 0x30);",
        "    PUSH32(esp, eax);",
        "    { uint32_t _icall_target = MEM32(esp + ecx * 4 + 8); PUSH32(esp, 0x001B5A3Au);"
        " RECOMP_ICALL_SAFE(_icall_target, _icall_esp); } /* indirect call */",
        "",
        "loc_001B5A3A: ;",
        "    esp = esp + 4;",
        "    POP32(esp, esi);",
    ]
    out = "\n".join(_fixup_icall_esp_save(lines))
    assert "RECOMP_ICALL_SAFE(_icall_target, _icall_esp - 4u);" in out, out


def test_icall_failure_rewinds_fully_for_a_callee_that_cleans_up():
    # No `add esp` after the call: the callee pops its own arguments
    # (stdcall, thiscall), and the full rewind stays right.
    lines = [
        "    PUSH32(esp, eax);",
        "    PUSH32(esp, 0x00120010u); RECOMP_ICALL_SAFE(edx, _icall_esp); /* indirect call */",
        "    ecx = eax;",
    ]
    out = "\n".join(_fixup_icall_esp_save(lines))
    assert "RECOMP_ICALL_SAFE(edx, _icall_esp);" in out, out


def test_icall_failure_keeps_the_full_rewind_when_the_cleanup_does_not_match():
    # MSVC sometimes defers: one `add esp` for several calls. A cleanup that is
    # not exactly this call's arguments proves nothing about this callee.
    lines = [
        "    PUSH32(esp, eax);",
        "    PUSH32(esp, 0x00120010u); RECOMP_ICALL_SAFE(edx, _icall_esp); /* indirect call */",
        "    esp = esp + 0xC;",
    ]
    out = "\n".join(_fixup_icall_esp_save(lines))
    assert "RECOMP_ICALL_SAFE(edx, _icall_esp);" in out, out


if __name__ == "__main__":
    _run()
