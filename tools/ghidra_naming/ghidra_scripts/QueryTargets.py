# Answer two questions about a handful of addresses, from the analysed program.
#
# Static analysis in the recompiler's own disassembler gives an incomplete
# caller list -- it records the call graph it happened to walk, so a function
# reached only through a pointer or from a section it did not sweep comes back
# with no callers at all. Ghidra has the whole reference database, and these
# are the two queries that matter during bring-up:
#
#   1. who references this address (callers, and pointer stores)
#   2. what does this function decompile to
#
# Usage (headless, project already analysed):
#   analyzeHeadless <proj> <name> -process xbe_flat.bin -noanalysis \
#       -scriptPath <dir> -postScript QueryTargets.py <out.txt> <addr>[,<addr>...]
#
# Addresses are hex, with or without 0x. Output is plain text so it can be read
# without a JSON round trip.
#@category Xbox

import sys

from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor


def main():
    args = getScriptArgs()
    if len(args) < 2:
        print("QueryTargets: need <out.txt> <addr,addr,...>")
        return

    out_path = args[0]
    targets = [t.strip() for t in args[1].split(",") if t.strip()]

    fm = currentProgram.getFunctionManager()
    rm = currentProgram.getReferenceManager()
    af = currentProgram.getAddressFactory()

    decomp = DecompInterface()
    decomp.openProgram(currentProgram)
    monitor = ConsoleTaskMonitor()

    lines = []
    for t in targets:
        text = t[2:] if t.lower().startswith("0x") else t
        addr = af.getAddress(text)
        lines.append("=" * 70)
        lines.append("TARGET 0x%s" % text.upper())

        if addr is None:
            lines.append("  (address does not parse)")
            continue

        fn = fm.getFunctionContaining(addr)
        if fn is not None:
            lines.append("  in function %s at %s"
                         % (fn.getName(), fn.getEntryPoint()))

        # Every reference to the address, with the function each comes from.
        refs = rm.getReferencesTo(addr)
        n = 0
        lines.append("  references to it:")
        for r in refs:
            frm = r.getFromAddress()
            owner = fm.getFunctionContaining(frm)
            lines.append("    %s  %-18s from %s"
                         % (frm, r.getReferenceType().getName(),
                            owner.getName() if owner else "(no function)"))
            n += 1
            if n >= 60:
                lines.append("    ... more")
                break
        if n == 0:
            lines.append("    (none)")

        # And the decompiled body, when the address is a function entry.
        fn = fm.getFunctionAt(addr)
        if fn is not None:
            res = decomp.decompileFunction(fn, 60, monitor)
            if res.decompileCompleted():
                lines.append("  decompiled:")
                for ln in res.getDecompiledFunction().getC().split("\n"):
                    lines.append("    " + ln)
            else:
                lines.append("  decompile failed: %s" % res.getErrorMessage())

    f = open(out_path, "w")
    f.write("\n".join(lines))
    f.write("\n")
    f.close()
    print("QueryTargets: wrote %s (%d lines)" % (out_path, len(lines)))


main()
