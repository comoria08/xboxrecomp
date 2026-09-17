/* Answer two questions about a handful of addresses, from the analysed program.
 *
 * The recompiler's own disassembler records only the call graph it managed to
 * walk, so a function reached through a pointer, or from a section its sweep
 * did not enter, comes back with no callers at all -- which during bring-up
 * reads as "nothing calls this" when the truth is "we did not see the call".
 * Ghidra has the whole reference database, and these are the two queries that
 * matter when chasing a chain that stops somewhere:
 *
 *   1. who references this address (calls, and pointer stores)
 *   2. what does the function at it decompile to
 *
 * Java rather than Python because Ghidra 12 dropped Jython: a .py script now
 * needs PyGhidra installed, while a .java one always runs.
 *
 * Usage (headless, project already analysed):
 *   analyzeHeadless <proj> <name> -process xbe_flat.bin -noanalysis \
 *       -scriptPath <dir> -postScript QueryTargets.java <out.txt> <addr>[,<addr>...]
 *
 * Addresses are hex, with or without 0x. Output is plain text.
 *
 * @category XboxRecomp
 */
import java.io.PrintWriter;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.ReferenceIterator;
import ghidra.program.model.symbol.ReferenceManager;

public class QueryTargets extends GhidraScript {

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            println("QueryTargets: need <out.txt> <addr,addr,...>");
            return;
        }

        String outPath = args[0];
        // analyzeHeadless splits a comma list into separate script arguments,
        // so every argument after the output path is a target, and each may
        // itself still be a comma list.
        java.util.List<String> list = new java.util.ArrayList<>();
        for (int a = 1; a < args.length; a++) {
            for (String t : args[a].split(",")) {
                list.add(t);
            }
        }
        String[] targets = list.toArray(new String[0]);

        FunctionManager fm = currentProgram.getFunctionManager();
        ReferenceManager rm = currentProgram.getReferenceManager();

        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);

        PrintWriter out = new PrintWriter(outPath);
        for (String raw : targets) {
            String t = raw.trim();
            if (t.isEmpty()) {
                continue;
            }
            if (t.toLowerCase().startsWith("0x")) {
                t = t.substring(2);
            }

            out.println("======================================================");
            out.println("TARGET 0x" + t.toUpperCase());

            Address addr = currentProgram.getAddressFactory().getAddress(t);
            if (addr == null) {
                out.println("  (address does not parse)");
                continue;
            }

            Function owner = fm.getFunctionContaining(addr);
            if (owner != null) {
                out.println("  inside " + owner.getName()
                            + " at " + owner.getEntryPoint());
            }

            out.println("  references to it:");
            ReferenceIterator it = rm.getReferencesTo(addr);
            int n = 0;
            while (it.hasNext() && n < 60) {
                Reference r = it.next();
                Address from = r.getFromAddress();
                Function f = fm.getFunctionContaining(from);
                out.println("    " + from + "  "
                            + r.getReferenceType().getName()
                            + "  from " + (f != null ? f.getName() : "(no function)"));
                n++;
            }
            if (n == 0) {
                out.println("    (none)");
            }

            Function entry = fm.getFunctionAt(addr);
            if (entry != null) {
                DecompileResults res = decomp.decompileFunction(entry, 60, monitor);
                if (res.decompileCompleted()) {
                    out.println("  decompiled:");
                    for (String line : res.getDecompiledFunction().getC().split("\n")) {
                        out.println("    " + line);
                    }
                } else {
                    out.println("  decompile failed: " + res.getErrorMessage());
                }
            }
        }
        out.close();
        decomp.dispose();
        println("QueryTargets: wrote " + outPath);
    }
}
