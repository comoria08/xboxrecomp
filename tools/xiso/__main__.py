"""
List or extract files from an Xbox disc image.

    py -3 -m tools.xiso ls      game.iso
    py -3 -m tools.xiso get     game.iso default.xbe -o game/
    py -3 -m tools.xiso unpack  game.iso -o extracted/
    py -3 -m tools.xiso pack    extracted/ -o game.iso [--exclude PATTERN ...]
"""

import argparse
import fnmatch
import os
import sys

from .xdvdfs import Xiso, XisoError


def cmd_ls(args):
    with Xiso(args.iso) as iso:
        print("partition base : 0x%08X" % iso.base)
        total = 0
        for d, e in sorted(iso.walk(), key=lambda x: (x[0], x[1].name)):
            print("  %10d  %s" % (e.size, (d + "/" + e.name).lstrip("/")))
            total += e.size
        print("  %10d  total" % total)
    return 0


def cmd_get(args):
    with Xiso(args.iso) as iso:
        # Allow a path so companion executables in subdirectories work too.
        path, _, name = args.name.replace("\\", "/").rpartition("/")
        e = iso.find(name, path)
        if e is None:
            print("not found in image: %s" % args.name, file=sys.stderr)
            return 1
        dest = os.path.join(args.output, e.name) if os.path.isdir(args.output) \
            or args.output.endswith(("/", "\\")) else args.output
        iso.extract(e, dest)
        print("%s -> %s (%d bytes)" % (e.name, dest, e.size))
    return 0


def cmd_unpack(args):
    with Xiso(args.iso) as iso:
        n = 0
        for d, e in iso.walk():
            dest = os.path.join(args.output, *(d.split("/") if d else []), e.name)
            iso.extract(e, dest)
            n += 1
        print("extracted %d files to %s" % (n, args.output))
    return 0


def cmd_pack(args):
    from .xdvdfs_write import pack
    patterns = args.exclude or []

    def exclude(rel, name):
        # A pattern with a slash matches the path from the image root, one
        # without matches a name anywhere: "UDATA", "*.orig", "assets/*.bak".
        return any(fnmatch.fnmatch(rel.lower() if "/" in p else name.lower(), p.lower())
                   for p in patterns)

    def progress(i, n, f):
        if i == n or i % 200 == 0:
            print("  %d/%d files" % (i, n))

    files, data = pack(args.dir, args.output, exclude, progress)
    print("packed %d files (%d bytes) into %s" % (files, data, args.output))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.xiso", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ls", help="list every file in the image")
    p.add_argument("iso")
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("get", help="extract one file")
    p.add_argument("iso")
    p.add_argument("name", help="e.g. default.xbe")
    p.add_argument("-o", "--output", default=".", help="file or directory")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("unpack", help="extract everything")
    p.add_argument("iso")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("pack", help="write an image of a directory")
    p.add_argument("dir")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--exclude", action="append", metavar="PATTERN",
                   help="leave out matching entries (repeatable)")
    p.set_defaults(func=cmd_pack)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except XisoError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
