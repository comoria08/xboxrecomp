"""The writer's images read back, file for file, through the reader."""

import os
import struct
import tempfile

from tools.xiso.xdvdfs import Xiso, SECTOR
from tools.xiso.xdvdfs_write import pack


def _tree(root):
    files = {
        "default.xbe": os.urandom(5000),
        "empty.txt": b"",
        "exact.bin": os.urandom(2 * SECTOR),      # ends exactly on a sector
        "assets/v2ip.viv": os.urandom(70000),
        "assets/deep/one.o": b"eagl",
        "UDATA/save.bin": b"not on a disc",
    }
    # Enough names in one directory for its table to need a second sector.
    for i in range(150):
        files["many/file_%03d_with_a_longish_name.dat" % i] = bytes([i]) * (i + 1)
    for rel, data in files.items():
        p = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
    os.makedirs(os.path.join(root, "nothing_here"))
    return files


def test_pack_reads_back():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        files = _tree(src)
        iso = os.path.join(tmp, "game.iso")
        pack(src, iso, exclude=lambda rel, name: name == "UDATA")

        with Xiso(iso) as x:
            assert x.base == 0
            seen = {}
            for d, e in x.walk():
                seen[(d + "/" + e.name).lstrip("/")] = x.read(e)
            names = {e.name for e in x.listdir()}
        assert "UDATA/save.bin" not in seen
        assert "UDATA" not in names
        assert "nothing_here" in names
        for rel, data in files.items():
            if rel.startswith("UDATA/"):
                continue
            assert seen[rel] == data, rel
        assert os.path.getsize(iso) % SECTOR == 0


def test_tree_links_follow_names():
    """The left/right links make a search tree: following them by name finds
    every entry, as the console and other extractors do."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        _tree(src)
        iso = os.path.join(tmp, "game.iso")
        pack(src, iso)
        with Xiso(iso) as x:
            many = x.find("many")
            table = x._read_at(many.sector, many.size)
            want = [e.name for e in x.listdir("many")]

        def lookup(name):
            off = 0
            while True:
                l, r, _, _, _, n = struct.unpack_from("<HHIIBB", table, off)
                here = table[off + 14:off + 14 + n].decode("ascii")
                if here.upper() == name.upper():
                    return True
                nxt = l if name.upper() < here.upper() else r
                if not nxt:
                    return False
                off = nxt * 4

        assert all(lookup(n) for n in want)
