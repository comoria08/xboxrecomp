"""
XDVDFS writer: a disc image from a directory tree.

The reader (xdvdfs.py) was enough while the pipeline only needed to take files
out of an image. Testing an installer that starts from an image needs one to
feed it, and a player who kept the extracted files but not the image can have
one rebuilt. The result is a bare game partition that this toolkit's reader
and the usual extractors read; it is not a byte copy of any original disc
(sector placement, timestamps and padding are this writer's own).

Layout, as the format expects:
  sectors 0-31   zero
  sector 32      volume descriptor: magic, root table sector and size,
                 a timestamp, the magic again at 0x7EC
  then           every directory table, then every file, each starting on a
                 sector boundary

A directory table is a binary search tree of entries ordered by name,
compared case-insensitively, with the root first. Each entry is 4-byte aligned
and never straddles a sector; the gaps are 0xFF.
"""

import os
import struct
import time

from .xdvdfs import SECTOR, MAGIC, ATTR_DIRECTORY

ATTR_NORMAL = 0x80


class _Node:
    __slots__ = ("name", "path", "is_dir", "size", "children", "sector",
                 "table", "table_size")

    def __init__(self, name, path, is_dir, size=0):
        self.name = name
        self.path = path
        self.is_dir = is_dir
        self.size = size
        self.children = []
        self.sector = 0
        self.table = b""
        self.table_size = 0


def _scan(path, rel, exclude):
    node = _Node(os.path.basename(path) if rel else "", path, True)
    for name in sorted(os.listdir(path), key=lambda n: n.upper()):
        sub = (rel + "/" + name).lstrip("/")
        if exclude(sub, name):
            continue
        name.encode("ascii")                    # the format has no other charset
        full = os.path.join(path, name)
        if os.path.isdir(full):
            node.children.append(_scan(full, sub, exclude))
        else:
            node.children.append(_Node(name, full, False, os.path.getsize(full)))
    return node


def _balanced(entries):
    """Pre-order of a balanced tree over entries sorted by name: (entry, left, right)."""
    if not entries:
        return None
    mid = len(entries) // 2
    return (entries[mid], _balanced(entries[:mid]), _balanced(entries[mid + 1:]))


def _layout(tree):
    """Place each entry of a tree in its table: {id(entry): byte offset}, table size."""
    order, stack = [], [tree]
    while stack:                                # pre-order, root first
        t = stack.pop()
        if t is None:
            continue
        order.append(t)
        stack.append(t[2])
        stack.append(t[1])
    pos, where = 0, {}
    for t in order:
        size = (0x0E + len(t[0].name) + 3) & ~3
        if pos % SECTOR + size > SECTOR:
            pos = (pos + SECTOR - 1) // SECTOR * SECTOR
        where[id(t[0])] = pos
        pos += size
    return order, where, max(SECTOR, (pos + SECTOR - 1) // SECTOR * SECTOR)


def _table(dirnode):
    entries = sorted(dirnode.children, key=lambda n: n.name.upper())
    tree = _balanced(entries)
    if tree is None:
        return b"\xff" * SECTOR
    order, where, size = _layout(tree)
    buf = bytearray(b"\xff" * size)
    for entry, left, right in order:
        off = where[id(entry)]
        name = entry.name.encode("ascii")
        struct.pack_into("<HHIIBB", buf, off,
                         where[id(left[0])] // 4 if left else 0,
                         where[id(right[0])] // 4 if right else 0,
                         entry.sector,
                         entry.table_size if entry.is_dir else entry.size,
                         ATTR_DIRECTORY if entry.is_dir else ATTR_NORMAL,
                         len(name))
        buf[off + 0x0E:off + 0x0E + len(name)] = name
        pad = ((0x0E + len(name) + 3) & ~3) - (0x0E + len(name))
        buf[off + 0x0E + len(name):off + 0x0E + len(name) + pad] = b"\xff" * pad
    return bytes(buf)


def pack(src_dir, out_path, exclude=None, progress=None):
    """Write an XDVDFS image of src_dir to out_path. exclude(relpath, name)
    returns True to leave an entry out. Returns (files, bytes of data)."""
    exclude = exclude or (lambda rel, name: False)
    root = _scan(src_dir, "", exclude)

    dirs, files, queue = [], [], [root]
    while queue:
        d = queue.pop(0)
        dirs.append(d)
        for c in d.children:
            (queue if c.is_dir else files).append(c)

    # Table sizes depend on names only, so they are known before any sector.
    for d in dirs:
        d.table_size = len(_table(d))
    sector = 33
    for d in dirs:
        d.sector = sector
        sector += d.table_size // SECTOR
    for f in files:
        f.sector = sector if f.size else 0
        sector += (f.size + SECTOR - 1) // SECTOR
    for d in dirs:
        d.table = _table(d)                     # now with every sector known

    with open(out_path, "wb") as out:
        out.write(b"\0" * (32 * SECTOR))
        desc = bytearray(SECTOR)
        desc[0:len(MAGIC)] = MAGIC
        filetime = int((time.time() + 11644473600) * 10000000)
        struct.pack_into("<IIQ", desc, 0x14, root.sector, root.table_size, filetime)
        desc[0x7EC:0x7EC + len(MAGIC)] = MAGIC
        out.write(desc)
        for d in dirs:
            out.seek(d.sector * SECTOR)
            out.write(d.table)
        data = 0
        for i, f in enumerate(files):
            if not f.size:
                continue
            out.seek(f.sector * SECTOR)
            with open(f.path, "rb") as src:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
            data += f.size
            if progress:
                progress(i + 1, len(files), f)
        out.seek(0, os.SEEK_END)                # the image ends on a whole sector
        if out.tell() < sector * SECTOR:
            out.truncate(sector * SECTOR)
    return len(files), data
