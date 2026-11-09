#!/usr/bin/env python3
"""Extract a departement zip like `unzip -o -q -DD`, using Python's zipfile.

Used by run_department.sh instead of UnZip 6.0, which refuses zip64 archives
whose entries end with 16-byte data descriptors ("overlapped components
(possible zip bomb)"), as written by some streaming zip tools.

Behaves like the unzip call it replaces, so output names stay identical:
  * file names are the raw stored bytes (UTF-8 flag or not), with embedded
    newlines removed, as UnZip does on Linux;
  * extracted files get the current time (like -DD);
  * every file's CRC-32 is verified while it is written (zipfile raises on a
    mismatch).
Safety checks kept: no absolute paths or '..', no overlapping entries, and no
implausible expansion (total size vs archive size).

    extract_zip.py ARCHIVE.zip DEST_DIR
"""
import os
import struct
import sys
import zipfile

MAX_EXPANSION = 20          # stored JPEGs barely compress; refuse anything wilder
CHUNK = 8 * 1024 * 1024


def raw_name(info):
    """The entry name as UnZip writes it on Linux: stored bytes, minus newlines."""
    b = info.filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437")
    return b.replace(b"\n", b"").replace(b"\r", b"")


def check_layout(zf, fh, size):
    infos = sorted((i for i in zf.infolist()), key=lambda i: i.header_offset)
    total = sum(i.file_size for i in infos)
    if total > MAX_EXPANSION * max(size, 1):
        sys.exit(f"ERROR: archive would expand to {total} bytes from {size}; refusing")
    for a, b in zip(infos, infos[1:]):
        fh.seek(a.header_offset)
        hdr = fh.read(30)
        n, e = struct.unpack("<HH", hdr[26:30])
        end = a.header_offset + 30 + n + e + a.compress_size
        if b.header_offset < end:
            sys.exit(f"ERROR: entries overlap ({a.filename!r} / {b.filename!r}); refusing")
    return infos


def main():
    archive, dest = sys.argv[1], sys.argv[2]
    dest_b = os.fsencode(os.path.abspath(dest))
    os.makedirs(dest_b, exist_ok=True)
    with open(archive, "rb") as fh, zipfile.ZipFile(archive) as zf:
        infos = check_layout(zf, fh, os.path.getsize(archive))
        n = 0
        for info in infos:
            name = raw_name(info)
            parts = [p for p in name.split(b"/") if p not in (b"", b".")]
            if name.startswith(b"/") or b".." in parts:
                sys.exit(f"ERROR: unsafe path in archive: {name!r}")
            target = os.path.join(dest_b, *parts) if parts else dest_b
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:   # CRC checked on read
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    out.write(buf)
            n += 1
    print(f"extracted {n} files from {os.path.basename(archive)} (CRC verified)")


if __name__ == "__main__":
    main()
