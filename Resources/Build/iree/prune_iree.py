# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Prunes an IREE include tree down to the headers Schola actually needs.

Roots are every ``iree/...`` header included by the ScholaIREE module sources (plus
``iree/runtime/api.h``, which IREE.Build.cs uses to detect the runtime). The transitive
closure of ``#include "iree/..."`` / ``#include <iree/...>`` is kept; everything else
under the include directory is deleted. The closure is deliberately over-approximate
(preprocessor conditionals are ignored), so it never removes a header a compile could need.

Usage: prune_iree.py <include_dir> <scholairee_source_dir>
"""
import re
import sys
from pathlib import Path

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"](iree/[^>"]+)[>"]', re.MULTILINE)
SOURCE_SUFFIXES = {".h", ".hpp", ".cpp", ".c", ".inl"}
ALWAYS_KEEP = {"iree/runtime/api.h"}


def includes_of(path: Path):
    try:
        return INCLUDE_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    include_dir = Path(sys.argv[1]).resolve()
    schola_dir = Path(sys.argv[2]).resolve()

    roots = set(ALWAYS_KEEP)
    for src in schola_dir.rglob("*"):
        if src.is_file() and src.suffix.lower() in SOURCE_SUFFIXES:
            roots.update(includes_of(src))

    keep, pending = set(), list(roots)
    while pending:
        rel = pending.pop()
        if rel in keep:
            continue
        header = include_dir / rel
        if not header.is_file():
            continue  # system or generated header that is not in the tree
        keep.add(rel)
        pending.extend(includes_of(header))

    if "iree/runtime/api.h" not in keep:
        print(f"error: iree/runtime/api.h not found under {include_dir}", file=sys.stderr)
        return 1

    removed = kept_bytes = removed_bytes = 0
    for path in sorted(include_dir.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        rel = path.relative_to(include_dir).as_posix()
        size = path.stat().st_size
        if rel in keep:
            kept_bytes += size
        else:
            path.unlink()
            removed += 1
            removed_bytes += size
    for d in sorted((p for p in include_dir.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()

    print(
        f"Kept {len(keep)} headers ({kept_bytes / 1048576:.2f} MB); "
        f"removed {removed} files ({removed_bytes / 1048576:.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
