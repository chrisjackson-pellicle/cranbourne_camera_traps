"""Subsample camera-trap images from nested folders.

Images are expected under a parent directory whose immediate subfolders are
the sampling groups (for example `WC001`, `WC002` beneath `01_wylies_creek`).
Deeper nesting such as date folders is searched recursively. Use `--subfolders`
to restrict sampling to a subset of those groups.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_WORKERS = 8


def _list_images(root: Path) -> list[Path]:
    """Return sorted image paths under `root`.

    Args:
        root: Directory to search recursively.

    Returns:
        Image paths found beneath `root`, sorted for stable sampling.
    """
    images: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                images.append(Path(dirpath, name))
    images.sort()
    return images


def collect_images(
    parent: Path,
    subfolders: Sequence[str] | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, list[Path]]:
    """Collect image files grouped by first-level subfolder.

    Args:
        parent: Directory whose immediate subfolders are sampling groups
            (for example camera-site folders).
        subfolders: Names of main subfolders to include. If `None`, every
            first-level subfolder under `parent` is considered.
        workers: Maximum number of threads used to walk subfolders in
            parallel. Directory listing is I/O-bound, so threads are used
            rather than processes.

    Returns:
        Mapping of each included subfolder name to a sorted list of image
        paths found beneath it. Empty subfolders are omitted unless they
        were named explicitly in `subfolders`.

    Raises:
        FileNotFoundError: If a name in `subfolders` is not an immediate
            subdirectory of `parent`.
    """
    available = {path.name: path for path in parent.iterdir() if path.is_dir()}
    if subfolders:
        missing = [name for name in subfolders if name not in available]
        if missing:
            available_list = ", ".join(sorted(available)) or "(none)"
            raise FileNotFoundError(
                f"Subfolder(s) not found under {parent}: {', '.join(missing)}. "
                f"Available: {available_list}"
            )
        selected = [available[name] for name in dict.fromkeys(subfolders)]
        include_empty = True
    else:
        selected = [available[name] for name in sorted(available)]
        include_empty = False

    worker_count = max(1, min(workers, len(selected)))
    if worker_count == 1:
        pairs = [(subfolder.name, _list_images(subfolder)) for subfolder in selected]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            pairs = list(pool.map(lambda path: (path.name, _list_images(path)), selected))

    grouped: dict[str, list[Path]] = {}
    for name, images in pairs:
        if images or include_empty:
            grouped[name] = images
    return grouped


def sample_images(
    grouped: dict[str, list[Path]],
    n: int,
    equal: bool,
    rng: random.Random,
) -> list[Path]:
    """Draw a subsample of image paths.

    Args:
        grouped: Image paths grouped by main subfolder name.
        n: Number of images requested.
        equal: If `True`, draw as evenly as possible from each group,
            redistributing leftover quota when a group runs out. If `False`,
            draw uniformly from the pooled set of all images.
        rng: Random generator used for sampling.

    Returns:
        Sampled image paths. Fewer than `n` paths are returned when the
        parent folder does not contain enough images.
    """
    all_images = [path for paths in grouped.values() for path in paths]
    if n >= len(all_images):
        rng.shuffle(all_images)
        return all_images
    if not equal:
        return rng.sample(all_images, n)
    return _sample_equal(grouped, n, rng)


def _sample_equal(
    grouped: dict[str, list[Path]],
    n: int,
    rng: random.Random,
) -> list[Path]:
    """Sample as evenly as possible across groups.

    Args:
        grouped: Image paths grouped by main subfolder name.
        n: Number of images requested.
        rng: Random generator used for shuffling.

    Returns:
        Sampled image paths.
    """
    pools = {name: list(paths) for name, paths in grouped.items() if paths}
    for paths in pools.values():
        rng.shuffle(paths)

    selected: list[Path] = []
    remaining = n
    while remaining > 0 and pools:
        names = list(pools)
        rng.shuffle(names)
        base, extra = divmod(remaining, len(names))
        exhausted: list[str] = []
        for i, name in enumerate(names):
            quota = base + (1 if i < extra else 0)
            if quota == 0:
                continue
            take = min(quota, len(pools[name]))
            selected.extend(pools[name][:take])
            del pools[name][:take]
            remaining -= take
            if not pools[name]:
                exhausted.append(name)
        for name in exhausted:
            del pools[name]
    return selected


def _copy_one(source: Path, parent: Path, output: Path) -> tuple[Path, Path]:
    """Copy one image into `output`, preserving its path relative to `parent`.

    Args:
        source: Image path to copy.
        parent: Parent directory used to compute the relative destination.
        output: Destination directory.

    Returns:
        The `(source, destination)` pair.
    """
    destination = output / source.relative_to(parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return source, destination


def copy_images(
    sampled: list[Path],
    parent: Path,
    output: Path,
    workers: int = DEFAULT_WORKERS,
) -> list[tuple[Path, Path]]:
    """Copy sampled images into `output`, preserving relative paths.

    Args:
        sampled: Image paths to copy.
        parent: Parent directory used to compute relative destinations.
        output: Destination directory.
        workers: Maximum number of threads used to copy files in parallel.

    Returns:
        Pairs of `(source, destination)` for each copied file.
    """
    worker_count = max(1, min(workers, len(sampled)))
    if worker_count == 1:
        return [_copy_one(source, parent, output) for source in sampled]
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        return list(
            pool.map(lambda source: _copy_one(source, parent, output), sampled)
        )


def write_manifest(
    copied: list[tuple[Path, Path]],
    parent: Path,
    manifest_path: Path,
) -> None:
    """Write a CSV mapping copied files back to their source paths.

    Args:
        copied: Pairs of `(source, destination)` paths.
        parent: Parent directory used to recover the main subfolder name.
        manifest_path: Destination CSV path.
    """
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "destination", "group"])
        for source, destination in copied:
            writer.writerow([str(source), str(destination), _group_name(source, parent)])


def _group_name(path: Path, parent: Path) -> str:
    """Return the first-level subfolder name for an image path.

    Args:
        path: Image path beneath `parent`.
        parent: Parent directory passed to the sampler.

    Returns:
        Name of the main subfolder that contains `path`.
    """
    return path.relative_to(parent).parts[0]


def _print_summary(
    grouped: dict[str, list[Path]],
    sampled: list[Path],
    parent: Path,
    requested: int,
) -> None:
    """Print available and sampled counts per main subfolder.

    Args:
        grouped: Image paths grouped by main subfolder name.
        sampled: Sampled image paths.
        parent: Parent directory passed to the sampler.
        requested: Number of images the user asked for.
    """
    sampled_counts: dict[str, int] = {name: 0 for name in grouped}
    for path in sampled:
        sampled_counts[_group_name(path, parent)] += 1

    print(f"Found {sum(len(paths) for paths in grouped.values())} images in {parent}")
    print(f"Requested {requested}, sampled {len(sampled)}")
    print()
    print(f"{'group':<20} {'available':>10} {'sampled':>10}")
    for name in grouped:
        print(f"{name:<20} {len(grouped[name]):>10} {sampled_counts[name]:>10}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse. Defaults to `sys.argv[1:]`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Subsample image files from nested camera-trap folders. "
            "Grouping uses the immediate subfolders of --parent "
            "(for example WC001, WC002)."
        )
    )
    parser.add_argument(
        "--parent",
        "-p",
        type=Path,
        required=True,
        help="Parent folder containing main subfolders such as WC001, WC002.",
    )
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        required=True,
        help="Number of images to recover.",
    )
    parser.add_argument(
        "--subfolders",
        "-f",
        nargs="+",
        metavar="NAME",
        help=(
            "Main subfolders to sample from (for example WC001 WC003). "
            "Defaults to all immediate subfolders of --parent."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Directory to copy sampled images into. Required unless --dry-run.",
    )
    parser.add_argument(
        "--equal",
        "-e",
        action="store_true",
        help=(
            "Draw an equal number of images from each main subfolder where "
            "possible. Remainder is redistributed when a subfolder has too few."
        ),
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sample summary without copying files.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Threads for directory scanning and file copies "
            f"(default: {DEFAULT_WORKERS}). Use 1 to disable parallelism. "
            "On a NAS, a small value is often faster than a large one."
        ),
    )
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is set")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run image subsampling from the command line.

    Args:
        argv: Argument list to parse. Defaults to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    parent = args.parent.expanduser().resolve()
    if not parent.is_dir():
        print(f"Parent folder does not exist or is not a directory: {parent}", file=sys.stderr)
        return 1

    print(f"Scanning images under {parent} ({args.workers} workers)...", flush=True)
    try:
        grouped = collect_images(parent, args.subfolders, workers=args.workers)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not any(grouped.values()):
        scope = (
            ", ".join(args.subfolders)
            if args.subfolders
            else str(parent)
        )
        print(f"No image files found in {scope}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    sampled = sample_images(grouped, args.n, args.equal, rng)
    sampled.sort()
    _print_summary(grouped, sampled, parent, args.n)

    if len(sampled) < args.n:
        print(
            f"\nWarning: only {len(sampled)} images available; requested {args.n}.",
            file=sys.stderr,
        )

    if args.dry_run:
        return 0

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied = copy_images(sampled, parent, output, workers=args.workers)
    manifest_path = output / "sample_manifest.csv"
    write_manifest(copied, parent, manifest_path)
    print(f"\nCopied {len(copied)} images to {output}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
