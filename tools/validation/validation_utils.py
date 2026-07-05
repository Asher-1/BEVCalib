"""Shared helpers for dataset validation (JPG/PNG image_2)."""

from pathlib import Path
from typing import List, Optional

# matplotlib scatter marker area (points^2); tiny for dense LiDAR overlays
PROJECTION_SCATTER_SIZE = 0.08
PROJECTION_SCATTER_ALPHA = 0.4


def list_sequence_images(image_dir: Path) -> List[Path]:
    """List image_2 frames; prefers .jpg over .png when both exist."""
    if not image_dir.exists():
        return []
    by_stem = {}
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for path in image_dir.glob(ext):
            stem = path.stem
            if stem not in by_stem or path.suffix.lower() in ('.jpg', '.jpeg'):
                by_stem[stem] = path
    return sorted(by_stem.values(), key=lambda p: p.stem)


def resolve_image_path(seq_dir: Path, frame: int) -> Optional[Path]:
    """Resolve a frame index to image_2 path (.jpg preferred)."""
    image_dir = seq_dir / 'image_2'
    for ext in ('.jpg', '.jpeg', '.png'):
        path = image_dir / f'{frame:06d}{ext}'
        if path.exists():
            return path
    return None
