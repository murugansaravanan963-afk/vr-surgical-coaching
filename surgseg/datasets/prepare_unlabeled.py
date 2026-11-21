"""Any unlabeled surgical footage -> unlabeled training frames for teacher distillation.

The student learns to imitate the teacher on these frames (no masks needed), so every
hour of surgical video helps. Good sources: Cholec80 / CholecT50 / HeiChole (request
forms), SurgeNetYoutube on Hugging Face (accept its terms), robotic recordings from
your lab, and - most valuable - recordings of YOUR simulator.

  python -m surgseg.datasets.prepare_unlabeled --src /path/videos_or_frames \
      --out $DATA/processed --name cholec80_unl --every 25 --max-per-video 2000
"""
from __future__ import annotations

from pathlib import Path

import cv2

from surgseg.datasets.common import SampleWriter, base_argparser, read_rgb

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".mpg", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# Folders that hold annotations, not footage (dataset releases mix both).
SKIP_DIRS = {"segmentation", "masks", "mask", "labels", "label", "ground_truth", "gt",
             "annotations", "semseg", "insseg"}


def key(path: Path, src: Path) -> str:
    """Unique id from the relative path: many releases name every file video_left.avi."""
    rel = path.relative_to(src).with_suffix("")
    return "_".join(rel.parts) if rel.parts else path.stem


def main() -> None:
    p = base_argparser(__doc__)
    p.add_argument("--name", required=True, help="dataset name to write")
    p.add_argument("--every", type=int, default=25, help="keep every Nth video frame")
    p.add_argument("--max-per-video", type=int, default=0)
    args = p.parse_args()
    src = Path(args.src)
    w = SampleWriter(args.out, args.name, args.max_side)
    files = sorted(q for q in src.rglob("*") if q.suffix.lower() in VIDEO_EXT | IMAGE_EXT
                   and not SKIP_DIRS & {d.lower() for d in q.relative_to(src).parts[:-1]})
    n = 0
    for f in (q for q in files if q.suffix.lower() in VIDEO_EXT):
        cap, idx, kept = cv2.VideoCapture(str(f)), 0, 0
        while cap.grab():
            if idx % args.every == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    vid = key(f, src)
                    w.add(f"{vid}_{idx:07d}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None,
                          group=vid, split="train", labeled=[])
                    kept += 1
                    n += 1
                if args.max_per_video and kept >= args.max_per_video:
                    break
            idx += 1
        cap.release()
    for f in (q for q in files if q.suffix.lower() in IMAGE_EXT):   # frame folders
        folder = key(f.parent, src) if f.parent != src else src.name
        w.add(key(f, src), read_rgb(f), None, group=folder, split="train", labeled=[])
        n += 1
        if args.limit and n >= args.limit:
            break
    w.close()


if __name__ == "__main__":
    main()
