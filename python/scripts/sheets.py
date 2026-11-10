# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Makes the frame-sheet fixtures, `python/tests/fixtures/sheets/`, with the video-thumbnail Plugin's service.

Run by hand (`pnpm sheets`) when the fixtures need remaking; never by `pnpm check`. It starts the `thumbs-collage`
service the video-thumbnail Plugin (`fyflow-ext.video-thumbnail`) runs, asks it for sheets of that Plugin's own test
clip, `test/fixtures/sample.mp4` (6 s, 320x180: one second of black, then ffmpeg's test pattern), with the requests
that Plugin's `VideoThumbnail` makes, and writes the JPEGs as the service hands them back: the frames' places and
timestamps in their comment (COM) segment.

The service is started with the clip's folder as its root, so a sheet's `video` is `sample.mp4`, not a path of the
machine that made it. Where the sibling repo is: `--thumbnails <folder>`, by default `../fyflow-plugin-video-thumbnails`
beside this repo; its service from `bin/<platform>-<arch>/` (`pnpm copy-service` there puts it in place).
"""

import argparse
import json
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

PYTHON = Path(__file__).resolve().parent.parent
OUT = PYTHON / "tests" / "fixtures" / "sheets"


def request(**overrides):
    """What `VideoThumbnail` asks the service for with the README's recommended settings (here a 3x3 grid of 480x270
    for a short clip), and `overrides`."""
    return {
        "path": "sample.mp4",
        # 7 frames in 9 cells, so the last two cells are null: no timestamp for them.
        "selection": {"mode": "uniform", "frames": 7, "start_offset": 0.5, "end_offset": 0.5},
        "collage": {"mode": "grid", "rows": 3, "cols": 3, "spacing": 0, "margin": 0, "width": 480, "height": 270},
        "fit": "stretch",
        "render_timestamps": False,
        "skip_blank_frames": True,
        **overrides,
    }


SHEETS = {
    "sample.jpg": request(),
    # As above, with the timestamps drawn on the frames.
    "sample-timestamps.jpg": request(render_timestamps=True),
}


def started(executable: Path, root: Path, font: Path) -> tuple[subprocess.Popen, str]:
    args = [str(executable), "--bind", "127.0.0.1:0", "--root-path", str(root), "--font-file", str(font)]
    args += ["--exit-on-stdin-close", "--max-workers", "2"]
    service = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=executable.parent
    )
    found: list[str] = []

    def read():
        for line in service.stdout:
            if not found and (match := re.search(r"Listening on (http://\S+)", line)):
                found.append(match.group(1))

    threading.Thread(target=read, daemon=True).start()
    deadline = time.monotonic() + 15
    while not found:
        if service.poll() is not None or time.monotonic() > deadline:
            service.kill()
            sys.exit(f"The service {executable} did not start")
        time.sleep(0.05)
    return service, found[0]


def main():
    machine = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[platform.machine().lower()]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--thumbnails", type=Path, default=PYTHON.parent.parent / "fyflow-plugin-video-thumbnails")
    args = parser.parse_args()
    exe = "thumbs-collage.exe" if sys.platform == "win32" else "thumbs-collage"
    executable = args.thumbnails / "bin" / f"{sys.platform}-{machine}" / exe
    clip = args.thumbnails / "test" / "fixtures" / "sample.mp4"
    font = args.thumbnails / "assets" / "fonts" / "DejaVuSans.ttf"
    for needed in (executable, clip, font):
        if not needed.is_file():
            sys.exit(f"Not found: {needed}")

    service, url = started(executable, clip.parent, font)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, body in SHEETS.items():
            post = urllib.request.Request(
                f"{url}/make_collage",
                data=json.dumps(body).encode(),
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(post, timeout=60) as response:
                (OUT / name).write_bytes(response.read())
            print(f"Wrote {OUT / name}")
    finally:
        service.stdin.close()
        try:
            service.wait(5)
        except subprocess.TimeoutExpired:
            service.kill()


if __name__ == "__main__":
    main()
