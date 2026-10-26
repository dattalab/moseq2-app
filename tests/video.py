"""Ask ffmpeg what is actually in a video file.

Both notebooks write mp4s a user is expected to open in a browser: the
extraction previews, and the crowd movies that show what a syllable looks
like. Whether those play is a property of the encoding rather than of the
frames, and nothing in the pipeline checks it.
"""

from __future__ import annotations

import subprocess


def probe(path, entries=("codec_name", "pix_fmt")):
    """What ffprobe reports about a video's first stream, in the order asked."""
    return subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=" + ",".join(entries),
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()


def index_is_at_the_front(path):
    """Whether the moov atom precedes the frame data.

    A player that meets the index first can show frame one and seek after
    reading a few kilobytes. With the index appended, it has to fetch the
    whole file before it can show anything.
    """
    data = open(path, "rb").read()
    return data.find(b"moov") < data.find(b"mdat")
