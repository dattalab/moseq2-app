"""Ask what is actually in a video file.

Both notebooks write mp4s a user is expected to open in a browser: the
extraction previews, and the crowd movies that show what a syllable looks
like. Whether those play is a property of the encoding rather than of the
frames, and nothing in the pipeline checks it.
"""

from __future__ import annotations

import av


def probe(path, entries=("codec_name", "pix_fmt")):
    """What a video's first stream reports, in the order asked.

    The names are ffprobe's, because that is what the values are usually
    read against: ``codec_name``, ``pix_fmt``, ``width``, ``height``.
    """
    fields = {
        "codec_name": lambda s: s.codec_context.name,
        "pix_fmt": lambda s: s.codec_context.pix_fmt,
        "width": lambda s: s.codec_context.width,
        "height": lambda s: s.codec_context.height,
    }
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [fields[entry](stream) for entry in entries]


def index_is_at_the_front(path):
    """Whether the moov atom precedes the frame data.

    A player that meets the index first can show frame one and seek after
    reading a few kilobytes. With the index appended, it has to fetch the
    whole file before it can show anything.
    """
    data = open(path, "rb").read()
    return data.find(b"moov") < data.find(b"mdat")
