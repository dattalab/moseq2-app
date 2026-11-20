"""A project folder that a file-layer step must neither read nor write.

The file layer (``moseq2_<package>.steps``) takes its inputs, outputs and
settings explicitly. A step run on recordings that sit inside a project folder
must still behave as if no project were there. ``trap_project`` writes the three
files a project holds, each with values that would change an output if a step
read them: a ``config.yaml`` whose ``extract`` section gives an unusable depth
range and whose ``pca`` section gives other filters, a ``session_config.yaml``
holding entries for the recordings, and a ``moseq2-project.yaml`` listing
sessions under the uuid ``"trap-uuid"`` with folders of their own. A step that
used any of them would write into a listed folder, carry the trap's uuid, or
record the trap's settings; one that wrote to them changes their bytes
(``Trap.unchanged``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from moseq2_common.yaml import write_yaml

#: The uuid every session the trap lists has.
TRAP_UUID = "trap-uuid"

#: An ``extract`` section no recording can be extracted with: a depth range a few millimetres from the camera.
TRAP_EXTRACT = {"bg_roi_depth_range": [1.0, 2.0], "manual_set_depth_range": True, "noise_tolerance": 99}

#: A ``pca`` section with filters no camera preset has.
TRAP_PCA = {"gaussfilter_space": [9.0, 9.0], "medfilter_space": [9], "tailfilter_size": [21, 21]}


@dataclass(frozen=True)
class Trap:
    """The trap's files and the bytes each was written with."""

    folder: Path
    written: dict

    @property
    def config_file(self) -> Path:
        return self.folder / "config.yaml"

    @property
    def session_config_file(self) -> Path:
        return self.folder / "session_config.yaml"

    @property
    def bookkeeping_file(self) -> Path:
        return self.folder / "moseq2-project.yaml"

    def unchanged(self) -> bool:
        """Whether every trap file still holds the bytes it was written with."""
        return all(path.is_file() and path.read_bytes() == data for path, data in self.written.items())


def trap_project(folder: Path, *, extract: Mapping = TRAP_EXTRACT, entries: Mapping = (),
                 sessions: Mapping = (), exclude: Sequence = ()) -> Trap:
    """Write a trap project in ``folder``.

    ``entries`` are ``session_config.yaml``'s entries (session key: settings).
    ``sessions`` maps each session key ``moseq2-project.yaml`` lists to its folder
    relative to ``folder``; each is listed with the uuid ``TRAP_UUID``. ``exclude``
    is ``config.yaml``'s ``exclude``.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    trap = Trap(folder=folder, written={})
    write_yaml({"camera_type": "kinect", "extract": dict(extract), "pca": dict(TRAP_PCA),
                "exclude": list(exclude)}, str(trap.config_file))
    write_yaml({str(key): dict(value) for key, value in dict(entries).items()}, str(trap.session_config_file))
    write_yaml({
        "sessions": {str(key): {"recording": str(key), "folder": str(where), "uuid": TRAP_UUID}
                     for key, where in dict(sessions).items()},
        "pca": {}, "models": {"fitted": [], "current": None, "analysis": {}}, "applied_models": [],
    }, str(trap.bookkeeping_file))
    for path in (trap.config_file, trap.session_config_file, trap.bookkeeping_file):
        trap.written[path] = path.read_bytes()
    return trap


def listing(root: Path) -> set[Path]:
    """Every file and folder under ``root``."""
    return set(Path(root).rglob("*"))
