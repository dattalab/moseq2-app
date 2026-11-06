"""Run records: what a step ran with, beside or inside the output it wrote.

Every pipeline step writes one record of the run that produced its output. A
step starts it before its work (``start_record``), notes what it detects on the
way (``Record.note``, ``Record.ran_with``), and finishes it when its outputs are
written (``Record.finish``). The record is one YAML text block: the
``/provenance`` attribute of an ``.h5`` output (``write_h5_record``), and a
``.yaml`` beside any other output (``write_sidecar``, ``sidecar_path``): the
file's name with ``.yaml`` for its extension (``model.p``: ``model.yaml``), a
``.yaml`` output's stem with ``.record.yaml`` (``syll_info.yaml``:
``syll_info.record.yaml``), or a figure's name without ``.png``/``.pdf``, or a
folder's name, with ``.yaml`` added.

A record holds:

- ``step``, ``front_end`` (``cli``, ``notebook`` or ``script``) and, for a
  command, ``command``: what was typed;
- ``started``, ``ended`` (with their time zone) and ``duration_seconds``;
- ``versions`` of the six moseq2 packages and of the libraries results depend
  on, and ``git`` (the commit and whether the checkout had uncommitted changes)
  only when the packages run from a git checkout;
- ``options``: the settings the step ran with, after every detection;
- ``inputs``: each input file with its absolute path, its path relative to the
  project, its ``fingerprint`` (size and content hash), for a file the pipeline
  wrote the ``run`` its own record names (``run_of``), and what describes its
  data (frame count, resolution, frame rate, camera) where the step knows it;
- ``notes``: anything else the step found worth saying (a missing metadata.json);
- ``outputs``: the outputs' array shapes and types, and the files written.

A fact that only describes the run and cannot be gathered is written as
``unknown: <reason>``; gathering a record never stops a run. Nothing in it
identifies the user or the machine beyond what its paths contain.

Records are read back to check, never to run on: a step runs on ``config.yaml``
and the recording's arena entry, and ``read_record`` serves display and checks.

No check reads a file's modification time. A file the pipeline wrote is
identified by the run its record names: the step and its start and end times
(``run_of``); a file carrying no record, such as a raw recording, which nothing
rewrites, by its size and the hash of its first and last MB (``fingerprint``).
``unchanged`` asks whether a file is still the one a record's input names, so a
project copied elsewhere with new file times is the same project.
``print-record`` (``print_record_command``), on all four programs, prints the
record of an output, a session's folder or a session key.

Outputs and their record are written together through a temporary name renamed
into place (``atomic_output``, with ``partial_name`` for a writer that tells the
format from the extension, or ``atomic_outputs`` for the files a writer names
from one stem), so an interrupted step leaves the previous output and record as
they were.

Each step checks the files it reads before any work (``check_inputs``): one that
does not exist, is empty or cannot be read stops it with a message naming the
step, the file and what to do (``InputError``). Every such check, here or in a
package, raises a ``StepInputError``, which the command line prints as a message.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shlex
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path

import click
import h5py
import numpy as np

from moseq2_common.yaml import dump_block_yaml, read_yaml

__all__ = ["RECORD_ATTRIBUTE", "InputError", "Record", "StepInputError", "atomic_output", "atomic_outputs",
           "check_inputs", "fingerprint", "find_record", "partial_name", "print_record_command", "read_record",
           "record_text", "run_of", "sidecar_path", "start_record", "unchanged", "write_h5_record", "write_record",
           "write_sidecar", "written_on"]

#: The attribute of an ``.h5`` output holding its record.
RECORD_ATTRIBUTE = "provenance"

#: Files up to this size are hashed in full; larger ones by their first and last ``_EDGE`` bytes.
SMALL_FILE = 8 * 1024 ** 2
_EDGE = 1024 ** 2

#: The packages whose versions a record holds: the six moseq2 packages, then the libraries results depend on.
#: A library is named by its distributions, the first one installed answering.
PACKAGES = ("moseq2-app", "moseq2-common", "moseq2-extract", "moseq2-pca", "moseq2-model", "moseq2-viz")
LIBRARIES = {
    "numpy": ("numpy",),
    "scipy": ("scipy",),
    "h5py": ("h5py",),
    "scikit-learn": ("scikit-learn",),
    "opencv": ("opencv-python-headless", "opencv-python", "opencv-contrib-python-headless",
               "opencv-contrib-python"),
    "av": ("av",),
    "jax": ("jax",),
    "jax-moseq": ("jax-moseq",),
}


def _unknown(reason: str) -> str:
    return f"unknown: {reason}"


def _now() -> datetime:
    return datetime.now().astimezone()


# ------------------------------------------------------------------ the run


def _versions() -> dict:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "not installed"
    for name, distributions in LIBRARIES.items():
        found[name] = "not installed"
        for distribution in distributions:
            try:
                found[name] = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
    return found


def _checkout(folder: Path) -> Path | None:
    """The git checkout ``folder`` is in, or None."""
    for candidate in (folder, *folder.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git() -> dict | str | None:
    """The commit of the checkout the packages run from, and whether it has uncommitted changes; None when
    they are installed rather than run from a checkout."""
    checkout = _checkout(Path(__file__).resolve().parent)
    if checkout is None:
        return None
    try:
        commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10)
        status = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return _unknown("the packages run from a git checkout, and git is not on PATH to ask for its commit")
    except (OSError, subprocess.SubprocessError) as error:
        return _unknown(f"git could not be run ({error})")
    if commit.returncode != 0:
        return _unknown(f"git rev-parse failed: {commit.stderr.strip()}")
    return {"commit": commit.stdout.strip(), "uncommitted_changes": bool(status.stdout.strip())}


def _invocation() -> tuple[str, str | None]:
    """The front end running the step, and the command typed when it is a command."""
    context = click.get_current_context(silent=True)
    if context is not None:
        name = context.find_root().info_name or ""
        program = os.path.basename(sys.argv[0]) if sys.argv else ""
        if program and (program == name or program.startswith("moseq2-")):
            return "cli", shlex.join([program, *sys.argv[1:]])
        return "cli", context.command_path
    shell = sys.modules.get("IPython")
    if shell is not None and getattr(shell, "get_ipython", lambda: None)() is not None:
        return "notebook", None
    return "script", None


# ------------------------------------------------------------------ files


def fingerprint(path: str | os.PathLike) -> dict:
    """``path``'s size and SHA-256: of the whole file up to ``SMALL_FILE``, else of its first and last 1 MB
    (about 0.02 s for any size). No file time: a copy of a file has its fingerprint."""
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        if stat.st_size <= SMALL_FILE:
            for block in iter(lambda: handle.read(_EDGE), b""):
                digest.update(block)
            hashed = "whole file"
        else:
            digest.update(handle.read(_EDGE))
            handle.seek(-_EDGE, os.SEEK_END)
            digest.update(handle.read(_EDGE))
            hashed = "first and last 1 MB"
    return {"size": stat.st_size, "sha256": digest.hexdigest(), "hashed": hashed}


def run_of(path: str | os.PathLike) -> dict | None:
    """The run that wrote ``path``, from the record it carries: its ``step``, ``started`` and ``ended``, and,
    for an extraction a flip classifier was applied to, when that was (``flip_corrected``). None when ``path``
    carries no record it can be read from (a raw recording, a downloaded flip classifier)."""
    try:
        record = read_record(path)
    except Exception:  # an unreadable record identifies nothing; the file is then known by its fingerprint
        return None
    if not isinstance(record, Mapping):
        return None
    run = {"step": record.get("step"), "started": record.get("started"), "ended": record.get("ended")}
    flip = (record.get("notes") or {}).get("flip_correction")
    if isinstance(flip, Mapping) and flip.get("applied"):
        run["flip_corrected"] = flip["applied"]
    return run


def unchanged(entry: Mapping, path: str | os.PathLike) -> bool:
    """Whether ``path`` is still the file a record's input ``entry`` describes: the same run (``run_of``) when
    either names one, else the same size and content hash. Never by file time."""
    run = run_of(path)
    if run is not None or entry.get("run") is not None:
        return run == entry.get("run")
    recorded = entry.get("fingerprint")
    if not isinstance(recorded, Mapping):
        return False
    now = fingerprint(path)
    return (now["size"], now["sha256"]) == (recorded.get("size"), recorded.get("sha256"))


def written_on(when) -> str:
    """A record's time (``ended``, ``flip_corrected``) as messages give it: ``2026-11-06 14:03``."""
    if not isinstance(when, datetime):
        when = datetime.fromisoformat(str(when))
    return when.strftime("%Y-%m-%d %H:%M")


def _relative(path: Path, project) -> str | None:
    if project is None:
        return None
    try:
        return Path(os.path.relpath(path, Path(project.folder).resolve())).as_posix()
    except ValueError:  # another drive on Windows
        return None


def _describe_input(item, project) -> dict:
    """One input: a path, or a mapping holding ``path`` and the facts that describe its data."""
    facts = dict(item) if isinstance(item, Mapping) else {"path": item}
    path = Path(os.path.abspath(facts.pop("path")))
    out = {"path": str(path), "relative_path": _relative(path, project)}
    if path.is_dir():
        out["kind"] = "folder"
    elif path.is_file():
        try:
            out["fingerprint"] = fingerprint(path)
        except OSError as error:
            out["fingerprint"] = _unknown(f"{path} could not be read ({error})")
        run = run_of(path)
        if run is not None:
            out["run"] = run
    else:
        out["fingerprint"] = _unknown(f"{path} does not exist")
    out.update({name: _plain(value) for name, value in facts.items()})
    return out


def _plain(value):
    """``value`` as YAML holds it: arrays and datasets by their shape and type, paths as text."""
    if isinstance(value, (h5py.Dataset, np.ndarray)):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ------------------------------------------------------------------ the record


class Record:
    """A run's record, gathered as the step runs; ``finish`` returns it as a dict."""

    def __init__(self, data: dict, started: datetime, project=None):
        self.data = data
        self._started = started
        self._project = project

    def ran_with(self, values: Mapping) -> None:
        """Set option values the step settled while running (a detected depth range, the floor depth)."""
        self.data["options"].update(_plain(dict(values)))

    def note(self, key: str, value: object) -> None:
        """Note a fact about the run that is not an option or an input."""
        self.data.setdefault("notes", {})[key] = _plain(value)

    def add_inputs(self, inputs: Sequence) -> None:
        """Inputs found once the step is under way (the extractions PCA reads)."""
        self.data["inputs"].extend(_describe_input(item, self._project) for item in inputs)

    def finish(self, outputs: Mapping[str, object]) -> dict:
        """The record, ended now, with ``outputs`` (arrays by shape and type, files by path)."""
        ended = _now()
        self.data["ended"] = ended.isoformat()
        self.data["duration_seconds"] = round((ended - self._started).total_seconds(), 3)
        self.data["outputs"] = _plain(dict(outputs))
        return dict(self.data)


def start_record(step: str, options, inputs: Sequence = (), *, project=None, front_end: str | None = None,
                 command: str | None = None) -> Record:
    """Begin the record of ``step``, running with ``options`` (an options object or a mapping) on ``inputs``.

    ``front_end`` and ``command`` are told from how the step was called when not given: a click command
    is ``cli``, an IPython kernel ``notebook``, anything else ``script``.
    """
    started = _now()
    detected_front_end, detected_command = _invocation()
    front_end = front_end or detected_front_end
    command = command if command is not None else detected_command
    data = {"step": step, "front_end": front_end}
    if front_end == "cli":
        data["command"] = command if command is not None else _unknown("the command line was not available")
    data["started"] = started.isoformat()
    data["project"] = str(Path(project.folder).resolve()) if project is not None else None
    data["versions"] = _versions()
    git = _git()
    if git is not None:
        data["git"] = git
    values = options.as_dict() if hasattr(options, "as_dict") else dict(options or {})
    data["options"] = _plain(values)
    data["inputs"] = [_describe_input(item, project) for item in inputs]
    return Record(data, started, project)


# ------------------------------------------------------------------ writing and reading


def record_text(record: Mapping) -> str:
    return dump_block_yaml(_plain(dict(record)))


def write_h5_record(h5: h5py.File, record: Mapping) -> None:
    """Store ``record`` in the open ``h5`` as its ``/provenance`` attribute."""
    h5.attrs[RECORD_ATTRIBUTE] = record_text(record)


def sidecar_path(output: str | os.PathLike) -> Path:
    """Where the record of ``output`` is written: beside a file under its name with ``.yaml`` for its
    extension (``model.p``: ``model.yaml``), beside a ``.yaml`` output under its stem with ``.record.yaml``
    (``syll_info.yaml``: ``syll_info.record.yaml``), and beside a figure's name without extension or a
    folder with ``.yaml`` added (``plots/usage``: ``plots/usage.yaml``)."""
    output = Path(str(output).rstrip("/\\"))
    if output.suffix == ".yaml":
        return output.with_name(f"{output.stem}.record.yaml")
    if output.is_file():
        return output.with_suffix(".yaml")
    return Path(f"{output}.yaml")


def write_sidecar(output: str | os.PathLike, record: Mapping) -> Path:
    """Write ``record`` beside ``output`` (``sidecar_path``), through a temporary file renamed into place."""
    path = sidecar_path(output)
    with atomic_output(path) as temporary:
        temporary.write_text(record_text(record))
    return path


def write_record(output: str | os.PathLike, record: Mapping) -> Path:
    """Attach ``record`` to ``output``: its ``/provenance`` attribute when ``output`` is an ``.h5`` file,
    else a ``.yaml`` beside it (``write_sidecar``). Returns the file holding the record."""
    output = Path(output)
    if output.suffix == ".h5" and output.is_file():
        with h5py.File(output, "a") as f:
            write_h5_record(f, record)
        return output
    return write_sidecar(output, record)


def partial_name(path: str | os.PathLike) -> Path:
    """The temporary name of ``path`` that keeps its extension (``.<stem>.partial<suffix>``), for
    ``atomic_output`` around a writer that tells the format from the extension."""
    path = Path(path)
    return path.with_name(f".{path.stem}.partial{path.suffix}")


@contextlib.contextmanager
def atomic_output(path: str | os.PathLike, temporary: str | os.PathLike | None = None) -> Iterator[Path]:
    """Yield a temporary name beside ``path`` to write to; rename it to ``path`` when the block completes.

    If the block stops (an error, Ctrl-C), the temporary file is deleted and ``path`` is as it was; a block
    that completes without writing the temporary file (an output switched off) leaves ``path`` alone.
    ``temporary`` defaults to ``.<name>.partial`` in ``path``'s folder; a writer that tells the format
    from the extension is given one that keeps it.
    """
    path = Path(path)
    temporary = Path(temporary) if temporary is not None else path.with_name(f".{path.name}.partial")
    try:
        yield temporary
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    if temporary.exists():
        os.replace(temporary, path)


@contextlib.contextmanager
def atomic_outputs(stem: str | os.PathLike, suffixes: Sequence[str]) -> Iterator[Path]:
    """Yield a temporary stem beside ``stem`` (``.<name>.partial``) for a writer that names its files from a
    stem; each ``<temporary stem><suffix>`` the block writes is renamed to ``<stem><suffix>`` when the block
    completes (``atomic_output`` for each), the first suffix last. If the block stops, every temporary file is
    deleted and the files under the final names are as they were."""
    stem = Path(stem)
    temporary = stem.with_name(f".{stem.name}.partial")
    with contextlib.ExitStack() as stack:
        for suffix in suffixes:
            stack.enter_context(atomic_output(f"{stem}{suffix}", f"{temporary}{suffix}"))
        yield temporary


# ------------------------------------------------------------------ checking inputs


class StepInputError(Exception):
    """A step's inputs or settings are not usable, found by a check at the top of the step before any work.

    Its message names what was found and says what to do; a ``ProjectCommand`` prints it as a message,
    not a traceback. The packages' checks subclass it (with ``ValueError``, as they raised before).
    """


class InputError(StepInputError, ValueError):
    """A file a step needs that does not exist, is empty or cannot be read."""


def _input_problem(path: Path) -> str | None:
    """What is wrong with ``path`` as an input (``does not exist``, ``is empty``, ``could not be read: ...``),
    or None. A folder is checked only for existing; an ``.h5`` is opened."""
    if not path.exists():
        return "does not exist"
    if path.is_dir():
        return None
    try:
        if path.stat().st_size == 0:
            return "is empty"
        with open(path, "rb") as handle:
            handle.read(1)
        if path.suffix == ".h5":
            with h5py.File(path, "r"):
                pass
    except OSError as error:
        return f"could not be read: {error}"
    return None


def check_inputs(step: str, inputs: Mapping[str | os.PathLike, str]) -> None:
    """Stop ``step`` before its work when a file it reads does not exist, is empty or cannot be read.

    ``inputs`` maps each file to what the person should do when it is unusable; the message names the step,
    the file and the problem, then says that.
    """
    for path, what_to_do in inputs.items():
        problem = _input_problem(Path(path))
        if problem is not None:
            raise InputError(f"{step} needs {path}, which {problem}. {what_to_do}")


def _stored_text(path: Path) -> str | None:
    """The record text stored for ``path``: an ``.h5``'s attribute, else its sidecar; None when it has none."""
    if path.is_file() and path.suffix == ".h5":
        with h5py.File(path, "r") as f:
            text = f.attrs.get(RECORD_ATTRIBUTE)
        if text is None:
            return None
        return text.decode() if isinstance(text, bytes) else str(text)
    sidecar = sidecar_path(path)
    return sidecar.read_text() if sidecar.is_file() else None


def read_record(path: str | os.PathLike) -> dict:
    """The record of the output at ``path``, for display and checks; never settings to run on."""
    text = _stored_text(Path(path))
    if text is None:
        raise FileNotFoundError(f"{path} carries no record.")
    return read_yaml(io.StringIO(text))


def find_record(target: str, cwd: str | os.PathLike | None = None) -> tuple[Path, str]:
    """The output ``target`` names and its record's text: an output file, a session's folder in
    aggregate_results (its ``results.h5``), or a session key listed in the project found from ``cwd``."""
    cwd = Path(cwd) if cwd is not None else Path.cwd()
    path = Path(target) if Path(target).is_absolute() else cwd / target
    candidates = [path]
    if path.is_dir():
        candidates.insert(0, path / "results.h5")
    from moseq2_common.bookkeeping import load, session_h5
    from moseq2_common.project import find_project

    project = find_project(cwd=cwd)
    if project is not None and Path(project.bookkeeping_file).is_file():
        session = load(project).sessions.get(Path(target).as_posix())
        if session is not None:
            candidates.append(session_h5(project, session))
    for candidate in candidates:
        text = _stored_text(candidate)
        if text is not None:
            return candidate, text
    where = f" or a session listed in {project.bookkeeping_file}" if project is not None else ""
    raise click.ClickException(
        f"No record found for {target}: it is not an output carrying a record, a session's folder in "
        f"aggregate_results{where}. Give the path of an .h5, a figure, a model or a crowd-movie folder, a "
        "session's folder, or (from the project folder) a session key such as cohort1/saline_example_4/depth.avi."
    )


def print_record_command() -> click.Command:
    """``print-record``, for each program's command group."""

    from moseq2_common.options import ProjectCommand, described_argument

    @click.command("print-record", cls=ProjectCommand, help="Print the record of the run that wrote an output: an .h5, a figure "
                   "or model, a session folder in aggregate_results, or a session key. Paste it into a "
                   "GitHub issue.")
    @described_argument(
        "output",
        missing="the output whose record to print: an .h5 (such as _pca/pca.h5), a figure, a model, a "
                "crowd-movie folder, a session's folder in aggregate_results, or (from the project folder) a "
                "session key such as cohort1/saline_example_4/depth.avi.",
    )
    def print_record(output):
        path, text = find_record(output)
        click.echo(f"# record of {path}")
        click.echo(text, nl=not text.endswith("\n"))

    return print_record
