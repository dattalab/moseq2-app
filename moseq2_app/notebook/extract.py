"""Step 3, Extract, as a GUI: what this run will do, where it runs, and its progress.

``extract()`` opens it on the project in the notebook's folder. Opening it reads
the project and writes nothing: "What this run will do" is
``moseq2_extract.project.plan`` (new, to redo and why, up to date), "Where it
runs" and "More settings" show the project's saved settings. Pressing the
button runs ``moseq2_extract.project.extract`` in the background with the
settings changed on the page, which it saves first, as ``moseq2 extract`` saves
the options typed. While it runs, each recording's row follows its extraction;
a recording that fails shows what happened and the fix in its row, and the rest
are extracted.

On a SLURM cluster the button submits one job per recording and the rows follow
the jobs: while the page shows them, it asks every few seconds (``poll``), and
each row reads its job's state and progress from
``moseq2_extract.project.cluster_runs``: in the queue, extracting with its
percentage, extracted, failed with what happened and the fix, or cancelled.
Stop cancels the jobs (``moseq2_extract.project.stop``). The jobs do not depend
on the notebook: a GUI opened while any job of the latest submission is queued
or running, or after one failed, shows that submission's rows again.

Each setting shown is declared once, with its control, in the options classes
(``moseq2_extract.options``): the computing block's ``cluster_type``, ``cores``,
``partition``, ``memory`` and ``wall_time``, and extraction's ``crop_size``,
``max_height`` and ``use_tracking_model``. Each recording is named by its session's
folder (``session_labels``), the way a person knows it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime

import traitlets

from moseq2_app.notebook._widget import RunningStep, problem_of, session_labels, static_text
from moseq2_common import capture
from moseq2_common.computing import available_cores
from moseq2_common.options import from_gui, gui_fields, read_settings, to_gui
from moseq2_common.project import Project

__all__ = ["ExtractWidget", "extract"]

#: The settings each section shows: the computing block's, then extraction's.
WHERE_IT_RUNS = ("cluster_type", "cores", "partition", "memory", "wall_time")
MORE_SETTINGS = ("crop_size", "max_height", "use_tracking_model")


def _length(seconds: float) -> str:
    return f"{round(seconds / 60)} min" if seconds >= 60 else f"{round(seconds)} s"


#: Where a cluster run happens, as the ``run`` trait says it.
CLUSTER = "the cluster"


def _item(run) -> dict:
    """A recording's row, from where its job stands (``moseq2_extract.project.RecordingRun``)."""
    done = run.total if run.state == "done" and run.total is not None else run.done
    return {"state": run.state, "done": done or 0, "total": run.total,
            "problem": problem_of(run.problem) if run.problem is not None else {}}


def _clock(text: str | None) -> str:
    """An ISO time as the GUI shows it, ``HH:MM``; empty when there is none."""
    try:
        return datetime.fromisoformat(str(text)).astimezone().strftime("%H:%M")
    except (TypeError, ValueError):
        return ""


class ExtractWidget(RunningStep):
    """The Extract step's GUI (module docstring)."""

    STEP = "extract"
    PHASE = "extraction"

    recordings = traitlets.List().tag(sync=True)
    _esm = static_text("kit.js", "steps.js", "extract.js")

    def __init__(self, project: Project | None = None):
        #: Where each recorded job stood when last read, so a progress file read from a cache never moves a row back.
        self._runs: dict = {}
        super().__init__(project)
        self.refresh()

    # ---- reading the project
    def _settings(self):
        from moseq2_extract.options import EXTRACT_STAGE, ExtractComputing, ExtractOptions

        computing = ExtractComputing.resolve(read_settings(self.project.config_file, ExtractComputing,
                                                           stage_classes=()))
        if computing.cores is None:
            computing = ExtractComputing.resolve(explicit={**computing.as_dict(), "cores": available_cores()})
        extraction = ExtractOptions.resolve(read_settings(self.project.config_file, ExtractOptions,
                                                          stage_classes=EXTRACT_STAGE))
        return computing, extraction

    def _read(self) -> None:
        from moseq2_extract import project as extraction_steps
        from moseq2_extract.options import ExtractComputing, ExtractOptions

        planned = extraction_steps.plan(self.project)
        seconds = {}
        for item in planned:
            try:
                seconds[item.key] = extraction_steps.recording_seconds(item.recording)
            except Exception:  # noqa: BLE001 -- a recording that cannot be measured shows no length
                pass
        labels = session_labels([p.key for p in planned])
        self.recordings = [{"id": p.key, "label": labels[p.key], "plan": p.plan, "reason": p.reason,
                            "changed": list(p.changed),
                            "meta": _length(seconds[p.key]) if p.key in seconds else ""} for p in planned]
        computing, extraction = self._settings()
        saved = {**to_gui(computing, WHERE_IT_RUNS), **to_gui(extraction, MORE_SETTINGS)}
        where = gui_fields(ExtractComputing, WHERE_IT_RUNS, saved)
        cores = available_cores()
        for field in where:
            if field["key"] == "cores":
                field.update(max=cores, unit=f"of {cores}")
            if field["key"] == "wall_time":
                to_extract = [seconds[p.key] for p in planned if p.plan in ("new", "redo") and p.key in seconds]
                if to_extract:
                    hours = max(1, -(-int(3 * max(to_extract)) // 3600))
                    field["hint"] = f"About {hours} {'hour' if hours == 1 else 'hours'} for these recordings."
        self.sections = [
            {"title": "Where it runs", "fields": where},
            {"title": "More settings", "fold": True, "fields": gui_fields(ExtractOptions, MORE_SETTINGS, saved)},
        ]
        self.saved = saved
        self._pick_up(planned)

    def _pick_up(self, planned) -> None:
        """When nothing has run in this GUI yet, show the latest cluster submission's rows: while any of its jobs is
        queued or running, as a run going on; when one failed and its recording is still to extract, as a run
        that finished."""
        from moseq2_extract import project as extraction_steps

        with self._lock:
            if self.run.get("state") != "idle" or not self._finished.is_set():
                return
        if not any(p.plan in ("in progress", "new", "redo") for p in planned):
            return
        runs = extraction_steps.cluster_runs(self.project, previous=self._runs)
        if not runs:
            return
        self._runs = runs
        latest = max((run.submitted or "" for run in runs.values()), default="")
        batch = {key: run for key, run in runs.items() if run.submitted == latest or run.live}
        to_extract = {p.key for p in planned if p.plan in ("new", "redo")}
        live = any(run.live for run in batch.values())
        if not live and not any(run.state == "failed" and key in to_extract for key, run in batch.items()):
            return
        run = {"state": "running" if live else "finished", "where": CLUSTER,
               "items": {key: _item(job) for key, job in sorted(batch.items())}}
        if not live:
            run["finished_at"] = max((_clock(job.updated or job.submitted) for job in batch.values()), default="")
        with self._lock:
            if self.run.get("state") == "idle":
                self.run = run

    # ---- following a cluster run
    def _cluster_running(self) -> bool:
        with self._lock:
            return self.run.get("where") == CLUSTER and self.run.get("state") == "running"

    def poll(self) -> None:
        """Read where the cluster jobs of the run shown stand, and update their rows; the page asks every few
        seconds while it shows them running. Nothing happens for a run on this computer, or while the jobs are
        being submitted."""
        if not self._cluster_running() or not self._finished.is_set():
            return
        from moseq2_extract import project as extraction_steps

        runs = extraction_steps.cluster_runs(self.project, previous=self._runs)
        self._runs = runs
        with self._lock:
            items = {key: dict(item) for key, item in self.run.get("items", {}).items()}
            for key in items:
                if key in runs:
                    items[key] = _item(runs[key])
            ended = not any(key in runs and runs[key].live for key in items)
            run = {**self.run, "items": items}
            if ended:
                run.update(state="finished", finished_at=time.strftime("%H:%M"))
            self.run = run
        if ended:
            from moseq2_app.notebook import _state

            _state.changed(self.project)

    def _on_poll(self, _content: Mapping) -> None:
        self.poll()

    def stop(self) -> None:
        """Stop the run: on this computer, at its next chunk; on a cluster, by cancelling its jobs that are queued
        or running (W15). Jobs still being submitted are cancelled as soon as they are."""
        if not self._cluster_running() or not self._finished.is_set():
            super().stop()
            return
        self._cancel()

    def _cancel(self) -> None:
        from moseq2_common.errors import ExpectedError
        from moseq2_extract import project as extraction_steps

        with self._lock:
            keys = list(self.run.get("items", {}))
        ids = extraction_steps.stop(self.project, keys)
        runs = extraction_steps.cluster_runs(self.project, previous=self._runs)
        self._runs = runs
        with self._lock:
            items = {key: (_item(runs[key]) if key in runs else dict(item))
                     for key, item in self.run.get("items", {}).items()}
            self.run = {**self.run, "items": items, "state": "stopped", "finished_at": time.strftime("%H:%M")}
        if ids:
            n = len(ids)
            self.report_problem(ExpectedError(
                f"Cancelled {n} cluster job{'' if n == 1 else 's'} at {time.strftime('%H:%M')}.",
                fix="press Extract to submit the recordings that hadn't finished."))
        from moseq2_app.notebook import _state

        _state.changed(self.project)

    # ---- running
    def _items(self) -> list[str]:
        return [r["id"] for r in self.recordings if r["plan"] in ("new", "redo")]

    def _where(self, settings: Mapping) -> str:
        cluster_type = settings.get("cluster_type", self.saved.get("cluster_type"))
        return "the cluster" if cluster_type == "slurm" else "this computer"

    def _typed(self, settings: Mapping) -> dict:
        """The settings the page changed from the saved ones, in their own form."""
        from moseq2_extract.options import ExtractComputing, ExtractOptions

        changed = {key: value for key, value in settings.items() if self.saved.get(key) != value}
        return {**from_gui(ExtractComputing, {k: v for k, v in changed.items() if k in WHERE_IT_RUNS}),
                **from_gui(ExtractOptions, {k: v for k, v in changed.items() if k in MORE_SETTINGS})}

    def _work(self, settings: Mapping, reporter):
        from moseq2_extract import project as extraction_steps

        return extraction_steps.extract(self.project, settings=self._typed(settings), reporter=reporter)

    def _finish(self, outcome: capture.Outcome) -> dict:
        from moseq2_common.errors import ExpectedError, StepStopped

        run = super()._finish(outcome)
        items = {key: dict(item) for key, item in run.get("items", {}).items()}
        if isinstance(outcome.error, StepStopped):
            self.report_problem(ExpectedError(
                f"Stopped at {time.strftime('%H:%M')}. Recordings that hadn't finished keep what they had before.",
                fix="press Extract to finish them."))
        result = outcome.value
        if result is not None:
            for key in result.extracted:
                items.setdefault(key, {}).update(state="done", problem={})
            for key, error in result.failed.items():
                items.setdefault(key, {}).update(state="failed", problem=problem_of(error))
        run["items"] = items
        if result is not None and result.jobs:
            return self._submitted(run, result)
        return run

    def _submitted(self, run: dict, result) -> dict:
        """The run once its jobs are submitted: going on, on the cluster, each row from its job. A Stop pressed
        while they were being submitted cancels them now."""
        from moseq2_extract import project as extraction_steps

        runs = extraction_steps.cluster_runs(self.project, previous=self._runs)
        self._runs = runs
        items = {key: _item(runs[key]) if key in runs else item for key, item in run["items"].items()}
        for key in result.jobs:
            if key in runs:
                items[key] = _item(runs[key])
        run = {key: value for key, value in run.items() if key != "finished_at"}
        run.update(state="running", where=CLUSTER, items=items)
        if self._stopping.is_set():
            with self._lock:
                self.run = run
            self._cancel()
            with self._lock:
                return dict(self.run)
        return run


def extract() -> ExtractWidget:
    """Open the Extract step: what this run will do, where it runs, then its progress."""
    return ExtractWidget()
