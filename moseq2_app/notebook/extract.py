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

Each setting shown is declared once, with its control, in the options classes
(``moseq2_extract.options``): the computing block's ``cluster_type``, ``cores``,
``partition``, ``memory`` and ``wall_time``, and extraction's ``crop_size``,
``max_height`` and ``use_tracking_model``. Each recording is named by its session's
folder (``session_labels``), the way a person knows it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import traitlets

from moseq2_app.notebook._widget import RunningStep, session_labels, static_text
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


class ExtractWidget(RunningStep):
    """The Extract step's GUI (module docstring)."""

    STEP = "extract"
    PHASE = "extraction"

    recordings = traitlets.List().tag(sync=True)
    _esm = static_text("kit.js", "steps.js", "extract.js")

    def __init__(self, project: Project | None = None):
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
        from moseq2_app.notebook._widget import problem_of
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
        return run


def extract() -> ExtractWidget:
    """Open the Extract step: what this run will do, where it runs, then its progress."""
    return ExtractWidget()
