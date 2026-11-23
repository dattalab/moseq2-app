// Step 3, Extract: what this run will do (one row per recording), where it runs, more settings
// folded away, and a footer with the button that starts it. While it runs, the rows show progress.
// State from Python: `recordings` (each with its plan), `sections`, `saved`, `run` and `problem`.

const PLAN = {
  new: { cls: "new", label: "New" },
  redo: { cls: "redo", label: "Redo" },
  done: { cls: "done", label: "Up to date" },
  "in progress": { cls: "muted", label: "In progress" },
};

function planSummary(recordings) {
  const n = (p) => recordings.filter((r) => r.plan === p).length;
  return h("div", { class: "m2-summary" },
    n("new") ? h("span", { class: "m2-chip new" }, `${n("new")} new`) : null,
    n("redo") ? h("span", { class: "m2-chip redo" }, `${n("redo")} to redo`) : null,
    h("span", { class: "m2-chip done" }, `${n("done")} up to date`));
}

function runSummary(items) {
  const states = Object.values(items);
  const n = (s) => states.filter((v) => v.state === s).length;
  return h("div", { class: "m2-summary" },
    n("done") ? h("span", { class: "m2-chip done" }, `${n("done")} extracted`) : null,
    n("running") ? h("span", { class: "m2-chip new" }, `${n("running")} extracting`) : null,
    n("waiting") ? h("span", { class: "m2-chip muted" }, `${n("waiting")} waiting`) : null,
    n("failed") ? h("span", { class: "m2-chip failed" }, `${n("failed")} failed`) : null);
}

function percent(item) {
  return item.total ? Math.round(100 * item.done / item.total) : 0;
}

function planRow(r, item) {
  let chip;
  if (item) {
    chip = item.state === "done" ? h("span", { class: "m2-chip done" }, "Extracted")
      : item.state === "failed" ? h("span", { class: "m2-chip failed" }, "Failed")
      : item.state === "waiting" || item.state === "stopped" ? h("span", { class: "m2-chip muted" }, "Waiting")
      : h("span", { class: "m2-chip new" }, `Extracting, ${percent(item)}%`);
  } else {
    const p = PLAN[r.plan] || PLAN.new;
    chip = h("span", { class: `m2-chip ${p.cls}`, title: r.changed && r.changed.length > 1 ? r.changed.join(", ") : null },
      r.plan === "redo" ? `Redo: ${r.reason}` : p.label);
  }
  const running = item && (item.state === "running" || item.state === "done");
  return h("div", { class: "m2-plan-row" },
    h("span", { class: r.plan === "done" && !item ? "m2-check-name m2-dim" : "m2-check-name" }, r.label),
    chip,
    r.meta ? h("span", { class: "m2-check-meta" }, r.meta) : null,
    running ? h("div", { class: "m2-bar" }, h("span", { style: `width: ${item.state === "done" ? 100 : percent(item)}%` })) : null,
    item && item.problem && item.problem.title ? problemPanel(item.problem, "m2-in-row") : null);
}

// Sorting the list: by name, or by what happens to each recording (problems and work first).
const STATE_ORDER = { failed: 0, running: 1, waiting: 2, stopped: 2, "in progress": 2, new: 3, redo: 4, done: 5 };

function sortRows(rows, run, sort) {
  const key = (r) => {
    const item = run.items && run.items[r.id];
    return item ? (item.state === "done" ? 5 : STATE_ORDER[item.state]) : STATE_ORDER[r.plan];
  };
  const sorted = [...rows].sort((a, b) => sort.by === "name"
    ? a.label.localeCompare(b.label, undefined, { numeric: true })
    : key(a) - key(b) || a.label.localeCompare(b.label, undefined, { numeric: true }));
  return sort.desc ? sorted.reverse() : sorted;
}

function sortHeader(sort, onSort) {
  const col = (by, label) => {
    const on = sort.by === by;
    return h("button", { class: on ? "m2-sort on" : "m2-sort", type: "button",
      "aria-sort": on ? (sort.desc ? "descending" : "ascending") : "none",
      onclick: () => onSort(on ? { by, desc: !sort.desc } : { by, desc: false }) },
      label, h("span", { class: "m2-sort-arrow", "aria-hidden": "true" }, on ? (sort.desc ? "▼" : "▲") : "▲"));
  };
  return h("div", { class: "m2-plan-head" }, col("name", "Recording"), col("state", "What happens"));
}

function render({ model, el }) {
  let draft = structuredClone(model.get("saved"));
  let sort = { by: "state", desc: false };
  const folds = {};
  const root = h("div", { class: `m2 phase-${model.get("phase")}` });
  el.append(root);

  function settingsSection(s) {
    const rows = s.fields.filter((f) => shown(f, draft)).map((f) => row(f, control(f, draft[f.key], (v) => {
      draft = { ...draft, [f.key]: v }; paint();
    })));
    if (!s.fold) return h("div", { class: "m2-section" }, h("div", { class: "m2-section-title" }, s.title), rows);
    const det = h("details", { class: "m2-fold", open: folds[s.title] || false,
      ontoggle: (e) => { folds[s.title] = e.target.open; } },
      h("summary", {}, s.title, h("span", { class: "m2-fold-hint" }, s.fold_hint || "")), rows);
    return h("div", { class: "m2-section" }, det);
  }

  function actionFooter(run, targets) {
    const start = () => model.send({ action: "start", settings: draft });
    const label = (n) => (n === 1 ? "Extract 1 recording" : `Extract ${n} recordings`);
    if (run.state === "running") {
      return h("div", { class: "m2-footer dirty" },
        h("div", { class: "m2-footer-text" },
          h("div", { class: "m2-footer-title" },
            run.where === "the cluster" ? "Extracting on the cluster" : "Extracting on this computer"),
          "You can keep using the notebook; this list keeps up."),
        h("div", { class: "m2-buttons" }, h("button", { class: "m2-button", type: "button",
          onclick: () => model.send({ action: "stop" }) }, "Stop")));
    }
    if (run.state === "finished") {
      const failed = Object.entries(run.items).filter(([, v]) => v.state === "failed").map(([k]) => k);
      return h("div", { class: "m2-footer" },
        h("div", { class: "m2-footer-text" },
          failed.length ? `Finished at ${run.finished_at}, with ${failed.length} failed. Fix it above, then extract it again.`
            : `Finished at ${run.finished_at}. Next: 4 Extraction diagnostics, the cell below.`),
        h("div", { class: "m2-buttons" },
          failed.length ? h("button", { class: "m2-button primary", type: "button", onclick: start }, label(failed.length))
            : null));
    }
    return h("div", { class: "m2-footer" },
      h("div", { class: "m2-footer-text m2-saved" },
        targets.length ? "The settings above are saved when extraction starts." : "Every recording is up to date."),
      h("div", { class: "m2-buttons" },
        h("button", { class: "m2-button primary", type: "button", disabled: targets.length === 0, onclick: start },
          targets.length ? label(targets.length) : "Nothing to extract")));
  }

  function warnings(lines) {
    const seen = [...new Set(lines.filter((l) => l.kind === "warning").map((l) => l.text))];
    if (!seen.length) return null;
    return h("div", { class: "m2-warnings" }, seen.map((t) => h("span", { class: "m2-note warn" }, t)));
  }

  function paint() {
    const recordings = model.get("recordings");
    const run = model.get("run");
    const targets = recordings.filter((r) => r.plan === "new" || r.plan === "redo").map((r) => r.id);
    const active = run.state === "running" || run.state === "finished";
    const shownRows = active ? recordings.filter((r) => r.id in run.items) : recordings;
    const extra = active ? Object.keys(run.items).filter((k) => !recordings.some((r) => r.id === k))
      .map((k) => ({ id: k, label: k, plan: "new" })) : [];
    root.replaceChildren(...[
      problemPanel(model.get("problem")),
      h("div", { class: "m2-section" },
        h("div", { class: "m2-section-title" }, active ? "Progress" : "What this run will do"),
        active ? runSummary(run.items) : planSummary(recordings),
        h("div", { class: "m2-plan" },
          sortHeader(sort, (s) => { sort = s; paint(); }),
          sortRows([...shownRows, ...extra], active ? run : {}, sort).map((r) => planRow(r, active ? run.items[r.id] : null))),
        active ? warnings(model.get("lines")) : null),
      ...model.get("sections").map(settingsSection),
      actionFooter(run, targets)].filter(Boolean));
  }

  model.on("change:run", paint);
  model.on("change:recordings", paint);
  model.on("change:problem", paint);
  model.on("change:sections", paint);
  model.on("change:saved", () => { draft = structuredClone(model.get("saved")); paint(); });
  paint();
  return () => { root.remove(); };
}

export default { render };
