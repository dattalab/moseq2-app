// Shared pieces for the steps that do long work and show results: settings sections (with folds),
// a status band, a progress line, a run footer, and small SVG charts. Appended after kit.js.

const SVG_NS = "http://www.w3.org/2000/svg";

function svgNode(parent, tag, attrs, text) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.append(e);
  return e;
}

function svgRoot(W, H, label) {
  const svg = svgNode(null, "svg", { viewBox: `0 0 ${W} ${H}`, class: "m2-chart", role: "img", "aria-label": label });
  return svg;
}

// Settings sections from declarations; a section with `fold` starts folded.
function settingsSections(sections, values, folds, setValue) {
  return sections.map((s) => {
    const rows = s.fields.filter((f) => shown(f, values)).map((f) => row(f, control(f, values[f.key], (v) => setValue(f.key, v))));
    if (!s.fold) return h("div", { class: "m2-section" }, h("div", { class: "m2-section-title" }, s.title), rows);
    return h("div", { class: "m2-section" },
      h("details", { class: "m2-fold", open: folds[s.title] || false, ontoggle: (e) => { folds[s.title] = e.target.open; } },
        h("summary", {}, s.title), rows));
  });
}

// Whether the step's saved result is current, and why not.
function statusBand(status) {
  const s = status.state === "current" ? STATUS.good : STATUS.warning;
  return h("div", { class: "m2-status" },
    h("div", { class: "m2-status-head" }, h("div", { class: "m2-section-title" }, "Status")),
    h("div", { class: "m2-status-line" }, h("span", { class: `m2-symbol-inline ${s.cls}` }, s.symbol), status.text));
}

function progressLine(run, verb) {
  if (run.state !== "running") return null;
  return h("div", { class: "m2-progress-line" }, h("span", { class: "m2-spinner", "aria-hidden": "true" }),
    `${verb} on ${run.where}, ${run.progress}%. You can keep using the notebook.`);
}

function runFooter(status, run, labels, onStart) {
  const label = status.state === "none" ? labels.first : status.state === "current" ? labels.again : labels.stale;
  return h("div", { class: "m2-footer" },
    h("div", { class: "m2-footer-text m2-saved" }, labels.note || "The settings above are saved when it starts."),
    h("div", { class: "m2-buttons" },
      h("button", { class: status.state === "current" ? "m2-button" : "m2-button primary", type: "button",
        disabled: run.state === "running" || !!status.blocked, onclick: onStart }, label)));
}

// A histogram of durations in seconds, with the median marked.
function durationHistogram(hist, edges, median, label) {
  const W = 640, H = 190, L = 40, R = 16, T = 16, B = 34;
  const max = Math.max(...hist) || 1;
  const n = hist.length;
  const bw = (W - L - R) / n;
  const x = (v) => L + ((v - edges[0]) / (edges[n] - edges[0])) * (W - L - R);
  const y = (v) => T + (1 - v / max) * (H - T - B);
  const svg = svgRoot(W, H, label);
  svgNode(svg, "line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "m2-grid" });
  hist.forEach((c, i) => {
    const r = svgNode(svg, "rect", { x: L + i * bw + 1, y: y(c), width: bw - 2, height: H - B - y(c), rx: 2, class: "m2-bar-fill" });
    svgNode(r, "title", {}, `${edges[i].toFixed(1)} to ${edges[i + 1].toFixed(1)} s: ${c}`);
  });
  for (const t of [0, 0.5, 1, 1.5, 2]) svgNode(svg, "text", { x: x(t), y: H - B + 16, class: "m2-tick", "text-anchor": "middle" }, `${t} s`);
  svgNode(svg, "line", { x1: x(median), x2: x(median), y1: T - 4, y2: H - B, class: "m2-median" });
  svgNode(svg, "text", { x: x(median) + 6, y: T + 6, class: "m2-chart-label" }, `median ${median.toFixed(2)} s`);
  return svg;
}

// A few seconds of signals (one line each), with dashed lines where changes were found.
function tracesWithChanges(series, seconds, changes, names, colors) {
  const W = 640, H = 200, L = 16, R = 16, T = 12, B = 30;
  const n = series[0].length;
  const all = series.flat();
  const lo = Math.min(...all), hi = Math.max(...all);
  const x = (i) => L + (i / (n - 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B);
  const svg = svgRoot(W, H, "Signals over a few seconds, with the changes found");
  for (const c of changes) {
    const cx = L + (c / seconds) * (W - L - R);
    svgNode(svg, "line", { x1: cx, x2: cx, y1: T, y2: H - B, class: "m2-change" });
  }
  series.forEach((s, k) => svgNode(svg, "polyline", { points: s.map((v, i) => `${x(i)},${y(v)}`).join(" "),
    class: "m2-trace", style: `stroke: ${colors[k]}` }));
  for (let t = 0; t <= seconds; t += 1) svgNode(svg, "text", { x: L + (t / seconds) * (W - L - R), y: H - 10, class: "m2-tick", "text-anchor": "middle" }, `${t} s`);
  const legend = h("div", { class: "m2-legend" },
    names.map((nm, k) => h("span", {}, h("span", { class: "m2-swatch", style: `background: ${colors[k]}` }), nm)),
    h("span", {}, h("span", { class: "m2-swatch m2-swatch-dash" }), "change found"));
  return h("div", {}, svg, legend);
}

// Bars, one per item, sorted as given; `highlight` items in the phase colour.
function rankedBars(items, valueOf, labelOf, highlight, label) {
  const W = 640, H = 170, L = 40, R = 8, T = 10, B = 28;
  const max = Math.max(...items.map(valueOf)) || 1;
  const bw = (W - L - R) / items.length;
  const y = (v) => T + (1 - v / max) * (H - T - B);
  const svg = svgRoot(W, H, label);
  for (const v of [0, max / 2, max]) {
    svgNode(svg, "line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "m2-grid" });
    svgNode(svg, "text", { x: L - 6, y: y(v) + 4, class: "m2-tick", "text-anchor": "end" }, `${(v * 100).toFixed(0)}%`);
  }
  items.forEach((it, i) => {
    const r = svgNode(svg, "rect", { x: L + i * bw + 1, y: y(valueOf(it)), width: bw - 2, height: H - B - y(valueOf(it)), rx: 2,
      class: highlight.has(labelOf(it)) ? "m2-bar-fill on" : "m2-bar-fill" });
    svgNode(r, "title", {}, `Syllable ${labelOf(it)}: ${(valueOf(it) * 100).toFixed(1)}% of frames`);
    if (i % 5 === 0) svgNode(svg, "text", { x: L + i * bw + bw / 2, y: H - 10, class: "m2-tick", "text-anchor": "middle" }, String(i + 1));
  });
  return svg;
}
