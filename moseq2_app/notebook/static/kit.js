// Shared controls for every step's GUI. Each takes a setting's declaration (sent from
// Python) and its current value, and calls onChange with the new value. A step's own
// module (setup.js and the like) is appended after this file and exports `render`.

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === undefined || c === null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// A link to a help page. The pages don't exist yet, so the link has no target.
function docLink(title, lead = "") {
  return h("span", {}, lead, h("a", { class: "m2-doc", title: "Help page, to be written" }, title));
}

// A settings row: label and help on the left, the control on the right.
function row(field, control) {
  return h("div", { class: "m2-row" },
    h("div", {},
      h("div", { class: "m2-row-label" }, field.label),
      field.help ? h("div", { class: "m2-row-help" }, field.help) : null,
      field.link ? h("div", { class: "m2-row-help" }, docLink(field.link)) : null),
    h("div", {}, control));
}

// One choice from a list: a dropdown, with an optional hint underneath.
function choiceControl(field, value, onChange) {
  const select = h("select", { class: "m2-select", "aria-label": field.label,
    onchange: (e) => onChange(e.target.value) },
    field.choices.map((c) => h("option", { value: c.value, selected: c.value === value }, c.label)));
  const chosen = field.choices.find((c) => c.value === value);
  const warn = chosen && chosen.warning;
  const hint = warn || field.hint;
  return h("div", {}, select, hint ? h("div", {}, h("span", { class: warn ? "m2-note warn" : "m2-note" }, hint)) : null);
}

// Items to include or leave out: a filterable checklist, ticked means included.
// `value` is the list of item ids left out.
const UI = {};  // per-control view state (filter text, scroll) that survives a repaint

function checklistControl(field, value, onChange) {
  const off = new Set(value);
  const items = field.items;
  const wrap = h("div", {});
  const ui = UI[field.key] || (UI[field.key] = { filter: "", scroll: 0 });

  const countText = h("span", { class: "m2-count-text" });
  const meterFill = h("span", {});
  const count = h("div", { class: "m2-count" }, countText, h("div", { class: "m2-meter" }, meterFill));
  const list = h("div", { class: "m2-checklist", role: "group", "aria-label": field.label,
    onscroll: (e) => { ui.scroll = e.target.scrollTop; } });

  function paint() {
    countText.textContent = `${items.length - off.size} of ${items.length} included`;
    meterFill.style.width = `${100 * (items.length - off.size) / items.length}%`;
    list.replaceChildren(...items
      .filter((it) => it.label.toLowerCase().includes(ui.filter))
      .map((it) => {
        const included = !off.has(it.id);
        const box = h("input", { class: "m2-box", type: "checkbox", checked: included,
          onchange: (e) => {
            if (e.target.checked) off.delete(it.id); else off.add(it.id);
            onChange([...off]);
          } });
        return h("label", { class: included ? "m2-check" : "m2-check off" },
          box,
          h("span", { class: "m2-check-name" }, it.label),
          included ? h("span", {}) : h("span", { class: "m2-tag" }, "left out"),
          h("span", { class: "m2-check-meta" }, it.meta || ""));
      }));
  }

  const search = h("input", { class: "m2-input", type: "search", placeholder: "Filter by name",
    value: ui.filter, "aria-label": `Filter ${field.label.toLowerCase()}`,
    oninput: (e) => { ui.filter = e.target.value.toLowerCase(); paint(); } });
  const all = h("button", { class: "m2-button quiet", type: "button", disabled: off.size === 0,
    onclick: () => { off.clear(); onChange([]); } }, "Include all");

  paint();
  wrap.append(h("div", { class: "m2-checklist-bar" }, count, search, all), list);
  requestAnimationFrame(() => { list.scrollTop = ui.scroll; });
  return wrap;
}

// A few choices side by side, one of them on.
function segmentedControl(field, value, onChange) {
  return h("div", { class: "m2-segmented", role: "radiogroup", "aria-label": field.label },
    field.choices.map((c) => h("button", { type: "button", role: "radio",
      class: c.value === value ? "m2-segment on" : "m2-segment", "aria-checked": c.value === value ? "true" : "false",
      onclick: () => onChange(c.value) }, c.label)));
}

// A number, with its unit, and a slider when the field asks for one.
function numberControl(field, value, onChange) {
  const set = (v) => {
    const n = Math.min(field.max ?? Infinity, Math.max(field.min ?? -Infinity, Number(v)));
    if (!Number.isNaN(n)) onChange(n);
  };
  const box = h("input", { class: "m2-input m2-number", type: "number", value, min: field.min, max: field.max,
    step: field.step || 1, "aria-label": field.label, onchange: (e) => set(e.target.value) });
  const slider = field.slider ? h("input", { class: "m2-slider", type: "range", value, min: field.min, max: field.max,
    step: field.step || 1, "aria-label": field.label, onchange: (e) => set(e.target.value) }) : null;
  return h("div", {},
    h("div", { class: "m2-number-row" }, slider, box, field.unit ? h("span", { class: "m2-unit" }, field.unit) : null),
    field.hint ? h("div", {}, h("span", { class: "m2-note" }, field.hint)) : null);
}

// Free text, saved when the person leaves the box.
function textControl(field, value, onChange) {
  return h("input", { class: "m2-input m2-text", type: "text", value: value ?? "", placeholder: field.placeholder || "",
    "aria-label": field.label, onchange: (e) => onChange(e.target.value.trim()) });
}

// On or off.
function toggleControl(field, value, onChange) {
  return h("label", { class: "m2-toggle" },
    h("input", { type: "checkbox", role: "switch", checked: !!value, "aria-label": field.label,
      onchange: (e) => onChange(e.target.checked) }),
    h("span", { class: "m2-toggle-track", "aria-hidden": "true" }),
    h("span", {}, value ? (field.on_label || "On") : (field.off_label || "Off")));
}

const CONTROLS = { choice: choiceControl, checklist: checklistControl, segmented: segmentedControl,
  number: numberControl, toggle: toggleControl, text: textControl };

// Whether a field shows, given the other settings: `show_if` names a setting and the value it needs.
function shown(field, values) {
  return !field.show_if || values[field.show_if.key] === field.show_if.value;
}

function control(field, value, onChange) {
  const make = CONTROLS[field.type];
  if (!make) return h("span", { class: "m2-note warn" }, `No control for setting type "${field.type}".`);
  return make(field, value, onChange);
}

// The installation readouts: all checks always shown, one tile each (a status symbol, then
// label above value). Each problem's explanation and fix are always shown below the tiles.
const STATUS = {
  good: { cls: "good", symbol: "✓" },
  warning: { cls: "warn", symbol: "!" },
  critical: { cls: "crit", symbol: "✕" },
};

function statusStrip(checks, onRecheck) {
  const tiles = h("div", { class: "m2-tiles" }, checks.map((c) => {
    const s = STATUS[c.status] || STATUS.warning;
    return h("div", { class: `m2-tile ${s.cls}` },
      h("span", { class: "m2-symbol", "aria-hidden": "true" }, s.symbol),
      h("span", { class: "m2-tile-label" }, c.label),
      h("span", { class: "m2-tile-value" }, c.value));
  }));
  const problems = checks.filter((c) => c.status !== "good").map((c) => {
    const s = STATUS[c.status] || STATUS.warning;
    return h("div", { class: `m2-detail ${s.cls}` },
      h("div", { class: "m2-detail-title" }, c.title),
      h("div", {}, c.explanation),
      c.fix ? h("div", { class: "m2-detail-fix" }, h("b", {}, "To fix it: "), c.fix) : null,
      c.help ? h("div", { class: "m2-detail-fix" }, docLink(c.help, "For complete setup instructions, see "), ".") : null);
  });
  return h("div", {},
    h("div", { class: "m2-status" },
      h("div", { class: "m2-status-head" },
        h("div", { class: "m2-section-title" }, "Installation checks"),
        h("button", { class: "m2-recheck", type: "button", onclick: onRecheck }, "↻ Check again")),
      tiles),
    problems);
}

// The footer: what is unsaved, and Revert and Save.
function footer({ changes, savedAt, onRevert, onSave, title }) {
  const dirty = changes.length > 0;
  const text = dirty
    ? h("div", { class: "m2-footer-text" },
        h("div", { class: "m2-footer-title" }, title || (changes.length === 1 ? "1 unsaved change" : `${changes.length} unsaved changes`)),
        h("ul", {}, changes.map((c) => h("li", {}, c))))
    : h("div", { class: "m2-footer-text m2-saved" },
        savedAt ? [h("b", {}, "✓ "), `Saved ${savedAt}.`] : "No unsaved changes.");
  return h("div", { class: dirty ? "m2-footer dirty" : "m2-footer" }, text,
    h("div", { class: "m2-buttons" },
      h("button", { class: "m2-button", type: "button", disabled: !dirty, onclick: onRevert }, "Revert"),
      h("button", { class: "m2-button primary", type: "button", disabled: !dirty, onclick: onSave }, "Save")));
}

// A problem: what happened, the fix, and a help page. An expected problem has an amber edge; a bug
// (kind "bug") a red one, with what a bug report needs folded underneath.
function problemPanel(problem, extraClass = "") {
  if (!problem || !problem.title) return null;
  const bug = problem.kind === "bug";
  return h("div", { class: `m2-detail ${bug ? "crit" : "warn"} ${extraClass}`.trim(), role: "alert" },
    h("div", { class: "m2-detail-title" }, problem.title),
    problem.text ? h("div", {}, problem.text) : null,
    problem.fix ? h("div", { class: "m2-detail-fix" }, h("b", {}, "To fix it: "), problem.fix) : null,
    problem.help ? h("div", { class: "m2-detail-fix" }, docLink(problem.help)) : null,
    bug && problem.details ? h("details", { class: "m2-fold m2-bug" },
      h("summary", {}, "Details for a bug report"),
      h("pre", { class: "m2-pre" }, problem.details)) : null);
}
