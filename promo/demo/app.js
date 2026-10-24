/* WhatIf local console.
   Vanilla JS, no build step, no network. The canvas view draws the same canvas
   JSON that Obsidian reads, but as an animated SVG graph. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  notes: [],
  activeId: null,
  tab: "brief",
  brief: null,
  retrieval: null,
  scoring: null,
  canvas: null,
  evalResults: null,
  busy: false,
};

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ------------------------------------------------------------------ utils */

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch (err) {
    payload = { error: text };
  }
  if (!response.ok) {
    const message = (payload && payload.error) || `Request failed (${response.status})`;
    throw new Error(message);
  }
  return payload;
}

function toast(message, kind = "info") {
  const node = el("div", { class: `toast${kind === "error" ? " is-error" : ""}`, text: message });
  $("#toasts").appendChild(node);
  setTimeout(() => {
    node.classList.add("is-out");
    setTimeout(() => node.remove(), 260);
  }, kind === "error" ? 7000 : 3800);
}

function fmtScore(value) {
  return Number(value).toFixed(3);
}

/* Split a line of prose into text and styled evidence tags. */
const TAG_RE = /\[(EVIDENCE|ESTIMATE|UNVERIFIED)(?::\s*([^\]]*))?\]/g;

function renderTagged(text) {
  const frag = document.createDocumentFragment();
  let last = 0;
  let match;
  TAG_RE.lastIndex = 0;
  while ((match = TAG_RE.exec(text)) !== null) {
    if (match.index > last) {
      frag.appendChild(document.createTextNode(text.slice(last, match.index)));
    }
    const kind = match[1].toLowerCase();
    const label =
      kind === "evidence" ? `[EVIDENCE: ${match[2]}]` : `[${match[1]}]`;
    frag.appendChild(el("span", { class: `tag tag-${kind}`, text: label }));
    last = match.index + match[0].length;
  }
  if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
  return frag;
}

function renderProse(text) {
  const box = el("div", { class: "prose" });
  const lines = (text || "").split("\n");
  let list = null;

  const flush = () => {
    if (list) { box.appendChild(list); list = null; }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flush(); continue; }
    if (line.startsWith("- ") || line.startsWith("* ")) {
      if (!list) list = el("ul");
      const item = el("li");
      item.appendChild(renderTagged(line.slice(2)));
      list.appendChild(item);
    } else {
      flush();
      const p = el("p");
      p.appendChild(renderTagged(line));
      box.appendChild(p);
    }
  }
  flush();
  return box;
}

/* ------------------------------------------------------------------ health */

async function refreshHealth() {
  const node = $("#health");
  try {
    const health = await api("/api/health");
    $("#vaultPath").textContent = health.vault;
    const text = $(".status-text", node);
    if (health.ollama_reachable && health.db_exists) {
      node.dataset.state = "ok";
      text.textContent = `ready · ${health.indexed}/${health.note_count} indexed`;
    } else if (health.ollama_reachable) {
      node.dataset.state = "warn";
      text.textContent = "index not built";
    } else {
      node.dataset.state = "down";
      text.textContent = "ollama offline";
    }
    return health;
  } catch (err) {
    node.dataset.state = "down";
    $(".status-text", node).textContent = "server error";
    return null;
  }
}

/* ------------------------------------------------------------------ notes */

function noteRow(note, index) {
  const badges = [];
  badges.push(el("span", { class: "badge", text: note.status }));
  if (note.failure_count > 0) {
    badges.push(
      el("span", {
        class: "badge badge-fail",
        text: `${note.failure_count} failed`,
      })
    );
  }
  const button = el("button", {
    class: "note",
    role: "option",
    "data-id": note.id,
    style: `animation-delay:${Math.min(index * 22, 320)}ms`,
    onclick: () => selectNote(note.id),
  });
  button.appendChild(
    el("div", { class: "note-row" }, [
      el("span", { class: "note-id", text: note.id }),
      el("span", { class: "note-domain", text: note.domain }),
    ])
  );
  if (note.headline) {
    button.appendChild(el("div", { class: "note-head", text: note.headline }));
  }
  button.appendChild(el("div", { class: "note-badges" }, badges));
  return el("li", {}, button);
}

function renderNotes(filter = "") {
  const list = $("#notes");
  const needle = filter.trim().toLowerCase();
  const visible = state.notes.filter((note) => {
    if (!needle) return true;
    return (
      note.id.toLowerCase().includes(needle) ||
      note.domain.toLowerCase().includes(needle) ||
      note.status.toLowerCase().includes(needle) ||
      (note.headline || "").toLowerCase().includes(needle)
    );
  });

  list.textContent = "";
  visible.forEach((note, index) => list.appendChild(noteRow(note, index)));
  $("#railEmpty").hidden = state.notes.length > 0;
  $("#noteCount").textContent = `${state.notes.length} notes`;
  $("#failCount").textContent = `${state.notes.filter((n) => n.has_failures).length} with failures`;
  if (state.activeId) {
    const active = $(`.note[data-id="${CSS.escape(state.activeId)}"]`, list);
    if (active) active.classList.add("is-active");
  }
}

async function loadNotes() {
  const data = await api("/api/notes");
  state.notes = data.notes || [];
  renderNotes($("#search").value);
}

/* ------------------------------------------------------------------ panels */

function panelFor(tab) {
  return $(`#panel-${tab}`);
}

function setTab(tab) {
  state.tab = tab;
  $$(".tab").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.tab === tab);
  });
  for (const name of ["brief", "canvas", "retrieval", "scoring", "evaluation"]) {
    const panel = panelFor(name);
    const active = name === tab;
    panel.hidden = !active;
    if (active) {
      panel.style.animation = "none";
      void panel.offsetWidth;
      panel.style.animation = "";
    }
  }
  if (tab === "evaluation") loadEvaluation();
  if (tab === "retrieval" && !state.retrieval) loadRetrieval();
  if (tab === "scoring" && !state.scoring) loadScoring();
  if (tab === "canvas" && !state.canvas) loadCanvas();
}

/* ------------------------------------------------------------------ brief */

function legendBlock(legendText) {
  const box = el("div", { class: "legend" });
  const rows = [
    ["[EVIDENCE: note-id]", "a claim supported by a past decision that was retrieved", "tag-evidence"],
    ["[ESTIMATE]", "a judgement with no supporting evidence in your notes", "tag-estimate"],
    ["[UNVERIFIED]", "a claim that failed validation and could not be confirmed", "tag-unverified"],
    ["COMPUTED", "a value calculated in code from your own scores and weights", "tag-evidence"],
  ];
  for (const [key, meaning, kind] of rows) {
    box.appendChild(
      el("div", { class: "legend-row" }, [
        el("span", { class: "legend-key" }, [
          el("span", { class: `tag ${kind}`, text: key }),
        ]),
        el("span", { class: "dim", text: meaning }),
      ])
    );
  }
  if (legendText) {
    box.appendChild(el("div", { class: "dim", text: legendText }));
  }
  return box;
}

function briefSkeleton() {
  const box = el("div", { class: "section" });
  for (let i = 0; i < 6; i += 1) {
    box.appendChild(
      el("div", { class: "sk-row" }, [
        el("div", { class: "skeleton" }),
        el("div", { class: "skeleton" }),
      ])
    );
  }
  return box;
}

function renderBrief(payload) {
  const panel = panelFor("brief");
  panel.textContent = "";

  if (!payload) {
    panel.appendChild(
      el("div", { class: "state" }, [
        el("h4", { text: "No brief yet" }),
        el("p", {
          text: "Generate a brief to retrieve similar decisions, surface the assumptions that failed before, and argue against your leading option.",
        }),
      ])
    );
    return;
  }

  if (payload.error) {
    panel.appendChild(
      el("div", { class: "state is-error" }, [
        el("h4", { text: "Could not generate the brief" }),
        el("p", { text: payload.error }),
      ])
    );
    return;
  }

  panel.appendChild(legendBlock());

  const validation = payload.validation || {};
  if (validation.issue_count > 0) {
    panel.appendChild(
      el("div", { class: "state is-error" }, [
        el("h4", {
          text: `${validation.issue_count} claim(s) failed validation and are marked [UNVERIFIED]`,
        }),
        el("p", {
          text: `The model was asked to fix them once and did not. The brief is still usable; the marked claims are not confirmed.`,
        }),
      ])
    );
  }

  const sections = payload.sections || {};

  const summary = el("section", { class: "section" }, [
    el("h3", { text: "Decision summary" }),
  ]);
  summary.appendChild(
    renderProse(sections.summary || "_Not generated._")
  );
  panel.appendChild(summary);

  const similar = el("section", { class: "section" }, [
    el("h3", { text: "Similar past decisions" }),
  ]);
  const similarRows = payload.similar || [];
  if (similarRows.length === 0) {
    similar.appendChild(
      el("div", { class: "state" }, [
        el("p", { text: "No past decisions were similar enough to surface." }),
      ])
    );
  } else {
    const wrap = el("div", { class: "table-wrap" });
    const table = el("table");
    table.appendChild(
      el("thead", {}, el("tr", {}, [
        el("th", { text: "Note" }),
        el("th", { text: "Similarity" }),
        el("th", { text: "Why it is relevant" }),
      ]))
    );
    const body = el("tbody");
    for (const note of similarRows) {
      const relevance = (payload.relevance || {})[note.note_id] || note.summary;
      body.appendChild(
        el("tr", {}, [
          el("td", {}, el("span", { class: "mono", text: note.note_id })),
          el("td", { class: "num", text: fmtScore(note.similarity) }),
          el("td", { text: relevance }),
        ])
      );
    }
    table.appendChild(body);
    wrap.appendChild(table);
    similar.appendChild(wrap);
  }
  panel.appendChild(similar);

  const failed = el("section", { class: "section" }, [
    el("h3", { text: "Assumptions that failed before" }),
  ]);
  const failedRows = payload.failed || [];
  if (failedRows.length === 0) {
    failed.appendChild(
      el("div", { class: "state" }, [
        el("p", { text: "No past assumption failures were close enough to this decision." }),
      ])
    );
  } else {
    const list = el("ul", { class: "prose" });
    for (const item of failedRows) {
      const li = el("li");
      li.appendChild(document.createTextNode(item.text + " "));
      li.appendChild(
        el("span", {
          class: "tag tag-evidence",
          text: `[EVIDENCE: ${item.source_note_id}]`,
        })
      );
      li.appendChild(
        el("span", { class: "dim", text: ` ${fmtScore(item.similarity)}` })
      );
      list.appendChild(li);
    }
    failed.appendChild(list);
  }
  panel.appendChild(failed);

  const scoring = payload.scoring || {};
  if (scoring.options && scoring.options.length) {
    const section = el("section", { class: "section" }, [
      el("h3", { text: "Weighted comparison" }),
    ]);
    section.appendChild(scoringTable(scoring));
    panel.appendChild(section);
  }

  const counter = el("section", { class: "section" }, [
    el("h3", { text: `Counterargument: against ${payload.leading || "your leading option"}` }),
  ]);
  counter.appendChild(renderProse(sections.counterargument || "_Not generated._"));
  panel.appendChild(counter);

  const questions = el("section", { class: "section" }, [
    el("h3", { text: "Open questions" }),
  ]);
  questions.appendChild(renderProse(sections.questions || "_Not generated._"));
  panel.appendChild(questions);
}

function scoringTable(scoring) {
  const wrap = el("div", { class: "table-wrap" });
  const table = el("table");
  const criteria = scoring.criteria || [];
  const options = scoring.options || [];

  const head = el("tr", {}, [el("th", { text: "Option" })]);
  for (const criterion of criteria) {
    head.appendChild(el("th", { text: `${criterion.name} (w${criterion.weight})` }));
  }
  head.appendChild(el("th", { text: "Total (COMPUTED)" }));
  table.appendChild(el("thead", {}, head));

  const body = el("tbody");
  for (const option of options) {
    const row = el("tr");
    if (option.option === scoring.leader) row.classList.add("is-leader");
    row.appendChild(el("td", { text: option.option }));
    for (const criterion of criteria) {
      const value = (criterion.scores || {})[option.option];
      row.appendChild(
        el("td", { class: "num", text: value === undefined ? "-" : String(value) })
      );
    }
    row.appendChild(el("td", { class: "total", text: String(option.total) }));
    body.appendChild(row);
  }
  table.appendChild(body);
  wrap.appendChild(table);
  wrap.appendChild(
    el("div", {
      class: "note-line",
      text: `COMPUTED: weighted total = sum of (score x weight). Maximum possible is ${scoring.max_points} points.`,
    })
  );
  return wrap;
}

/* ------------------------------------------------------------------ canvas */

function renderCanvas(canvas) {
  const panel = panelFor("canvas");
  panel.textContent = "";

  if (!canvas) {
    panel.appendChild(
      el("div", { class: "state" }, [
        el("h4", { text: "No canvas yet" }),
        el("p", { text: "Open the Canvas tab to build one." }),
      ])
    );
    return;
  }

  const stage = el("div", { class: "canvas-stage" });
  const nodes = canvas.nodes || [];
  const edges = canvas.edges || [];

  const pad = 60;
  const minX = Math.min(...nodes.map((n) => n.x), 0);
  const minY = Math.min(...nodes.map((n) => n.y), 0);
  const maxX = Math.max(...nodes.map((n) => n.x + n.width), 800);
  const maxY = Math.max(...nodes.map((n) => n.y + n.height), 600);
  const width = maxX - minX + pad * 2;
  const height = maxY - minY + pad * 2;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `${minX - pad} ${minY - pad} ${width} ${height}`);

  const byId = new Map(nodes.map((n) => [n.id, n]));

  // Edges first so the nodes sit on top.
  edges.forEach((edge, index) => {
    const from = byId.get(edge.fromNode);
    const to = byId.get(edge.toNode);
    if (!from || !to) return;
    const x1 = from.x + from.width / 2;
    const y1 = from.y + from.height;
    const x2 = to.x + to.width / 2;
    const y2 = to.y;
    const midY = (y1 + y2) / 2;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`);
    path.setAttribute("class", "cedge");
    path.style.setProperty("--len", "600");
    path.style.animationDelay = `${120 + index * 55}ms`;
    svg.appendChild(path);
  });

  nodes.forEach((node, index) => {
    // The position lives on the outer group as an SVG transform attribute. The
    // animation lives on an inner group, because a CSS transform would override
    // the attribute and collapse every node onto the origin.
    const position = document.createElementNS("http://www.w3.org/2000/svg", "g");
    position.setAttribute("transform", `translate(${node.x} ${node.y})`);

    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "cnode");
    group.style.animationDelay = `${index * 60}ms`;
    if (node.type === "file") group.classList.add("is-root");
    if (/\*\*Assumption\*\*/.test(node.text || "")) group.classList.add("is-assumption");
    if (/\*\*Risk\*\*/.test(node.text || "")) group.classList.add("is-risk");

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("width", String(node.width));
    rect.setAttribute("height", String(node.height));
    rect.setAttribute("rx", "10");
    group.appendChild(rect);

    const body = canvasNodeBody(node);
    const titleEl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    titleEl.setAttribute("x", "16");
    titleEl.setAttribute("y", "30");
    titleEl.setAttribute("font-weight", "600");
    titleEl.textContent =
      body.title.length > 38 ? `${body.title.slice(0, 37)}…` : body.title;
    group.appendChild(titleEl);

    body.lines.slice(0, 5).forEach((line, lineIndex) => {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("class", "cnode-sub");
      t.setAttribute("x", "16");
      t.setAttribute("y", String(52 + lineIndex * 19));
      t.textContent = line.length > 46 ? `${line.slice(0, 45)}…` : line;
      group.appendChild(t);
    });

    position.appendChild(group);
    svg.appendChild(position);
  });

  stage.appendChild(svg);
  panel.appendChild(el("section", { class: "section" }, [
    el("h3", { text: "Branching canvas" }),
    stage,
    el("div", {
      class: "note-line",
      text: "The same JSON Obsidian Canvas reads, written to Canvases/<id>.canvas.",
    }),
  ]));
}

/* Turn a canvas node's Markdown into a title and a few readable lines. */
function canvasNodeBody(node) {
  if (node.type === "file") {
    const parts = (node.file || "").split("/");
    return {
      title: parts[parts.length - 1].replace(/\.md$/, ""),
      lines: [parts.slice(0, -1).join("/") || "decision note"],
    };
  }

  const raw = (node.text || "").split("\n");
  const title = (raw[0] || "").replace(/^#+\s*/, "").replace(/\*/g, "").trim();
  const lines = [];

  for (const rawLine of raw.slice(1)) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.startsWith("|")) {
      const cells = line.split("|").map((c) => c.trim()).filter(Boolean);
      if (cells.length < 2) continue;
      // Skip the header and the separator row of the Markdown table.
      if (cells[0].toLowerCase() === "criterion") continue;
      if (cells.every((c) => /^-+$/.test(c))) continue;
      const [criterion, score, weight, points] = cells;
      if (points !== undefined) {
        lines.push(`${criterion}  ${score} x ${weight} = ${points}`);
      } else {
        lines.push(cells.join("  "));
      }
      continue;
    }

    lines.push(line.replace(/\*\*/g, ""));
  }

  return { title, lines };
}

/* ------------------------------------------------------------------ retrieval */

function renderRetrieval(payload) {
  const panel = panelFor("retrieval");
  panel.textContent = "";

  if (!payload) {
    panel.appendChild(
      el("div", { class: "state" }, [
        el("h4", { text: "Loading retrieval" }),
        el("p", { text: "Finding the decisions most like this one." }),
      ])
    );
    return;
  }

  const left = el("section", { class: "section" }, [el("h3", { text: "Similar decisions" })]);
  const similar = payload.similar || [];
  if (!similar.length) {
    left.appendChild(el("div", { class: "state" }, [el("p", { text: "Nothing similar enough." })]));
  }
  for (const note of similar) {
    const card = el("div", { class: "row-card" });
    card.appendChild(
      el("div", { class: "row-top" }, [
        el("span", { class: "row-id", text: note.note_id }),
        el("span", { class: "row-sim", text: fmtScore(note.similarity) }),
      ])
    );
    card.appendChild(el("div", { class: "row-text", text: note.summary }));
    const meter = el("div", { class: "meter" }, el("i"));
    card.appendChild(meter);
    left.appendChild(card);
    requestAnimationFrame(() => {
      $("i", meter).style.width = `${Math.max(2, note.similarity * 100)}%`;
    });
  }

  const right = el("section", { class: "section" }, [
    el("h3", { text: "Assumptions that failed before" }),
  ]);
  const failed = payload.failed || [];
  if (!failed.length) {
    right.appendChild(
      el("div", { class: "state" }, [el("p", { text: "No close past failures found." })])
    );
  }
  for (const item of failed) {
    const card = el("div", { class: "row-card" });
    card.appendChild(
      el("div", { class: "row-top" }, [
        el("span", { class: "row-id", text: item.source_note_id }),
        el("span", { class: "row-sim", text: fmtScore(item.similarity) }),
      ])
    );
    card.appendChild(el("div", { class: "row-text", text: item.text }));
    const meter = el("div", { class: "meter" }, el("i"));
    card.appendChild(meter);
    right.appendChild(card);
    requestAnimationFrame(() => {
      $("i", meter).style.width = `${Math.max(2, item.similarity * 100)}%`;
    });
    if (item.above_threshold === false) {
      card.appendChild(
        el("div", {
          class: "note-line",
          text: `below the ${payload.threshold} threshold, so a brief would not show this`,
        })
      );
      card.style.opacity = "0.62";
    }
  }

  panel.appendChild(el("div", { class: "split" }, [left, right]));
}

/* ------------------------------------------------------------------ scoring */

function renderScoring(payload) {
  const panel = panelFor("scoring");
  panel.textContent = "";

  if (!payload) {
    panel.appendChild(
      el("div", { class: "state" }, [
        el("h4", { text: "Loading scoring" }),
        el("p", { text: "Computing the weighted comparison." }),
      ])
    );
    return;
  }

  const scoring = payload.result || {};
  if (!scoring.options || !scoring.options.length) {
    panel.appendChild(
      el("div", { class: "state" }, [
        el("h4", { text: "Nothing to score" }),
        el("p", { text: "This note has no criteria or no scores yet." }),
      ])
    );
    return;
  }

  const section = el("section", { class: "section" }, [
    el("h3", { text: "Weighted comparison" }),
  ]);
  section.appendChild(scoringTable(scoring));
  panel.appendChild(section);

  const criteria = scoring.criteria || [];
  const options = scoring.options || [];
  const criteriaBox = el("div", { class: "criteria" });

  for (const criterion of criteria) {
    const block = el("div", { class: "criterion" });
    block.appendChild(
      el("div", { class: "criterion-head" }, [
        el("span", { class: "criterion-name", text: criterion.name }),
        el("span", { class: "criterion-weight", text: `weight ${criterion.weight}` }),
      ])
    );
    for (const option of options) {
      const score = (criterion.scores || {})[option.option];
      const value = score === undefined ? 0 : score;
      const line = el("div", { class: "criterion-row" });
      line.appendChild(
        el("span", {
          class: "criterion-option",
          text: option.option,
        })
      );
      const track = el("div", { class: "criterion-track" });
      const fill = el("div", { class: "criterion-fill" });
      track.appendChild(fill);
      line.appendChild(track);
      line.appendChild(
        el("span", {
          class: "criterion-score",
          text: score === undefined ? "-" : String(score),
        })
      );
      block.appendChild(line);
      requestAnimationFrame(() => {
        fill.style.width = `${(value / 5) * 100}%`;
        if (option.option === scoring.leader) fill.classList.add("is-leader");
      });
    }
    criteriaBox.appendChild(block);
  }

  panel.appendChild(
    el("section", { class: "section" }, [el("h3", { text: "Criteria" }), criteriaBox])
  );

  if ((payload.warnings || []).length) {
    const warn = el("section", { class: "section" }, [el("h3", { text: "Notes" })]);
    const list = el("ul", { class: "prose" });
    for (const warning of payload.warnings) {
      list.appendChild(el("li", { text: warning }));
    }
    warn.appendChild(list);
    panel.appendChild(warn);
  }
}

/* ------------------------------------------------------------------ evaluation */

function renderEvaluation(results) {
  const panel = panelFor("evaluation");
  panel.textContent = "";

  if (!results) {
    panel.appendChild(
      el("div", { class: "state" }, [
        el("h4", { text: "No evaluation yet" }),
        el("p", {
          text: "Run the leave-one-out evaluation to see how often a past failure is surfaced when it is relevant.",
        }),
        el("p", {}, [
          el("button", {
            class: "btn btn-primary",
            text: "Run evaluation",
            onclick: runEvaluation,
          }),
        ]),
      ])
    );
    return;
  }

  const stats = el("div", { class: "stat-row" });
  const rate = results.hit_rate === undefined ? 0 : results.hit_rate;
  const stat = (label, value, sub, tone) =>
    el("div", { class: "stat" }, [
      el("div", { class: "stat-label", text: label }),
      el("div", { class: `stat-value${tone ? ` is-${tone}` : ""}`, text: value }),
      el("div", { class: "stat-sub", text: sub }),
    ]);
  stats.appendChild(
    stat(
      "Hit rate",
      `${Math.round(rate * 100)}%`,
      `${results.hits} of ${results.total} notes`,
      rate > 0 ? "good" : "bad"
    )
  );
  stats.appendChild(stat("Threshold", String(results.threshold), "cosine similarity"));
  stats.appendChild(
    stat("Embedder", results.embed_model || "unknown", "produced these numbers", "text")
  );
  stats.appendChild(
    stat("Skipped", String(results.notes_skipped), `${results.notes_total} notes in vault`)
  );
  panel.appendChild(stats);

  const sweep = el("section", { class: "section" }, [
    el("h3", { text: "Hit rate by threshold" }),
  ]);
  const bars = el("div", { class: "bars" });
  for (const row of results.rates_by_threshold || []) {
    const line = el("div", { class: "bar-row" });
    line.appendChild(el("span", { class: "bar-label", text: row.threshold.toFixed(2) }));
    const track = el("div", { class: "bar-track" });
    const fill = el("div", { class: `bar-fill${row.rate === 0 ? " is-empty" : ""}` });
    track.appendChild(fill);
    line.appendChild(track);
    line.appendChild(
      el("span", { class: "bar-val", text: `${row.hits}/${row.total} = ${Math.round(row.rate * 100)}%` })
    );
    bars.appendChild(line);
    requestAnimationFrame(() => {
      fill.style.width = `${row.rate * 100}%`;
    });
  }
  sweep.appendChild(bars);
  panel.appendChild(sweep);

  const cases = el("section", { class: "section" }, [el("h3", { text: "Per note" })]);
  for (const item of results.cases || []) {
    const card = el("div", { class: `case ${item.hit ? "is-hit" : "is-miss"}` });
    card.appendChild(
      el("div", { class: "case-head" }, [
        el("span", { class: "case-verdict", text: item.hit ? "HIT" : "MISS" }),
        el("span", { class: "case-id", text: item.note_id }),
        el("span", { class: "case-sim", text: `best ${fmtScore(item.best_match)}` }),
      ])
    );
    const list = el("dl", { class: "case-pair" });
    list.appendChild(el("dt", { text: "true" }));
    list.appendChild(el("dd", { text: item.best_match_true || "-" }));
    list.appendChild(el("dt", { text: "surfaced" }));
    list.appendChild(el("dd", {}, [
      document.createTextNode(item.best_match_surfaced || "-"),
      el("em", { text: `  [${item.best_match_source || "-"}]` }),
    ]));
    list.appendChild(el("dt", { text: "retrieved" }));
    list.appendChild(
      el("dd", {
        text: `${item.retrieved_count} past failures, ${item.surfaced_count} above the threshold`,
      })
    );
    card.appendChild(list);
    cases.appendChild(card);
  }
  panel.appendChild(cases);
}

/* ------------------------------------------------------------------ loaders */

async function loadBrief() {
  const panel = panelFor("brief");
  panel.textContent = "";
  panel.appendChild(briefSkeleton());
  try {
    const payload = await api(`/api/brief/${encodeURIComponent(state.activeId)}`);
    state.brief = payload;
    renderBrief(payload.exists ? payload : null);
  } catch (err) {
    panel.textContent = "";
    panel.appendChild(
      el("div", { class: "state is-error" }, [
        el("h4", { text: "Could not load the brief" }),
        el("p", { text: err.message }),
      ])
    );
  }
}

async function loadScoring() {
  const panel = panelFor("scoring");
  panel.textContent = "";
  panel.appendChild(briefSkeleton());
  try {
    state.scoring = await api(`/api/scoring/${encodeURIComponent(state.activeId)}`);
    renderScoring(state.scoring);
  } catch (err) {
    panel.textContent = "";
    panel.appendChild(
      el("div", { class: "state is-error" }, [
        el("h4", { text: "Could not compute the scoring" }),
        el("p", { text: err.message }),
      ])
    );
  }
}

async function loadRetrieval() {
  const panel = panelFor("retrieval");
  panel.textContent = "";
  panel.appendChild(briefSkeleton());
  try {
    state.retrieval = await api(`/api/retrieval/${encodeURIComponent(state.activeId)}`);
    renderRetrieval(state.retrieval);
  } catch (err) {
    panel.textContent = "";
    panel.appendChild(
      el("div", { class: "state is-error" }, [
        el("h4", { text: "Could not run retrieval" }),
        el("p", { text: err.message }),
      ])
    );
  }
}

async function loadCanvas() {
  const panel = panelFor("canvas");
  panel.textContent = "";
  panel.appendChild(briefSkeleton());
  try {
    state.canvas = await api(`/api/canvas/${encodeURIComponent(state.activeId)}`);
    renderCanvas(state.canvas.canvas);
  } catch (err) {
    panel.textContent = "";
    panel.appendChild(
      el("div", { class: "state is-error" }, [
        el("h4", { text: "Could not build the canvas" }),
        el("p", { text: err.message }),
      ])
    );
  }
}

async function loadEvaluation() {
  const panel = panelFor("evaluation");
  panel.textContent = "";
  panel.appendChild(briefSkeleton());
  try {
    const payload = await api("/api/eval");
    state.evalResults = payload.exists ? payload.results : null;
    renderEvaluation(state.evalResults);
  } catch (err) {
    panel.textContent = "";
    panel.appendChild(
      el("div", { class: "state is-error" }, [
        el("h4", { text: "Could not load the evaluation" }),
        el("p", { text: err.message }),
      ])
    );
  }
}

/* ------------------------------------------------------------------ actions */

async function generateBrief() {
  const button = $("#btnGenerate");
  if (state.busy) return;
  state.busy = true;
  button.classList.add("is-busy");
  button.disabled = true;

  const panel = panelFor("brief");
  panel.textContent = "";
  const steps = [
    "Reading the note",
    "Retrieving similar decisions",
    "Surfacing assumptions that failed",
    "Computing the weighted table",
    "Writing the counterargument",
    "Validating every claim",
  ];
  const stepNodes = steps.map((label) =>
    el("div", { class: "progress-step" }, [
      el("span", { class: "progress-tick" }),
      el("span", { text: label }),
    ])
  );
  const progress = el("div", { class: "progress" }, [
    el("div", { class: "progress-steps" }, stepNodes),
  ]);
  panel.appendChild(progress);

  let index = 0;
  const ticker = setInterval(() => {
    if (index < stepNodes.length) {
      if (index > 0) {
        stepNodes[index - 1].classList.remove("is-active");
        stepNodes[index - 1].classList.add("is-done");
      }
      stepNodes[index].classList.add("is-active");
      index += 1;
    }
  }, 620);

  try {
    const payload = await api(`/api/brief/${encodeURIComponent(state.activeId)}`, {
      method: "POST",
      body: JSON.stringify({ k: 5 }),
    });
    state.brief = payload;
    renderBrief(payload);
    const issues = (payload.validation || {}).issue_count || 0;
    toast(
      issues > 0
        ? `Brief written with ${issues} claim(s) marked [UNVERIFIED]`
        : "Brief written and every claim validated"
    );
  } catch (err) {
    panel.textContent = "";
    panel.appendChild(
      el("div", { class: "state is-error" }, [
        el("h4", { text: "Could not generate the brief" }),
        el("p", { text: err.message }),
      ])
    );
    toast(err.message, "error");
  } finally {
    clearInterval(ticker);
    stepNodes.forEach((node) => {
      node.classList.remove("is-active");
      node.classList.add("is-done");
    });
    button.classList.remove("is-busy");
    button.disabled = false;
    state.busy = false;
  }
}

async function runEvaluation() {
  const button = $("#btnIndex");
  button.classList.add("is-busy");
  button.disabled = true;
  try {
    const payload = await api("/api/eval", { method: "POST", body: "{}" });
    state.evalResults = payload.results;
    renderEvaluation(state.evalResults);
    toast(`Evaluation finished: ${payload.results.hits}/${payload.results.total}`);
  } catch (err) {
    toast(err.message, "error");
  } finally {
    button.classList.remove("is-busy");
    button.disabled = false;
  }
}

async function rebuildIndex() {
  const button = $("#btnIndex");
  button.classList.add("is-busy");
  button.disabled = true;
  try {
    const stats = await api("/api/index", { method: "POST", body: "{}" });
    toast(`Index rebuilt: ${stats.indexed} indexed, ${stats.skipped} skipped, ${stats.failed} failed`);
    await loadNotes();
    await refreshHealth();
  } catch (err) {
    toast(err.message, "error");
  } finally {
    button.classList.remove("is-busy");
    button.disabled = false;
  }
}

async function selectNote(noteId) {
  state.activeId = noteId;
  state.brief = null;
  state.retrieval = null;
  state.scoring = null;
  state.canvas = null;

  $$(".note").forEach((node) => {
    node.classList.toggle("is-active", node.dataset.id === noteId);
  });

  const note = state.notes.find((n) => n.id === noteId);
  $("#placeholder").hidden = true;
  $("#doc").hidden = false;
  $("#docId").textContent = noteId;
  const tags = $("#docTags");
  tags.textContent = "";
  if (note) {
    tags.appendChild(el("span", { class: "badge", text: note.domain }));
    tags.appendChild(el("span", { class: "badge", text: note.status }));
    tags.appendChild(el("span", { class: "badge", text: note.created }));
  }
  $("#btnGenerate").disabled = false;

  setTab("brief");
  loadBrief();
}

/* ------------------------------------------------------------------ wiring */

function wire() {
  $("#search").addEventListener("input", (event) => renderNotes(event.target.value));

  $$(".tab").forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tab));
  });

  $("#btnGenerate").addEventListener("click", generateBrief);
  $("#btnIndex").addEventListener("click", rebuildIndex);

  $("#btnServe").addEventListener("click", () => {
    window.open("http://127.0.0.1:8765", "_blank");
  });

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea")) return;
    if (event.key === "/") {
      event.preventDefault();
      $("#search").focus();
      return;
    }
    const tabs = ["brief", "canvas", "retrieval", "scoring", "evaluation"];
    const index = tabs.indexOf(state.tab);
    if (event.key === "j" && state.activeId) {
      const current = state.notes.findIndex((n) => n.id === state.activeId);
      const next = state.notes[(current + 1) % state.notes.length];
      if (next) selectNote(next.id);
    }
    if (event.key === "k" && state.activeId) {
      const current = state.notes.findIndex((n) => n.id === state.activeId);
      const previous = state.notes[(current - 1 + state.notes.length) % state.notes.length];
      if (previous) selectNote(previous.id);
    }
    if (event.key === "g") setTab(tabs[(index + 1) % tabs.length]);
  });
}

async function boot() {
  wire();
  await refreshHealth();
  try {
    await loadNotes();
  } catch (err) {
    toast(err.message, "error");
  }
  if (state.notes.length) {
    selectNote(state.notes[0].id);
  }
}

document.addEventListener("DOMContentLoaded", boot);
