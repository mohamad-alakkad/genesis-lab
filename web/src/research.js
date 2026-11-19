// GENESIS Research Console. Every value shown comes from one of four labelled sources:
//   LIVE  - the running simulation and the network's actual outputs this tick
//   LOG   - recorded training/evaluation (research.json, built from the checkpoint store)
//   RUN   - experiments executed here, in this browser, on stated seeds (reproducible)
//   DERIVED - computed from the above (formula stated)
// Nothing here changes the policy's weights. Sensory interventions are labelled out-of-distribution.
import * as SIM from "./sim/world.js";
const { World, CONSTANTS: C, N_CH } = SIM;
const N_MEM = SIM.N_MEM || 0;  // memory inputs exist from world 2.2.0

const SEED_COLORS = ["#1FA89C", "#8A6CF0", "#C77A40", "#4A86D8"];  // validated: dark surface, CVD ΔE ≥ 11
const ACTIONS = C.actions.map((a) => a.replace("_", " "));
const CH = ["solid", "seed", "vent", "shelter", "predator", "water"];
const CH_COLOR = ["#8fa3ad", "#FFB36B", "#F2978A", "#46E7D7", "#ff5f6d", "#6f9cff"];
const SELF = ["energy", "stamina", "speed v/vmax", "turn ω/ωmax", "in shelter", "in vent", "in water", "contact", "light", "raining",
  "sin(day phase)", "cos(day phase)", "heard: proximity", "heard: sin bearing", "heard: cos bearing", "injury (recovery slowed)"].slice(0, SIM.N_SELF || 15);
const MEM = ["memory: recency of last predator percept", "memory: sin bearing to where it was", "memory: cos bearing to where it was",
  "memory: remembered proximity", "memory: it was pouncing"].slice(0, N_MEM);
const el = (tag, attrs = {}, ...kids) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v); } for (const c of kids) e.append(c); return e; };
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const tag = (kind) => `<span class="src src-${kind.toLowerCase()}" title="${{ LIVE: "live from the running simulation", LOG: "recorded training/evaluation", RUN: "run in this browser on stated seeds", DERIVED: "computed from live or recorded values" }[kind]}">${kind}</span>`;

export function initResearch(api) {
  let research = null, open = false, tab = "overview", sense = { blindPredator: false, deaf: false };
  const btn = el("button", { class: "rc-open", "aria-expanded": "false", "aria-controls": "rc" }, el("span", { class: "rc-t" }, "Research Console"), el("span", { class: "rc-s" }, "Explore the intelligence behind Vela"));
  const panel = el("section", { id: "rc", class: "rc", "aria-label": "Research Console", hidden: "" });
  document.body.append(btn, panel);
  const tabs = ["overview", "live", "training", "experiments", "architecture"];
  const head = el("div", { class: "rc-head" }, el("div", { class: "rc-title" }, "Research Console"),
    el("nav", { class: "rc-tabs", role: "tablist" }, ...tabs.map((t) => el("button", { role: "tab", "data-tab": t, "aria-selected": "false" }, t))),
    el("button", { class: "rc-close", "aria-label": "Close research console" }, "×"));
  const body = el("div", { class: "rc-body" });
  panel.append(head, body);
  const setOpen = (v) => { open = v; panel.hidden = !v; btn.setAttribute("aria-expanded", String(v)); document.body.classList.toggle("rc-on", v); if (v) render(); };
  btn.onclick = () => setOpen(!open);
  head.querySelector(".rc-close").onclick = () => setOpen(false);
  head.querySelectorAll("[data-tab]").forEach((b) => (b.onclick = () => { tab = b.dataset.tab; render(); }));
  addEventListener("keydown", (e) => { if (e.key === "Escape" && open) setOpen(false); });
  fetch("public/research.json").then((r) => r.ok ? r.json() : null).then((j) => { research = j; if (open) render(); }).catch(() => {});

  api.setSensory = (obs) => {  // experimental interventions on the observation vector (policy weights untouched)
    if (!sense.blindPredator && !sense.deaf) return obs;
    const o = Float64Array.from(obs);
    if (sense.blindPredator) for (let k = 0; k < C.n_rays; k++) o[N_CH * k + 4] = 0;
    if (sense.deaf) { const S = N_CH * C.n_rays; o[S + 12] = 0; o[S + 13] = 0; o[S + 14] = 0; }
    return o;
  };

  function render() {
    head.querySelectorAll("[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
    body.replaceChildren(({ overview, live, training, experiments, architecture })[tab]());
  }
  setInterval(() => { if (open && tab === "live") updateLive(); }, 200);

  // ---------------------------------------------------------------- overview
  function overview() {
    const b = api.brainMeta(), params = api.paramCount();
    const R = research?.reports || {}, g5 = R["gen5-families"], g4 = R["gen4-results"], g3 = R["gen3-diagnosis"], g2 = R["gen2-diagnosis"];
    const it5 = g5?.interpretation || {}, refs = g5?.references_world_3_0 ? Object.values(g5.references_world_3_0)[0] : null;
    const arm = (a, f) => g5?.arms?.[a]?.[f];
    const row = (label, r) => r ? [label, `${fmt(r.ticks.mean, 0)} [${fmt(r.ticks.ci95[0], 0)}, ${fmt(r.ticks.ci95[1], 0)}]`, pct(r.caught.mean ?? r.caught), fmt(r.food.mean ?? r.food, 1),
      r.shelter_when_threat ? pct(r.shelter_when_threat.mean ?? r.shelter_when_threat, 1) : "—"] : null;
    const refRow = (label, r) => r ? [label, fmt(r.ticks, 0), pct(r.caught), fmt(r.food, 1), r.shelter_when_threat != null ? pct(r.shelter_when_threat, 1) : "—"] : null;
    const rows = g5 ? [
      row("standard · gen5 (trained vs fast predators)", arm("gen5-broad", "standard")), row("standard · gen5 (standard training)", arm("gen5-std", "standard")),
      refRow("standard · scripted survivor", refs?.standard?.["scripted survivor"]), refRow("standard · privileged preemptive hider", refs?.standard?.["privileged preemptive hider"]),
      row("relentless · gen5 (trained vs fast predators)", arm("gen5-broad", "relentless")), row("relentless · gen5 (standard training)", arm("gen5-std", "relentless")),
      refRow("relentless · scripted survivor", refs?.relentless?.["scripted survivor"]), refRow("relentless · privileged preemptive hider", refs?.relentless?.["privileged preemptive hider"]),
    ].filter(Boolean) : [];
    const g6 = research?.lineages?.gen6 || [];
    return el("div", { class: "rc-sec" },
      el("p", { class: "rc-lede", html: `Vela is a <b>PPO</b> agent: two tanh MLPs (policy and value, ${b ? b.hidden.join("×") : "?"} hidden units, ${params.toLocaleString()} parameters) trained <b>from random weights</b> in a NumPy simulation, then run here unchanged: the browser executes the exported weights once per 0.1 s world tick and takes the argmax action.` }),
      el("div", { class: "rc-tiles" },
        tile("brain on stage", b ? `${b.milestone}` : "none", b ? `${b.source} · ${b.step.toLocaleString()} steps · world ${b.spec}` : "", "LIVE"),
        tile("world", b ? b.spec : "—", "fertility cycles · withering thickets · fissures · injury · five shared acts", "LOG"),
        tile("now training", g6.length ? "gen6" : "—", g6.length ? "objective ablation: reward for food vs reward for staying alive" : "", "LOG")),
      el("h3", {}, "Latest finding: why Vela flees instead of hiding"),
      g5 ? el("div", { class: "rc-callout", html: `${tag("LOG")} <b>gen5 hypothesis:</b> ${it5.hypothesis_gen5}<br><b>Generalization:</b> ${it5.generalization}<br><b>Why:</b> ${it5.why}<br><b>Next:</b> ${it5.next}` }) : el("p", {}, "loading…"),
      rows.length ? table(["condition · policy", "survival ticks (95% CI)", "caught", "seeds eaten", "sheltered when threatened"], rows) : "",
      el("p", { class: "rc-note", html: g5 ? `${g5.seeds}. 100 episodes per run per family; arms pool 4 seeds (400 episodes). References: 100 episodes on the same fresh seeds; the hider uses privileged world state and is evaluation-only.` : "" }),
      el("h3", {}, "How we got here"),
      el("ul", { class: "rc-list", html: [
        g2 && `<li><b>gen2 (world 2.0.0):</b> ${g2.finding}</li>`,
        g3 && `<li><b>gen3 (world 2.1.0):</b> ${g3.finding}</li>`,
        g4 && `<li><b>gen4 (world 2.2.0):</b> ${g4.interpretation?.horizon} ${g4.interpretation?.memory}</li>`,
        g5 && `<li><b>gen5 (world 3.0.0):</b> faster predators in training did not create conditional hiding; the food objective favors evasion even where hiding would keep Vela alive longer.</li>`,
      ].filter(Boolean).join("") }));
  }
  const pct = (x, d = 0) => x == null ? "—" : `${(100 * x).toFixed(d)}%`;
  function tile(k, v, sub, src) { return el("div", { class: "rc-tile", html: `<div class="k">${k} ${tag(src)}</div><div class="v">${v}</div><div class="s">${sub}</div>` }); }
  function table(hdr, rows) { const t = el("table", { class: "rc-table" }); t.append(el("tr", {}, ...hdr.map((h) => el("th", {}, h)))); for (const r of rows) t.append(el("tr", {}, ...r.map((c) => el("td", {}, String(c))))); return t; }

  // ---------------------------------------------------------------- live
  let liveEls = null;
  function live() {
    const wrap = el("div", { class: "rc-sec rc-live" });
    const svg = svgEl("svg", { viewBox: "-110 -110 220 220", class: "rc-rays", role: "img", "aria-label": "Ray sensors: what Vela senses in each direction" });
    const probs = el("div", { class: "rc-probs" });
    const selfT = el("div", { class: "rc-self" });
    const scal = el("div", { class: "rc-scal" });
    const ev = el("ol", { class: "rc-events", reversed: "" });
    const spark = svgEl("svg", { viewBox: "0 0 300 60", class: "rc-spark", preserveAspectRatio: "none" });
    const senses = el("fieldset", { class: "rc-senses" }, el("legend", { html: `Sensory interventions ${tag("LIVE")} <span class="ood">out-of-distribution</span>` }),
      chk("blind to the predator (ray channel 4 → 0)", "blindPredator"), chk("deaf (hearing inputs 78–80 → 0)", "deaf"),
      el("p", { class: "rc-note" }, "These zero inputs before the network sees them. The network never trained this way, so behavior here measures robustness, not a skill."));
    wrap.append(
      el("div", { class: "rc-grid2" },
        el("div", {}, el("h3", { html: `Senses ${tag("LIVE")}` }), svg, el("div", { class: "rc-legend" }, ...CH.map((c, i) => el("span", { html: `<i style="background:${CH_COLOR[i]}"></i>${c}` }))),
          el("p", { class: "rc-note" }, "11 rays over ±120°. Each wedge segment = proximity (1 − d/6 u) of the nearest object of that kind; nothing behind Vela is seen, the predator only within light-dependent sight (2.5–6 u). Hearing covers all directions.")),
        el("div", {}, el("h3", { html: `Policy output ${tag("LIVE")}` }), probs, scal)),
      el("h3", { html: `Observation vector (${SIM.OBS_DIM} inputs, raw, before normalization) ${tag("LIVE")}` }), selfT,
      el("div", { class: "rc-grid2" },
        el("div", {}, el("h3", { html: `Reward this episode ${tag("LIVE")}` }), spark, el("p", { class: "rc-note" }, "Cumulative training reward: +1 seed, −0.05/tick in an active vent, −1 starvation, −2 caught. Nothing rewards hiding or resting.")),
        el("div", {}, el("h3", { html: `Behavioral events ${tag("LIVE")}` }), ev)),
      senses,
      el("div", { class: "rc-row" }, el("button", { class: "rc-btn", id: "rcReplay" }, "Restart this episode's seed"), el("span", { class: "rc-note", id: "rcSeed" })),
      el("p", { class: "rc-note", id: "rcVisitor" }));
    liveEls = { svg, probs, selfT, scal, ev, spark };
    wrap.querySelector("#rcReplay").onclick = () => api.restart(api.seed());
    updateLive();
    return wrap;
  }
  function chk(label, key) { const c = el("input", { type: "checkbox" }); c.checked = sense[key]; c.onchange = () => { sense[key] = c.checked; api.log({ kind: "intervention", text: `${label}: ${c.checked ? "on" : "off"}` }); }; return el("label", { class: "rc-chk" }, c, " " + label); }
  function updateLive() {
    if (!liveEls) return;
    const s = api.state(); if (!s || !s.obs) return;
    const o = s.obs, svg = liveEls.svg;
    // rays polar plot
    svg.replaceChildren();
    for (const r of [33, 66, 100]) svg.append(svgEl("circle", { r, class: "ring" }));
    C.n_rays && api.rayRel().forEach((rel, k) => {
      const a = rel - Math.PI / 2;  // heading points up
      for (let ch = 0; ch < N_CH; ch++) {
        const v = o[N_CH * k + ch]; if (v <= 0) continue;
        const r0 = 100 * (1 - v), w = 0.07, aa = a + (ch - 2.5) * w * 0.33;
        svg.append(svgEl("line", { x1: Math.cos(aa) * 100, y1: Math.sin(aa) * 100, x2: Math.cos(aa) * Math.max(r0, 6), y2: Math.sin(aa) * Math.max(r0, 6), stroke: CH_COLOR[ch], "stroke-width": 3, "stroke-linecap": "round" }));
      }
    });
    const S = N_CH * C.n_rays;
    if (o[S + 12] > 0) { const b = Math.atan2(o[S + 13], o[S + 14]) - Math.PI / 2; svg.append(svgEl("circle", { cx: Math.cos(b) * 104, cy: Math.sin(b) * 104, r: 5, fill: "#ff5f6d", class: "heard" })); }
    svg.append(svgEl("path", { d: "M0,-9 L6,7 L0,3 L-6,7 Z", fill: "#46E7D7" }));
    // probabilities
    const p = s.probs, chosen = s.action;
    liveEls.probs.replaceChildren(...ACTIONS.map((a, i) => el("div", { class: `rc-pr${i === chosen ? " on" : ""}`, html: `<span>${a}</span><b><i style="width:${p ? (p[i] * 100).toFixed(1) : 0}%"></i></b><em>${p ? (p[i] * 100).toFixed(1) : "—"}%</em>` })));
    const ent = p ? -p.reduce((acc, q) => acc + (q > 0 ? q * Math.log(q) : 0), 0) : null;
    liveEls.scal.innerHTML = `<div>selected: <b>${ACTIONS[chosen] ?? "—"}</b> (argmax)</div>
      <div>V(s) critic estimate: <b>${fmt(s.value, 3)}</b> ${tag("LIVE")} <span class="rc-note">discounted return the critic expects (γ=0.99); not a probability of survival</span></div>
      <div>policy entropy: <b>${fmt(ent, 3)}</b> nats of max ${fmt(Math.log(8), 3)} ${tag("DERIVED")}</div>`;
    // observation table
    const rows = [];
    for (let k = 0; k < C.n_rays; k++) rows.push(`<tr><td>ray ${k} (${fmt(api.rayRel()[k] * 180 / Math.PI, 0)}°)</td>${CH.map((c, ch) => `<td class="${o[N_CH * k + ch] > 0 ? "hot" : ""}">${fmt(o[N_CH * k + ch], 2)}</td>`).join("")}</tr>`);
    liveEls.selfT.innerHTML = `<table class="rc-table obs"><tr><th>input</th>${CH.map((c) => `<th>${c}</th>`).join("")}</tr>${rows.join("")}</table>
      <table class="rc-table obs"><tr><th>#</th><th>self / environment input</th><th>value</th></tr>${SELF.map((n, i) => `<tr><td>${S + i}</td><td>${n}</td><td>${fmt(o[S + i], 3)}</td></tr>`).join("")}
      ${MEM.map((n, i) => `<tr><td>${S + SELF.length + i}</td><td>${n}</td><td>${fmt(o[S + SELF.length + i], 3)}</td></tr>`).join("")}
      <tr><td>${S + SELF.length + N_MEM}–${S + SELF.length + N_MEM + 7}</td><td>last action (one-hot)</td><td>${ACTIONS[Array.from(o.slice(S + SELF.length + N_MEM)).indexOf(1)] ?? "—"}</td></tr></table>`;
    // reward sparkline
    const rw = s.rewardTrace; const n = rw.length;
    if (n > 1) { const mn = Math.min(0, ...rw), mx = Math.max(1, ...rw); const pts = rw.map((v, i) => `${(i / (n - 1)) * 300},${58 - ((v - mn) / (mx - mn)) * 54}`).join(" ");
      liveEls.spark.innerHTML = `<polyline points="${pts}" fill="none" stroke="#FFB36B" stroke-width="2" vector-effect="non-scaling-stroke"/><text x="296" y="10" text-anchor="end">${fmt(rw[n - 1], 2)}</text>`; }
    liveEls.ev.replaceChildren(...s.events.slice(-14).reverse().map((e) => el("li", { html: `<code>t=${e.tick ?? "—"}</code> ${e.text}` })));
    const vEl = document.getElementById("rcVisitor");
    if (vEl) vEl.innerHTML = s.visitorActs ? `${tag("LIVE")} This episode includes ${s.visitorActs} visitor intervention${s.visitorActs > 1 ? "s" : ""} (marked [visitor] above). Visitor episodes are never part of the evaluation data; nature performs the same acts at fixed rates during training and evaluation.` : "";
    document.getElementById("rcSeed") && (document.getElementById("rcSeed").textContent = `episode seed ${api.seed()} · world ${api.brainMeta()?.spec ?? ""} · family standard`);
  }

  // ---------------------------------------------------------------- training
  function training() {
    const wrap = el("div", { class: "rc-sec" });
    if (!research) { wrap.append(el("p", {}, "loading training records…")); return wrap; }
    const lin = research.lineages;
    for (const name of ["gen6", "gen5", "gen4", "gen3", "gen2", "gen1"]) {
      const runs = lin[name]; if (!runs?.length) { if (name === "gen3") wrap.append(el("p", { class: "rc-note", html: `${tag("LOG")} gen3 (world 2.1.0): no published evaluations yet.` })); continue; }
      const v = research.world_versions[name] || runs[0].world_spec_version;
      const key = runs[0].eval.some((e) => e.score != null) ? "score" : "food_mean";
      wrap.append(el("h3", { html: `${name} · world ${v} · ${runs.length} independent seeds ${tag("LOG")}` }));
      const ref = name === "gen2" ? { y: research.reports["gen2-diagnosis"]?.rows?.["scripted survivor (evaluation-only baseline)"]?.ticks, label: "scripted survivor (60 ep.)" } : null;
      wrap.append(lineChart(runs, key, key === "score" ? "held-out survival (ticks, of 1500)" : "held-out seeds eaten per episode", ref));
      if (key === "score") wrap.append(lineChart(runs, "caught", "held-out capture rate (fraction of episodes)", null, true));
      wrap.append(table(["seed", "steps", "best", "latest", "latest caught", "last publish", "cfg"], runs.map((r) => {
        const ev = r.eval, best = Math.max(...ev.map((e) => e[key] ?? 0)), last = ev[ev.length - 1] || {};
        const age = (Date.now() - Date.parse(r.published_at)) / 60000;
        return [r.run_id, r.global_step.toLocaleString(), fmt(best, key === "score" ? 0 : 1), fmt(last[key], key === "score" ? 0 : 1), last.caught == null ? "—" : pct(last.caught),
          `${age < 1 ? "<1" : age.toFixed(0)} min ago`, `${r.config.hidden?.join("×")} · ${r.config.n_envs} envs · lr ${r.config.lr}`];
      })));
      wrap.append(el("p", { class: "rc-note" }, `Each point: deterministic argmax policy on a fixed held-out seed set (eval range ≥ 3e9, never trained on), ${runs[0].config.eval_episodes ?? 100} episodes, family "standard". Lineages run in different worlds and are not directly comparable.`));
    }
    wrap.append(el("h3", { html: `Method ${tag("LOG")}` }), el("ul", { class: "rc-list", html: `
      <li>PPO, clipped objective (ε 0.2), GAE (γ 0.99, λ 0.95), 4 epochs, minibatch 2048, Adam lr 3e-4 → 5% linear decay, entropy 0.01, value 0.5, grad-norm 0.5.</li>
      <li>Hand-written NumPy implementation; gradients verified against finite differences in the test suite.</li>
      <li>256 parallel environments × 128 steps per update; observation running mean/variance normalization (frozen at export).</li>
      <li>Baselines are evaluation-only scripts (random, forager, survivor) and are excluded from the browser bundle.</li>
      <li>Checkpoints: write → SHA-256 → load-test → manifest → published to a git store; cloud cycles resume from it (verified across fresh sandboxes).</li>` }));
    return wrap;
  }
  // color follows identity: the run (≤ 4 runs) or the experimental arm (> 4 runs; the arm's second seed is drawn fainter)
  const armOf = (id) => id.replace(/-s\d+$/, "");
  const styleOf = (runs) => { const arms = [...new Set(runs.map((r) => armOf(r.run_id)))], byArm = runs.length > 4;
    return runs.map((r, i) => byArm ? { color: SEED_COLORS[arms.indexOf(armOf(r.run_id)) % 4], op: runs.filter((q, j) => j < i && armOf(q.run_id) === armOf(r.run_id)).length ? 0.5 : 1 }
      : { color: SEED_COLORS[i % 4], op: 1 }); };
  function lineChart(runs, key, title, ref, isFrac = false) {
    const sty = styleOf(runs);
    const W = 560, H = 190, L = 44, R = 12, T = 12, B = 30;
    const pts = runs.map((r) => r.eval.filter((e) => e[key] != null).map((e) => [e.step, e[key]]));
    const xmax = Math.max(...pts.flat().map((p) => p[0]), 1), ys = pts.flat().map((p) => p[1]).concat(ref?.y ?? []);
    const ymax = isFrac ? 1 : Math.ceil(Math.max(...ys) / 100) * 100 || 1, ymin = 0;
    const X = (x) => L + (x / xmax) * (W - L - R), Y = (y) => T + (1 - (y - ymin) / (ymax - ymin)) * (H - T - B);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "rc-chart", role: "img", "aria-label": title });
    for (let i = 0; i <= 4; i++) { const y = ymin + (ymax - ymin) * i / 4; svg.append(svgEl("line", { x1: L, x2: W - R, y1: Y(y), y2: Y(y), class: "grid" }), svgText(L - 6, Y(y) + 3, isFrac ? `${(y * 100).toFixed(0)}%` : y.toFixed(0), "end")); }
    for (let i = 0; i <= 4; i++) { const x = xmax * i / 4; svg.append(svgText(X(x), H - 10, `${(x / 1e6).toFixed(0)}M`, "middle")); }
    svg.append(svgText(W - R, H - 10 + 0, "", "end"));
    if (ref?.y) svg.append(svgEl("line", { x1: L, x2: W - R, y1: Y(ref.y), y2: Y(ref.y), class: "ref" }), svgText(W - R - 2, Y(ref.y) - 5, `${ref.label}: ${ref.y.toFixed(0)}`, "end", "refl"));
    pts.forEach((p, i) => { if (!p.length) return; svg.append(svgEl("polyline", { points: p.map(([x, y]) => `${X(x)},${Y(y)}`).join(" "), fill: "none", stroke: sty[i].color, "stroke-opacity": sty[i].op, "stroke-width": 2 }));
      const [lx, ly] = p[p.length - 1]; svg.append(svgEl("circle", { cx: X(lx), cy: Y(ly), r: 4, fill: sty[i].color, "fill-opacity": sty[i].op, stroke: "#0E1822", "stroke-width": 2 })); });
    // hover: crosshair + tooltip with every seed's value at the nearest evaluation
    const hair = svgEl("line", { y1: T, y2: H - B, class: "hair", visibility: "hidden" }); svg.append(hair);
    const tip = el("div", { class: "rc-tip", hidden: "" });
    const box = el("figure", { class: "rc-fig" }, el("figcaption", { html: `${title} ${tag("LOG")}` }), svg, tip,
      el("div", { class: "rc-legend" }, ...runs.map((r, i) => el("span", { html: `<i style="background:${sty[i].color};opacity:${sty[i].op}"></i>${r.run_id}` }))));
    svg.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect(), x = ((e.clientX - rect.left) / rect.width) * W; if (x < L || x > W - R) { hair.setAttribute("visibility", "hidden"); tip.hidden = true; return; }
      const step = ((x - L) / (W - L - R)) * xmax;
      hair.setAttribute("x1", x); hair.setAttribute("x2", x); hair.setAttribute("visibility", "visible");
      tip.hidden = false; tip.style.left = `${(x / W) * 100}%`;
      tip.innerHTML = `<b>${(step / 1e6).toFixed(1)}M steps</b>` + runs.map((r, i) => { const near = r.eval.filter((q) => q[key] != null).reduce((a, q) => Math.abs(q.step - step) < Math.abs(a.step - step) ? q : a, r.eval[0]);
        return `<div><i style="background:${sty[i].color};opacity:${sty[i].op}"></i>${r.run_id}: ${isFrac ? pct(near[key]) : fmt(near[key], 0)}${near.score_ci95 && !isFrac ? ` [${fmt(near.score_ci95[0], 0)}, ${fmt(near.score_ci95[1], 0)}]` : ""} <span>@${(near.step / 1e6).toFixed(1)}M</span></div>`; }).join("");
    });
    svg.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); tip.hidden = true; });
    return box;
  }

  // ---------------------------------------------------------------- experiments
  function experiments() {
    const brains = api.brains();
    const wrap = el("div", { class: "rc-sec" });
    const picks = el("div", { class: "rc-row wrap" }, ...brains.map((b, i) => { const c = el("input", { type: "checkbox", value: b.key }); c.checked = i < 2 || b.key === "step0"; return el("label", { class: "rc-chk" }, c, ` ${b.label}`); }));
    const n = el("select", {}, ...[10, 20, 50].map((v) => el("option", { value: v }, `${v} episodes`)));
    const base = el("input", { type: "number", value: "4100000000", min: "4000000000", max: "4294967295", step: "1", "aria-label": "first seed" });
    const fam = el("select", {}, ...Object.keys(C.families).filter((f) => f !== "train").map((f) => el("option", { value: f }, f)));
    const cond = el("select", {}, el("option", { value: "none" }, "senses intact"), el("option", { value: "blind" }, "blind to predator (OOD)"), el("option", { value: "deaf" }, "deaf (OOD)"));
    const go = el("button", { class: "rc-btn" }, "Run controlled comparison");
    const out = el("div", {});
    wrap.append(el("p", { class: "rc-lede", html: `Run several brains on <b>the same seeds</b> (same layouts, same predator, same weather) in this browser, without rendering. Results are reproducible: same seeds + same brain → same outcome. ${tag("RUN")}` }),
      el("h3", {}, "Brains"), picks, el("div", { class: "rc-row wrap" }, n, el("label", {}, "first seed ", base), fam, cond, go), out);
    go.onclick = async () => {
      const chosen = [...picks.querySelectorAll("input:checked")].map((c) => c.value); if (!chosen.length) return;
      go.disabled = true; const N = +n.value, s0 = +base.value, family = fam.value, condition = cond.value;
      out.replaceChildren(el("p", { class: "rc-note", id: "rcProg" }, "running…"));
      const res = {};
      for (const key of chosen) res[key] = await runBatch(api.brainByKey(key), family, s0, N, condition, (i) => (out.querySelector("#rcProg").textContent = `${api.brainLabel(key)}: episode ${i}/${N}`));
      go.disabled = false;
      const ci = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length; const sd = Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / Math.max(1, a.length - 1)); return [m, 1.96 * sd / Math.sqrt(a.length)]; };
      const row = (k) => { const r = res[k]; const [t, th] = ci(r.ticks), [f, fh] = ci(r.food), [c] = ci(r.caught); const shT = r.shelterThreat.filter((x) => x != null); return [api.brainLabel(k), `${fmt(t, 0)} ± ${fmt(th, 0)}`, pct(c), `${fmt(f, 1)} ± ${fmt(fh, 1)}`, shT.length ? pct(ci(shT)[0], 1) : "—"]; };
      out.replaceChildren(table(["brain", "survival ticks (95% CI)", "caught", "seeds eaten", "sheltered when threatened"], chosen.map(row)),
        el("p", { class: "rc-note", html: `${tag("RUN")} world ${C.world_spec_version}, family "${family}", seeds ${s0}…${s0 + N - 1}, deterministic argmax, condition: ${condition === "none" ? "senses intact" : condition + " (out-of-distribution)"}, cap 1500 ticks. "Threatened" = predator hunting or searching within 6 u (simulation state, not an input).` }));
    };
    return wrap;
  }
  function runBatch(brain, family, s0, N, condition, progress) {
    return new Promise((resolve) => {
      const r = { ticks: [], food: [], caught: [], shelterThreat: [] }; let i = 0;
      const S = N_CH * C.n_rays;
      const step = () => {
        const t0 = performance.now();
        while (i < N && performance.now() - t0 < 30) {
          const w = new World(family); w.reset(s0 + i);
          let threat = 0, shelt = 0;
          while (!w.dead && !w.truncated) {
            let o = w.observe();
            if (condition === "blind") for (let k = 0; k < C.n_rays; k++) o[N_CH * k + 4] = 0;
            if (condition === "deaf") { o[S + 12] = 0; o[S + 13] = 0; o[S + 14] = 0; }
            w.step(brain.act(o).action); w.events.length = 0;
            const p = w.pred; if (p.on && (p.mode === 1 || p.mode === 2) && Math.hypot(p.x - w.x, p.y - w.y) < 6) { threat++; if (w.inShelter) shelt++; }
          }
          r.ticks.push(w.t); r.food.push(w.epFood); r.caught.push(w.caught ? 1 : 0); r.shelterThreat.push(threat ? shelt / threat : null);
          i++; progress(i);
        }
        if (i < N) setTimeout(step, 0); else resolve(r);
      };
      step();
    });
  }

  // ---------------------------------------------------------------- architecture
  function architecture() {
    const wrap = el("div", { class: "rc-sec" });
    wrap.append(el("figure", { class: "rc-fig", html: archSVG() }),
      el("h3", {}, "Verified"),
      el("ul", { class: "rc-list", html: `
        <li><b>One world, two implementations.</b> Python (NumPy, batch-vectorized, float64) is the training reference; this page runs a JavaScript port. Parity test: 20 recorded episodes × 900 ticks incl. visitor actions match within 1e-6 per observation.</li>
        <li><b>Portable randomness.</b> Each episode is a mulberry32 stream seeded by an integer, identical in both languages, so a seed reproduces a world exactly.</li>
        <li><b>Browser inference.</b> The exported JSON bundle holds the normalization statistics and both MLPs; the policy runs in float64, once per tick, argmax.</li>
        <li><b>Training compute.</b> Claude Code Cloud sandboxes (4 CPU, 16 GB) run 45-minute cycles of a plain Python trainer: 4 seeds in parallel at ≈13k steps/s each. Cycles publish verified checkpoints every 8 minutes and the next cycle resumes from them.</li>
        <li><b>Rendering and sound.</b> three.js scene and procedural Web Audio. Both read simulation state and the chosen action; neither can choose an action.</li>` }),
      el("h3", {}, "Not yet verified / planned"),
      el("ul", { class: "rc-list", html: `<li>gen3 training (world 2.1.0) is in progress; no gen3 claims are made here.</li><li>Phone performance and the audio experience have not been formally evaluated.</li><li>Generalization families beyond "standard" for gen3, reward-hacking audit, and final evaluation report: pending.</li>` }),
      el("p", { class: "rc-note", html: `Source and documentation: <a href="https://github.com/mohamad-alakkad/genesis-lab" target="_blank" rel="noopener">github.com/mohamad-alakkad/genesis-lab</a> (world spec: docs/WORLD_SPEC.md, decisions: docs/DECISIONS.md).` }));
    return wrap;
  }
  function archSVG() {
    const box = (x, y, w, h, t, s) => `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/><text x="${x + w / 2}" y="${y + 20}" class="t">${t}</text><text x="${x + w / 2}" y="${y + 37}" class="s">${s}</text></g>`;
    const arrow = (x1, y1, x2, y2, l) => `<g><line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#ah)"/><text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 5}" class="l">${l}</text></g>`;
    return `<svg viewBox="0 0 600 250" class="rc-arch" role="img" aria-label="GENESIS architecture">
      <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z"/></marker></defs>
      ${box(10, 20, 170, 50, "World (Python)", "vectorized NumPy, float64")}
      ${box(10, 110, 170, 50, "PPO trainer", "NumPy MLPs, from scratch")}
      ${box(10, 195, 170, 46, "Cloud cycles", "4 seeds × 45 min")}
      ${box(220, 110, 160, 50, "Checkpoint store", "SHA-256 · git branch")}
      ${box(420, 110, 170, 50, "Policy bundle", "JSON weights + stats")}
      ${box(420, 20, 170, 50, "World (JavaScript)", "parity-tested port")}
      ${box(420, 195, 170, 46, "This page", "3D render · audio · console")}
      ${arrow(95, 70, 95, 108, "observations, rewards")}
      ${arrow(95, 160, 95, 193, "runs in")}
      ${arrow(180, 135, 218, 135, "verified ckpt")}
      ${arrow(380, 135, 418, 135, "export")}
      ${arrow(505, 160, 505, 193, "argmax / tick")}
      ${arrow(505, 70, 505, 108, "observation")}
      ${arrow(180, 45, 418, 45, "same spec + seeds (parity)")}
    </svg>`;
  }
}
function svgEl(tag, attrs = {}) { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; }
function svgText(x, y, t, anchor = "start", cls = "ax") { const e = svgEl("text", { x, y, "text-anchor": anchor, class: cls }); e.textContent = t; return e; }
