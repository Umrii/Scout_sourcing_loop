"use strict";

// ───────────────────────── helpers ─────────────────────────
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const KEY_STORE = "scout_api_key";

function getKey() {
  return $("#apiKey").value.trim();
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": getKey(),
      ...(opts.headers || {}),
    },
  });
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
    if (res.status === 401) detail = "Unauthorized — check the X-API-Key (top right).";
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.status === 204 ? null : res.json();
}

function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), 3500);
}

const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cell = (v) => (v === null || v === undefined || v === "" ? `<td class="null">null</td>` : `<td>${esc(v)}</td>`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ───────────────────────── health ─────────────────────────
async function checkHealth() {
  try {
    const h = await fetch("/health").then((r) => r.json());
    $("#healthDot").className = "dot ok";
    $("#healthText").textContent = h.llm_mode + " · " + h.database;
    $("#backend").textContent = `${h.llm_mode} LLM, ${h.database}`;
  } catch {
    $("#healthDot").className = "dot bad";
    $("#healthText").textContent = "offline";
  }
}

// ───────────────────────── tabs ─────────────────────────
$$(".tab-btn").forEach((btn) =>
  btn.addEventListener("click", () => {
    $$(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
    const tab = btn.dataset.tab;
    $$(".tab").forEach((s) => s.classList.toggle("hidden", s.id !== "tab-" + tab));
    if (tab === "experts" && !$("#expertsResults").dataset.loaded) searchExperts();
    if (tab === "activity") loadRuns();
  })
);

// ───────────────────────── source loop ─────────────────────────
function setSteps(state) {
  // state: "" | running stage name | "done"
  const order = ["classify", "route", "enrich", "outreach"];
  $$("#steps li").forEach((li) => {
    li.classList.remove("active", "done");
    if (state === "done") li.classList.add("done");
    else if (state && order.indexOf(li.dataset.step) <= order.indexOf(state)) li.classList.add("active");
  });
}

$("#briefForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const brief = {
    title: $("#b-title").value.trim(),
    description: $("#b-desc").value.trim(),
    required_domains: $("#b-domains").value.split(",").map((s) => s.trim()).filter(Boolean),
    min_seniority: $("#b-seniority").value || null,
    num_experts_needed: parseInt($("#b-num").value, 10) || 3,
  };
  const btn = $("#briefForm button");
  btn.disabled = true;
  $("#runMeta").innerHTML = "";
  $("#results").innerHTML = `<div class="empty"><span class="spinner"></span> running the loop…</div>`;
  setSteps("classify");
  try {
    const project = await api("/projects", { method: "POST", body: JSON.stringify(brief) });
    const run = await api(`/projects/${project.id}/source`, { method: "POST" });
    $("#runMeta").innerHTML = `run_id <b>${run.run_id.slice(0, 8)}</b> · project <b>${project.id.slice(0, 8)}</b>`;
    setSteps("outreach");
    const matches = await pollMatches(project.id, brief.num_experts_needed);
    setSteps("done");
    renderMatches(matches, project);
  } catch (err) {
    setSteps("");
    $("#results").innerHTML = "";
    toast(err.message, true);
  } finally {
    btn.disabled = false;
  }
});

async function pollMatches(projectId, want) {
  for (let i = 0; i < 12; i++) {
    const matches = await api(`/projects/${projectId}/matches`);
    if (matches.length >= want || (matches.length > 0 && i >= 3)) return matches;
    await sleep(500);
  }
  return api(`/projects/${projectId}/matches`);
}

function renderMatches(matches, project) {
  if (!matches.length) {
    $("#results").innerHTML = `<div class="empty">No matches — ingest experts in this domain first (Org memory tab).</div>`;
    return;
  }
  const head = `<div class="section-title"><h2>Shortlist for “${esc(project.title)}”</h2>
    <span class="muted">${matches.length} ranked</span></div>`;
  $("#results").innerHTML =
    head +
    matches
      .map((m, i) => {
        const e = m.expert;
        const sub = [e.current_title, e.company].filter(Boolean).join(" · ") || "—";
        const pct = Math.round(m.overall_score * 100);
        return `<div class="match">
        <div class="match-head">
          <span class="rank">#${i + 1}</span>
          <div>
            <div class="match-name">${esc(e.name)}</div>
            <div class="match-sub">${esc(sub)}</div>
          </div>
          <span class="spacer"></span>
          <span class="badge ${m.relevance}">${m.relevance}</span>
        </div>
        <div class="scorebar"><i style="width:${pct}%"></i></div>
        <div class="scorerow">
          <span>score <b>${m.overall_score.toFixed(2)}</b></span>
          <span>domain ${m.domain_match_score.toFixed(2)}</span>
          <span>seniority ${m.seniority_fit.toFixed(2)}</span>
        </div>
        <p class="rationale">${esc(m.rationale)}</p>
        <details class="outreach">
          <summary>Outreach draft</summary>
          <div class="draft" id="draft-${m.id}">${esc(m.outreach_draft) || "—"}</div>
          <div style="margin-top:8px"><button class="ghost" data-regen="${m.id}">Regenerate</button></div>
        </details>
      </div>`;
      })
      .join("");

  $$("[data-regen]").forEach((b) =>
    b.addEventListener("click", async () => {
      b.disabled = true;
      try {
        const m = await api(`/matches/${b.dataset.regen}/outreach`, { method: "POST" });
        $(`#draft-${m.id}`).textContent = m.outreach_draft;
        toast("Outreach regenerated");
      } catch (err) {
        toast(err.message, true);
      } finally {
        b.disabled = false;
      }
    })
  );
}

// ───────────────────────── org memory ─────────────────────────
async function searchExperts() {
  const p = new URLSearchParams();
  if ($("#s-domain").value.trim()) p.set("domain", $("#s-domain").value.trim());
  if ($("#s-seniority").value) p.set("seniority", $("#s-seniority").value);
  if ($("#s-q").value.trim()) p.set("q", $("#s-q").value.trim());
  const box = $("#expertsResults");
  box.dataset.loaded = "1";
  box.innerHTML = `<div class="empty"><span class="spinner"></span></div>`;
  try {
    const rows = await api("/experts/search?" + p.toString());
    if (!rows.length) { box.innerHTML = `<div class="empty">No experts found.</div>`; return; }
    box.innerHTML = `<table><thead><tr>
        <th>Name</th><th>Title</th><th>Company</th><th>Seniority</th><th>Domains</th><th>Yrs</th><th>Location</th>
      </tr></thead><tbody>${rows.map(expertRow).join("")}</tbody></table>`;
  } catch (err) {
    box.innerHTML = "";
    toast(err.message, true);
  }
}

function expertRow(e) {
  const domains = (e.domains || []).map((d) => `<span class="chip">${esc(d)}</span>`).join("") || "—";
  return `<tr>${cell(e.name)}${cell(e.current_title)}${cell(e.company)}${cell(e.seniority)}
    <td>${domains}</td>${cell(e.years_experience)}${cell(e.location)}</tr>`;
}

$("#searchBtn").addEventListener("click", searchExperts);
$$("#s-domain, #s-q").forEach((i) => i.addEventListener("keydown", (e) => { if (e.key === "Enter") searchExperts(); }));

$("#ingestBtn").addEventListener("click", async () => {
  const raw = $("#ingestBio").value.trim();
  if (!raw) return toast("Paste a bio first", true);
  const out = $("#ingestResult");
  $("#ingestBtn").disabled = true;
  try {
    const res = await api("/experts/ingest", { method: "POST", body: JSON.stringify({ bios: [{ raw_bio: raw, source: "manual" }] }) });
    out.textContent = JSON.stringify(res, null, 2);
    out.classList.remove("hidden");
    toast("Extracted & stored");
    $("#expertsResults").dataset.loaded = "";
  } catch (err) {
    toast(err.message, true);
  } finally {
    $("#ingestBtn").disabled = false;
  }
});

// ───────────────────────── activity ─────────────────────────
async function loadRuns() {
  const p = new URLSearchParams({ limit: "60" });
  if ($("#r-stage").value) p.set("stage", $("#r-stage").value);
  const box = $("#runsResults");
  box.innerHTML = `<div class="empty"><span class="spinner"></span></div>`;
  try {
    const rows = await api("/runs?" + p.toString());
    if (!rows.length) { box.innerHTML = `<div class="empty">No runs yet — start a sourcing run.</div>`; return; }
    box.innerHTML = `<table><thead><tr>
        <th>Time</th><th>Stage</th><th>Model</th><th>Prompt</th><th>Latency</th><th>Tokens</th><th>Status</th>
      </tr></thead><tbody>${rows.map(runRow).join("")}</tbody></table>`;
  } catch (err) {
    box.innerHTML = "";
    toast(err.message, true);
  }
}

function runRow(r) {
  const t = new Date(r.created_at).toLocaleTimeString();
  const tokens = [r.input_tokens, r.output_tokens].some((x) => x != null) ? `${r.input_tokens ?? "?"}/${r.output_tokens ?? "?"}` : "—";
  const status = `<span class="status-${r.status}">${r.status}</span>`;
  return `<tr><td>${esc(t)}</td><td>${esc(r.stage)}</td>${cell(r.model)}${cell(r.prompt_version)}
    <td>${r.latency_ms ?? "—"} ms</td><td>${tokens}</td><td>${status}</td></tr>`;
}

$("#refreshRuns").addEventListener("click", loadRuns);
$("#r-stage").addEventListener("change", loadRuns);

// ───────────────────────── init ─────────────────────────
(function init() {
  const saved = localStorage.getItem(KEY_STORE);
  $("#apiKey").value = saved !== null ? saved : "dev-secret-key";
  $("#apiKey").addEventListener("change", () => localStorage.setItem(KEY_STORE, getKey()));
  checkHealth();
})();
