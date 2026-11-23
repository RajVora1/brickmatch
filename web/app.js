/* BrickMatch website.
 *
 * Runs in the browser: browsing builds, editing counts and ranking builds (the same rules as brickmatch/match.py).
 * Asks the API (server/app.py) for the things that need Python: finding pieces in a photo, step pictures,
 * the booklet PDF and the 3D view. The API address comes from config.js.
 */
(() => {
  "use strict";

  const API = (window.BRICKMATCH_API || "").replace(/\/+$/, "");
  const MAX_SIDE = 2048; // photos are shrunk before upload; the server does the same
  const CATEGORIES = [["all", "All"], ["house", "Houses"], ["tower", "Towers"], ["wall", "Walls"], ["bridge", "Bridges"]];
  const BOX_COLORS = ["#ff4d4d", "#ffcd03", "#3fa9ff", "#2ee07a", "#ff8a00", "#c77dff", "#ff5fb0", "#00e0d0"];

  const $ = (s) => document.querySelector(s);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid);
    return n;
  };

  const state = {
    catalog: null,
    names: {}, parts: {}, colors: {},
    api: "unknown",       // unknown | waking | ready | nomodels | down
    found: null,          // what the model counted ({piece: n}), null if the user typed counts in
    have: null,           // what builds are matched against; null = nothing entered yet
    detections: null,     // [{box, piece, score, color}] in photo pixels
    photo: null,          // canvas with the (shrunk) photo
    sample: null,         // sample photo info when one was used (has the real counts)
    filter: "all",
    buildId: null,
    buildData: null,
    buildReq: 0,
  };

  /* ---------------- helpers ---------------- */
  function toast(msg, ms = 3500) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => t.classList.remove("show"), ms);
  }

  function pieceName(id) { return state.names[id] || id; }
  function shortName(id) { return pieceName(id).replace("Slope 45", "Slope").replace(/ x /g, "x"); }
  function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }
  function total(counts) { return Object.values(counts || {}).reduce((a, b) => a + b, 0); }

  function pieceImg(id, cls = "piece-img") {
    const p = state.catalog.pieces.find((x) => x.id === id);
    if (p && p.icon) return el("img", { class: cls, src: `generated/pieces/${id}.png`, alt: "", loading: "lazy" });
    // Pieces without a picture yet: a little flat brick in the piece's color.
    const box = el("span", { class: "piece-fallback" });
    box.innerHTML = '<svg viewBox="0 0 58 44" width="58" height="44"><rect x="8" y="14" width="42" height="22" rx="3" fill="#cfd3d8" stroke="#1b1d24" stroke-width="2"/><rect x="14" y="9" width="10" height="7" rx="2" fill="#cfd3d8" stroke="#1b1d24" stroke-width="2"/><rect x="34" y="9" width="10" height="7" rx="2" fill="#cfd3d8" stroke="#1b1d24" stroke-width="2"/></svg>';
    return box;
  }

  async function api(path, opts = {}, timeoutMs = 90000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(API + path, { ...opts, signal: ctrl.signal });
      if (!res.ok) {
        let msg = `The server said ${res.status}.`;
        try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : msg; } catch (_) { /* not JSON */ }
        const err = new Error(msg);
        err.status = res.status;
        throw err;
      }
      return res;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("The server took too long to answer. It may still be waking up: try again in a minute.");
      if (e instanceof TypeError) throw new Error("Couldn't reach the BrickMatch server. It may be waking up (free servers nap when idle): try again in a minute.");
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  /* ---------------- server status ---------------- */
  function setPill(kind, text) {
    const pill = $("#server-pill");
    pill.className = `pill pill-${kind}`;
    pill.querySelector("span").textContent = text;
  }

  async function checkServer(attempt = 0) {
    try {
      const res = await api("/api/health", {}, 15000);
      const h = await res.json();
      if (h.models === "ready") { state.api = "ready"; setPill("ok", "Brick sorter ready"); return; }
      if (h.models === "loading") {
        state.api = "waking"; setPill("wait", "Loading the models…");
      } else {
        state.api = "nomodels"; setPill("bad", "Photo finder offline");
        console.warn("BrickMatch API:", h.message);
        return;
      }
    } catch (_) {
      state.api = attempt < 30 ? "waking" : "down";
      setPill(state.api === "down" ? "bad" : "wait", state.api === "down" ? "Server unreachable" : "Waking up the brick sorter…");
      if (state.api === "down") return;
    }
    setTimeout(() => checkServer(attempt + 1), attempt < 6 ? 4000 : 8000);
  }

  /* ---------------- matching (same rules as brickmatch/match.py) ---------------- */
  function scoreBuild(build, inv) {
    let need = 0, have = 0;
    const missing = {};
    for (const p of build.parts) {
      const owned = inv[p.piece] || 0;
      need += p.count;
      have += Math.min(owned, p.count);
      if (owned < p.count) missing[p.piece] = p.count - owned;
    }
    return { coverage: round3(have / need), have, need, missing };
  }

  // Python's round(x, 3): halves go to the even neighbour (0.5625 -> 0.562), so ties rank the same way.
  function round3(x) {
    const k = x * 1000;
    let r = Math.round(k);
    if (Math.abs(k - Math.floor(k) - 0.5) < 1e-9 && r % 2) r -= 1;
    return r / 1000;
  }

  function rankBuilds(inv) {
    return state.catalog.builds
      .map((b) => ({ ...b, ...scoreBuild(b, inv) }))
      .sort((a, b) => (b.coverage - a.coverage) || (total(a.missing) - total(b.missing)));
  }
  window.BrickMatchRank = (inv) => rankBuilds(inv).map((b) => ({ id: b.id, coverage: b.coverage, missing: b.missing }));

  /* ---------------- step 1: photo ---------------- */
  async function loadSamples() {
    try {
      const samples = await (await fetch("samples/samples.json")).json();
      const box = $("#samples");
      for (const s of samples) {
        box.append(el("button", { class: "sample", type: "button", title: `${s.label}: ${total(s.answer)} pieces`, onclick: () => useSample(s) },
          el("img", { src: s.thumb, alt: `${s.label}, a photo of loose pieces`, loading: "lazy" }),
          el("span", { text: s.label })));
      }
    } catch (_) { $("#samples").closest(".side-card").querySelector("h3").textContent = "Try it"; }
  }

  async function useSample(s) {
    try {
      const blob = await (await fetch(s.photo)).blob();
      await handlePhoto(blob, s);
    } catch (e) { showStatus(`Couldn't load that sample: ${e.message}`, true); }
  }

  function showStatus(msg, error = false, busy = false) {
    const st = $("#detect-status");
    st.hidden = !msg;
    st.className = `status${error ? " error" : ""}`;
    st.replaceChildren();
    if (busy) st.append(el("span", { class: "loader" }, el("i"), el("i"), el("i"), el("i")));
    st.append(el("span", { text: msg }));
  }

  async function toCanvas(file) {
    let bmp;
    try { bmp = await createImageBitmap(file, { imageOrientation: "from-image" }); } catch (_) {
      bmp = await new Promise((ok, bad) => {
        const img = new Image();
        img.onload = () => ok(img);
        img.onerror = () => bad(new Error("That file isn't a photo the browser can open."));
        img.src = URL.createObjectURL(file);
      });
    }
    const w = bmp.width, h = bmp.height, k = Math.min(1, MAX_SIDE / Math.max(w, h));
    const c = document.createElement("canvas");
    c.width = Math.round(w * k);
    c.height = Math.round(h * k);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    return c;
  }

  async function handlePhoto(file, sample = null) {
    if (!file || !(file.type || "").startsWith("image/")) { showStatus("Please choose a photo (JPG or PNG).", true); return; }
    let canvas;
    try { canvas = await toCanvas(file); } catch (e) { showStatus(e.message, true); return; }
    showStatus(state.api === "ready" ? "Finding and naming your pieces…" : "Waking up the brick sorter and finding your pieces. The first photo can take up to a minute…", false, true);
    $("#drop").scrollIntoView({ behavior: "smooth", block: "start" });
    const blob = await new Promise((ok) => canvas.toBlob(ok, "image/jpeg", 0.9));
    const form = new FormData();
    form.append("photo", blob, "photo.jpg");
    form.append("conf", $("#conf").value);
    let result;
    try {
      result = await (await api("/api/detect", { method: "POST", body: form }, 150000)).json();
    } catch (e) {
      showStatus(e.status === 503 ? `The photo finder isn't running right now (${e.message}) You can still type in your pieces.` : e.message, true);
      return;
    }
    if (state.api !== "ready") checkServer();
    showStatus("");
    const k = canvas.width / result.width;
    state.detections = result.pieces.map((p) => ({ ...p, box: p.box.map((v) => v * k) }));
    state.photo = canvas;
    state.sample = sample;
    state.found = { ...result.counts };
    state.have = { ...result.counts };
    showCheck();
    const n = result.pieces.length;
    toast(n ? `Found ${plural(n, "piece")} in ${result.seconds}s` : "No pieces found. Try a lower confidence or a plainer background.");
  }

  function typeIn() {
    state.found = null; state.detections = null; state.photo = null; state.sample = null;
    state.have = state.have || {};
    showCheck();
  }

  /* ---------------- step 2: check counts ---------------- */
  function showCheck() {
    $("#check").hidden = false;
    const hasPhoto = !!state.photo;
    $("#photo-wrap").hidden = !hasPhoto;
    $(".check-grid").classList.toggle("no-photo", !hasPhoto);
    $("#check-intro").textContent = hasPhoto
      ? "The model can get a piece wrong (a 1x4 brick seen from the side can look like a 2x4). Fix any count with − and +: to fix a wrong type, take one off the type it said and add one to the right type. Builds update as you go."
      : "Type in how many of each piece you have, or use − and +. Builds update as you go.";
    if (hasPhoto) drawPhoto();
    renderCounts();
    renderBuilds();
    requestAnimationFrame(() => $("#check").scrollIntoView({ behavior: "smooth", block: "start" }));
  }

  function boxColors() {
    const order = state.catalog.pieces.map((p) => p.id);
    const out = {};
    order.forEach((id, i) => { out[id] = BOX_COLORS[i % BOX_COLORS.length]; });
    return out;
  }

  function drawPhoto() {
    const c = $("#photo"), src = state.photo;
    c.width = src.width; c.height = src.height;
    const g = c.getContext("2d");
    g.drawImage(src, 0, 0);
    const colors = boxColors();
    const lw = Math.max(3, src.width / 320), fs = Math.max(16, Math.round(src.width / 55));
    g.font = `700 ${fs}px Nunito, system-ui, sans-serif`;
    const labels = $("#show-labels").checked;
    for (const d of state.detections) {
      const [x1, y1, x2, y2] = d.box, col = colors[d.piece] || "#fff";
      g.lineWidth = lw + 3; g.strokeStyle = "rgba(0,0,0,.55)"; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
      g.lineWidth = lw; g.strokeStyle = col; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if (!labels) continue;
      const text = `${shortName(d.piece)} ${Math.round(d.score * 100)}%`;
      const tw = g.measureText(text).width + 12, th = fs + 8, ty = Math.max(0, y1 - th);
      g.fillStyle = col; g.fillRect(x1 - lw / 2, ty, tw, th);
      g.fillStyle = "#111"; g.fillText(text, x1 + 6 - lw / 2, ty + fs);
    }
    const n = state.detections.length, unsure = state.detections.filter((d) => d.score < 0.6).length;
    $("#photo-summary").textContent = n
      ? `Found ${plural(n, "piece")} of ${plural(Object.keys(state.found).length, "type")}${unsure ? `, unsure about ${unsure}` : ""}.`
      : "No pieces found in this photo.";
  }

  function colorsSeen(piece) {
    if (!state.detections) return [];
    const c = {};
    for (const d of state.detections) if (d.piece === piece) c[d.color] = (c[d.color] || 0) + 1;
    return Object.entries(c).sort((a, b) => b[1] - a[1]);
  }

  function countRow(id) {
    const found = state.found ? state.found[id] || 0 : null;
    const have = state.have[id] || 0;
    const meta = el("div", { class: "count-meta" });
    if (found !== null) meta.append(el("span", { text: `model saw ${found}` }));
    for (const [col, n] of colorsSeen(id)) {
      const rgb = state.colors[col] || [200, 200, 200];
      meta.append(el("span", { title: col.replace(/_/g, " ") }, el("i", { class: "swatch", style: `background:rgb(${rgb.join(",")})` }), `${n}`));
    }
    const p = state.catalog.pieces.find((x) => x.id === id);
    if (p && p.part) meta.append(el("span", { text: `#${p.part}` }));
    const input = el("input", { type: "number", min: "0", max: "999", inputmode: "numeric", value: String(have), "aria-label": `How many ${pieceName(id)}` });
    const set = (n) => {
      n = Math.max(0, Math.min(999, Math.floor(Number(n) || 0)));
      if (n) state.have[id] = n; else delete state.have[id];
      renderCounts(); renderBuilds(); refreshBuildIfOpen();
    };
    input.addEventListener("change", () => set(input.value));
    input.addEventListener("focus", () => input.select());
    const changed = found !== null && found !== have;
    return el("li", { class: `count-row${changed ? " changed" : ""}${have ? "" : " zero"}` },
      pieceImg(id),
      el("div", {}, el("div", { class: "count-name", text: pieceName(id) }), meta),
      el("div", { class: "stepper" },
        el("button", { type: "button", "aria-label": `One less ${pieceName(id)}`, disabled: have === 0, onclick: () => set(have - 1) }, "−"),
        input,
        el("button", { type: "button", "aria-label": `One more ${pieceName(id)}`, onclick: () => set(have + 1) }, "+")));
  }

  function renderCounts() {
    const pieces = state.catalog.pieces;
    const found = state.found || {};
    const foundOrder = Object.keys(found).sort((a, b) => found[b] - found[a]);
    const tier1 = pieces.filter((p) => p.tier === 1 && !(p.id in found)).map((p) => p.id);
    const tier2 = pieces.filter((p) => p.tier !== 1 && !(p.id in found)).map((p) => p.id);
    const extra = Object.keys(state.have).filter((id) => !pieces.some((p) => p.id === id) && !(id in found));
    $("#counts").replaceChildren(...[...foundOrder, ...tier1, ...extra].map(countRow));
    $("#counts-more").replaceChildren(...tier2.map(countRow));
    $("#total-pill").textContent = total(state.have);
    const changed = state.found && JSON.stringify(sortObj(state.found)) !== JSON.stringify(sortObj(state.have));
    const reset = $("#reset-counts");
    reset.textContent = state.found ? "Undo my changes" : "Clear all";
    reset.disabled = state.found ? !changed : total(state.have) === 0;
    renderAnswer();
  }

  function sortObj(o) { return Object.fromEntries(Object.entries(o).filter(([, v]) => v > 0).sort()); }

  function renderAnswer() {
    const box = $("#answer");
    if (!state.sample || !state.found) { box.hidden = true; return; }
    const ans = state.sample.answer, ids = new Set([...Object.keys(ans), ...Object.keys(state.found)]);
    let right = 0;
    for (const id of ids) right += Math.min(ans[id] || 0, state.found[id] || 0);
    const off = [...ids].filter((id) => (ans[id] || 0) !== (state.found[id] || 0))
      .map((id) => `${shortName(id)}: really ${ans[id] || 0}, saw ${state.found[id] || 0}`);
    box.hidden = false;
    box.replaceChildren(
      el("div", { text: `Answer key for ${state.sample.label}: ${total(ans)} pieces. The model got ${right} of them right.` }),
      off.length ? el("div", { class: "muted small", text: off.join(" · ") }) : el("div", { class: "small", text: "Every count matches." }));
  }

  /* ---------------- step 3: builds ---------------- */
  function renderTabs() {
    const ranked = state.catalog.builds;
    $("#tabs").replaceChildren(...CATEGORIES.map(([key, label]) => {
      const n = key === "all" ? ranked.length : ranked.filter((b) => b.category === key).length;
      return el("button", { class: "tab", role: "tab", type: "button", "aria-selected": String(state.filter === key),
        onclick: () => { state.filter = key; renderTabs(); renderBuilds(); } }, label, el("span", { text: n }));
    }));
  }

  function meter(coverage) {
    const on = Math.round(coverage * 10);
    const cls = coverage >= 0.999 ? "" : coverage >= 0.6 ? " mid" : " low";
    return el("div", { class: `meter${cls}`, "aria-hidden": "true" }, Array.from({ length: 10 }, (_, i) => el("i", { class: i < on ? "on" : "" })));
  }

  function renderBuilds() {
    const counting = state.have && total(state.have) > 0;
    const list = (counting ? rankBuilds(state.have) : state.catalog.builds)
      .filter((b) => state.filter === "all" || b.category === state.filter);
    const ready = counting ? rankBuilds(state.have).filter((b) => b.coverage >= 0.999).length : 0;
    $("#builds-title").textContent = counting ? (ready ? `You can build ${plural(ready, "thing")} right now` : "Your closest builds") : "Pick a build";
    $("#builds-intro").textContent = counting
      ? "Ranked by how many of each build's pieces you already have. Red tags are the pieces you're missing."
      : `All ${state.catalog.builds.length} builds use only the 8 basic piece types. Find your pieces first and they'll be ranked by how many you already have.`;
    const grid = $("#build-grid");
    if (!list.length) { grid.replaceChildren(el("div", { class: "empty", text: "No builds in this group yet." })); return; }
    grid.replaceChildren(...list.map((b) => {
      const body = el("div", { class: "card-body" }, el("h3", { text: b.name }), el("p", { text: b.description }));
      if (counting) {
        body.append(meter(b.coverage), el("div", { class: "have", text: `You have ${b.have} of ${b.need} pieces` }));
        const miss = Object.entries(b.missing);
        if (miss.length) body.append(el("div", { class: "missing-chips" }, miss.map(([id, n]) => el("span", { class: "chip", text: `${n}× ${shortName(id)}` }))));
      } else {
        body.append(el("div", { class: "have muted", text: `${b.total} pieces · ${plural(b.steps, "step")}` }));
      }
      body.append(el("div", { class: "card-foot", text: "Build it →" }));
      return el("button", { class: `build-card${b.id === state.buildId ? " active" : ""}`, type: "button", onclick: () => openBuild(b.id) },
        el("div", { class: "thumb" },
          el("span", { class: `cat cat-${b.category}`, text: b.category }),
          counting && b.coverage >= 0.999 ? el("span", { class: "stamp", text: "Ready!" }) : null,
          b.steps ? el("img", { src: `generated/builds/${b.id}.png`, alt: `Picture of ${b.name}`, loading: "lazy" }) : null),
        body);
    }));
  }

  /* ---------------- step 4: one build ---------------- */
  function checkedCounts() { return state.have && total(state.have) > 0 ? state.have : null; }

  function renderParts(build) {
    const inv = checkedCounts();
    const items = build.parts.map((p) => {
      const owned = inv ? inv[p.piece] || 0 : null;
      const short = inv && owned < p.count;
      const part = state.parts[p.piece];
      return el("li", {}, pieceImg(p.piece),
        el("div", {}, el("div", { class: "pname", text: pieceName(p.piece) }), part ? el("div", { class: "partno", text: `Part #${part}` }) : null),
        el("div", { class: `qty${short ? " short" : ""}`, text: inv ? `${Math.min(owned, p.count)}/${p.count}` : `×${p.count}` }));
    });
    $("#parts").replaceChildren(...items);
    const missing = inv ? scoreBuild(build, inv).missing : {};
    $("#shop-note").textContent = !inv
      ? "Find your pieces to see which ones you're missing."
      : Object.keys(missing).length
        ? "Missing pieces can be bought one by one with the part numbers above (for example from LEGO Pick a Brick or BrickLink)."
        : "You have every piece for this build.";
  }

  function setBuildStatus(build) {
    const inv = checkedCounts(), st = $("#build-status");
    if (!inv) { st.className = "build-status"; st.textContent = `${build.total} pieces in ${plural(build.steps, "step")}. Find or type in your pieces to have them checked off.`; return; }
    const s = scoreBuild(build, inv), miss = total(s.missing);
    const data = state.buildData && state.buildData.id === build.id ? state.buildData : null;
    if (!miss) { st.className = "build-status ok"; st.textContent = "You have every piece. Let's build!"; return; }
    let text = `You're missing ${plural(miss, "piece")}.`;
    if (data && data.stuck) text += data.stuck === 1 ? " They're needed from step 1." : ` You can build up to step ${data.stuck - 1}.`;
    st.className = `build-status ${s.coverage >= 0.6 ? "part" : "none"}`;
    st.textContent = text + " Missing pieces are red in the 3D view and the steps.";
  }

  async function openBuild(id) {
    const build = state.catalog.builds.find((b) => b.id === id);
    if (!build) return;
    state.buildId = id;
    $("#build").hidden = false;
    $("#build-name").textContent = build.name;
    $("#build-desc").textContent = build.description;
    renderParts(build);
    setBuildStatus(build);
    renderBuilds();
    $("#build").scrollIntoView({ behavior: "smooth", block: "start" });
    await loadBuild(build);
  }

  function refreshBuildIfOpen() {
    if (!state.buildId || $("#build").hidden) return;
    const build = state.catalog.builds.find((b) => b.id === state.buildId);
    renderParts(build);
    setBuildStatus(build);
    clearTimeout(refreshBuildIfOpen.t);
    refreshBuildIfOpen.t = setTimeout(() => loadBuild(build), 700); // wait for the clicking to stop
  }

  async function loadBuild(build) {
    const req = ++state.buildReq;
    if (!build.steps) {
      $("#viewer").replaceChildren(el("div", { class: "viewer-empty", text: "This build doesn't have step-by-step instructions yet." }));
      $("#steps").replaceChildren();
      return;
    }
    if (!state.buildData || state.buildData.id !== build.id) {
      $("#viewer").replaceChildren(el("div", { class: "viewer-empty" }, el("span", { class: "loader" }, el("i"), el("i"), el("i"), el("i")), el("p", { text: "Building the 3D view…" })));
      $("#steps").replaceChildren(el("li", { class: "skeleton" }), el("li", { class: "skeleton" }));
    }
    let data;
    try {
      data = await (await api("/api/build", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: build.id, counts: checkedCounts() }) }, 120000)).json();
    } catch (e) {
      if (req !== state.buildReq) return;
      $("#viewer").replaceChildren(el("div", { class: "viewer-empty" }, el("p", { text: e.message }),
        el("button", { class: "brick-btn blue small", type: "button", onclick: () => loadBuild(build) }, "Try again")));
      $("#steps").replaceChildren();
      return;
    }
    if (req !== state.buildReq) return; // a newer request is on its way
    state.buildData = data;
    setBuildStatus(build);
    const frame = el("iframe", { title: `3D view of ${build.name}`, sandbox: "allow-scripts", loading: "lazy" });
    frame.srcdoc = data.viewer;
    $("#viewer").replaceChildren(frame);
    $("#steps").replaceChildren(...data.steps.map((s) => el("li", { class: `step${s.missing ? " short-step" : ""}` },
      el("img", { src: s.image, alt: `Picture of ${s.title}`, onclick: () => lightbox(s.image) }),
      el("div", { class: "step-text" },
        el("div", { class: "big", text: s.number }),
        el("h4", { text: s.title.replace(/^Step \d+: /, "") }),
        el("ul", {}, s.lines.map((l) => el("li", { text: l }))),
        s.missing ? el("span", { class: "need", text: `${plural(s.missing, "piece")} missing in this step` }) : null))));
  }

  function lightbox(src) {
    const box = el("div", { class: "lightbox", onclick: () => box.remove() }, el("img", { src, alt: "" }));
    document.addEventListener("keydown", function esc(e) { if (e.key === "Escape") { box.remove(); document.removeEventListener("keydown", esc); } });
    document.body.append(box);
  }

  function download(blob, name) {
    const a = el("a", { href: URL.createObjectURL(blob), download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }

  async function downloadBooklet() {
    const btn = $("#dl-booklet");
    if (!state.buildId) return;
    btn.disabled = true;
    const label = btn.textContent;
    btn.textContent = "Making your booklet…";
    try {
      const res = await api("/api/booklet", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: state.buildId, counts: checkedCounts() }) }, 120000);
      download(await res.blob(), `${state.buildId}_booklet.pdf`);
    } catch (e) { toast(e.message, 6000); } finally { btn.disabled = false; btn.textContent = label; }
  }

  function download3d() {
    if (!state.buildData) { toast("The 3D view hasn't loaded yet."); return; }
    download(new Blob([state.buildData.viewer], { type: "text/html" }), `${state.buildData.id}_3d.html`);
  }

  /* ---------------- start ---------------- */
  async function init() {
    checkServer();
    try {
      state.catalog = await (await fetch("generated/catalog.json")).json();
    } catch (_) {
      $("#build-grid").replaceChildren(el("div", { class: "empty", text: "Couldn't load the builds list. Run python -m server.site_assets to make it." }));
      return;
    }
    for (const p of state.catalog.pieces) { state.names[p.id] = p.name; state.parts[p.id] = p.part; }
    for (const c of state.catalog.colors) state.colors[c.id] = c.rgb;
    $("#fact-builds").textContent = state.catalog.builds.length;
    renderTabs();
    renderBuilds();
    loadSamples();

    const file = $("#file"), drop = $("#drop");
    file.addEventListener("change", () => { if (file.files[0]) handlePhoto(file.files[0]); file.value = ""; });
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) handlePhoto(f); });
    $("#type-btn").addEventListener("click", typeIn);
    $("#conf").addEventListener("input", (e) => { $("#conf-out").textContent = Number(e.target.value).toFixed(2); });
    $("#show-labels").addEventListener("change", drawPhoto);
    $("#reset-counts").addEventListener("click", () => {
      state.have = state.found ? { ...state.found } : {};
      renderCounts(); renderBuilds(); refreshBuildIfOpen();
    });
    $("#dl-booklet").addEventListener("click", downloadBooklet);
    $("#dl-3d").addEventListener("click", download3d);
  }

  init();
})();
