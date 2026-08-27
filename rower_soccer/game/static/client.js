import { SceneView, streamPoses } from "/static/render3d.js";
/* Browser client. Sends the high level only: (skill, target, aim).
 *
 * The token lives in localStorage, so a phone that sleeps or drops wifi rejoins
 * its own seat instead of a new one -- and the token is the ONLY thing that names
 * a slot: the client never sends a slot id with an input, so it is structurally
 * unable to drive someone else's creature.
 *
 * The click->world affine lives on the server (MatchSim.uv_to_world); the client
 * sends normalized (u, v) in [0,1] read off the <img>'s own bounding box, so
 * resizing the window, rotating a phone, or changing the render resolution cannot
 * desync input from picture.
 */
const KEYS = ["follow", "dribble", "kick", "shoot", "scripted"];
const S = {
  token: localStorage.getItem("tok") || null,
  name: localStorage.getItem("name") || "",
  // Remembered so a phone that sleeps mid-match does not ask for a passphrase to
  // come back. Sent in the POST body only -- never in a URL, which would put it in
  // browser history and in the tunnel's request log.
  code: localStorage.getItem("code") || "",
  slot: null, skill: "idle", available: [], state: null,
  drag: null, lastEvent: 0, streamAt: 0, streamTries: 0, movedAt: 0, movedFrame: 0,
};

const $ = (id) => document.getElementById(id);
const view = $("view"), overlay = $("overlay"), ctx = overlay.getContext("2d");

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(Object.assign({ token: S.token }, body || {})),
  });
  return r.json().catch(() => ({}));
}

// --- lobby -----------------------------------------------------------------
// One join at a time. poll() re-joins on a 403 five times a second, and without
// this a server that says no would stack up window.prompt() dialogs forever.
let _joining = null;
function join() {
  if (!_joining) _joining = _join().finally(() => { _joining = null; });
  return _joining;
}

async function _join() {
  S.name = ($("name").value || S.name || "player").slice(0, 24);
  localStorage.setItem("name", S.name);
  let r = await post("/join", { name: S.name, token: S.token, code: S.code });
  // Only a gated server that did not accept the remembered code gets this far, so
  // ask once and try again with a clean slate. (A merely *stale* token -- server
  // restarted, or --client-ttl reaped us -- never lands here: the remembered code
  // is still good, so the server just issues a new token.)
  if (r.need_code) {
    localStorage.removeItem("code");
    S.code = window.prompt("join code") || "";
    if (!S.code) { log("! a join code is required"); return false; }
    localStorage.setItem("code", S.code);
    r = await post("/join", { name: S.name, token: null, code: S.code });
  }
  if (!r.token) { log("! " + (r.error || "join failed")); return false; }
  S.token = r.token; S.slot = r.slot;
  localStorage.setItem("tok", S.token);
  openStream();
  return true;
}

async function claim(slot) {
  if (!S.token) await join();
  const r = await post("/claim", { slot });
  if (!r.ok) log("! " + r.error); else S.slot = r.slot;
}

async function release() { await post("/release", {}); S.slot = null; }

// --- the video stream ------------------------------------------------------
// On a LAN the MJPEG stream is opened once in the markup and never thought about
// again. Over the internet it is a long-lived HTTP response crossing a CDN, a
// home router and a phone that suspends radios in the background -- it WILL be
// cut, sometimes with an error event and sometimes by simply going quiet. So the
// stream is opened from script (it also needs the token on a gated server) and
// re-opened whenever it dies or silently freezes.
function openStream() {
  if (!S.token) return;
  const now = Date.now();
  if (now - S.streamAt < 1500) return;          // never hammer it
  S.streamAt = now;
  S.pix = null; S.movedAt = 0; S.movedFrame = 0;
  // Cache-buster: without it a re-set src can be served from the image cache and
  // the picture stays dead. The token is a per-session capability, not the code.
  view.src = "/stream?token=" + encodeURIComponent(S.token) + "&r=" + now;
}

view.addEventListener("error", () => {
  S.streamTries++;
  const wait = Math.min(1000 * S.streamTries, 8000);
  log("! video dropped; retrying");
  setTimeout(openStream, wait);
});
view.addEventListener("load", () => { S.streamTries = 0; });
// A backgrounded tab gets its stream throttled or torn down with no error; the
// picture is then permanently stale when the player comes back to it.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { S.streamAt = 0; openStream(); }
});
window.addEventListener("online", () => { S.streamAt = 0; openStream(); });

// Freeze detector. `error` does not fire for a stream that is merely silent (a
// CDN idling out a long response, a phone that suspended its radio), and an <img>
// gives no portable per-frame event -- so sample a few pixels instead.
//
// The trick is not to mistake a *still* picture for a *dead* one. The server
// legitimately stops sending when it drops frames under host load: measured, a
// busy box freezes the picture for ~5 s at a stretch while the match itself runs
// on perfectly. So the comparison is against the server's own frame counter from
// /state: only "the server rendered 40+ new frames and our pixels never moved"
// means the bytes are being lost between there and here.
const _probe = document.createElement("canvas");
_probe.width = 24; _probe.height = 16;
const _pctx = _probe.getContext("2d", { willReadFrequently: true });
function checkFrozen(st) {
  if (document.hidden || !view.naturalWidth) return;
  let sig;
  try {
    _pctx.drawImage(view, 0, 0, _probe.width, _probe.height);
    const d = _pctx.getImageData(0, 0, _probe.width, _probe.height).data;
    sig = 0;
    for (let i = 0; i < d.length; i += 17) sig = (sig * 31 + d[i]) | 0;
  } catch (e) { return; }               // tainted or not yet decodable
  if (sig !== S.pix) { S.pix = sig; S.movedAt = Date.now(); S.movedFrame = st.frame; }
  else if (S.movedAt && Date.now() - S.movedAt > 6000 &&
           st.frame - S.movedFrame > 40) {
    log("! video froze; reconnecting");
    openStream();
  }
}

// --- input -----------------------------------------------------------------
async function setSkill(s) {
  if (!S.slot) { log("! claim a seat first"); return; }
  if (s !== "idle" && !S.available.includes(s)) { log(`! ${s} not trained yet`); return; }
  S.skill = s; paintSkills();
  const r = await post("/input", { skill: s });
  if (!r.ok) log("! " + r.error);
}

function uv(ev) {
  const box = R3 ? R3.canvas : view;
  const r = box.getBoundingClientRect();
  const p = ev.touches ? ev.touches[0] : ev;
  return { u: (p.clientX - r.left) / r.width, v: (p.clientY - r.top) / r.height };
}

/** Client-side pick: uv -> world xy through OUR camera. Null when 2-D. */
function pickWorld(p) {
  if (!R3 || !R3.desc) return null;
  return R3.pickGround(p.u * 2 - 1, 1 - p.v * 2);
}

function dragStart(ev) { ev.preventDefault(); S.drag = { a: uv(ev), b: uv(ev) }; }
function dragMove(ev) { if (S.drag) { ev.preventDefault(); S.drag.b = uv(ev); } }
async function dragEnd(ev) {
  if (!S.drag) return;
  ev.preventDefault();
  const { a, b } = S.drag; S.drag = null;
  if (!S.slot) { log("! claim a seat first"); return; }
  // A drag sets the target at the RELEASE point and the aim along the drag; a
  // plain tap is the same thing with zero aim. Aim is what kick/shoot will steer
  // by once WS1 trains them; follow ignores it.
  // With client-side rendering the browser OWNS the camera, so it resolves the
  // click itself and sends world xy. The server's uv->world path exists only
  // for the MJPEG client, where the server owned the camera -- and keeping two
  // projections in sync is exactly what went wrong with the server-side chase
  // camera. One owner, one projection.
  const wb = pickWorld(b), wa = pickWorld(a);
  let body;
  if (wb) {
    body = { x: wb.x, y: wb.y };
    if (wa) {
      const dx = wb.x - wa.x, dy = wb.y - wa.y, n = Math.hypot(dx, dy);
      if (n > 1e-6) { body.aim_x = dx / n; body.aim_y = dy / n; }
    }
  } else {
    body = { u: b.u, v: b.v, aim_u: b.u - a.u, aim_v: b.v - a.v };
  }
  const r = await post("/input", body);
  if (!r.ok) log("! " + r.error);
}

// --- painting --------------------------------------------------------------
function paintSkills() {
  const box = $("skills");
  if (box.childElementCount !== KEYS.length + 1) {
    box.innerHTML = "";
    KEYS.concat(["idle"]).forEach((s, i) => {
      const b = document.createElement("button");
      b.id = "sk_" + s;
      b.innerHTML = `${s}<span class="k">${i < KEYS.length ? i + 1 : "esc"}</span>`;
      b.onclick = () => setSkill(s);
      box.appendChild(b);
    });
  }
  KEYS.concat(["idle"]).forEach((s) => {
    const b = $("sk_" + s);
    b.classList.toggle("on", S.skill === s);
    b.disabled = s !== "idle" && !S.available.includes(s);
  });
}

function paintSeats(st) {
  const seats = $("seats");
  seats.innerHTML = "";
  (st.lobby ? st.lobby.seats : []).forEach((s) => {
    const d = document.createElement("div");
    const team = s.slot.startsWith("home") ? "home" : "away";
    d.className = `seat ${team}` + (s.taken ? "" : " free") + (S.slot === s.slot ? " mine" : "");
    const p = (st.players || []).find((x) => x.slot === s.slot) || {};
    d.innerHTML = `<div class="n">${s.slot.replace("_", " ")} &middot; ${p.controller || "?"}</div>` +
                  `<div class="w">${s.taken ? s.name : "free &mdash; tap to claim"}</div>` +
                  `<div class="n">${p.skill || ""}</div>`;
    d.onclick = () => (S.slot === s.slot ? release() : claim(s.slot));
    seats.appendChild(d);
  });
}

function paintOverlay(st) {
  const w = view.clientWidth, h = view.clientHeight;
  if (!w || !h) return;
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
  ctx.clearRect(0, 0, w, h);
  // Ring my own creature and draw a line to its target, so a 4-player screen is
  // readable at a glance -- everything else is already in the render.
  const me = (st.players || []).find((p) => p.slot === S.slot);
  if (me) {
    const x = me.u * w, y = me.v * h;
    // The aim line follows `me.path`, a ground polyline the server already
    // projected through the active camera. Drawing a single straight segment
    // to (tu, tv) instead was only right for the topdown affine -- under the
    // chase cameras it lifted off the pitch and cut through the scene.
    ctx.strokeStyle = me.color || "#4cc38a"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 16, 0, 6.284); ctx.stroke();
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    const pts = me.path || [[me.u, me.v], [me.tu, me.tv]];
    pts.forEach(([pu, pv], i) => (i ? ctx.lineTo(pu * w, pv * h)
                                   : ctx.moveTo(pu * w, pv * h)));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (S.drag) {
    const a = S.drag.a, b = S.drag.b;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a.u * w, a.v * h); ctx.lineTo(b.u * w, b.v * h); ctx.stroke();
    ctx.beginPath(); ctx.arc(b.u * w, b.v * h, 8, 0, 6.284); ctx.stroke();
  }
}

function banner(text) {
  const b = $("banner");
  b.textContent = text; b.classList.add("show");
  clearTimeout(banner._t);
  banner._t = setTimeout(() => b.classList.remove("show"), 1600);
}

function log(line) {
  const d = document.createElement("div");
  d.textContent = line;
  $("log").prepend(d);
  while ($("log").childElementCount > 40) $("log").lastChild.remove();
}

// --- poll ------------------------------------------------------------------
async function poll() {
  try {
    const r = await fetch("/state?token=" + encodeURIComponent(S.token || ""));
    if (r.status === 403) {
      // Gated server, and our token is not (or no longer) one it knows: it was
      // restarted, or --client-ttl reaped us while the laptop was shut. Re-join
      // with the remembered code; only a wrong code reaches the prompt.
      if (!S.code) { $("phase").textContent = "join code required"; return; }
      $("phase").textContent = "reconnecting";
      await join();
      return;
    }
    const st = await r.json();
    S.state = st;
    S.available = st.available_skills || [];
    if (st.me) { S.slot = st.me.slot; }
    $("score").textContent = `${(st.score || [0, 0])[0]} – ${(st.score || [0, 0])[1]}`;
    $("clock").textContent = st.phase === "countdown"
      ? String(Math.ceil(st.countdown || 0))
      : (st.time_left != null ? st.time_left.toFixed(1) : "--");
    $("phase").textContent = st.phase + (st.stats ? `  ${st.stats.realtime}x rt` : "");
    const whoEl = $("who");
    whoEl.textContent = S.slot ? `you: ${S.slot}` : "spectator";
    whoEl.classList.toggle("seated", !!S.slot);
    const me = (st.players || []).find((p) => p.slot === S.slot);
    if (me && me.skill !== S.skill) { S.skill = me.skill; }
    $("cambtn").textContent = "cam: " + (st.camera === "broadcast" ? "tv" : "top");
    paintSkills(); paintSeats(st); paintOverlay(st); checkFrozen(st);
    (st.events || []).forEach((e) => {
      const key = e.tick + e.type + (e.slot || "") + (e.team || "");
      if (poll._seen.has(key)) return;
      poll._seen.add(key);
      if (poll._seen.size > 200) poll._seen = new Set();
      if (e.type === "goal") banner("GOAL " + e.team.toUpperCase());
      if (e.type === "match_start") banner("GO");
      if (e.type === "match_end") banner("FULL TIME");
      if (["goal", "match_start", "match_end", "slot_claim", "ball_touch"].includes(e.type))
        log(`${e.t.toFixed(1)}s ${e.type} ${e.slot || e.team || ""}`);
    });
  } catch (err) { $("phase").textContent = "disconnected"; }
}
poll._seen = new Set();

// --- wire up ---------------------------------------------------------------
view.addEventListener("mousedown", dragStart);
window.addEventListener("mousemove", dragMove);
window.addEventListener("mouseup", dragEnd);
view.addEventListener("touchstart", dragStart, { passive: false });
view.addEventListener("touchmove", dragMove, { passive: false });
view.addEventListener("touchend", dragEnd, { passive: false });
view.addEventListener("dragstart", (e) => e.preventDefault());

async function unflip() {
  const r = await post("/input", { action: "unflip" });
  if (!r.ok) log("! " + (r.error || "unflip failed"));
}

document.addEventListener("keydown", (e) => {
  if (document.activeElement === $("name")) return;
  const i = "12345".indexOf(e.key);
  if (i >= 0) setSkill(KEYS[i]);
  else if (e.key === "Escape") setSkill("idle");
  else if (e.key === "r" || e.key === "R") unflip();
});

$("joinbtn").onclick = join;
$("unflipbtn").onclick = unflip;
$("cambtn").onclick = () => post("/control", { action: "camera" });
$("startbtn").onclick = () => post("/control", { action: "start" });
$("stopbtn").onclick = () => post("/control", { action: "stop" });
$("name").value = S.name;

// --- client-side rendering --------------------------------------------------
let R3 = null;

async function start3d() {
  const canvas = $("webgl");
  // Everything up to the first successful draw is inside the fallback: a
  // missing WebGL context, a blocked module import, a gated /scene and a
  // driver that dies inside build() all have to end with the player watching
  // the MJPEG stream rather than a blank box.
  let desc;
  try {
    R3 = new SceneView(canvas);
    const res = await fetch("/scene?token=" + encodeURIComponent(S.token || ""));
    if (!res.ok) throw new Error("scene " + res.status);
    desc = await res.json();
    if (!desc || !desc.geoms) throw new Error("no geoms");
    R3.build(desc);
    R3.resize();
    R3.draw();
  } catch (e) {
    log("! 3-D unavailable (" + e.message + "), using the video stream");
    R3 = null;
    return false;
  }
  view.hidden = true;
  canvas.hidden = false;

  let latest = null;
  const pump = () => streamPoses(
      "/poses?token=" + encodeURIComponent(S.token || ""),
      (f) => { latest = f; },
  ).catch(() => {}).finally(() => setTimeout(pump, 500));   // reconnect
  pump();

  const loop = () => {
    if (latest) {
      R3.apply(latest);
      // Follow my own seat when I have one; spectators get the wide view.
      R3.follow = S.slot ? desc.slots.indexOf(S.slot) : null;
      const i = R3.follow;
      let p = null;
      if (i !== null && i >= 0 && S.state) {
        const me = (S.state.players || [])[i];
        if (me && me.world) p = { x: me.world[0], y: me.world[1] };
      }
      R3.aim(p);
    }
    R3.resize();
    R3.draw();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  return true;
}

// Server-side rendering is the DEFAULT again. It became affordable when EGL
// turned out to work here (2.2 ms a frame on the GPU against 46 ms on the CPU
// rasteriser), and it keeps one authority for physics AND for what everyone
// sees -- each player now gets their own POV camera rendered server-side.
// `?client3d=1` opts into the browser renderer, which is still useful when the
// server has no GPU.
(async () => {
  await join();
  const want3d = new URLSearchParams(location.search).get("client3d") === "1";
  if (!want3d || !(await start3d())) {
    view.hidden = false;
    $("webgl").hidden = true;
  }
  setInterval(poll, 200);
  poll();
})();
