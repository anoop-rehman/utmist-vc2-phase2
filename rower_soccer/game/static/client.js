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
  slot: null, skill: "idle", available: [], state: null,
  drag: null, lastEvent: 0,
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
async function join() {
  S.name = ($("name").value || S.name || "player").slice(0, 24);
  localStorage.setItem("name", S.name);
  const r = await post("/join", { name: S.name, token: S.token });
  S.token = r.token; S.slot = r.slot;
  localStorage.setItem("tok", S.token);
}

async function claim(slot) {
  if (!S.token) await join();
  const r = await post("/claim", { slot });
  if (!r.ok) log("! " + r.error); else S.slot = r.slot;
}

async function release() { await post("/release", {}); S.slot = null; }

// --- input -----------------------------------------------------------------
async function setSkill(s) {
  if (!S.slot) { log("! claim a seat first"); return; }
  if (s !== "idle" && !S.available.includes(s)) { log(`! ${s} not trained yet`); return; }
  S.skill = s; paintSkills();
  const r = await post("/input", { skill: s });
  if (!r.ok) log("! " + r.error);
}

function uv(ev) {
  const r = view.getBoundingClientRect();
  const p = ev.touches ? ev.touches[0] : ev;
  return { u: (p.clientX - r.left) / r.width, v: (p.clientY - r.top) / r.height };
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
  const body = { u: b.u, v: b.v, aim_u: b.u - a.u, aim_v: b.v - a.v };
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
    const x = me.u * w, y = me.v * h, tx = me.tu * w, ty = me.tv * h;
    ctx.strokeStyle = "#4cc38a"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 16, 0, 6.284); ctx.stroke();
    ctx.setLineDash([5, 5]);
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(tx, ty); ctx.stroke();
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
    paintSkills(); paintSeats(st); paintOverlay(st);
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

document.addEventListener("keydown", (e) => {
  if (document.activeElement === $("name")) return;
  const i = "12345".indexOf(e.key);
  if (i >= 0) setSkill(KEYS[i]);
  else if (e.key === "Escape") setSkill("idle");
});

$("joinbtn").onclick = join;
$("startbtn").onclick = () => post("/control", { action: "start" });
$("stopbtn").onclick = () => post("/control", { action: "stop" });
$("name").value = S.name;

(async () => { await join(); setInterval(poll, 200); poll(); })();
