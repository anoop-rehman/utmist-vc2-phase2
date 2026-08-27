"""Four real seats, four open streams, continuous clicks -- the load that crashed us.

The 32 unit tests in `run_tests.py` all passed while the live server was
segfaulting every few minutes, because the bug was a DATA RACE: `/input` on an
HTTP handler thread reached dm_control, which lazily called `physics.forward()`
on the same MjData the sim thread was stepping. Nothing single-threaded can see
that, and neither can a test with one client -- only a click un-projected
through a *chase* camera touched the offending path, and a chase camera only
exists once someone takes a seat.

So this is the shape of the reproducer, and it is the shape the bug needs: four
seats CLAIMED (not just joined), four long-lived MJPEG readers, and clicks
going in the whole time. Against the unfixed server it died in about a second;
against the fixed one it runs indefinitely.

Run it against a scratch server, never a live match -- it takes all four seats.

    PYTHONPATH=. .venv/bin/python -m rower_soccer.game.tests.stress_multiplayer \
        --port 8095 --seconds 240

Add `--code` when the server was started with `--join-code`. Writes SURVIVED or
DIED to `--out`; exit status is 0 or 1, so CI can gate on it.
"""
import argparse, json, threading, time, urllib.request, sys
_a = argparse.ArgumentParser()
_a.add_argument("--port", type=int, default=8095)
_a.add_argument("--seconds", type=float, default=240)
_a.add_argument("--code", default=None)   # gated servers need the join code
_a.add_argument("--out", default="/tmp/stress.txt")
A = _a.parse_args()
B = f"http://localhost:{A.port}"
def post(p, d):
    return json.load(urllib.request.urlopen(urllib.request.Request(
        B+p, json.dumps(d).encode(), {"Content-Type":"application/json"}), timeout=20))
SLOTS = ["home_1","home_2","away_1","away_2"]
toks = []
for i, s in enumerate(SLOTS):
    t = post("/join", {"name": f"p{i}", "code": A.code})["token"]
    post("/claim", {"slot": s, "token": t}); toks.append(t)
post("/control", {"action":"start","token":toks[0]})

stop = threading.Event()
def stream(t):
    # a long-lived MJPEG reader, like a browser tab
    while not stop.is_set():
        try:
            r = urllib.request.urlopen(f"{B}/stream?token={t}", timeout=30)
            while not stop.is_set():
                if not r.read(4096): break
        except Exception:
            time.sleep(0.5)
for t in toks:
    threading.Thread(target=stream, args=(t,), daemon=True).start()

t0=time.time(); n=0
while time.time()-t0 < A.seconds:
    for i,t in enumerate(toks):
        try:
            post("/input", {"u":0.2+0.15*i, "v":0.7, "token":t}); n+=1
        except Exception as e:
            open(A.out,"w").write(
                f"DIED after {time.time()-t0:.0f}s, {n} inputs: {e}\n")
            stop.set(); sys.exit(1)
    time.sleep(0.4)
stop.set()
open(A.out,"w").write(f"SURVIVED {A.seconds:.0f}s, {n} inputs, 4 streams\n")
