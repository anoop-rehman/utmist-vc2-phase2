"""Faithful port of the Unity (uvc2) Karl-Sims creature BRAIN to Python.

Ground truth: Assets/Scripts/Phenotype/Creature.cs, Segment.cs,
CreatureSpawner.cs and Genotype/CreatureGenotype.cs of the uvc2 repo.
This module reproduces, at the instance level, exactly what Unity builds when
it spawns a creature from a CreatureGenotype, including its bugs -- the
creatures were evolved against those bugs, so they are part of the genome's
semantics.

Semantics replicated (with receipts):

* Update order (Creature.FixedUpdate): FeedForward() is called TWICE per
  physics step.  Each FeedForward pass:
      1. every sensor Neuron:  SetSensorOutputs()   (outValue <- world)
      2. every hidden Neuron:  GetInputs()          (latch a,b,c from inputs)
      3. every hidden Neuron:  SetOutput()          (compute outValue)
      4. every effector:       GetInputs()          (latch + apply motor)
  i.e. the hidden layer is a synchronous two-phase update, and effectors read
  the freshly computed outputs.  Both passes of one step see the same
  Time.time (Unity returns fixedTime inside FixedUpdate).

* Fixed timestep: uvc2/ProjectSettings/TimeManager.asset says
  "Fixed Timestep: 0.0333333" -- dt = 1/30 s, NOT the Unity default 0.02.

* Time origin: oscillate-wave/saw use absolute Time.time.  During evolution
  creatures were spawned at arbitrary global times, so the phase is not part
  of the genome; we canonically start at t = 0 (t = step * dt).

* Contact sensors ALWAYS read -1 ("empty").  Segment.cs has script execution
  order -50 (Segment.cs.meta), Creature.cs has 0.  Per fixed step Unity runs:
  Segment.FixedUpdate (resets the isXEmpty flags to true) -> Creature.
  FixedUpdate (brain reads them) -> internal physics -> OnTriggerStay (sets
  the flags).  So the flags set by last step's triggers are wiped before the
  brain ever reads them: GetContact returns (empty ? -1 : 1) == -1 forever.
  We replicate the constant, and keep a switch to turn real contacts on.

* Neuron input latching (Neuron.GetInputs): a = neuronA.outValue*weights[0]
  etc., but ONLY if that input neuron exists; a,b,c are initialised to 1 and
  keep their old value otherwise.  Unresolved references therefore mean a
  constant input of 1.0, not 0.

* Reference resolution (Creature.ConnectNeurons/GetNeuron) with the
  genotype-sharing quirk: CreatureSpawner.SetupSegment stamps
  ng.nr.connectionPath = <path of the instance being spawned> IN PLACE on the
  genotype's NeuronGenotype, which is SHARED by all instances of that segment
  type.  The last spawned instance wins.  Consequently SELF/CHILD lookups
  (which compare stamped paths) match every instance of a segment type and
  return the FIRST one in the global list -- e.g. all copies of a recursive
  limb are driven by the first copy's oscillator.  Replicated exactly.
  - Sensor references that fail to resolve still occupy an input slot
    (inputNeuronsToAdd.Add(null)); hidden/effector references that fail are
    skipped, shifting later inputs into earlier weight slots.
  - >3 resolved inputs: none are connected.
  - PARENT walks the *instance* parent chain, decrementing relativeLevel only
    when the parent's segment TYPE id differs from the current one, then
    matches by neuron id within that segment instance's own neuron list.

* Root segment: in KSS stage Unity force-adds photosensor neurons 9,10,11 to
  the root SegmentGenotype at spawn time; photosensors read 0 when the env
  has no light source tagged "Photosensor" (OceanEnv/FloorEnv spawn none:
  photosensorSpawnLocations is empty).

* Neuron ops (Neuron.SetOutput), float32 like Unity/C#: after computing,
  NaN -> 0, then clamp to [-300, 300]; integrate/differentiate then store
  dummy1 = a.  Mathf.Min/Max/Sign ternary semantics kept (Sign(0) = +1,
  NaN propagation via the C# ternaries).

* Effector (id 12, Neuron.GetInputs): res = a+b+c; if (res <= 15) res += 10;
  targetVelocity = 10 * clamp(res, -40, 40)  [HingeJoint units: deg/s],
  motor force cap 400 N*m (Segment.AttachHingeJoint), hinge limits +-75 deg.
  The res += 10 quirk is replicated exactly.

Usage:
    from rower_soccer.tools.creature_format import load_creature
    from rower_soccer.tools.sims_brain import SimsBrain
    brain = SimsBrain(load_creature("creature_configs/TWO_ARM_ROWER.creature"))
    targets = brain.fixed_update(t)   # {segment_uid: targetVelocity deg/s}
"""

import math
from dataclasses import dataclass, field

import numpy as np

F = np.float32
UNITY_DT = F(1.0 / 30.0)  # TimeManager.asset "Fixed Timestep: 0.0333333"

GHOST, PARENT, SELF, CHILD, TRACED = range(5)


def _nullable(sn):
    """Decode the SN<...> serializable-nullable wrapper from the .creature file."""
    if isinstance(sn, dict):
        return sn["v"] if sn.get("hasValue") else None
    return sn


@dataclass
class NRef:
    """NeuronReference (input reference, kept as stored in the genotype)."""
    relativity: int | None
    relative_level: int | None
    connection_path: list | None
    id: int

    @classmethod
    def from_dict(cls, d):
        return cls(
            relativity=_nullable(d.get("relativityNullable")),
            relative_level=_nullable(d.get("relativeLevelNullable")),
            connection_path=(None if d.get("connectionPath") is None
                             else list(d["connectionPath"])),
            id=d["id"],
        )


@dataclass
class NGShared:
    """One NeuronGenotype record, SHARED across all instances of its segment
    type -- mirrors Unity, where spawning mutates the genotype in place."""
    type: int
    inputs: list          # list[NRef]
    weights: list         # list[float]
    id: int               # nr.id
    # spawn-time stamps (SetupSegment): last spawned instance wins
    relativity: int = TRACED
    stamped_path: tuple | None = None


@dataclass
class SegInstance:
    uid: int
    type_id: int
    parent: "SegInstance | None"
    path: tuple | None    # None for root (Unity leaves it null)
    neurons: list = field(default_factory=list)


class NeuronInst:
    __slots__ = ("ng", "segment", "segment_id", "a", "b", "c", "out",
                 "dummy1", "in_a", "in_b", "in_c", "last_effector_cmd")

    def __init__(self, ng, segment, segment_id):
        self.ng = ng
        self.segment = segment          # SegInstance or None (ghost)
        self.segment_id = segment_id    # segment TYPE byte (0 = ghost)
        self.a = F(1.0)
        self.b = F(1.0)
        self.c = F(1.0)
        self.out = F(0.0)
        self.dummy1 = F(0.0)
        self.in_a = self.in_b = self.in_c = None
        self.last_effector_cmd = F(0.0)  # deg/s, effectors only


def spawn_instances(genotype):
    """Segment-instance expansion, mirroring CreatureSpawner.SpawnSegment:
    DFS in stored connection order; recursiveLimitValues dict decremented on
    spawn and cloned per child; runTerminalOnly when the limit hits 0 or the
    type has no self-connection; terminalOnly connections skipped unless
    runTerminalOnly."""
    segs = {s["id"]: s for s in genotype["segments"]}
    out = []

    def rec(conn, limits, path, parent):
        tid = 1 if conn is None else conn["destination"]
        sg = segs.get(tid)
        if sg is None:
            return
        inst = SegInstance(uid=len(out), type_id=tid, parent=parent,
                           path=None if path is None else tuple(path))
        out.append(inst)
        limits = dict(limits)
        limits[tid] -= 1
        run_terminal_only = (limits[tid] == 0 or
                             not any(c["destination"] == tid
                                     for c in sg["connections"]))
        for c in sg["connections"]:
            if limits[c["destination"]] > 0:
                if not run_terminal_only and c["terminalOnly"]:
                    continue
                child_path = ([] if path is None else list(path)) + [c["id"]]
                rec(c, limits, child_path, inst)

    limits0 = {s["id"]: s["recursiveLimit"] for s in genotype["segments"]}
    rec(None, limits0, None, None)
    return out


class SimsBrain:
    """Instance-level replica of Unity's Creature neuron graph."""

    def __init__(self, genotype, contact_value=-1.0, photosensor_value=0.0,
                 joint_axis_fn=None):
        # contact_value=-1: the Unity execution-order bug (see module docs).
        # joint_axis_fn(uid, axis_index 0..2) -> Unity-style degrees; only
        # called if the genome actually references joint-angle sensors 6-8.
        self.contact_value = F(contact_value)
        self.photosensor_value = F(photosensor_value)
        self.joint_axis_fn = joint_axis_fn or (lambda uid, ax: 0.0)

        # --- shared NeuronGenotype records per segment type (Unity shares) ---
        shared = {}
        for sg in genotype["segments"]:
            ngs = []
            for n in sg["neurons"]:
                nr = NRef.from_dict(n["nr"])
                ngs.append(NGShared(
                    type=n["type"],
                    inputs=[NRef.from_dict(i) for i in (n["inputs"] or [])],
                    weights=[F(w) for w in (n["weights"] or [])],
                    id=nr.id,
                ))
            shared[sg["id"]] = ngs
        # KSS: root segment gets photosensors 9,10,11 force-added if missing
        if genotype.get("stage", 0) == 0 and 1 in shared:
            have = {ng.id for ng in shared[1]}
            for pid in (9, 10, 11):
                if pid not in have:
                    shared[1].append(NGShared(type=0, inputs=[], weights=[],
                                              id=pid))

        self.sensors, self.neurons, self.effectors = [], [], []

        def add_neuron(ng, segment, segment_id):
            n = NeuronInst(ng, segment, segment_id)
            if ng.id >= 13:
                self.neurons.append(n)
            elif ng.id == 12:
                self.effectors.append(n)
            else:
                self.sensors.append(n)
            return n

        # ghost neurons first (SpawnCreature), relativity forced to GHOST,
        # path NOT stamped
        for ng in shared.get(0, []):
            ng.relativity = GHOST
            add_neuron(ng, None, 0)

        # spawn segments depth-first; stamp shared records in spawn order
        self.instances = spawn_instances(genotype)
        for inst in self.instances:
            for ng in shared.get(inst.type_id, []):
                ng.relativity = TRACED
                ng.stamped_path = inst.path      # last spawn wins (shared!)
                n = add_neuron(ng, inst, inst.type_id)
                inst.neurons.append(n)

        # InitializeCreature: connect hidden neurons, then effectors
        self._connect(self.neurons)
        self._connect(self.effectors)

    # --- Creature.GetNeuron -------------------------------------------------
    def _get_neuron(self, neuron_list, guiding, requesting):
        req_ng = requesting.ng
        if guiding.relativity == GHOST:
            for n in neuron_list:
                if n.segment_id == 0 and n.ng.id == guiding.id:
                    return n
        elif guiding.relativity == PARENT:
            left = guiding.relative_level or 0
            cur = requesting.segment
            while left != 0:
                nxt = cur.parent if cur is not None else None
                if nxt is None:          # Unity: exception -> break
                    break
                if nxt.type_id != cur.type_id:   # type-id equality quirk
                    left -= 1
                cur = nxt
            if cur is not None:
                for n in cur.neurons:
                    if n.ng.id == guiding.id:
                        return n
        elif guiding.relativity == SELF:
            for n in neuron_list:
                if n.ng.id == guiding.id:
                    p = n.ng.stamped_path
                    if p is None:
                        if req_ng.stamped_path is None:
                            return n
                    elif (req_ng.stamped_path is not None and
                          tuple(p) == tuple(req_ng.stamped_path)):
                        return n
        elif guiding.relativity == CHILD:
            if req_ng.stamped_path is None:
                child_path = (None if guiding.connection_path is None
                              else tuple(guiding.connection_path))
            else:
                child_path = tuple(req_ng.stamped_path) + tuple(
                    guiding.connection_path or ())
            for n in neuron_list:
                if n.ng.id == guiding.id:
                    p = n.ng.stamped_path
                    if p is None:
                        if child_path is None:
                            return n
                    elif child_path is not None and tuple(p) == child_path:
                        return n
        # TRACED (or anything else): Unity has no branch -> null
        return None

    # --- Creature.ConnectNeurons -------------------------------------------
    def _connect(self, neuron_list):
        for n in neuron_list:
            found = []
            for ref in n.ng.inputs:
                if ref.id >= 13:
                    hit = self._get_neuron(self.neurons, ref, n)
                    if hit is not None:
                        found.append(hit)
                elif ref.id == 12:
                    hit = self._get_neuron(self.effectors, ref, n)
                    if hit is not None:
                        found.append(hit)
                else:  # sensor: added even when unresolved (slot-shift quirk)
                    found.append(self._get_neuron(self.sensors, ref, n))
            # SetInputNeurons: >3 -> nothing connected
            if len(found) >= 1 and len(found) <= 3:
                n.in_a = found[0]
                if len(found) >= 2:
                    n.in_b = found[1]
                if len(found) == 3:
                    n.in_c = found[2]

    # --- per-pass update ----------------------------------------------------
    def _set_sensor_outputs(self):
        for s in self.sensors:
            i = s.ng.id
            if i <= 5:
                s.out = self.contact_value
            elif i <= 8:
                s.out = F(self.joint_axis_fn(s.segment.uid, i - 6))
            elif i <= 11:
                s.out = self.photosensor_value
            else:
                s.out = F(0.0)

    @staticmethod
    def _get_inputs(n):
        w = n.ng.weights
        if n.in_a is not None:
            n.a = F(n.in_a.out * w[0])
        if n.in_b is not None:
            n.b = F(n.in_b.out * w[1])
        if n.in_c is not None:
            n.c = F(n.in_c.out * w[2])
        if n.ng.id == 12:  # effector applies its motor inside GetInputs
            res = F(F(n.a + n.b) + n.c)
            if res <= F(15.0):  # the evolved-against quirk; NaN falls through
                res = F(res + F(10.0))
            res = F(-40.0) if res < F(-40.0) else (F(40.0) if res > F(40.0)
                                                   else res)
            n.last_effector_cmd = F(F(10.0) * res)  # deg/s

    @staticmethod
    def _set_output(n, t, dt):
        a, b, c, out, d1 = n.a, n.b, n.c, n.out, n.dummy1
        ty = n.ng.type
        with np.errstate(all="ignore"):
            if ty == 0:
                out = F(a + b)
            elif ty == 1:
                out = F(a * b)
            elif ty == 2:
                out = F(a / b)
            elif ty == 3:                       # Mathf.Min(a+b, c)
                s = F(a + b)
                out = s if s < c else c
            elif ty == 4:
                out = F(1.0) if a > b else F(-1.0)
            elif ty == 5:                       # Mathf.Sign: >=0 -> 1
                out = F(1.0) if a >= F(0.0) else F(-1.0)
            elif ty == 6:
                out = a if a < b else b
            elif ty == 7:
                out = a if a > b else b
            elif ty == 8:
                out = F(abs(a))
            elif ty == 9:
                out = b if a > F(0.0) else c
            elif ty == 10:
                out = F(a + F(F(b - a) * c))
            elif ty == 11:
                out = F(math.sin(a))
            elif ty == 12:
                out = F(math.cos(a))
            elif ty == 13:
                out = F(math.atan(a))
            elif ty == 14:
                aa = float(abs(a))
                out = F(math.log10(aa)) if aa > 0.0 else (
                    F(-math.inf) if aa == 0.0 else F(math.nan))
            elif ty == 15:
                try:
                    out = F(math.exp(a))
                except OverflowError:
                    out = F(math.inf)
            elif ty == 16:
                try:
                    e = F(math.exp(F(-a)))
                except OverflowError:
                    e = F(math.inf)
                out = F(F(1.0) / F(F(1.0) + e))
            elif ty == 17:                      # integrate
                out = F(out + F(dt * F(F(a + d1) * F(0.5))))
            elif ty == 18:                      # differentiate
                out = F(F(a - d1) / dt)
            elif ty == 19:                      # smooth
                out = F(out + F(F(a - out) * F(0.5)))
            elif ty == 20:                      # memory
                out = a
            elif ty == 21:                      # oscillate-wave
                out = F(F(b * F(math.sin(F(t * a)))) + c)
            elif ty == 22:                      # oscillate-saw
                x = F(t * a)
                out = F(F(b * F(x - F(math.floor(x)))) + c)
            else:
                out = F(0.0)
        if math.isnan(out):
            out = F(0.0)
        out = F(-300.0) if out < F(-300.0) else (F(300.0) if out > F(300.0)
                                                 else out)
        n.out = out
        if ty in (17, 18):
            n.dummy1 = a

    def feed_forward(self, t, dt=UNITY_DT):
        t = F(t)
        self._set_sensor_outputs()
        for n in self.neurons:
            self._get_inputs(n)
        for n in self.neurons:
            self._set_output(n, t, dt)
        for e in self.effectors:
            self._get_inputs(e)

    def fixed_update(self, t, dt=UNITY_DT):
        """One Unity physics step: FeedForward twice at the same Time.time.
        Returns {segment_uid: hinge targetVelocity in deg/s} (last write of
        the step wins, as in Unity)."""
        self.feed_forward(t, dt)
        self.feed_forward(t, dt)
        return {e.segment.uid: float(e.last_effector_cmd)
                for e in self.effectors if e.segment is not None}

    # --- introspection ------------------------------------------------------
    def describe(self):
        lines = []
        def name(n):
            if n is None:
                return "None"
            seg = "ghost" if n.segment is None else f"uid{n.segment.uid}"
            return f"[{seg}:n{n.ng.id}]"
        for label, lst in (("SENSOR", self.sensors), ("HIDDEN", self.neurons),
                           ("EFFECTOR", self.effectors)):
            for n in lst:
                lines.append(
                    f"{label} {name(n)} type={n.ng.type} "
                    f"w={[round(float(w), 4) for w in n.ng.weights]} "
                    f"A={name(n.in_a)} B={name(n.in_b)} C={name(n.in_c)}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from rower_soccer.tools.creature_format import load_creature

    path = sys.argv[1] if len(sys.argv) > 1 else \
        "creature_configs/TWO_ARM_ROWER.creature"
    brain = SimsBrain(load_creature(path))
    print(f"{len(brain.sensors)} sensors, {len(brain.neurons)} hidden, "
          f"{len(brain.effectors)} effectors")
    print(brain.describe())
    print("\nfirst steps (t, H=hidden[0].out, effector cmds deg/s):")
    for step in range(8):
        t = step * float(UNITY_DT)
        cmds = brain.fixed_update(t)
        print(f"t={t:6.3f}  H={float(brain.neurons[0].out):9.3f}  "
              + " ".join(f"u{k}:{v:7.1f}" for k, v in sorted(cmds.items())))
