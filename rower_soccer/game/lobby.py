"""Lobby: 4 claimable seats + unlimited spectators, reconnect-safe.

Input isolation is structural, not checked: a client sends a **token**, never a slot
id, and the server derives the slot from the token.  There is therefore no request a
client can construct that drives someone else's creature -- the classic bug in
"pass your player index" designs, and the one that would silently corrupt a demo
file (mislabelled BC data is worse than missing BC data).

Reconnect: identity is the token, not the socket.  A phone that sleeps, drops wifi
and comes back with the same token resumes its seat.  A seat whose client has been
silent for `claim_timeout` seconds is *stale* and may be taken by someone else --
but only when someone actually asks for it, so a brief blip costs nothing.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

SLOTS = ("home_1", "home_2", "away_1", "away_2")


@dataclass
class Client:
    token: str
    name: str
    slot: str | None = None
    joined: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class Lobby:
    def __init__(self, slots=SLOTS, claim_timeout=25.0):
        self.slots = tuple(slots)
        self.claim_timeout = float(claim_timeout)
        self._lock = threading.RLock()
        self._clients: dict[str, Client] = {}
        self._by_slot: dict[str, str] = {}      # slot -> token

    # -- membership --------------------------------------------------------
    def join(self, name, token=None):
        """Idempotent. Re-joining with a known token keeps the seat (reconnect)."""
        with self._lock:
            if token and token in self._clients:
                c = self._clients[token]
                c.last_seen = time.time()
                if name:
                    c.name = name[:24]
                return c
            token = secrets.token_urlsafe(16)
            c = Client(token=token, name=(name or "player")[:24])
            self._clients[token] = c
            return c

    def get(self, token):
        with self._lock:
            c = self._clients.get(token)
            if c is not None:
                c.last_seen = time.time()
            return c

    def heartbeat(self, token):
        return self.get(token) is not None

    def leave(self, token):
        with self._lock:
            c = self._clients.pop(token, None)
            if c and c.slot and self._by_slot.get(c.slot) == token:
                del self._by_slot[c.slot]
            return c

    # -- seats -------------------------------------------------------------
    def _stale(self, token, now):
        c = self._clients.get(token)
        return c is None or (now - c.last_seen) > self.claim_timeout

    def claim(self, token, slot):
        """Take `slot`. Returns (ok, message). A client holds at most one seat."""
        with self._lock:
            c = self._clients.get(token)
            if c is None:
                return False, "unknown token; join first"
            if slot not in self.slots:
                return False, f"no such slot {slot!r}"
            now = time.time()
            c.last_seen = now
            holder = self._by_slot.get(slot)
            if holder == token:
                return True, "already yours"
            if holder is not None and not self._stale(holder, now):
                return False, f"slot {slot} is taken by {self._clients[holder].name}"
            if holder is not None:
                self._clients[holder].slot = None       # reclaim a dead seat
            if c.slot:
                self._by_slot.pop(c.slot, None)
            c.slot = slot
            self._by_slot[slot] = token
            return True, "ok"

    def release(self, token):
        with self._lock:
            c = self._clients.get(token)
            if c is None or c.slot is None:
                return None
            slot, c.slot = c.slot, None
            self._by_slot.pop(slot, None)
            return slot

    def slot_of(self, token):
        with self._lock:
            c = self._clients.get(token)
            return None if c is None else c.slot

    def occupant(self, slot):
        """The live client in `slot`, or None (a stale holder reads as empty)."""
        with self._lock:
            tok = self._by_slot.get(slot)
            if tok is None:
                return None
            c = self._clients.get(tok)
            if c is None or (time.time() - c.last_seen) > self.claim_timeout:
                return None
            return c

    def state(self):
        with self._lock:
            now = time.time()
            seats = []
            for s in self.slots:
                tok = self._by_slot.get(s)
                c = self._clients.get(tok) if tok else None
                seats.append(dict(
                    slot=s, name=c.name if c else None,
                    taken=c is not None,
                    stale=bool(c and (now - c.last_seen) > self.claim_timeout),
                    idle=round(now - c.last_seen, 1) if c else None))
            spectators = [c.name for c in self._clients.values() if c.slot is None]
            return dict(seats=seats, spectators=spectators,
                        n_clients=len(self._clients))
