"""Command interface shared by the low-level controller, the play server,
the BC dataset, and the high-level RL policy.

A command is (a_cmd, r_cmd, kick):
  a_cmd in {-1, 0, +1}: backward / coast / forward acceleration intent (S/W)
  r_cmd in {-1, 0, +1}: clockwise / none / counter-clockwise rotation (D/A)
  kick  in {0, 1}:      burst intent; only *accepted* when cooldown expired

The kick channel mirrors the discrete "jump" channel of the 2019 BoxHead:
a binary event with a cooldown, giving both humans and the high-level policy
a way to transfer maximum energy into the ball along the current facing.
"""

from dataclasses import dataclass

import numpy as np

KICK_COOLDOWN_SECONDS = 2.0
KICK_ACTIVE_SECONDS = 0.4  # duration the low-level holds the kick mode


@dataclass
class Command:
    a_cmd: int = 0
    r_cmd: int = 0
    kick: bool = False

    def as_array(self):
        return np.array([self.a_cmd, self.r_cmd, float(self.kick)], dtype=np.float32)


class KickState:
    """Per-player kick cooldown / active-burst state machine.

    tick() advances by dt and returns the *effective* kick signal the
    low-level policy should see (1.0 while a burst is active).
    """

    def __init__(self, cooldown=KICK_COOLDOWN_SECONDS, active=KICK_ACTIVE_SECONDS):
        self.cooldown = cooldown
        self.active = active
        self.reset()

    def reset(self):
        self._cooldown_left = 0.0
        self._active_left = 0.0

    def request(self):
        """Attempt to trigger a kick. Returns True if accepted."""
        if self._cooldown_left <= 0.0 and self._active_left <= 0.0:
            self._active_left = self.active
            self._cooldown_left = self.cooldown
            return True
        return False

    def tick(self, dt):
        effective = 1.0 if self._active_left > 0.0 else 0.0
        self._active_left = max(0.0, self._active_left - dt)
        self._cooldown_left = max(0.0, self._cooldown_left - dt)
        return effective

    @property
    def cooldown_fraction(self):
        """1.0 = just used, 0.0 = ready. Exposed in observations and UI."""
        return self._cooldown_left / self.cooldown if self.cooldown > 0 else 0.0

    @property
    def ready(self):
        return self._cooldown_left <= 0.0 and self._active_left <= 0.0


def command_obs(command: Command, kick_state: KickState):
    """Command-related observation vector for the low-level policy:
    [a_cmd, r_cmd, effective_kick_active, cooldown_fraction]."""
    return np.array(
        [command.a_cmd, command.r_cmd,
         1.0 if kick_state._active_left > 0.0 else 0.0,
         kick_state.cooldown_fraction],
        dtype=np.float32)


# MultiDiscrete([3,3,2]) encoding used by BC / high-level RL.
def command_to_multidiscrete(command: Command):
    return np.array([command.a_cmd + 1, command.r_cmd + 1, int(command.kick)])


def multidiscrete_to_command(md):
    return Command(a_cmd=int(md[0]) - 1, r_cmd=int(md[1]) - 1, kick=bool(md[2]))
