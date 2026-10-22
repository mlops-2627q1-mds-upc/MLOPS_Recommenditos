"""When a load test user sends its requests (docs/docs/load-testing.md, issue #143).

Each user sends at its share of the total rate, with exponential gaps between
its sends, so the users together send like a Poisson process at the total rate:
the requests are spread in time the way independent clients' requests are.
A fixed gap would let every user started in the same second send in the same
second, and again every gap, so the API would see the whole rate as bursts.
"""

import random


class Arrivals:
    """The send times of one user, at `rate` requests per second on average."""

    def __init__(self, rate: float, rng: random.Random) -> None:
        if rate <= 0:
            raise ValueError(f"the rate must be positive, not {rate}")
        self.rate = rate
        self.rng = rng
        self._last_send: float | None = None

    def next_wait(self, now: float) -> float:
        """Seconds from `now` until the next send.

        The first send follows the user's start by an exponential gap, so users
        started together do not send together. Each later send follows the
        previous send by a new gap, less the time the previous answer took;
        when the answer took longer than the gap, the user sends at once, so a
        slow target gets fewer requests rather than a growing queue.
        """
        gap = self.rng.expovariate(self.rate)
        start = now if self._last_send is None else self._last_send
        wait = max(0.0, start + gap - now)
        self._last_send = now + wait
        return wait
