from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..v5 import gumbel_from_uniform, stable_words


class EventRandom:
    """Stateless, event-addressed random streams with bounded scalar caching."""

    def __init__(self, seed: int):
        self.seed = int(seed)

    @staticmethod
    @lru_cache(maxsize=32_768)
    def identity_words(identity: str) -> tuple[int, ...]:
        return stable_words(identity)

    def generator(self, *coordinates: int) -> np.random.Generator:
        seed = np.random.SeedSequence(
            self.seed,
            spawn_key=tuple(int(value) for value in coordinates),
        )
        return np.random.Generator(np.random.PCG64(seed))

    def stable_generator(
        self, stream: int, *coordinates: int, identity: str
    ) -> np.random.Generator:
        return self.generator(
            stream,
            *coordinates,
            *self.identity_words(str(identity)),
        )

    @lru_cache(maxsize=131_072)
    def gumbel(
        self,
        stream: int,
        event_address: tuple[int, ...],
        identity: str,
    ) -> float:
        rng = self.stable_generator(
            stream, *event_address, identity=identity
        )
        return gumbel_from_uniform(float(rng.random()))
