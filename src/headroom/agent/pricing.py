"""Token pricing, so cost is measured rather than estimated.

Every number the project reports about money comes from this module applied to
usage the API actually returned, and it is written into the trace per step. The
alternative -- estimating from token counts after the fact -- is how a project
ends up publishing a cost that never happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Multiplier on the input rate for a cache read. A cached prefix is billed at
#: a tenth of the input rate, which is what makes a large static system prompt
#: plus tool schemas the right place to put the invariant half of the context.
CACHE_READ_MULTIPLIER: Final[float] = 0.1
#: Multipliers for writing a cache entry, by time-to-live. Paid once per
#: prefix. Distinguished because the two differ by a factor of 1.6 and this
#: project runs a 1-hour TTL -- an evaluation reuses the same role prefix for
#: as long as it takes to grind through the matrix, which is exactly the case
#: the longer entry exists for. Charging the 5-minute rate for a 1-hour write
#: would understate the bill.
CACHE_WRITE_MULTIPLIER_5M: Final[float] = 1.25
CACHE_WRITE_MULTIPLIER_1H: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per million tokens."""

    input_per_mtok: float
    output_per_mtok: float


#: Published list prices. Keyed by the model id passed to the API.
PRICES: Final[dict[str, Price]] = {
    "claude-opus-5": Price(5.0, 25.0),
    "claude-sonnet-5": Price(2.0, 10.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
    "claude-haiku-4-5-20251001": Price(1.0, 5.0),
}


class UnknownModelError(KeyError):
    """Raised rather than guessed.

    A missing price would otherwise surface as a cost of zero, which is the
    single most misleading number this project could print.
    """


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    #: Cache writes, split by TTL because they are priced differently. When the
    #: API returns only the undifferentiated total it is attributed to the 5m
    #: bucket, which is the cheaper of the two, so a mis-attribution shows up
    #: as an understated cost rather than a flattering one -- and the split is
    #: reported, so it is visible either way.
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_5m_tokens=self.cache_write_5m_tokens + other.cache_write_5m_tokens,
            cache_write_1h_tokens=self.cache_write_1h_tokens + other.cache_write_1h_tokens,
        )

    @property
    def cache_write_tokens(self) -> int:
        return self.cache_write_5m_tokens + self.cache_write_1h_tokens

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


def price_for(model: str) -> Price:
    if model in PRICES:
        return PRICES[model]
    # Dated snapshots share the base model's price: claude-sonnet-5-20260114
    # is priced as claude-sonnet-5.
    for known, price in PRICES.items():
        if model.startswith(known):
            return price
    raise UnknownModelError(
        f"no price for {model!r}; add it to PRICES rather than letting cost read zero"
    )


def cost_usd(model: str, usage: Usage) -> float:
    """Dollars for one call's usage."""
    price = price_for(model)
    per_input = price.input_per_mtok / 1_000_000.0
    return (
        usage.input_tokens * per_input
        + usage.cache_read_tokens * per_input * CACHE_READ_MULTIPLIER
        + usage.cache_write_5m_tokens * per_input * CACHE_WRITE_MULTIPLIER_5M
        + usage.cache_write_1h_tokens * per_input * CACHE_WRITE_MULTIPLIER_1H
        + usage.output_tokens * price.output_per_mtok / 1_000_000.0
    )
