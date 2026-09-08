"""The chain model.

Processing is an ordered, declarative list of typed ops. The agent never
mutates audio; it edits a chain, and the renderer applies the whole chain to
the *original* source every time. That buys four things:

1. **No cumulative degradation.** Re-rendering from source means 40 iterations
   do not stack 40 generations of quantization and filter ringing.
2. **Revision instead of correction-stacking.** A 240 Hz cut can be changed
   from -3 dB to -1.5 dB. A destructive pipeline could only add a +1.5 dB
   boost on top, which is a different and worse filter.
3. **The chain is the artifact** -- readable, diffable, exportable, and the
   thing a real engineer would want to inspect.
4. **Determinism.** Same chain plus same source equals the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from .ops import ChainStage, Op, OpKind, edit_op

if TYPE_CHECKING:
    from headroom.audio import AudioBuffer


class ChainError(ValueError):
    """Raised for a structurally invalid chain edit, e.g. an unknown op id."""


@dataclass(frozen=True, slots=True)
class Reposition:
    """A record of the system correcting the agent's ordering.

    An agent that puts the limiter first should be corrected, not trusted --
    and the correction is logged rather than silent, because the rate at which
    it happens is evidence about how well the agent understands the chain.
    """

    op_id: str
    kind: OpKind
    from_index: int
    to_index: int


class Chain(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ops: tuple[Op, ...] = Field(default=())

    # --- inspection ---------------------------------------------------------
    def find(self, op_id: str) -> Op:
        for op in self.ops:
            if op.id == op_id:
                return op
        raise ChainError(f"no op with id {op_id!r}; have {[o.id for o in self.ops]}")

    def index_of(self, op_id: str) -> int:
        for i, op in enumerate(self.ops):
            if op.id == op_id:
                return i
        raise ChainError(f"no op with id {op_id!r}")

    def of_kind(self, kind: OpKind) -> tuple[Op, ...]:
        return tuple(op for op in self.ops if op.kind == kind)

    def fingerprint(self) -> str:
        """Stable hash of the audible content of the chain.

        Op ids are excluded: two chains with identical parameters render to
        identical audio regardless of how their ops were named, so they must
        share a render-cache entry.
        """
        payload = [
            {k: v for k, v in op.model_dump(mode="json").items() if k != "id"} for op in self.ops
        ]
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()

    # --- editing: every method returns a new Chain --------------------------
    def add(self, op: Op, position: int | None = None) -> Chain:
        ops = list(self.ops)
        ops.insert(len(ops) if position is None else position, op)
        return Chain(ops=tuple(ops))

    def edit(self, op_id: str, **params: object) -> Chain:
        i = self.index_of(op_id)
        ops = list(self.ops)
        ops[i] = edit_op(ops[i], **params)
        return Chain(ops=tuple(ops))

    def remove(self, op_id: str) -> Chain:
        i = self.index_of(op_id)
        return Chain(ops=tuple(self.ops[:i] + self.ops[i + 1 :]))

    def canonical(self) -> tuple[Chain, tuple[Reposition, ...]]:
        """Reorder to canonical mastering order, reporting every move.

        Order is gain staging -> EQ -> dynamics -> stereo -> limiter. The sort
        is stable, so ops within a stage keep their relative order and an agent
        that deliberately puts one EQ before another is respected.
        """
        indexed = list(enumerate(self.ops))
        ordered = sorted(indexed, key=lambda pair: (ChainStage(pair[1].stage), pair[0]))
        moves = tuple(
            Reposition(op_id=op.id, kind=OpKind(op.kind), from_index=old_index, to_index=new_index)
            for new_index, (old_index, op) in enumerate(ordered)
            if old_index != new_index
        )
        return Chain(ops=tuple(op for _, op in ordered)), moves

    def describe(self) -> str:
        """Readable signal flow. This is what goes in the report."""
        if not self.ops:
            return "source -> (empty chain) -> render"
        parts: list[str] = []
        for op in self.ops:
            d = op.model_dump(exclude={"id", "kind"})
            if op.kind == OpKind.EQ:
                bands = ", ".join(
                    f"{b['shape']} {b['freq_hz']:.0f}Hz {b['gain_db']:+.1f}dB Q{b['q']:.2f}"
                    for b in d["bands"]
                )
                parts.append(f"eq[{bands}]")
            else:
                parts.append(f"{op.kind}[{_format_params(d)}]")
        return "source -> " + " -> ".join(parts) + " -> render"


def _format_params(params: dict[str, object]) -> str:
    """Render op parameters for human reading.

    Only level-like quantities get an explicit sign: "+2.00 dB" is meaningful
    because gain has a direction, whereas "width=+1.30" or "release_ms=+50.00"
    just looks wrong.
    """
    signed_suffixes = ("_db", "_dbtp", "_dbfs")
    out: list[str] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, float):
            fmt = f"{value:+.2f}" if key.endswith(signed_suffixes) else f"{value:.2f}"
            out.append(f"{key}={fmt}")
        else:
            out.append(f"{key}={value}")
    return " ".join(out)


def render(source: AudioBuffer, chain: Chain, use_cache: bool = True) -> AudioBuffer:
    """Apply a chain to the original source. Re-exported from the backend so
    callers never import a backend directly."""
    from .backends.pedalboard import render_chain

    return render_chain(source, chain, use_cache=use_cache)
