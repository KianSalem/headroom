"""Specialist roles and the ownership tables that keep them disjoint.

The architectural bet of this system is stated in one place, here. A single
agent shown all 28 dimensions at once must reason about level, tone, dynamics
and image simultaneously, and its context grows with every measurement it has
ever seen. Four specialists, each shown only the dimensions it can act on,
each holding exclusive write access to the ops that move them, trade that for
a coordination problem -- which is a problem a *deterministic* supervisor can
solve, because routing by largest weighted contribution needs no judgement.

Two invariants make the trade safe, and both are asserted by tests rather than
left to discipline:

**Every scored feature has exactly one owner.** No dimension is unowned (which
would make it permanently unfixable) and none is owned twice (which would let
two specialists fight over it across turns -- the exact oscillation the critic
would then have to clean up).

**Every op kind has exactly one owner.** Two specialists editing the same op
is the failure SPEC 8.3 warns about: each sees the other's edit as drift and
corrects it, and the chain oscillates without either specialist doing anything
wrong.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from headroom.analysis.spectral import N_BANDS
from headroom.dsp.ops import OpKind
from headroom.target.distance import SCORED


class Role(StrEnum):
    EQ = "eq"
    DYNAMICS = "dynamics"
    STEREO = "stereo"
    LOUDNESS = "loudness"


#: Op kinds each role may write. Disjoint by construction, and enforced in the
#: tool layer: a specialist is not merely *asked* not to touch another's ops,
#: it is not given the tools.
#:
#: The limiter belongs to Loudness alone. SPEC 8.3 assigns ``op_limiter`` to
#: Dynamics and the limiter *ceiling* to Loudness, which contradicts its own
#: rule that two specialists must never edit the same op type. Loudness wins
#: the tie because ``true_peak_dbtp`` is the feature the limiter exists to
#: control, and it is a loudness dimension.
OWNED_OPS: Final[dict[Role, frozenset[OpKind]]] = {
    Role.EQ: frozenset({OpKind.EQ}),
    Role.DYNAMICS: frozenset({OpKind.COMPRESSOR, OpKind.EXPANDER}),
    Role.STEREO: frozenset({OpKind.STEREO_WIDTH}),
    Role.LOUDNESS: frozenset({OpKind.GAIN, OpKind.LIMITER}),
}


def _feature_owners() -> dict[str, Role]:
    """Map every scored dimension to the role that can move it.

    Ownership follows *the op that changes the number*, not the family the
    feature is filed under, and those differ in three places worth naming:

    * ``lra`` is a loudness-family measurement of dynamic range. Only a
      compressor or expander moves it, so Dynamics owns it.
    * ``attack_log2_ms`` and ``percussive_logit`` are transient measurements.
      The parameters that move them are compressor attack and release.
    * ``flatness_logit`` is a spectral summary, and EQ is what flattens or
      peaks a spectrum.
    """
    owners: dict[str, Role] = {
        "lufs_integrated": Role.LOUDNESS,
        "true_peak_dbtp": Role.LOUDNESS,
        "lra": Role.DYNAMICS,
        "crest_factor_db": Role.DYNAMICS,
        "crest_short_p50": Role.DYNAMICS,
        "flatness_logit": Role.EQ,
        "correlation_z": Role.STEREO,
        "mono_compat_db": Role.STEREO,
        "attack_log2_ms": Role.DYNAMICS,
        "percussive_logit": Role.DYNAMICS,
    }
    for i in range(N_BANDS):
        owners[f"band_clr_{i}"] = Role.EQ
        owners[f"width_{i}"] = Role.STEREO
    return owners


FEATURE_OWNER: Final[dict[str, Role]] = _feature_owners()

OWNED_FEATURES: Final[dict[Role, tuple[str, ...]]] = {
    role: tuple(s.name for s in SCORED if FEATURE_OWNER[s.name] is role) for role in Role
}

#: One line per role, used verbatim in the system prompt. Kept next to the
#: ownership table so a change to what a role owns cannot silently leave the
#: prompt describing the old boundary.
ROLE_BRIEF: Final[dict[Role, str]] = {
    Role.EQ: (
        "You control tone. You own the equalizer and nothing else. Your job is to "
        "match the target's spectral balance, band by band."
    ),
    Role.DYNAMICS: (
        "You control dynamics. You own the compressor and the expander and nothing "
        "else. Your job is to match the target's crest factor, loudness range and "
        "transient character."
    ),
    Role.STEREO: (
        "You control the stereo image. You own the width processor and nothing else. "
        "Your job is to match the target's per-band width, channel correlation and "
        "mono compatibility."
    ),
    Role.LOUDNESS: (
        "You control level and peak. You own the output gain and the true-peak "
        "limiter and nothing else. Your job is to hit the target's integrated "
        "loudness without exceeding its true-peak ceiling."
    ),
}

OWNER_OF_OP: Final[dict[OpKind, Role]] = {
    kind: role for role, kinds in OWNED_OPS.items() for kind in kinds
}
