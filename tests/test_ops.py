from __future__ import annotations

import pytest
from pydantic import ValidationError

from headroom.dsp.ops import (
    STAGE_OF,
    BoundViolationError,
    ChainStage,
    EqOp,
    OpKind,
    edit_op,
    make_op,
    op_eq,
    op_gain,
)


def test_typed_constructor_gives_a_concrete_type() -> None:
    """``make_op`` returns the ``Op`` union, so ``.gain_db`` cannot type-check.
    The typed constructors exist so internal callers -- and the specialists --
    get a concrete type."""
    op = op_gain(-2.1)
    assert op.gain_db == -2.1
    assert op.stage is ChainStage.GAIN_STAGING


def test_make_op_dispatches_from_a_string_kind() -> None:
    """The tool layer receives the kind as a string from the model, so the
    dynamic path has to exist alongside the typed constructors."""
    op = make_op("gain", gain_db=-2.1)
    assert op.kind is OpKind.GAIN
    assert op.stage is ChainStage.GAIN_STAGING


@pytest.mark.parametrize(
    ("kind", "params", "field"),
    [
        (OpKind.GAIN, {"gain_db": 40.0}, "gain_db"),
        (OpKind.GAIN, {"gain_db": -40.0}, "gain_db"),
        (
            OpKind.EQ,
            {"bands": [{"shape": "peak", "freq_hz": 240, "gain_db": -30.0}]},
            "bands.0.gain_db",
        ),
        (OpKind.EQ, {"bands": [{"shape": "peak", "freq_hz": 240, "q": 50.0}]}, "bands.0.q"),
        (OpKind.EQ, {"bands": [{"shape": "peak", "freq_hz": 30000}]}, "bands.0.freq_hz"),
        (OpKind.COMPRESSOR, {"threshold_db": -18.0, "ratio": 100.0}, "ratio"),
        (OpKind.COMPRESSOR, {"threshold_db": 10.0, "ratio": 4.0}, "threshold_db"),
        (OpKind.LIMITER, {"ceiling_dbtp": 0.0}, "ceiling_dbtp"),
        (OpKind.STEREO_WIDTH, {"width": 3.0}, "width"),
        (OpKind.STEREO_WIDTH, {"width": 1.0, "band": 12}, "band"),
        (OpKind.EXPANDER, {"threshold_db": -20.0, "ratio": 20.0}, "ratio"),
    ],
)
def test_out_of_bounds_is_rejected_not_clamped(
    kind: OpKind, params: dict[str, object], field: str
) -> None:
    """Rejection rather than silent clamping. A clamp would tell the agent its
    move succeeded and then show it a measurement that disagrees, which is the
    hardest class of bug to attribute inside a feedback loop."""
    with pytest.raises(BoundViolationError) as exc:
        make_op(kind, **params)
    assert any(v["field"] == field for v in exc.value.violations)


def test_bound_violation_is_machine_readable() -> None:
    with pytest.raises(BoundViolationError) as exc:
        make_op(OpKind.GAIN, gain_db=40.0)
    payload = exc.value.as_tool_error()
    assert payload["error"] == "bound_violation"
    violation = exc.value.violations[0]
    assert violation["value"] == 40.0
    assert "24" in str(violation["constraint"])


def test_editing_revises_in_place_instead_of_stacking() -> None:
    """Changing a -3 dB cut to -1.5 dB must edit the filter, not add a +1.5 dB
    boost on top, which would be a different and worse filter."""
    eq = op_eq([{"shape": "peak", "freq_hz": 240, "gain_db": -3.0, "q": 1.4}])
    revised = edit_op(eq, bands=({"shape": "peak", "freq_hz": 240, "gain_db": -1.5, "q": 1.4},))
    assert isinstance(revised, EqOp)
    assert len(revised.bands) == 1
    assert revised.bands[0].gain_db == -1.5
    assert revised.id == eq.id


def test_editing_is_also_bounded() -> None:
    with pytest.raises(BoundViolationError):
        edit_op(op_gain(0.0), gain_db=99.0)


def test_ops_are_immutable() -> None:
    op = op_gain(-2.0)
    with pytest.raises(ValidationError):
        op.gain_db = 5.0


MINIMAL_PARAMS: dict[OpKind, dict[str, object]] = {
    OpKind.GAIN: {"gain_db": 0.0},
    OpKind.EQ: {"bands": [{"shape": "peak", "freq_hz": 1000.0, "gain_db": 0.0}]},
    OpKind.COMPRESSOR: {"threshold_db": -18.0, "ratio": 2.0},
    OpKind.EXPANDER: {"threshold_db": -18.0, "ratio": 2.0},
    OpKind.STEREO_WIDTH: {"width": 1.0},
    OpKind.LIMITER: {},
}


def test_every_op_kind_is_constructible_and_has_a_stage() -> None:
    """Guards against adding an op kind without wiring it into the canonical
    ordering, which would leave its position in the chain undefined."""
    assert set(MINIMAL_PARAMS) == set(OpKind)
    for kind, params in MINIMAL_PARAMS.items():
        op = make_op(kind, **params)
        assert isinstance(op.stage, ChainStage)
        assert STAGE_OF[kind] is op.stage


def test_dynamics_ops_share_a_stage() -> None:
    """Compressor and expander are both dynamics, so ordering between them is
    the agent's choice rather than the system's."""
    assert STAGE_OF[OpKind.COMPRESSOR] is STAGE_OF[OpKind.EXPANDER]


def test_stage_order_is_the_canonical_mastering_chain() -> None:
    assert [
        STAGE_OF[k]
        for k in (OpKind.GAIN, OpKind.EQ, OpKind.COMPRESSOR, OpKind.STEREO_WIDTH, OpKind.LIMITER)
    ] == sorted(
        [
            STAGE_OF[k]
            for k in (
                OpKind.GAIN,
                OpKind.EQ,
                OpKind.COMPRESSOR,
                OpKind.STEREO_WIDTH,
                OpKind.LIMITER,
            )
        ]
    )
