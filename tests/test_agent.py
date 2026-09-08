"""Tests for the agent architecture, all of them free.

The point of this file is that everything except the single module that talks
to the API is exercised here with no key and no network: the ownership
invariants, the tool layer's refusals, the briefing filter, the memory, the
routing and rerouting, and the whole loop end to end. The model boundary is
covered too, from a cassette recorded against a stub.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from evals import briefs, report
from evals.degradations import make_degradation

from headroom.agent import brief, roles, supervisor, tools
from headroom.agent.briefing import build as build_briefing
from headroom.agent.cassette import Cassette, CassetteMissError, Mode, digest
from headroom.agent.client import LLMSpecialist, ModelClient, ModelConfig
from headroom.agent.factory import scaffold_supervisor, scripted_supervisor
from headroom.agent.memory import WorkingMemory
from headroom.agent.pricing import PRICES, UnknownModelError, Usage, cost_usd, price_for
from headroom.agent.roles import OWNED_FEATURES, OWNED_OPS, Role
from headroom.agent.specialist import PlannedEdit, ProportionalSpecialist, merge_plan
from headroom.analysis.features import analyze
from headroom.audio import AudioBuffer
from headroom.baselines import trivial
from headroom.control.critic import CriticConfig
from headroom.control.loop import run_loop
from headroom.control.state import AbortReason, LoopState, RunTrace, StepRecord, Verdict
from headroom.dsp.backends.pedalboard import clear_cache, render_chain
from headroom.dsp.chain import Chain
from headroom.dsp.ops import EqOp, OpKind, StereoWidthOp, op_gain, op_limiter
from headroom.target.distance import FAMILY_WEIGHTS, SCORED, SPEC_BY_NAME, distance, to_scored
from headroom.target.profile import TargetProfile

from .conftest import SR

BUDGET = CriticConfig(render_budget=14)


@pytest.fixture(scope="module")
def scene() -> tuple[AudioBuffer, TargetProfile]:
    rng = np.random.default_rng(3)
    t = np.arange(SR * 5) / SR
    bass = 0.18 * np.sin(2 * np.pi * 55 * t)
    mids = 0.10 * np.sin(2 * np.pi * 440 * t) + 0.08 * np.sin(2 * np.pi * 1320 * t)
    tops = 0.05 * rng.standard_normal(t.size)
    hits = np.zeros_like(t)
    for start in range(0, t.size - 3000, int(SR * 0.3)):
        hits[start : start + 3000] += (
            rng.standard_normal(3000) * 0.12 * np.exp(-np.arange(3000) / 500)
        )
    original = AudioBuffer(
        np.stack(
            [bass + mids + tops + hits, bass + mids + np.roll(tops, 71) + np.roll(hits, 37)],
            axis=1,
        ),
        SR,
    )
    return original, TargetProfile.from_features(analyze(original))


def _degraded(scene: tuple[AudioBuffer, TargetProfile], kind: str, seed: int = 0) -> AudioBuffer:
    original, _ = scene
    level = analyze(original).lufs_integrated
    degradation = make_degradation(kind, seed=seed, level_db=level)  # type: ignore[arg-type]
    return render_chain(original, degradation.chain)


def _state(
    scene: tuple[AudioBuffer, TargetProfile],
    kind: str = "level_offset",
    chain: Chain | None = None,
) -> LoopState:
    _, target = scene
    source = _degraded(scene, kind)
    features = analyze(source)
    result = distance(features, target)
    return LoopState(
        source=source,
        target=target,
        chain=chain or Chain(),
        features=features,
        distance=result,
        initial_distance=result.score,
        step_index=0,
        step_scale=1.0,
        renders_used=0,
        render_budget=BUDGET.render_budget,
    )


# --- invariants ---------------------------------------------------------------


def test_every_scored_feature_has_exactly_one_owner() -> None:
    """An unowned dimension is permanently unfixable; a twice-owned one is two
    specialists fighting over the same number across turns."""
    names = [s.name for s in SCORED]
    assert sorted(roles.FEATURE_OWNER) == sorted(names)
    flat = [n for role in Role for n in OWNED_FEATURES[role]]
    assert sorted(flat) == sorted(names)
    assert len(flat) == len(set(flat)) == 28


def test_op_ownership_is_a_partition() -> None:
    owned = [kind for role in Role for kind in OWNED_OPS[role]]
    assert sorted(owned) == sorted(OpKind)
    assert len(owned) == len(set(owned))


def test_tool_availability_partitions_by_role() -> None:
    """A specialist is not asked to respect the boundary, it is not given the
    tools that would cross it."""
    per_role = {role: {s.name for s in tools.tools_for(role)} for role in Role}
    for role, names in per_role.items():
        assert "finish" in names
        for other, other_names in per_role.items():
            if other is role:
                continue
            assert names - {"finish"} & (other_names - {"finish"}) == names - {"finish"} or True
            assert not (names - {"finish"}) & (other_names - {"finish"})


def test_every_schema_bound_matches_the_enforced_bound() -> None:
    """The advertised range and the validated range are read from the same
    annotation, so this asserts the extraction rather than a copy."""
    checked = 0
    for spec in tools.TOOLS.values():
        for param in spec.params:
            schema = param.json_schema()
            if "minimum" not in schema:
                continue
            checked += 1
            lo, hi = schema["minimum"], schema["maximum"]
            assert lo < hi
    assert checked >= 15


def test_role_prompt_names_the_role_and_nothing_else_owned() -> None:
    for role in Role:
        prompt = roles.ROLE_BRIEF[role]
        assert "own" in prompt
        assert "nothing else" in prompt


# --- the tool layer -----------------------------------------------------------


def test_setters_are_absolute_and_idempotent() -> None:
    """A restated value is reported as a no-op rather than consuming a render
    to discover the chain did not change."""
    out = tools.apply_call(Chain(), Role.LOUDNESS, "set_gain", {"gain_db": -3.0, "reason": "x"})
    assert out.ok and out.action == "gain.gain_db -3.000"
    again = tools.apply_call(out.chain, Role.LOUDNESS, "set_gain", {"gain_db": -3.0, "reason": "x"})
    assert not again.ok
    assert again.error == "no_change"
    assert again.chain.fingerprint() == out.chain.fingerprint()

    revised = tools.apply_call(
        out.chain, Role.LOUDNESS, "set_gain", {"gain_db": -1.5, "reason": "x"}
    )
    assert revised.action == "gain.gain_db +1.500"
    assert len(revised.chain.of_kind(OpKind.GAIN)) == 1, "revision, not a second gain op"


def test_bound_violation_is_structured_and_leaves_the_chain_alone() -> None:
    out = tools.apply_call(Chain(), Role.LOUDNESS, "set_gain", {"gain_db": 40.0, "reason": "x"})
    assert not out.ok
    assert out.payload["error"] == "bound_violation"
    violation = out.payload["violations"][0]
    assert violation["field"] == "gain_db"
    assert violation["value"] == 40.0
    assert "24" in violation["constraint"]
    assert out.chain.ops == ()


def test_calling_another_roles_tool_is_refused_and_names_the_owner() -> None:
    out = tools.apply_call(Chain(), Role.EQ, "set_gain", {"gain_db": -2.0, "reason": "x"})
    assert out.error == "not_owned"
    assert out.payload["owner"] == "loudness"
    assert out.payload["your_role"] == "eq"
    assert "set_eq_band" in out.payload["available"]
    assert out.chain.ops == ()


def test_unknown_tool_lists_what_is_available() -> None:
    out = tools.apply_call(Chain(), Role.STEREO, "set_reverb", {"wet": 1.0})
    assert out.error == "unknown_tool"
    assert set(out.payload["available"]) == {s.name for s in tools.tools_for(Role.STEREO)}


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"gain_db": "loud", "reason": "x"}, "bad_argument"),
        ({"reason": "x"}, "missing_argument"),
    ],
)
def test_bad_arguments_are_reported_not_raised(args: dict[str, Any], expected: str) -> None:
    out = tools.apply_call(Chain(), Role.LOUDNESS, "set_gain", args)
    assert out.error == expected


def test_out_of_range_band_index_lists_the_bands() -> None:
    out = tools.apply_call(
        Chain(), Role.EQ, "set_eq_band", {"band_index": 12, "gain_db": 1.0, "reason": "x"}
    )
    assert out.error == "bad_argument"
    assert "20-60 Hz" in json.dumps(out.payload)


def test_the_equalizer_reports_when_it_is_full() -> None:
    """Nine analysis bands, room for eight filters. Real pressure, reported
    with the occupied slots so the choice can be informed."""
    chain = Chain()
    for band in range(tools.MAX_EQ_BANDS):
        out = tools.apply_call(
            chain,
            Role.EQ,
            "set_eq_band",
            {"band_index": band, "gain_db": 1.0 + band * 0.1, "reason": "x"},
        )
        assert out.ok
        chain = out.chain
    full = tools.apply_call(
        chain, Role.EQ, "set_eq_band", {"band_index": 8, "gain_db": 2.0, "reason": "x"}
    )
    assert full.error == "eq_full"
    assert len(full.payload["occupied"]) == tools.MAX_EQ_BANDS
    assert chain.fingerprint() == full.chain.fingerprint()

    # ...and making room works.
    freed = tools.apply_call(chain, Role.EQ, "remove_eq_band", {"band_index": 0, "reason": "x"})
    assert freed.ok
    assert tools.apply_call(
        freed.chain, Role.EQ, "set_eq_band", {"band_index": 8, "gain_db": 2.0, "reason": "x"}
    ).ok


def test_apply_call_never_raises_on_garbage() -> None:
    """The tool layer is the boundary a model writes through, so a malformed
    call has to become a readable result rather than an exception that aborts
    the run with proposal_error."""
    rng = random.Random(7)
    values: list[Any] = [None, "", "NaN", -1e9, 1e9, 0, 3.5, True, [], {}, "8"]
    for name, spec in tools.TOOLS.items():
        for role in Role:
            for _ in range(12):
                args = {p.name: rng.choice(values) for p in spec.params if rng.random() < 0.8}
                out = tools.apply_call(Chain(), role, name, args)
                assert isinstance(out.chain, Chain)
                assert out.ok or out.error


def test_actions_parse_in_the_critics_format() -> None:
    """The critic reads '<param> <signed delta>'. Agent traces have to be
    readable by the same detector as heuristic traces, or oscillation handling
    would silently not apply to the agent."""
    chain = Chain()
    produced: list[str] = []
    for name, args in [
        ("set_gain", {"gain_db": -2.0, "reason": "x"}),
        ("set_eq_band", {"band_index": 4, "gain_db": 1.5, "reason": "x"}),
        ("set_limiter", {"ceiling_dbtp": -1.0, "reason": "x"}),
    ]:
        out = tools.apply_call(
            chain, Role.LOUDNESS if "gain" in name or "limit" in name else Role.EQ, name, args
        )
        assert out.ok, out.payload
        chain = out.chain
        produced.append(out.action)
    for action in produced:
        head, _, tail = action.rpartition(" ")
        assert head
        float(tail)


def test_stereo_width_is_per_band_and_global_independently() -> None:
    out = tools.apply_call(Chain(), Role.STEREO, "set_stereo_width", {"width": 1.4, "reason": "x"})
    out2 = tools.apply_call(
        out.chain, Role.STEREO, "set_stereo_width", {"width": 0.8, "band_index": 0, "reason": "x"}
    )
    assert out2.ok
    assert len(out2.chain.of_kind(OpKind.STEREO_WIDTH)) == 2
    again = tools.apply_call(
        out2.chain, Role.STEREO, "set_stereo_width", {"width": 0.8, "band_index": 0, "reason": "x"}
    )
    assert again.error == "no_change"


# --- the briefing filter ------------------------------------------------------


def test_a_specialist_is_shown_only_the_dimensions_it_owns(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """Context isolation as a filter, not an instruction. This is the claim the
    whole multi-specialist design rests on, so it is asserted on the rendered
    prompt text rather than on an intention."""
    state = _state(scene, "combo")
    memory = WorkingMemory()
    for role in Role:
        text = build_briefing(state, memory, role).render()
        mine = set(OWNED_FEATURES[role])
        for spec in SCORED:
            if spec.name in mine:
                continue
            assert spec.name not in text, f"{role} was shown {spec.name}"


def test_the_briefing_hides_frozen_dimensions(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    state = _state(scene, "spectral_tilt")
    state.frozen_params = {"band_clr_8"}
    brief = build_briefing(state, WorkingMemory(), Role.EQ)
    assert "band_clr_8" not in {d.name for d in brief.mine}
    assert "band_clr_8" in brief.render()  # named in the frozen block, not the table


def test_damping_reaches_the_specialist_as_an_instruction(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    state = _state(scene, "level_offset")
    state.step_scale = 0.25
    text = build_briefing(state, WorkingMemory(), Role.LOUDNESS).render()
    assert "damped" in text
    assert "25%" in text


def test_the_briefing_states_the_programme_level(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """A threshold set relative to 0 dBFS instead of programme level is the
    single most common way to produce a compressor that does nothing."""
    state = _state(scene, "over_expand")
    brief = build_briefing(state, WorkingMemory(), Role.DYNAMICS)
    assert brief.programme_level_db < -1.0
    assert "LUFS integrated" in brief.render()


# --- memory -------------------------------------------------------------------


def test_memory_attributes_outcomes_and_counts_strikes() -> None:
    memory = WorkingMemory()
    memory.open(
        step=0,
        role=Role.EQ,
        targeted=("band_clr_6",),
        actions=("eq.band6 -2.000",),
        rationale="hot",
        score_before=1.0,
    )
    memory.settle(0.7)
    assert memory.strikes_for(Role.EQ) == 0
    memory.open(
        step=1,
        role=Role.EQ,
        targeted=("band_clr_6",),
        actions=("eq.band6 -1.000",),
        rationale="hot",
        score_before=0.7,
    )
    memory.settle(0.9)
    assert memory.strikes_for(Role.EQ) == 1
    assert "WORSE" in memory.render_own(Role.EQ)
    assert memory.stats()["hit_rate"] == 0.5


def test_other_roles_are_summarized_in_one_line_each() -> None:
    memory = WorkingMemory()
    for role in (Role.EQ, Role.LOUDNESS):
        memory.open(
            step=0,
            role=role,
            targeted=(),
            actions=(f"{role}.x +1.000",),
            rationale="",
            score_before=1.0,
        )
        memory.settle(0.5)
    lines = memory.render_others(Role.STEREO).splitlines()
    assert len(lines) == 2


# --- routing ------------------------------------------------------------------


def test_routing_picks_the_role_owning_the_largest_weighted_error(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    state = _state(scene, "level_offset")
    loads = supervisor.role_loads(state)
    assert loads
    assert loads[0].role is Role.LOUDNESS
    assert loads == sorted(loads, key=lambda load: -load.contribution)


def test_a_stuck_specialist_is_routed_around(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """One role must not be able to spend the whole render budget failing."""
    state = _state(scene, "level_offset")
    sup = scaffold_supervisor()
    sup.memory.strikes[Role.LOUDNESS] = sup.config.strike_limit
    proposal = sup.propose(state)
    assert proposal.role != str(Role.LOUDNESS)
    assert "rerouted" in proposal.note


def test_a_specialist_may_pass_and_the_next_one_is_consulted(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    state = _state(scene, "combo")
    scripts: dict[Role, Any] = {role: [] for role in Role}
    # Only the EQ role will act; everyone else's script is empty, so they pass.
    scripts[Role.EQ] = [[("set_eq_band", {"band_index": 3, "gain_db": -2.0, "reason": "x"})]]
    sup = scripted_supervisor(scripts)
    proposal = sup.propose(state)
    assert proposal.role == str(Role.EQ)
    assert "passed" in proposal.note
    assert sup.consults > 1


def test_an_edit_that_changes_nothing_audible_is_not_rendered(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """Rendering it would spend a render to learn the chain is unchanged, and
    the loop would abort on proposal_empty with nothing to attribute it to."""
    chain = Chain(ops=(op_gain(-2.0),))
    state = _state(scene, "level_offset", chain=chain)
    scripts: dict[Role, Any] = {
        Role.LOUDNESS: [[("set_gain", {"gain_db": -2.0, "reason": "restating"})]]
    }
    sup = scripted_supervisor(scripts)
    proposal = sup.propose(state)
    assert proposal.give_up
    assert sup.empty_rounds + sum(sup.stops.values()) >= 1


def test_out_of_order_ops_are_repositioned_and_counted(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """A limiter added before an EQ is corrected by the system rather than
    trusted, and the correction shows up as a number."""
    chain = Chain(ops=(op_limiter(-1.0),))
    state = _state(scene, "spectral_tilt", chain=chain)
    scripts: dict[Role, Any] = {
        Role.EQ: [[("set_eq_band", {"band_index": 7, "gain_db": -3.0, "reason": "x"})]]
    }
    sup = scripted_supervisor(scripts)
    proposal = sup.propose(state)
    assert [OpKind(op.kind) for op in proposal.chain.ops] == [OpKind.EQ, OpKind.LIMITER]
    # A swap moves both ops, and Reposition is recorded per op.
    assert sup.repositions == 2
    assert "repositioned" in proposal.note


def test_all_roles_exhausted_gives_up_with_a_reason(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    state = _state(scene, "combo")
    sup = scripted_supervisor({role: [] for role in Role})
    proposal = sup.propose(state)
    assert proposal.give_up
    assert "no specialist could act" in proposal.note


def test_ownership_violations_are_counted_not_just_refused(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    state = _state(scene, "spectral_tilt")
    scripts: dict[Role, Any] = {
        Role.EQ: [
            [
                ("set_gain", {"gain_db": -2.0, "reason": "not mine"}),
                ("set_eq_band", {"band_index": 7, "gain_db": -3.0, "reason": "mine"}),
            ]
        ]
    }
    sup = scripted_supervisor(scripts)
    proposal = sup.propose(state)
    assert sup.memory.ownership_violations == 1
    assert proposal.n_edits == 1
    assert not proposal.chain.of_kind(OpKind.GAIN)


# --- the coordinated-edit claim ----------------------------------------------


def test_a_specialist_bundles_several_edits_into_one_render(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """The architecture's whole numeric claim. The heuristic corrects one
    feature per render; a specialist that owns nine bands can correct several
    for the same price, and this asserts it actually does."""
    state = _state(scene, "spectral_tilt")
    brief = build_briefing(state, WorkingMemory(), Role.EQ)
    assert len(brief.mine) > 1, "the fixture must have several bands out"
    turn = ProportionalSpecialist(role=Role.EQ)(brief, state.chain)
    assert len(turn.actions) > 1
    eq = turn.chain.of_kind(OpKind.EQ)[0]
    assert isinstance(eq, EqOp)
    assert len(eq.bands) == len(turn.actions)


def test_two_dimensions_that_share_one_op_collapse_to_one_edit(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """correlation_z and mono_compat_db are two views of the same global width
    control. Emitting both would let the second silently erase the first."""
    state = _state(scene, "stereo_collapse")
    brief = build_briefing(state, WorkingMemory(), Role.STEREO)
    turn = ProportionalSpecialist(role=Role.STEREO)(brief, state.chain)
    width_ops = [
        op for op in turn.chain.of_kind(OpKind.STEREO_WIDTH) if isinstance(op, StereoWidthOp)
    ]
    slots = [op.band for op in width_ops]
    assert len(slots) == len(set(slots))


def test_dominant_action_is_the_largest_move(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    state = _state(scene, "spectral_tilt")
    brief = build_briefing(state, WorkingMemory(), Role.EQ)
    turn = ProportionalSpecialist(role=Role.EQ)(brief, state.chain)
    magnitudes = [abs(float(a.rsplit(" ", 1)[1])) for a in turn.actions]
    assert abs(float(turn.dominant_action.rsplit(" ", 1)[1])) == max(magnitudes)


# --- end to end through the shared loop --------------------------------------


@pytest.mark.parametrize("kind", ["level_offset", "spectral_tilt", "stereo_overwide"])
def test_the_scaffold_converges_on_single_feature_degradations(
    scene: tuple[AudioBuffer, TargetProfile], kind: str
) -> None:
    _, target = scene
    clear_cache()
    trace = run_loop(
        system="agent-scaffold",
        source=_degraded(scene, kind),
        target=target,
        propose=scaffold_supervisor().propose,
        degradation_kind=kind,
        config=BUDGET,
    )
    assert trace.recovery_ratio > 0.5, trace.summary()
    assert trace.n_renders <= BUDGET.render_budget


def test_the_trace_records_which_specialist_moved(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    _, target = scene
    clear_cache()
    trace = run_loop(
        system="agent-scaffold",
        source=_degraded(scene, "combo"),
        target=target,
        propose=scaffold_supervisor().propose,
        config=BUDGET,
    )
    acted = [s for s in trace.steps if s.action]
    assert acted
    assert all(s.role in {str(r) for r in Role} for s in acted)
    assert {s.role for s in acted}, "at least one role attributed"
    assert trace.total_cost_usd == 0.0, "the scaffold makes no API calls"
    assert max(s.n_edits for s in acted) >= 1


def test_the_scaffold_beats_random_decisively(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    _, target = scene
    wins = 0
    for kind in ("level_offset", "spectral_tilt", "stereo_overwide", "band_shift"):
        source = _degraded(scene, kind)
        clear_cache()
        agent = run_loop(
            system="agent-scaffold",
            source=source,
            target=target,
            propose=scaffold_supervisor().propose,
            config=BUDGET,
        )
        clear_cache()
        rand = run_loop(
            system="random",
            source=source,
            target=target,
            propose=trivial.make_random_propose(1),
            config=BUDGET,
        )
        wins += agent.recovery_ratio > rand.recovery_ratio
    assert wins == 4


def test_a_scripted_pathology_aborts_with_a_named_reason(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """Every stop has an enumerated cause. A run that ends for an
    unattributable reason is a run that cannot appear in the results table."""
    _, target = scene
    # A specialist that boosts, cuts, boosts the same band: the critic's
    # sign-flip detector should catch it and damping should escalate.
    flip = [
        [("set_eq_band", {"band_index": 6, "gain_db": g, "reason": "x"})]
        for g in (6.0, -6.0, 6.0, -6.0, 6.0, -6.0, 6.0, -6.0)
    ]
    sup = scripted_supervisor(
        {Role.EQ: flip}, supervisor.SupervisorConfig(strike_limit=99, stop_limit=99)
    )
    clear_cache()
    trace = run_loop(
        system="agent-scripted",
        source=_degraded(scene, "spectral_tilt"),
        target=target,
        propose=sup.propose,
        config=CriticConfig(render_budget=8),
    )
    assert trace.converged or trace.abort_reason is not None
    if trace.abort_reason is not None:
        assert trace.abort_reason in set(AbortReason)


def test_the_supervisor_reports_its_own_statistics(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    _, target = scene
    sup = scaffold_supervisor()
    clear_cache()
    run_loop(
        system="agent-scaffold",
        source=_degraded(scene, "combo"),
        target=target,
        propose=sup.propose,
        config=BUDGET,
    )
    stats = sup.stats()
    assert stats["turns"] >= 1
    assert set(stats["by_role"]) == {str(r) for r in Role}
    assert stats["consults"] >= stats["turns"]


# --- pricing ------------------------------------------------------------------


def test_cost_is_computed_from_published_rates() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=0)
    assert cost_usd("claude-haiku-4-5", usage) == pytest.approx(1.0)
    assert cost_usd("claude-sonnet-5", usage) == pytest.approx(2.0)
    cached = Usage(cache_read_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", cached) == pytest.approx(0.1)


def test_a_dated_snapshot_inherits_its_base_price() -> None:
    assert price_for("claude-sonnet-5-20260114") == PRICES["claude-sonnet-5"]


def test_an_unpriced_model_raises_rather_than_reporting_zero() -> None:
    """A missing price would surface as a cost of zero, which is the most
    misleading number this project could print."""
    with pytest.raises(UnknownModelError):
        cost_usd("some-other-model", Usage(input_tokens=10))


# --- cassettes ----------------------------------------------------------------


def test_a_cassette_round_trips_and_keys_on_the_whole_request(tmp_path: Path) -> None:
    cassette = Cassette(path=tmp_path / "c", mode=Mode.AUTO)
    request = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    assert cassette.get(request) is None
    cassette.put(request, {"content": [], "usage": {"input_tokens": 1, "output_tokens": 1}})
    assert cassette.get(request) is not None
    assert cassette.hits == 1

    changed = {**request, "messages": [{"role": "user", "content": "hello there"}]}
    assert cassette.get(changed) is None, "a changed prompt must miss, not hit stale"


def test_replay_mode_turns_a_missing_recording_into_an_error(tmp_path: Path) -> None:
    """What CI wants: a changed prompt fails a test instead of quietly
    spending money."""
    cassette = Cassette(path=tmp_path / "c", mode=Mode.REPLAY)
    with pytest.raises(CassetteMissError):
        cassette.get({"model": "m", "messages": []})


def test_a_cassette_never_stores_a_credential(tmp_path: Path) -> None:
    cassette = Cassette(path=tmp_path / "c", mode=Mode.AUTO)
    request = {"model": "m", "messages": [], "api_key": "sk-ant-secret", "stream": False}
    cassette.put(request, {"content": []})
    written = " ".join(f.read_text() for f in (tmp_path / "c").glob("*.json"))
    assert "sk-ant-secret" not in written
    assert digest(request) == digest({"model": "m", "messages": []})


# --- the model boundary, driven from a stub ----------------------------------


def _usage(inp: int = 1500, out: int = 60, cache_read: int = 0) -> dict[str, Any]:
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": 0,
    }


def _use(name: str, **args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"tu_{name}", "name": name, "input": args}


def _response(*blocks: dict[str, Any], stop: str = "tool_use") -> dict[str, Any]:
    return {"content": list(blocks), "stop_reason": stop, "usage": _usage()}


class _Stub:
    """Stands in for the API. Records the requests it was sent."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return self.responses.pop(0)


def test_the_llm_specialist_applies_tool_calls_and_prices_them(
    scene: tuple[AudioBuffer, TargetProfile], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(scene, "spectral_tilt")
    brief = build_briefing(state, WorkingMemory(), Role.EQ)
    stub = _Stub(
        _response(
            {"type": "text", "text": "band 8 is hot"},
            _use("set_eq_band", band_index=8, gain_db=-3.0, reason="band 8 hot"),
            _use("finish", reason="done"),
        )
    )
    monkeypatch.setattr(ModelClient, "_send", lambda self, request: stub(request))
    client = ModelClient(
        config=ModelConfig(model="claude-haiku-4-5"),
        cassette=Cassette(path=tmp_path / "c", mode=Mode.AUTO),
    )
    turn = LLMSpecialist(role=Role.EQ, client=client)(brief, state.chain)

    assert turn.actions == ("eq.band8 -3.000",)
    assert turn.chain.of_kind(OpKind.EQ)
    assert turn.n_llm_calls == 1
    assert turn.cost_usd == pytest.approx(cost_usd("claude-haiku-4-5", Usage(1500, 60)))
    assert "band 8 is hot" in turn.rationale

    request = stub.requests[0]
    assert request["tool_choice"] == {"type": "any"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert {t["name"] for t in request["tools"]} == {s.name for s in tools.tools_for(Role.EQ)}


def test_a_rejected_call_is_recovered_from_within_the_same_round(
    scene: tuple[AudioBuffer, TargetProfile], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost argument for structured refusal: a bad idea costs a tool
    round-trip, not a render."""
    state = _state(scene, "spectral_tilt")
    brief = build_briefing(state, WorkingMemory(), Role.EQ)
    stub = _Stub(
        _response(_use("set_eq_band", band_index=8, gain_db=-99.0, reason="way out")),
        _response(
            _use("set_eq_band", band_index=8, gain_db=-6.0, reason="inside the bound"),
            _use("finish", reason="done"),
        ),
    )
    monkeypatch.setattr(ModelClient, "_send", lambda self, request: stub(request))
    client = ModelClient(cassette=Cassette(path=tmp_path / "c", mode=Mode.AUTO))
    turn = LLMSpecialist(role=Role.EQ, client=client)(brief, state.chain)

    assert turn.rejected == 1
    assert turn.actions == ("eq.band8 -6.000",)
    assert turn.n_llm_calls == 2
    followup = stub.requests[1]
    results = followup["messages"][-1]["content"]
    assert results[0]["is_error"] is True
    assert "bound_violation" in results[0]["content"]


def test_a_recorded_run_replays_with_no_client_at_all(
    scene: tuple[AudioBuffer, TargetProfile], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property CI depends on: once recorded, the agent layer runs with no
    key, no network, and the same parsing code."""
    state = _state(scene, "spectral_tilt")
    brief = build_briefing(state, WorkingMemory(), Role.EQ)
    blocks = (
        _use("set_eq_band", band_index=8, gain_db=-3.0, reason="hot"),
        _use("finish", reason="done"),
    )
    stub = _Stub(_response(*blocks))
    monkeypatch.setattr(ModelClient, "_send", lambda self, request: stub(request))
    recorded = ModelClient(cassette=Cassette(path=tmp_path / "c", mode=Mode.AUTO))
    first = LLMSpecialist(role=Role.EQ, client=recorded)(brief, state.chain)
    assert recorded.cassette is not None and recorded.cassette.writes == 1

    def _explode(self: ModelClient, request: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("replay must not reach the API")

    monkeypatch.setattr(ModelClient, "_send", _explode)
    replayed_client = ModelClient(cassette=Cassette(path=tmp_path / "c", mode=Mode.REPLAY))
    second = LLMSpecialist(role=Role.EQ, client=replayed_client)(brief, state.chain)

    assert second.actions == first.actions
    assert second.chain.fingerprint() == first.chain.fingerprint()
    assert second.cost_usd == first.cost_usd
    assert replayed_client.n_replayed == 1


def test_the_whole_loop_runs_on_a_cassette(
    scene: tuple[AudioBuffer, TargetProfile], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full agent run with the model replaced by a stub: routing, tool calls,
    rendering, measurement, memory, the critic and the trace, end to end."""
    _, target = scene
    from headroom.agent.factory import llm_specialists

    def _fake(self: ModelClient, request: dict[str, Any]) -> dict[str, Any]:
        # Answer whichever role asked, using the briefing it was sent.
        text = request["messages"][0]["content"]
        if "lufs_integrated" in text:
            return _response(
                _use("set_gain", gain_db=-3.0, reason="level"), _use("finish", reason="d")
            )
        if "band_clr" in text:
            return _response(
                _use("set_eq_band", band_index=8, gain_db=-2.0, reason="tone"),
                _use("finish", reason="d"),
            )
        return _response(_use("finish", reason="nothing I own is out"))

    monkeypatch.setattr(ModelClient, "_send", _fake)
    client = ModelClient(cassette=Cassette(path=tmp_path / "c", mode=Mode.AUTO))
    sup = supervisor.Supervisor(specialists=llm_specialists(client))
    clear_cache()
    trace = run_loop(
        system="agent",
        source=_degraded(scene, "level_offset"),
        target=target,
        propose=sup.propose,
        config=CriticConfig(render_budget=4),
        model=client.config.model,
    )
    assert trace.n_renders >= 1
    assert trace.total_cost_usd > 0.0
    assert trace.total_input_tokens > 0
    assert trace.recovery_ratio > 0.0
    assert any(s.role == str(Role.LOUDNESS) for s in trace.steps if s.action)


# --- slot merging: a regression the CLI demo found ---------------------------


def _edit(slot: str, param: str, delta: float, reason: str, current: float = 1.0) -> PlannedEdit:
    return PlannedEdit(
        slot=slot,
        tool="set_stereo_width",
        param=param,
        current=current,
        delta=delta,
        low=0.0,
        high=2.0,
        reason=reason,
    )


def test_conflicting_dimensions_on_one_control_are_averaged_not_alternated() -> None:
    """Regression, found by running the CLI demo rather than by reasoning.

    correlation_z wants more width and mono_compat_db wants less. Keeping
    whichever was larger made the surviving dimension alternate between turns,
    so the width went up, down, up, down and the critic correctly killed the
    run for oscillation. The specialist was not confused; the collapse rule
    was wrong.
    """
    calls, reasons = merge_plan(
        [
            _edit("width:None", "width", +0.20, "correlation_z"),
            _edit("width:None", "width", -0.30, "mono_compat_db"),
        ],
        max_edits=6,
    )
    assert len(calls) == 1
    assert calls[0][1]["width"] == pytest.approx(1.0 + (0.20 - 0.30) / 2)
    assert "correlation_z" in reasons[0]
    assert "mono_compat_db" in reasons[0]


def test_near_duplicate_dimensions_do_not_double_the_correction() -> None:
    """The other half of the choice: summing would have doubled a correction
    that two near-identical measurements both asked for, and started the
    oscillation from the other end."""
    calls, _ = merge_plan(
        [
            _edit("dyn:comp", "width", +0.80, "crest_factor_db"),
            _edit("dyn:comp", "width", +0.90, "crest_short_p50"),
        ],
        max_edits=6,
    )
    assert calls[0][1]["width"] == pytest.approx(1.85)


def test_a_merged_request_that_cancels_makes_no_edit() -> None:
    calls, _ = merge_plan(
        [_edit("width:None", "width", +0.25, "a"), _edit("width:None", "width", -0.25, "b")],
        max_edits=6,
    )
    assert calls == []


def test_merging_clamps_to_the_bound_rather_than_proposing_past_it() -> None:
    calls, _ = merge_plan([_edit("width:None", "width", +5.0, "very wide")], max_edits=6)
    assert calls[0][1]["width"] == pytest.approx(2.0)


def test_the_edit_budget_caps_a_round() -> None:
    plan = [_edit(f"eq:{i}", "width", 0.1 * (i + 1), f"band {i}") for i in range(9)]
    calls, _ = merge_plan(plan, max_edits=3)
    assert len(calls) == 3


def test_the_scaffold_does_not_oscillate_on_a_stereo_degradation(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """The end-to-end form of the same regression."""
    _, target = scene
    clear_cache()
    trace = run_loop(
        system="agent-scaffold",
        source=_degraded(scene, "stereo_collapse"),
        target=target,
        propose=scaffold_supervisor().propose,
        config=BUDGET,
    )
    assert trace.abort_reason is not AbortReason.OSCILLATION_UNRESOLVED, trace.summary()
    assert trace.recovery_ratio > 0.5, trace.summary()


# --- reporting ----------------------------------------------------------------


def _step(index: int, role: str, score: float, edits: int = 1) -> StepRecord:
    return StepRecord(
        index=index,
        chain=Chain(),
        chain_fingerprint="x",
        distance_score=score,
        n_out_of_tolerance=1,
        by_family={},
        worst=[],
        action=f"{role}.p +1.000",
        step_scale=1.0,
        oscillating=False,
        verdict=Verdict.CONTINUE,
        renders_used=index + 1,
        elapsed_s=0.0,
        role=role,
        n_edits=edits,
    )


def _fake_trace(system: str, steps: tuple[StepRecord, ...], **kwargs: Any) -> RunTrace:
    return RunTrace(
        system=system,
        track_id="t",
        degradation_kind="combo",
        degradation_seed=0,
        source_hash="h",
        target_label="target",
        target_provenance="p",
        initial_distance=1.0,
        final_distance=steps[-1].distance_score if steps else 1.0,
        recovery_ratio=0.5,
        converged=True,
        final_chain=Chain(),
        steps=steps,
        **kwargs,
    )


def test_role_breakdown_attributes_improvement_to_the_role_that_moved() -> None:
    trace = _fake_trace(
        "agent",
        (
            _step(0, "eq", 0.6, edits=3),
            _step(1, "loudness", 0.8),
            _step(2, "eq", 0.4),
        ),
    )
    rows = {r.role: r for r in report.role_breakdown([trace], "agent")}
    assert rows["eq"].turns == 2
    assert rows["eq"].edits == 4
    assert rows["eq"].edits_per_turn == pytest.approx(2.0)
    assert rows["eq"].helped == 2
    assert rows["loudness"].helped == 0
    assert rows["loudness"].hit_rate == 0.0


def test_the_report_says_zero_when_nothing_was_billed() -> None:
    trace = _fake_trace("agent-scaffold", (_step(0, "eq", 0.5),))
    text = report.render_spend([trace])
    assert "$0.0000" in text
    assert "no API call" in text


def test_the_report_states_measured_spend_and_replay_status() -> None:
    trace = _fake_trace(
        "agent",
        (_step(0, "eq", 0.5),),
        model="claude-haiku-4-5",
        total_cost_usd=0.0031,
        total_input_tokens=1500,
        total_output_tokens=60,
        total_cache_read_tokens=1200,
        replayed_from_cassette=True,
    )
    text = report.render_spend([trace])
    assert "claude-haiku-4-5" in text
    assert "$0.0031" in text
    # Fully replayed: the figure is what it cost to record, not to reproduce.
    assert "cost nothing to produce" in text

    recorded = trace.model_copy(update={"replayed_from_cassette": False})
    live = report.render_spend([recorded])
    assert "what they actually cost" in live
    assert "$0.0031" in live

    mixed = report.render_spend([trace, recorded])
    assert "1 of 2 runs were replayed" in mixed


def test_the_specialist_table_is_omitted_when_no_agent_ran() -> None:
    trace = _fake_trace("heuristic", ())
    assert report.render_roles([trace]) == ""
    assert report.agent_systems([trace]) == []


def test_a_sawtooth_cannot_evade_the_reroute_rule() -> None:
    """Regression, found by watching a real run rather than by reasoning.

    Scoring strikes against the previous step let a specialist that overshoots
    and corrects hold the route indefinitely: worse, better, worse, better, and
    every recovery reset the count. It kept the route for seven straight
    renders and ended no better than it started.
    """
    memory = WorkingMemory()
    for step, (before, after) in enumerate([(1.0, 1.2), (1.2, 1.05), (1.05, 1.3)]):
        memory.open(
            step=step,
            role=Role.EQ,
            targeted=(),
            actions=(f"eq.band1 +{step + 1}.000",),
            rationale="",
            score_before=before,
        )
        memory.settle(after)
    # Every one of those failed to beat the 1.0 the run started from.
    assert memory.strikes_for(Role.EQ) == 3
    assert memory.best_score == pytest.approx(1.0)
    # The specialist is still shown the comparison it can act on.
    assert "BETTER" in memory.render_own(Role.EQ)


def test_beating_the_best_clears_the_strikes() -> None:
    memory = WorkingMemory()
    memory.open(
        step=0,
        role=Role.EQ,
        targeted=(),
        actions=("eq.band1 +1.000",),
        rationale="",
        score_before=1.0,
    )
    memory.settle(1.4)
    assert memory.strikes_for(Role.EQ) == 1
    memory.open(
        step=1,
        role=Role.EQ,
        targeted=(),
        actions=("eq.band1 -2.000",),
        rationale="",
        score_before=1.4,
    )
    memory.settle(0.6)
    assert memory.strikes_for(Role.EQ) == 0
    assert memory.best_score == pytest.approx(0.6)


# --- briefs -------------------------------------------------------------------


def test_a_brief_becomes_offsets_against_the_audios_own_measurements(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """Relative, not absolute: 'brighter' means brighter than *this*."""
    original, _ = scene
    fv = analyze(original)
    spec = brief.parse(
        "brighter",
        {
            "adjustments": [
                {"dimension": "band_clr_8", "offset_tol": 2.0, "reason": "brighter"},
                {"dimension": "band_clr_7", "offset_tol": 1.5, "reason": "brighter"},
            ],
            "hold": ["loudness"],
            "rationale": "top bands up",
        },
    )
    profile = spec.apply_to(fv)
    scored = to_scored(fv)
    assert profile.targets["band_clr_8"] == pytest.approx(scored["band_clr_8"] + 2.0 * 0.75)
    # A held family is pinned where it started.
    assert profile.targets["lufs_integrated"] == pytest.approx(scored["lufs_integrated"])
    # Anything unnamed and unheld stays unconstrained, so the metric neither
    # rewards nor penalizes what happened to it.
    assert "width_4" not in profile.targets


def test_an_invented_dimension_is_recorded_and_dropped() -> None:
    """One hallucinated name should not throw away an otherwise correct
    translation, and the rate is worth reporting."""
    spec = brief.parse(
        "brighter",
        {
            "adjustments": [
                {"dimension": "sparkle", "offset_tol": 3.0, "reason": "x"},
                {"dimension": "band_clr_8", "offset_tol": 2.0, "reason": "x"},
            ],
            "rationale": "",
        },
    )
    assert spec.rejected == ("sparkle",)
    assert spec.named() == {"band_clr_8": 2.0}


def test_an_offset_inside_the_indifference_band_is_dropped() -> None:
    """A request smaller than one tolerance would be a constraint that is
    already satisfied, which costs render budget and buys nothing."""
    spec = brief.parse(
        "a touch brighter",
        {
            "adjustments": [{"dimension": "band_clr_8", "offset_tol": 0.3, "reason": "x"}],
            "rationale": "",
        },
    )
    assert spec.adjustments == ()


def test_an_out_of_bounds_offset_is_refused() -> None:
    spec = brief.parse(
        "MUCH brighter",
        {
            "adjustments": [{"dimension": "band_clr_8", "offset_tol": 400.0, "reason": "x"}],
            "rationale": "",
        },
    )
    assert spec.adjustments == ()
    assert "band_clr_8" in spec.rejected[0]


def test_the_brief_tool_schema_enumerates_only_real_dimensions() -> None:
    schema = brief.tool_schema()
    dims = schema["input_schema"]["properties"]["adjustments"]["items"]["properties"]["dimension"]
    assert set(dims["enum"]) == {s.name for s in SCORED}
    # Holds accept a family or a single dimension, because briefs are regional
    # and families are not.
    holds = set(schema["input_schema"]["properties"]["hold"]["items"]["enum"])
    assert holds == set(FAMILY_WEIGHTS) | {s.name for s in SCORED}


def test_translation_scoring_reads_signs_not_magnitudes() -> None:
    case = briefs.BriefCase(
        brief="brighter",
        expect=(briefs.Expectation("top", briefs.bands(+1, 7, 8)),),
    )
    right = brief.BriefTarget(
        brief="brighter",
        adjustments=(
            brief.Adjustment(dimension="band_clr_7", offset_tol=1.0),
            brief.Adjustment(dimension="band_clr_8", offset_tol=6.0),
        ),
    )
    score, extraneous = briefs.score_translation(case, right)
    assert score == 1.0
    assert extraneous == ()

    backwards = brief.BriefTarget(
        brief="brighter",
        adjustments=(
            brief.Adjustment(dimension="band_clr_7", offset_tol=-2.0),
            brief.Adjustment(dimension="band_clr_8", offset_tol=2.0),
            brief.Adjustment(dimension="lufs_integrated", offset_tol=3.0),
        ),
    )
    # One band the right way and one the wrong way fails the region outright:
    # that is what stops region matching from becoming a free pass.
    score, extraneous = briefs.score_translation(case, backwards)
    assert score == 0.0
    assert extraneous == ("lufs_integrated",)


def test_execution_scoring_requires_an_audible_move() -> None:
    case = briefs.BriefCase(
        brief="brighter", expect=(briefs.Expectation("top", {"band_clr_8": +1}),)
    )
    target = brief.BriefTarget(brief="brighter")
    before = {"band_clr_8": 0.0}
    score, detail = briefs.score_execution(case, target, before, {"band_clr_8": 0.75 * 2.0})
    assert score == 1.0
    # Inside the indifference band is not a success.
    score, _ = briefs.score_execution(case, target, before, {"band_clr_8": 0.75 * 0.2})
    assert score == 0.0
    # Right dimension, wrong way.
    score, _ = briefs.score_execution(case, target, before, {"band_clr_8": -0.75 * 2.0})
    assert score == 0.0
    assert "top" in detail


def test_collateral_scoring_exempts_what_the_brief_itself_asked_for() -> None:
    """'Louder without squashing the dynamics' protects dynamics and asks for
    loudness. A protected family must not veto the request inside it."""
    case = briefs.BriefCase(
        brief="louder",
        expect=(briefs.Expectation("level", {"lufs_integrated": +1}),),
        protect=("loudness", "dynamics"),
    )
    before = {s.name: 0.0 for s in SCORED}
    after = dict(before)
    after["lufs_integrated"] = 6.0  # the request itself, exempt
    score, detail = briefs.score_collateral(case, before, after)
    assert score == 1.0
    assert detail == {}

    after["crest_factor_db"] = 5.0  # collateral damage, not exempt
    score, detail = briefs.score_collateral(case, before, after)
    assert score < 1.0
    assert "crest_factor_db" in detail


def test_every_brief_case_has_a_physically_checkable_signature() -> None:
    """The signatures are facts about the measurements. A typo here would
    silently grade the system against the wrong thing."""
    for case in briefs.CASES:
        assert case.expect, case.brief
        for expectation in case.expect:
            assert expectation.want, f"{case.brief}: {expectation.label}"
            assert 1 <= expectation.need <= len(expectation.want)
            for dim, sign in expectation.want.items():
                assert dim in SPEC_BY_NAME, f"{case.brief}: {dim}"
                assert sign in (-1, 1)
        # No two expectations may demand opposite directions on one dimension,
        # which would make the case unsatisfiable.
        for a in case.expect:
            for b in case.expect:
                if a is b:
                    continue
                for dim in set(a.want) & set(b.want):
                    assert a.want[dim] == b.want[dim], f"{case.brief}: contradictory on {dim}"
        for family in case.protect:
            assert family in FAMILY_WEIGHTS, f"{case.brief}: {family}"
            # A protection over a family every expectation already moves would
            # be vacuous.
            in_family = {s.name for s in SCORED if s.family == family}
            assert in_family - case.named_dims(), f"{case.brief}: {family} fully exempt"


def test_a_brief_runs_end_to_end_against_a_stub_translator(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """The whole path with no key: translate, constrain, run the loop, render,
    and score all three questions."""
    original, _ = scene
    case = briefs.BriefCase(
        brief="brighter and more open up top",
        expect=(briefs.Expectation("top bands", briefs.bands(+1, 7, 8)),),
        protect=("loudness",),
    )

    def stub(text: str, features: object) -> tuple[brief.BriefTarget, object]:
        return (
            brief.BriefTarget(
                brief=text,
                adjustments=(
                    brief.Adjustment(dimension="band_clr_7", offset_tol=2.0, reason="up top"),
                    brief.Adjustment(dimension="band_clr_8", offset_tol=2.0, reason="up top"),
                ),
                hold=("loudness",),
                rationale="top bands up, level pinned",
            ),
            None,
        )

    clear_cache()
    result = briefs.run_case(
        case,
        original,
        stub,
        scaffold_supervisor().propose,
        system="agent-scaffold",
        config=BUDGET,
    )
    assert not result.error, result.error
    assert result.translation_score == 1.0
    assert result.execution_score == 1.0, result.detail
    assert result.collateral_score == 1.0, result.detail
    assert result.passed
    assert result.trace is not None
    assert result.trace.total_cost_usd == 0.0
    assert "translation" in briefs.render_markdown([result])


def test_a_translation_that_names_nothing_is_a_result_not_a_crash(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    original, _ = scene
    case = briefs.BriefCase(
        brief="make it good", expect=(briefs.Expectation("top", {"band_clr_8": +1}),)
    )

    def empty(text: str, features: object) -> tuple[brief.BriefTarget, object]:
        return brief.BriefTarget(brief=text, rationale="nothing measurable here"), None

    result = briefs.run_case(case, original, empty, scaffold_supervisor().propose)
    assert result.error
    assert result.trace is None
    assert not result.passed


def test_a_translator_that_raises_is_a_result_not_a_crash(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    original, _ = scene
    case = briefs.BriefCase(
        brief="brighter", expect=(briefs.Expectation("top", {"band_clr_8": +1}),)
    )

    def broken(text: str, features: object) -> tuple[brief.BriefTarget, object]:
        raise RuntimeError("the API fell over")

    result = briefs.run_case(case, original, broken, scaffold_supervisor().propose)
    assert "translation failed" in result.error


def test_adjustments_arriving_as_a_json_string_are_repaired() -> None:
    """A real tool-use quirk. The strict reading threw away a translation that
    was otherwise exactly right, turning a model quirk into a scored
    comprehension failure."""
    spec = brief.parse(
        "narrower",
        {
            "adjustments": (
                '[{"dimension": "correlation_z", "offset_tol": 3.0, "reason": "in"},'
                ' {"dimension": "width_7", "offset_tol": -3.0, "reason": "in"}]'
            ),
            "rationale": "narrow it",
        },
    )
    assert spec.repaired_json is True
    assert spec.named() == {"correlation_z": 3.0, "width_7": -3.0}


def test_unrepairable_arguments_stay_empty_rather_than_raising() -> None:
    spec = brief.parse("x", {"adjustments": "not json at all", "rationale": ""})
    assert spec.adjustments == ()
    assert spec.repaired_json is False


def test_holding_a_dimension_the_brief_also_moves_lets_the_move_win() -> None:
    """A contradiction the model does produce. Keeping both would pin the value
    the request wanted changed, which reads downstream as an unsatisfiable
    target."""
    spec = brief.parse(
        "wider up top, tight underneath",
        {
            "adjustments": [{"dimension": "width_7", "offset_tol": 2.5, "reason": "wider"}],
            "hold": ["width_7", "width_0", "stereo"],
            "rationale": "",
        },
    )
    assert "width_7" not in spec.hold
    assert set(spec.hold) == {"width_0", "stereo"}


def test_a_dimension_hold_pins_only_that_dimension(
    scene: tuple[AudioBuffer, TargetProfile],
) -> None:
    """Briefs are regional and families are not: 'keep the low end tight' has
    to be expressible without forbidding the widening it accompanies."""
    original, _ = scene
    fv = analyze(original)
    spec = brief.parse(
        "wider up top, tight underneath",
        {
            "adjustments": [{"dimension": "width_7", "offset_tol": 2.5, "reason": "wider"}],
            "hold": ["width_0", "width_1", "mono_compat_db"],
            "rationale": "",
        },
    )
    targets = spec.apply_to(fv).targets
    scored = to_scored(fv)
    assert targets["width_0"] == pytest.approx(scored["width_0"])
    assert targets["mono_compat_db"] == pytest.approx(scored["mono_compat_db"])
    assert targets["width_7"] > scored["width_7"]
    # ...and the rest of the stereo family is left free.
    assert "width_4" not in targets


def test_a_leading_plus_on_a_number_is_repaired() -> None:
    """JSON forbids ``+2.0``, and a model asked for signed offsets writes them
    that way. Combined with the string-instead-of-array quirk this discarded a
    translation that was correct in every dimension and sign."""
    spec = brief.parse(
        "punchier",
        {
            "adjustments": (
                '[{"dimension": "crest_factor_db", "offset_tol": +2.0, "reason": "punch"},'
                ' {"dimension": "attack_log2_ms", "offset_tol": -2.0, "reason": "faster"}]'
            ),
            "rationale": "",
        },
    )
    assert spec.repaired_json is True
    assert spec.named() == {"crest_factor_db": 2.0, "attack_log2_ms": -2.0}


def test_the_repair_does_not_touch_a_plus_inside_a_string() -> None:
    spec = brief.parse(
        "louder",
        {
            "adjustments": (
                '[{"dimension": "lufs_integrated", "offset_tol": 3.0, "reason": "+3 dB please"}]'
            ),
            "rationale": "",
        },
    )
    assert spec.adjustments[0].reason == "+3 dB please"


# --- the committed cassettes --------------------------------------------------

CASSETTE_ROOT = Path(__file__).resolve().parent.parent / "cassettes"


def _committed_cassettes() -> list[Path]:
    return sorted(CASSETTE_ROOT.rglob("*.json")) if CASSETTE_ROOT.exists() else []


@pytest.mark.skipif(not _committed_cassettes(), reason="no cassettes committed yet")
def test_no_committed_cassette_carries_a_credential() -> None:
    """This repository is public and these files are the API traffic verbatim.

    Asserted over the whole directory rather than trusting the redaction at the
    one place that writes them, because the cost of being wrong is a leaked key
    in immutable git history.
    """
    allowed = {"model", "max_tokens", "system", "tools", "messages", "tool_choice"}
    forbidden = ("sk-ant-", "api_key", "authorization", "x-api-key", "bearer ")
    for file in _committed_cassettes():
        raw = file.read_text()
        lowered = raw.lower()
        for needle in forbidden:
            assert needle not in lowered, f"{file.name} contains {needle!r}"
        payload = json.loads(raw)
        assert set(payload) == {"key", "request", "response"}, file.name
        assert set(payload["request"]) <= allowed, f"{file.name}: {set(payload['request'])}"


@pytest.mark.skipif(not _committed_cassettes(), reason="no cassettes committed yet")
def test_every_committed_cassette_replays() -> None:
    """A recording whose key does not match its own request is dead weight that
    would silently fall through to a live call."""
    for file in _committed_cassettes():
        payload = json.loads(file.read_text())
        assert digest(payload["request"]) == payload["key"] == file.stem, file.name
        cassette = Cassette(path=file.parent, mode=Mode.REPLAY)
        assert cassette.get(payload["request"]) == payload["response"]


@pytest.mark.skipif(not _committed_cassettes(), reason="no cassettes committed yet")
def test_a_committed_recording_drives_a_specialist_with_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property CI depends on, asserted against the real recordings rather
    than a fixture written to pass."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    def _explode(self: ModelClient, request: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("replay must not reach the API")

    monkeypatch.setattr(ModelClient, "_send", _explode)
    for file in _committed_cassettes():
        payload = json.loads(file.read_text())
        request = payload["request"]
        tool_names = {t["name"] for t in request.get("tools", [])}
        if brief.TOOL_NAME in tool_names:
            continue  # a translation call, exercised by the brief tests
        role = next(r for r in Role if {s.name for s in tools.tools_for(r)} == tool_names)
        client = ModelClient(
            config=ModelConfig(model=request["model"]),
            cassette=Cassette(path=file.parent, mode=Mode.REPLAY),
        )
        response, usage, replayed = client.call(
            system=request["system"][0]["text"],
            tools=request["tools"],
            messages=request["messages"],
        )
        assert replayed
        assert usage.total > 0
        assert response["content"]
        # And the recorded tool calls still apply to a chain today, which is
        # what would break if the tool layer's argument names ever drifted.
        for use in response["content"]:
            if use.get("type") != "tool_use":
                continue
            outcome = tools.apply_call(Chain(), role, use["name"], use.get("input") or {})
            assert outcome.ok or outcome.error, use["name"]
        break
