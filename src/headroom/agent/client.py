"""The model boundary: the only module in the package that imports ``anthropic``.

That is enforced, not merely intended. ``.importlinter`` forbids the analyzer,
the DSP layer, the target metric, the control loop, the tool layer, the
routing and the memory from reaching this module's dependency. Two things
follow, and both are claims the project needs to be able to make:

*The objective function is independent of the thing being graded.* There is no
path by which a model can influence a measurement, so "no LLM-as-judge in the
primary metric" is a property of the import graph rather than a promise.

*The architecture is testable without a key.* Everything except this file runs
in CI, on a fork's pull request, at zero cost.

Cost discipline lives here too. The static half of the prompt -- the role
instructions and the tool schemas -- is identical on every call a role ever
makes and is marked for caching, so it is billed at a tenth of the input rate
after the first hit. The per-turn briefing, the part that actually differs, is
a few hundred tokens. Every call is recorded to a cassette, so a second run of
the same evaluation costs nothing.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Final

from headroom.analysis.features import FeatureVector
from headroom.dsp.chain import Chain
from headroom.target.distance import SPEC_BY_NAME, to_scored

from .brief import TOOL_NAME as BRIEF_TOOL_NAME
from .brief import BriefTarget
from .brief import parse as parse_brief
from .brief import system_prompt as brief_system_prompt
from .brief import tool_schema as brief_tool_schema
from .briefing import Briefing, role_system_prompt
from .cassette import Cassette, Mode
from .pricing import Usage, cost_usd
from .roles import Role
from .specialist import SpecialistTurn, ToolRecord, run_calls
from .tools import schemas_for

#: Haiku 4.5 by default: this project's whole point is that the interesting
#: question is architectural, and the budget is real. The model is a
#: configuration value, recorded in every trace, so a sweep is a flag rather
#: than a code change.
DEFAULT_MODEL: Final[str] = "claude-haiku-4-5"

#: Environment variable read for the credential. Never a constructor argument
#: and never written to a trace or a cassette: this repository is public.
API_KEY_ENV: Final[str] = "ANTHROPIC_API_KEY"

_RETRYABLE: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504, 529})


class ModelError(RuntimeError):
    """A call failed after retries, or returned something unusable."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model: str = DEFAULT_MODEL
    #: Enough for a handful of tool calls and a short reason. Tool calls are
    #: small; a large ceiling only buys the chance to pay for a ramble.
    max_tokens: int = 1024
    #: Sent as ``output_config.effort`` when set. Left empty by default
    #: because Haiku 4.5 does not accept it.
    effort: str = ""
    #: Extended thinking. Off by default: on a task whose entire input is a
    #: table of six numbers, thinking tokens are billed at the output rate for
    #: reasoning the tool schema already encodes.
    thinking: bool = False
    thinking_budget: int = 1024
    #: Cache the role prompt and tool schemas. ``""`` disables it.
    #:
    #: Measured, not assumed: a cache breakpoint only engages once the prefix
    #: clears a per-model minimum, and on Haiku 4.5 that minimum is above this
    #: system's ~2.3k-token role prefix -- probing it with the real prefix
    #: returned zero cache writes on every TTL, while quadrupling the prefix
    #: wrote and then read an entry. So on Haiku this setting costs nothing and
    #: buys nothing, and it starts paying on a model with a lower threshold
    #: without a code change. The hit rate is in the client's reported stats
    #: either way, which is how the claim stays checkable rather than assumed.
    #:
    #: A 1-hour entry is the right TTL for an evaluation, which reuses one
    #: prefix for as long as the matrix takes, and it is priced accordingly --
    #: see ``pricing.CACHE_WRITE_MULTIPLIER_1H``.
    cache_ttl: str = "1h"
    #: Tool round-trips inside one round. Three is enough to recover from a
    #: rejected call and still finish; more just pays for indecision.
    max_tool_rounds: int = 3
    max_retries: int = 3
    retry_base_s: float = 1.0


@dataclass
class ModelClient:
    """Cassette-aware wrapper. Owns pricing, retries and usage accounting."""

    config: ModelConfig = field(default_factory=ModelConfig)
    cassette: Cassette | None = None
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    n_calls: int = 0
    n_replayed: int = 0
    _api: Any = None

    def _client(self) -> Any:
        if self._api is None:
            import anthropic

            key = os.environ.get(API_KEY_ENV, "")
            if not key:
                raise ModelError(
                    f"{API_KEY_ENV} is not set. Either export it, or run with a "
                    "cassette in replay mode, which needs no credential."
                )
            self._api = anthropic.Anthropic(api_key=key, max_retries=0)
        return self._api

    def _request(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tool_choice: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self.config
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if cfg.cache_ttl:
            # The breakpoint sits on the system block, which caches the tool
            # schemas ahead of it as well: together they are the large,
            # byte-identical part of every call this role makes.
            system_blocks[0]["cache_control"] = {"type": "ephemeral", "ttl": cfg.cache_ttl}
        request: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "system": system_blocks,
            "tools": tools,
            "messages": messages,
            "tool_choice": tool_choice,
        }
        if cfg.effort:
            request["output_config"] = {"effort": cfg.effort}
        if cfg.thinking:
            # Extended thinking and a forced tool call are mutually exclusive,
            # so enabling one relaxes the other.
            request["thinking"] = {"type": "enabled", "budget_tokens": cfg.thinking_budget}
            request["tool_choice"] = {"type": "auto"}
        return request

    def call(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Usage, bool]:
        """One model call. Returns the response as a plain dict, its usage, and
        whether it came from a cassette.

        The response is handled as a dict on both paths -- recorded and live --
        so replay exercises exactly the same parsing code as a real call, which
        is the only way a cassette is worth anything as a test fixture.
        """
        request = self._request(
            system=system,
            tools=tools,
            messages=messages,
            tool_choice=tool_choice or {"type": "any"},
        )
        if self.cassette is not None:
            recorded = self.cassette.get(request)
            if recorded is not None:
                usage = _usage_of(recorded)
                self._account(usage, replayed=True)
                return recorded, usage, True

        response = self._send(request)
        if self.cassette is not None and self.cassette.mode is not Mode.OFF:
            self.cassette.put(request, response)
        usage = _usage_of(response)
        self._account(usage, replayed=False)
        return response, usage, False

    def _send(self, request: dict[str, Any]) -> dict[str, Any]:
        import anthropic

        client = self._client()
        last: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                message = client.messages.create(**request)
                dumped: dict[str, Any] = message.model_dump(mode="json")
                return dumped
            except anthropic.APIStatusError as exc:
                if exc.status_code not in _RETRYABLE:
                    raise ModelError(f"{exc.status_code} from the API: {exc}") from exc
                last = exc
            except anthropic.APIConnectionError as exc:
                last = exc
            time.sleep(self.config.retry_base_s * (2**attempt))
        raise ModelError(f"gave up after {self.config.max_retries} attempts: {last}")

    def _account(self, usage: Usage, *, replayed: bool) -> None:
        self.usage = self.usage + usage
        # The recorded usage is billed as it was when recorded. Reporting zero
        # on replay would understate what the published numbers cost to
        # produce; the trace carries a replay flag so the two are separable.
        self.cost += cost_usd(self.config.model, usage)
        self.n_calls += 1
        if replayed:
            self.n_replayed += 1

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "calls": self.n_calls,
            "replayed": self.n_replayed,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "cache_write_5m_tokens": self.usage.cache_write_5m_tokens,
            "cache_write_1h_tokens": self.usage.cache_write_1h_tokens,
            "cost_usd": round(self.cost, 6),
            "cassette": self.cassette.stats() if self.cassette else None,
        }


def _usage_of(response: dict[str, Any]) -> Usage:
    raw = response.get("usage") or {}
    total_writes = int(raw.get("cache_creation_input_tokens") or 0)
    # The API reports cache writes both as a total and, when it has the
    # breakdown, split by TTL. The two are priced differently, so the split is
    # preferred and the total is only a fallback.
    breakdown = raw.get("cache_creation") or {}
    write_1h = int(breakdown.get("ephemeral_1h_input_tokens") or 0)
    write_5m = int(breakdown.get("ephemeral_5m_input_tokens") or 0)
    if write_1h + write_5m == 0:
        write_5m = total_writes
    return Usage(
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        cache_read_tokens=int(raw.get("cache_read_input_tokens") or 0),
        cache_write_5m_tokens=write_5m,
        cache_write_1h_tokens=write_1h,
    )


def _text_of(response: dict[str, Any]) -> str:
    return " ".join(
        str(block.get("text", "")).strip()
        for block in response.get("content", [])
        if block.get("type") == "text"
    ).strip()


def _tool_uses(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in response.get("content", []) if b.get("type") == "tool_use"]


def _result_text(record: ToolRecord) -> str:
    """What a tool result says back to the model.

    A failure gets the whole structured payload: the field, the value and the
    bound are what it needs to recover. A success gets one short line. The
    payload's readable chain description is useful in a trace and useless here,
    and every byte of it is re-billed on the next round-trip -- the round where
    a specialist made six edits cost 11k input tokens before this.
    """
    if not record.ok:
        return json.dumps(record.payload, default=str)[:1200]
    return f"ok: {record.action}"


def _assistant_echo(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the assistant turn with only the fields the API accepts back.

    A dumped response carries nulls and read-only fields that a request would
    reject, so the blocks are reconstructed explicitly rather than echoed.
    """
    out: list[dict[str, Any]] = []
    for block in response.get("content", []):
        kind = block.get("type")
        if kind == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif kind == "tool_use":
            out.append(
                {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                }
            )
        elif kind in ("thinking", "redacted_thinking"):
            out.append(dict(block))
    return out


@dataclass
class LLMSpecialist:
    """A specialist backed by a model call.

    Interchangeable with :class:`~headroom.agent.specialist.ProportionalSpecialist`
    and :class:`~headroom.agent.specialist.ScriptedSpecialist`: the supervisor
    cannot tell which it is holding, which is what makes the three-way
    comparison in the results table a controlled one.
    """

    role: Role
    client: ModelClient

    def __call__(self, briefing: Briefing, chain: Chain) -> SpecialistTurn:
        system = role_system_prompt(self.role)
        tools = schemas_for(self.role)
        messages: list[dict[str, Any]] = [{"role": "user", "content": briefing.render()}]
        records: list[ToolRecord] = []
        rationale: list[str] = []
        usage = Usage()
        replayed_all = True
        calls_made = 0

        for _ in range(self.client.config.max_tool_rounds):
            response, call_usage, replayed = self.client.call(
                system=system, tools=tools, messages=messages
            )
            usage = usage + call_usage
            replayed_all = replayed_all and replayed
            calls_made += 1

            text = _text_of(response)
            if text:
                rationale.append(text)
            uses = _tool_uses(response)

            if not uses:
                if response.get("stop_reason") == "max_tokens":
                    rationale.append("(response truncated at max_tokens)")
                break

            chain, new_records, finished = run_calls(
                self.role, chain, [(u.get("name", ""), u.get("input") or {}) for u in uses]
            )
            records.extend(new_records)
            rationale.extend(
                str(u.get("input", {}).get("reason", "")).strip()
                for u in uses
                if str(u.get("input", {}).get("reason", "")).strip()
            )
            if finished:
                break

            messages.append({"role": "assistant", "content": _assistant_echo(response)})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": use.get("id", ""),
                            "content": _result_text(record),
                            "is_error": not record.ok,
                        }
                        for use, record in zip(uses, new_records, strict=False)
                    ],
                }
            )

        applied = [r for r in records if r.ok and r.action]
        return SpecialistTurn(
            role=self.role,
            chain=chain,
            calls=tuple(records),
            rationale=" | ".join(dict.fromkeys(r for r in rationale if r))[:600],
            stop=not applied,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost_usd=cost_usd(self.client.config.model, usage),
            model=self.client.config.model,
            n_llm_calls=calls_made,
        )


@dataclass
class BriefTranslator:
    """Turns a natural-language brief into a measurable target.

    One call, one forced tool, one bounded output. The model is not given the
    audio, is not given any processing tools, and cannot score anything: it
    reads a table of measurements and emits signed offsets against dimensions
    the deterministic metric already defines. The loop and the metric are
    unchanged downstream, which is what makes the result gradable.
    """

    client: ModelClient

    def __call__(self, brief: str, fv: FeatureVector) -> tuple[BriefTarget, Usage]:
        response, usage, _ = self.client.call(
            system=brief_system_prompt(),
            tools=[brief_tool_schema()],
            messages=[{"role": "user", "content": _brief_message(brief, fv)}],
            tool_choice={"type": "tool", "name": BRIEF_TOOL_NAME},
        )
        for block in _tool_uses(response):
            if block.get("name") == BRIEF_TOOL_NAME:
                return parse_brief(brief, dict(block.get("input") or {})), usage
        raise ModelError(f"the translation call returned no {BRIEF_TOOL_NAME} call")


def _brief_message(brief: str, fv: FeatureVector) -> str:
    """The brief plus where the audio currently sits.

    The offsets are relative, so in principle the current values are not
    needed. They are supplied anyway because they are what lets the model
    notice that the brief is asking for something the audio already has, or
    asking to widen something already at the top of its range -- and because
    "brighter" from a dull starting point is a bigger ask than from a bright
    one.
    """
    scored = to_scored(fv)
    rows = [
        f"  {name:18s} {scored[name]:+9.3f} {SPEC_BY_NAME[name].unit:9s} "
        f"(tolerance {SPEC_BY_NAME[name].tolerance:g})"
        for name in sorted(scored)
    ]
    return "\n".join(
        [
            f"Brief: {brief}",
            "",
            "Where the audio sits now:",
            *rows,
            "",
            "Translate the brief into offsets against these values.",
        ]
    )
