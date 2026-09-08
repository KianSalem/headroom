"""Wiring. The one place that decides which kind of specialist gets used.

Three systems come out of the same architecture, and keeping them one function
apart is what makes the comparison in the results table mean something:

``agent-scaffold``  four :class:`ProportionalSpecialist`. The full supervisor,
tool layer, memory and critic, with arithmetic where the model would be. Free.

``agent``  four :class:`LLMSpecialist`. Identical everywhere else.

``agent-scripted``  four :class:`ScriptedSpecialist`. For tests that need a
specific pathology on demand.

The difference between the first two isolates the model's contribution from the
architecture's. Neither number means much alone: an agent that beats the
heuristic has not shown the model is doing the work, and an agent that loses to
it has not shown the model is useless.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from headroom.control.loop import Proposer
from headroom.control.state import RunTrace

from .cassette import Cassette, Mode
from .client import DEFAULT_MODEL, LLMSpecialist, ModelClient, ModelConfig
from .roles import Role
from .specialist import ProportionalSpecialist, Script, ScriptedSpecialist, Specialist
from .supervisor import DEFAULT_CONFIG, Supervisor, SupervisorConfig

#: Where cassettes live. Committed, so CI and a stranger's checkout both replay
#: rather than call.
CASSETTE_ROOT: Final[Path] = Path("cassettes")

MODE_ENV: Final[str] = "HEADROOM_CASSETTE_MODE"
ROOT_ENV: Final[str] = "HEADROOM_CASSETTE_ROOT"

SCAFFOLD_NAME: Final[str] = "agent-scaffold"
AGENT_NAME: Final[str] = "agent"


def cassette_for(name: str, mode: Mode | str | None = None, root: Path | None = None) -> Cassette:
    """Build a cassette, defaulting its mode from the environment.

    ``auto`` locally so the first run records, ``replay`` in CI so a changed
    prompt fails a test instead of quietly spending money.
    """
    chosen = mode if mode is not None else os.environ.get(MODE_ENV, Mode.AUTO)
    base = root or Path(os.environ.get(ROOT_ENV, str(CASSETTE_ROOT)))
    return Cassette(path=base / name, mode=Mode(chosen))


def scaffold_specialists() -> dict[Role, Specialist]:
    return {role: ProportionalSpecialist(role=role) for role in Role}


def llm_specialists(client: ModelClient) -> dict[Role, Specialist]:
    """One specialist per role, sharing a client so cost is accounted once.

    Sharing is deliberate: the shared client is what makes "this run cost
    $0.0031" a single measured number rather than four that have to be added
    up correctly.
    """
    return {role: LLMSpecialist(role=role, client=client) for role in Role}


def scripted_specialists(scripts: Mapping[Role, Script]) -> dict[Role, Specialist]:
    return {role: ScriptedSpecialist(role=role, script=scripts.get(role, ())) for role in Role}


def scaffold_supervisor(config: SupervisorConfig = DEFAULT_CONFIG) -> Supervisor:
    return Supervisor(specialists=scaffold_specialists(), config=config)


def agent_supervisor(
    *,
    model: str = DEFAULT_MODEL,
    model_config: ModelConfig | None = None,
    cassette: Cassette | None = None,
    config: SupervisorConfig = DEFAULT_CONFIG,
) -> Supervisor:
    cfg = model_config or ModelConfig(model=model)
    client = ModelClient(config=cfg, cassette=cassette or cassette_for(cfg.model))
    return Supervisor(specialists=llm_specialists(client), config=config)


def scripted_supervisor(
    scripts: Mapping[Role, Script], config: SupervisorConfig = DEFAULT_CONFIG
) -> Supervisor:
    return Supervisor(specialists=scripted_specialists(scripts), config=config)


def make_proposer(system: str, **kwargs: object) -> Proposer:
    """Look up an agent system by the name the runner and the report use."""
    if system == SCAFFOLD_NAME:
        return scaffold_supervisor().propose
    if system == AGENT_NAME:
        model = str(kwargs.get("model", DEFAULT_MODEL))
        effort = str(kwargs.get("effort", ""))
        return agent_supervisor(model_config=ModelConfig(model=model, effort=effort)).propose
    raise KeyError(f"{system!r} is not an agent system; have {SCAFFOLD_NAME!r}, {AGENT_NAME!r}")


@dataclass
class AgentSystem:
    """A named agent system plus the handles the runner needs afterwards.

    The shared loop deliberately knows nothing about specialists, cassettes or
    token prices, which is what keeps every system held to identical rules. The
    consequence is that agent-specific provenance -- routing statistics,
    whether the run was replayed rather than billed -- has to be attached after
    the fact, and this is the seam where that happens.
    """

    name: str
    supervisor: Supervisor
    client: ModelClient | None = None

    @property
    def propose(self) -> Proposer:
        return self.supervisor.propose

    @property
    def fully_replayed(self) -> bool:
        client = self.client
        return bool(client and client.n_calls and client.n_replayed == client.n_calls)

    def annotate(self, trace: RunTrace) -> RunTrace:
        stats: dict[str, Any] = dict(self.supervisor.stats())
        if self.client is not None:
            stats["client"] = self.client.stats()
        return trace.model_copy(
            update={"system_stats": stats, "replayed_from_cassette": self.fully_replayed}
        )


def build_system(
    system: str,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = "",
    cassette: Cassette | None = None,
    config: SupervisorConfig = DEFAULT_CONFIG,
) -> AgentSystem:
    """Construct one of the agent systems by the name the report uses."""
    if system == SCAFFOLD_NAME:
        return AgentSystem(name=system, supervisor=scaffold_supervisor(config))
    if system == AGENT_NAME:
        cfg = ModelConfig(model=model, effort=effort)
        client = ModelClient(config=cfg, cassette=cassette or cassette_for(cfg.model))
        return AgentSystem(
            name=system,
            supervisor=Supervisor(specialists=llm_specialists(client), config=config),
            client=client,
        )
    raise KeyError(f"{system!r} is not an agent system; have {SCAFFOLD_NAME!r}, {AGENT_NAME!r}")
