"""The agent: a deterministic supervisor over four bounded specialists.

Read in this order:

``roles``       who owns which ops and which measurements. Two invariants.
``tools``       the only surface a model can change audio through.
``briefing``    what a specialist is shown, which is a filter, not a log.
``memory``      what it remembers, since it gets no conversation history.
``specialist``  the interface, plus the two model-free implementations.
``supervisor``  routing, rerouting, and refusing a pointless render.
``client``      the single module that imports ``anthropic``.
``cassette``    record and replay, so this is all reproducible for nothing.
``factory``     which specialist kind a named system gets.
"""

from .briefing import Briefing, build, role_system_prompt
from .cassette import Cassette, CassetteMissError, Mode
from .factory import (
    AGENT_NAME,
    SCAFFOLD_NAME,
    agent_supervisor,
    cassette_for,
    make_proposer,
    scaffold_supervisor,
    scripted_supervisor,
)
from .memory import MemoryEntry, WorkingMemory
from .pricing import PRICES, Usage, cost_usd
from .roles import FEATURE_OWNER, OWNED_FEATURES, OWNED_OPS, OWNER_OF_OP, Role
from .specialist import (
    ProportionalSpecialist,
    ScriptedSpecialist,
    Specialist,
    SpecialistTurn,
    ToolRecord,
)
from .supervisor import Supervisor, SupervisorConfig, role_loads
from .tools import TOOLS, ToolOutcome, ToolSpec, apply_call, schemas_for, tools_for

__all__ = [
    "AGENT_NAME",
    "FEATURE_OWNER",
    "OWNED_FEATURES",
    "OWNED_OPS",
    "OWNER_OF_OP",
    "PRICES",
    "SCAFFOLD_NAME",
    "TOOLS",
    "Briefing",
    "Cassette",
    "CassetteMissError",
    "MemoryEntry",
    "Mode",
    "ProportionalSpecialist",
    "Role",
    "ScriptedSpecialist",
    "Specialist",
    "SpecialistTurn",
    "Supervisor",
    "SupervisorConfig",
    "ToolOutcome",
    "ToolRecord",
    "ToolSpec",
    "Usage",
    "WorkingMemory",
    "agent_supervisor",
    "apply_call",
    "build",
    "cassette_for",
    "cost_usd",
    "make_proposer",
    "role_loads",
    "role_system_prompt",
    "scaffold_supervisor",
    "schemas_for",
    "scripted_supervisor",
    "tools_for",
]
