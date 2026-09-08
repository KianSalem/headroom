"""Record and replay of model calls.

A public repository has three cost problems and this module is the answer to
all three.

*Development.* Debugging a nine-step agent run by re-billing every call on
every iteration is how a ten-dollar experiment becomes a hundred-dollar one.
Recorded once, replayed free.

*Continuous integration.* A fork's pull request cannot be given a secret, so
without replay the agent layer would be the one part of the system CI never
tested -- exactly the part most likely to break.

*Reproduction.* Someone who reads the results table can re-run the evaluation
that produced it, from committed data, for nothing. That is a stronger claim
than a table of numbers with a note saying trust me.

The key is a digest of everything that determines the response: model, system
prompt, messages, tool schemas and sampling parameters. Change any of them and
the entry misses, which is the desired behaviour -- a silent hit on a stale
prompt would be worse than an API bill.

Requests are stored in full alongside their responses. That makes the cassette
directory a readable record of every prompt the system has ever sent, which is
worth more as documentation than it costs in repository size.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

#: Keys stripped before hashing and before storage. Nothing here changes the
#: response, and one of them is a credential.
_VOLATILE: Final[frozenset[str]] = frozenset({"api_key", "metadata", "stream"})


class Mode(StrEnum):
    #: Replay a hit, call the API on a miss, record the result. The default.
    AUTO = "auto"
    #: Replay only. A miss is an error, which is what CI wants: it turns "the
    #: prompt changed and nobody re-recorded" into a failing test rather than
    #: an unexpected charge.
    REPLAY = "replay"
    #: Always call the API and overwrite. For re-recording deliberately.
    RECORD = "record"
    #: No cassette at all.
    OFF = "off"


class CassetteMissError(KeyError):
    """A request had no recording and the mode forbids calling the API."""


def digest(request: dict[str, Any]) -> str:
    payload = {k: v for k, v in sorted(request.items()) if k not in _VOLATILE}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()


@dataclass
class Cassette:
    """A directory of one JSON file per recorded call.

    One file per call rather than one file per cassette: recordings made by
    different runs never touch the same bytes, so two people adding cases
    cannot conflict in git.
    """

    path: Path
    mode: Mode = Mode.AUTO
    hits: int = 0
    misses: int = 0
    writes: int = 0
    #: Digests seen this session, for pruning recordings nothing replays.
    seen: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def enabled(self) -> bool:
        return self.mode is not Mode.OFF

    def _file(self, key: str) -> Path:
        return self.path / f"{key}.json"

    def get(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Return a recorded response, or ``None`` to mean "call the API"."""
        if self.mode in (Mode.OFF, Mode.RECORD):
            return None
        key = digest(request)
        self.seen.add(key)
        file = self._file(key)
        if not file.exists():
            self.misses += 1
            if self.mode is Mode.REPLAY:
                raise CassetteMissError(
                    f"no recording for {key} in {self.path}. The prompt or tool schema "
                    "changed since the cassette was made; re-record with "
                    "HEADROOM_CASSETTE_MODE=record."
                )
            return None
        self.hits += 1
        loaded: dict[str, Any] = json.loads(file.read_text())
        response: dict[str, Any] = loaded["response"]
        return response

    def put(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        if self.mode is Mode.OFF:
            return
        key = digest(request)
        self.seen.add(key)
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "request": {k: v for k, v in request.items() if k not in _VOLATILE},
            "response": response,
        }
        self._file(key).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
        self.writes += 1

    def stats(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "path": str(self.path),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "on_disk": len(list(self.path.glob("*.json"))) if self.path.exists() else 0,
        }

    def unused(self) -> list[Path]:
        """Recordings on disk that nothing asked for this session."""
        if not self.path.exists():
            return []
        return sorted(f for f in self.path.glob("*.json") if f.stem not in self.seen)
