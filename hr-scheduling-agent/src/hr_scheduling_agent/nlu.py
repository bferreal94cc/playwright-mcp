"""Understanding what a manager or employee actually asked for.

Both input modes in the design land here as text: typed messages arrive
directly, spoken ones pass through a :class:`SpeechTranscriber` first. Audio
transcription is deliberately a port, not an implementation -- it belongs to
whichever speech service the deployment already pays for.

Classification runs rules first. Shift-work vocabulary is small and repetitive
("I'm sick", "swap Tuesday", "who's on nights"), so rules resolve the common
cases instantly, for free, and identically every time. The LLM is the fallback
for the long tail, and only when a provider is configured -- which keeps the
system fully functional offline.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum

from .providers import LLMProvider


class Intent(str, Enum):
    CREATE_SCHEDULE = "create_schedule"
    REPORT_ABSENCE = "report_absence"
    SWAP_SHIFT = "swap_shift"
    REQUEST_TIME_OFF = "request_time_off"
    POST_JOB = "post_job"
    QUERY_SCHEDULE = "query_schedule"
    UNKNOWN = "unknown"


@dataclass
class ParsedRequest:
    """What the speaker wanted, how sure we are, and what we pulled out."""

    intent: Intent
    confidence: float
    entities: dict[str, object] = field(default_factory=dict)
    raw_text: str = ""
    source: str = "rules"

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.5


class SpeechTranscriber(ABC):
    """Port for speech-to-text. Implementations wrap a real STT service."""

    @abstractmethod
    def transcribe(self, audio: bytes) -> str:
        """Return the transcript for a chunk of audio."""


class ScriptedTranscriber(SpeechTranscriber):
    """Test double that replays a fixed list of transcripts in order."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self.calls = 0

    def transcribe(self, audio: bytes) -> str:
        if not self._transcripts:
            return ""
        index = min(self.calls, len(self._transcripts) - 1)
        self.calls += 1
        return self._transcripts[index]


# Weighted keyword signatures. Multi-word phrases are weighted above single
# words because "off" alone means little and "time off" means a lot.
_SIGNATURES: dict[Intent, tuple[tuple[str, float], ...]] = {
    Intent.REPORT_ABSENCE: (
        ("call out", 3.0), ("calling out", 3.0), ("can't come", 3.0), ("cant come", 3.0),
        ("won't make it", 3.0), ("wont make it", 3.0), ("no show", 3.0), ("not coming", 3.0),
        ("i'm sick", 2.5), ("im sick", 2.5), ("sick", 1.5), ("absent", 2.0), ("unwell", 1.5),
    ),
    Intent.SWAP_SHIFT: (
        ("swap", 3.0), ("trade shift", 3.0), ("cover my", 2.5), ("switch shift", 3.0),
        ("give away", 2.0), ("pick up my", 2.0), ("drop my shift", 2.5),
    ),
    Intent.REQUEST_TIME_OFF: (
        ("time off", 3.0), ("day off", 3.0), ("days off", 3.0), ("vacation", 3.0),
        ("pto", 3.0), ("annual leave", 3.0), ("holiday request", 2.5), ("leave request", 2.5),
    ),
    Intent.CREATE_SCHEDULE: (
        ("build the schedule", 3.0), ("create the schedule", 3.0), ("create a schedule", 3.0),
        ("build a schedule", 3.0), ("generate the schedule", 3.0), ("make the schedule", 3.0),
        ("draft the schedule", 3.0), ("schedule for next week", 2.5), ("roster for", 2.0),
        ("staff the", 2.0),
    ),
    Intent.POST_JOB: (
        ("post a job", 3.0), ("job posting", 3.0), ("open position", 3.0), ("hire", 2.5),
        ("hiring", 2.5), ("recruit", 2.5), ("need another", 2.0), ("vacancy", 2.5),
    ),
    Intent.QUERY_SCHEDULE: (
        ("when am i working", 3.0), ("what's my schedule", 3.0), ("whats my schedule", 3.0),
        ("my shifts", 2.5), ("who is on", 2.5), ("who's on", 2.5), ("am i working", 2.5),
        ("show me the schedule", 2.5),
    ),
}

_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_LLM_SYSTEM = (
    "You classify workforce scheduling messages. Reply with JSON only, using the keys "
    '"intent" and "confidence". intent must be one of: '
    + ", ".join(i.value for i in Intent)
    + ". confidence is a number between 0 and 1."
)


class RuleBasedParser:
    """Keyword scoring plus regex entity extraction. No network, no cost."""

    def __init__(self, known_roles: set[str] | None = None, known_names: dict[str, str] | None = None):
        self.known_roles = {r.lower() for r in (known_roles or set())}
        # Maps a lowercase display name to an employee id.
        self.known_names = {k.lower(): v for k, v in (known_names or {}).items()}

    def parse(self, text: str, reference: date | None = None) -> ParsedRequest:
        reference = reference or date.today()
        lowered = text.lower()

        scores: dict[Intent, float] = {}
        for intent, signature in _SIGNATURES.items():
            score = sum(weight for phrase, weight in signature if phrase in lowered)
            if score:
                scores[intent] = score

        if not scores:
            return ParsedRequest(Intent.UNKNOWN, 0.0, self._entities(text, lowered, reference), text)

        intent, best = max(scores.items(), key=lambda kv: (kv[1], kv[0].value))
        # Confidence rises with the strength of the winner and falls when a
        # rival intent scores nearly as well.
        runner_up = max((v for k, v in scores.items() if k is not intent), default=0.0)
        margin = (best - runner_up) / best if best else 0.0
        confidence = min(0.95, (min(best, 3.0) / 3.0) * (0.6 + 0.4 * margin))
        return ParsedRequest(intent, round(confidence, 3), self._entities(text, lowered, reference), text)

    # -- entity extraction -------------------------------------------------

    def _entities(self, text: str, lowered: str, reference: date) -> dict[str, object]:
        entities: dict[str, object] = {}
        dates = self._dates(lowered, reference)
        if dates:
            entities["dates"] = [d.isoformat() for d in dates]
        times = self._times(lowered)
        if times:
            entities["times"] = [t.strftime("%H:%M") for t in times]
        role = self._role(lowered)
        if role:
            entities["role"] = role
        people = self._people(lowered)
        if people:
            entities["employee_ids"] = people
        count = self._headcount(lowered, role)
        if count is not None:
            entities["headcount"] = count
        if "next week" in lowered:
            entities["week_start"] = (
                reference + timedelta(days=(7 - reference.weekday()))
            ).isoformat()
        return entities

    def _dates(self, lowered: str, reference: date) -> list[date]:
        found: list[date] = []

        if re.search(r"\btoday\b", lowered):
            found.append(reference)
        if re.search(r"\btomorrow\b", lowered):
            found.append(reference + timedelta(days=1))

        for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered):
            try:
                found.append(date(int(match[1]), int(match[2]), int(match[3])))
            except ValueError:
                continue

        for match in re.finditer(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", lowered):
            year = int(match[3]) if match[3] else reference.year
            year = year + 2000 if year < 100 else year
            try:
                found.append(date(year, int(match[1]), int(match[2])))
            except ValueError:
                continue

        month_pattern = "|".join(_MONTHS)
        for match in re.finditer(rf"\b({month_pattern})[a-z]*\.?\s+(\d{{1,2}})\b", lowered):
            try:
                found.append(date(reference.year, _MONTHS[match[1]], int(match[2])))
            except ValueError:
                continue

        for match in re.finditer(rf"\b(next\s+)?({'|'.join(_WEEKDAYS)})\b", lowered):
            found.append(self._weekday_on_or_after(reference, _WEEKDAYS[match[2]], bool(match[1])))

        deduped: list[date] = []
        for value in found:
            if value not in deduped:
                deduped.append(value)
        return deduped

    @staticmethod
    def _weekday_on_or_after(reference: date, weekday: int, force_next_week: bool) -> date:
        delta = (weekday - reference.weekday()) % 7
        if force_next_week:
            # "next Tuesday" means the one in the following week, not today.
            delta = delta + 7 if delta < 7 else delta
            if delta == 0:
                delta = 7
        return reference + timedelta(days=delta)

    @staticmethod
    def _times(lowered: str) -> list[time]:
        found: list[time] = []
        for match in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lowered):
            hour = int(match[1]) % 12
            if match[3] == "pm":
                hour += 12
            minute = int(match[2]) if match[2] else 0
            if 0 <= hour < 24 and 0 <= minute < 60:
                found.append(time(hour, minute))
        for match in re.finditer(r"\b(\d{1,2}):(\d{2})\b(?!\s*(?:am|pm))", lowered):
            hour, minute = int(match[1]), int(match[2])
            if 0 <= hour < 24 and 0 <= minute < 60:
                candidate = time(hour, minute)
                if candidate not in found:
                    found.append(candidate)
        return found

    def _role(self, lowered: str) -> str | None:
        for role in sorted(self.known_roles, key=len, reverse=True):
            if re.search(rf"\b{re.escape(role)}s?\b", lowered):
                return role
        return None

    def _people(self, lowered: str) -> list[str]:
        found: list[str] = []
        for name, employee_id in self.known_names.items():
            if re.search(rf"\b{re.escape(name)}\b", lowered) and employee_id not in found:
                found.append(employee_id)
        return found

    @staticmethod
    def _headcount(lowered: str, role: str | None) -> int | None:
        """How many *people* are being asked for.

        Anchored to the detected role so that "two days off" counts days, not
        staff. Without a role in the sentence there is nothing to count.
        """
        if not role:
            return None
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        qualifier = r"(?:more\s+|extra\s+|additional\s+)*"
        noun = rf"{re.escape(role)}s?\b"
        match = re.search(rf"\b(\d{{1,3}})\s+{qualifier}{noun}", lowered)
        if match:
            return int(match[1])
        for word, value in words.items():
            if re.search(rf"\b{word}\s+{qualifier}{noun}", lowered):
                return value
        return None


class NLUPipeline:
    """Rules first, LLM second, never neither."""

    def __init__(
        self,
        parser: RuleBasedParser | None = None,
        provider: LLMProvider | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.parser = parser or RuleBasedParser()
        self.provider = provider
        self.threshold = threshold

    def understand(self, text: str, reference: date | None = None) -> ParsedRequest:
        """Classify ``text``, escalating to the LLM only when rules are unsure."""
        parsed = self.parser.parse(text, reference=reference)
        if parsed.confidence >= self.threshold or self.provider is None:
            return parsed
        return self._ask_llm(text, parsed)

    def understand_speech(
        self, audio: bytes, transcriber: SpeechTranscriber, reference: date | None = None
    ) -> ParsedRequest:
        """Transcribe then classify. The voice path is the text path plus STT."""
        transcript = transcriber.transcribe(audio)
        parsed = self.understand(transcript, reference=reference)
        parsed.entities["transcript"] = transcript
        return parsed

    def _ask_llm(self, text: str, fallback: ParsedRequest) -> ParsedRequest:
        assert self.provider is not None
        prompt = f"Message: {text!r}\nRespond with JSON only."
        try:
            payload = self.provider.complete_json(prompt, system=_LLM_SYSTEM)
        except Exception:
            # A provider outage must not take scheduling down; the rule-based
            # answer is degraded but usable.
            return fallback
        if not payload:
            return fallback
        try:
            intent = Intent(str(payload.get("intent", "")).strip().lower())
        except ValueError:
            return fallback
        try:
            confidence = float(payload.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return ParsedRequest(
            intent=intent,
            confidence=max(0.0, min(1.0, confidence)),
            entities=fallback.entities,
            raw_text=text,
            source=f"llm:{self.provider.name}",
        )


def parser_for_context(context) -> RuleBasedParser:
    """Build a parser primed with a scenario's roles and staff names."""
    roles = {role for employee in context.employees for role in employee.roles}
    roles |= {shift.role for shift in context.shifts}
    names = {employee.name: employee.id for employee in context.employees}
    return RuleBasedParser(known_roles=roles, known_names=names)
