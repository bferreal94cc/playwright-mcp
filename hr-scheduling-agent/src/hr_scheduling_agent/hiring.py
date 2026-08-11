"""The hiring workflow: job posting, screening and interview scheduling.

Everything here degrades gracefully. When an LLM provider is configured the
agent uses it to draft postings and grade free-text answers; when one is not,
deterministic templates and an explicit rubric take over. A hiring manager on a
plane with no connectivity still gets a usable posting and a defensible score.

Screening scores are advisory. The agent ranks and explains; a person decides.
Nothing here auto-rejects a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .domain import TimeWindow
from .providers import LLMProvider


class Recommendation(str, Enum):
    ADVANCE = "advance"
    HOLD = "hold"
    DECLINE = "decline"


@dataclass(frozen=True)
class JobPosting:
    """A drafted opening, ready for a human to edit and publish."""

    role: str
    location_id: str
    headcount: int
    title: str
    body: str
    must_haves: tuple[str, ...] = ()
    nice_to_haves: tuple[str, ...] = ()
    source: str = "template"


@dataclass(frozen=True)
class Candidate:
    """An applicant and what they bring."""

    id: str
    name: str
    years_experience: float
    skills: frozenset[str] = frozenset()
    certifications: frozenset[str] = frozenset()
    available_slot_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class InterviewSlot:
    """A window an interviewer has opened up."""

    id: str
    interviewer: str
    window: TimeWindow


@dataclass(frozen=True)
class ScreeningResult:
    """A scored pre-screen, with the reasoning that produced the score."""

    candidate_id: str
    score: float
    recommendation: Recommendation
    strengths: tuple[str, ...]
    gaps: tuple[str, ...]
    source: str = "rubric"

    def describe(self) -> str:
        strengths = ", ".join(self.strengths) or "none recorded"
        gaps = ", ".join(self.gaps) or "none recorded"
        return (
            f"{self.candidate_id}: {self.score:.0f}/100 ({self.recommendation.value}) -- "
            f"strengths: {strengths}; gaps: {gaps}"
        )


@dataclass
class InterviewBooking:
    candidate_id: str
    slot_id: str
    interviewer: str


@dataclass
class InterviewPlan:
    """Result of matching candidates to interview slots."""

    bookings: list[InterviewBooking] = field(default_factory=list)
    unscheduled: list[str] = field(default_factory=list)
    unused_slots: list[str] = field(default_factory=list)


_POSTING_SYSTEM = (
    "You write concise, inclusive job postings for shift-based roles. "
    "Return JSON with keys 'title' and 'body'. No markdown fences."
)


class InterviewAgent:
    """Drafts postings, screens applicants and books interviews."""

    def __init__(self, provider: LLMProvider | None = None, advance_at: float = 70.0,
                 decline_below: float = 40.0) -> None:
        self.provider = provider
        self.advance_at = advance_at
        self.decline_below = decline_below

    # -- posting ----------------------------------------------------------

    def draft_posting(
        self,
        role: str,
        location_id: str,
        headcount: int = 1,
        must_haves: tuple[str, ...] = (),
        nice_to_haves: tuple[str, ...] = (),
    ) -> JobPosting:
        """Draft an opening. Uses the LLM when available, a template otherwise."""
        template = self._template_posting(role, location_id, headcount, must_haves, nice_to_haves)
        if self.provider is None:
            return template

        prompt = (
            f"Role: {role}\nLocation: {location_id}\nOpenings: {headcount}\n"
            f"Required: {', '.join(must_haves) or 'none specified'}\n"
            f"Preferred: {', '.join(nice_to_haves) or 'none specified'}\n"
            "Write the posting."
        )
        try:
            payload = self.provider.complete_json(prompt, system=_POSTING_SYSTEM)
        except Exception:
            return template
        if not payload or not payload.get("body"):
            return template
        return JobPosting(
            role=role,
            location_id=location_id,
            headcount=headcount,
            title=str(payload.get("title") or template.title),
            body=str(payload["body"]),
            must_haves=must_haves,
            nice_to_haves=nice_to_haves,
            source=f"llm:{self.provider.name}",
        )

    @staticmethod
    def _template_posting(
        role: str,
        location_id: str,
        headcount: int,
        must_haves: tuple[str, ...],
        nice_to_haves: tuple[str, ...],
    ) -> JobPosting:
        plural = "s" if headcount != 1 else ""
        lines = [
            f"We are hiring {headcount} {role}{plural} at {location_id}.",
            "",
            "What the job involves:",
            f"  Working scheduled {role} shifts as part of a rota, alongside a small team.",
            "",
            "What you need:",
        ]
        lines += [f"  - {item}" for item in (must_haves or ("Reliability and a willingness to learn",))]
        if nice_to_haves:
            lines += ["", "Nice to have:"] + [f"  - {item}" for item in nice_to_haves]
        lines += [
            "",
            "Shifts are published in advance and you can set your availability and",
            "swap shifts from your phone. We welcome applicants from every background.",
        ]
        return JobPosting(
            role=role,
            location_id=location_id,
            headcount=headcount,
            title=f"{role.title()} ({headcount} opening{plural}) - {location_id}",
            body="\n".join(lines),
            must_haves=must_haves,
            nice_to_haves=nice_to_haves,
        )

    # -- screening --------------------------------------------------------

    def screening_questions(self, role: str, must_haves: tuple[str, ...] = ()) -> list[str]:
        """The short pre-screen asked over voice or text."""
        questions = [
            f"How much experience do you have working as a {role}?",
            "Which days and times are you generally available?",
            "Tell us about a busy shift you handled well.",
        ]
        questions += [f"Do you hold or can you obtain: {item}?" for item in must_haves]
        return questions

    def score_screening(
        self,
        candidate: Candidate,
        required_skills: frozenset[str] = frozenset(),
        required_certifications: frozenset[str] = frozenset(),
        minimum_years: float = 0.0,
        answers: dict[str, str] | None = None,
    ) -> ScreeningResult:
        """Score one candidate against the role's stated requirements."""
        strengths: list[str] = []
        gaps: list[str] = []

        matched_skills = required_skills & candidate.skills
        missing_skills = required_skills - candidate.skills
        if matched_skills:
            strengths.append("skills: " + ", ".join(sorted(matched_skills)))
        if missing_skills:
            gaps.append("missing skills: " + ", ".join(sorted(missing_skills)))
        skill_ratio = len(matched_skills) / len(required_skills) if required_skills else 1.0

        matched_certs = required_certifications & candidate.certifications
        missing_certs = required_certifications - candidate.certifications
        if matched_certs:
            strengths.append("certified: " + ", ".join(sorted(matched_certs)))
        if missing_certs:
            gaps.append("missing certifications: " + ", ".join(sorted(missing_certs)))
        cert_ratio = (
            len(matched_certs) / len(required_certifications) if required_certifications else 1.0
        )

        if minimum_years > 0:
            experience_ratio = min(1.0, candidate.years_experience / minimum_years)
            if candidate.years_experience >= minimum_years:
                strengths.append(f"{candidate.years_experience:g} years experience")
            else:
                gaps.append(
                    f"{candidate.years_experience:g} of {minimum_years:g} years experience"
                )
        else:
            experience_ratio = 1.0

        score = 100.0 * (0.45 * skill_ratio + 0.35 * cert_ratio + 0.20 * experience_ratio)
        if answers:
            strengths.append(f"answered {len(answers)} screening question(s)")

        return ScreeningResult(
            candidate_id=candidate.id,
            score=round(score, 1),
            recommendation=self._recommend(score),
            strengths=tuple(strengths),
            gaps=tuple(gaps),
        )

    def _recommend(self, score: float) -> Recommendation:
        if score >= self.advance_at:
            return Recommendation.ADVANCE
        if score < self.decline_below:
            return Recommendation.DECLINE
        return Recommendation.HOLD

    def rank(self, results: list[ScreeningResult]) -> list[ScreeningResult]:
        """Highest score first; ties broken by id so ordering is stable."""
        return sorted(results, key=lambda r: (-r.score, r.candidate_id))

    # -- interview scheduling ---------------------------------------------

    def schedule_interviews(
        self, candidates: list[Candidate], slots: list[InterviewSlot]
    ) -> InterviewPlan:
        """Match candidates to slots they said they can make.

        Scarcest candidates first -- someone who can only do one slot gets it
        before someone who could take any of five.
        """
        plan = InterviewPlan()
        taken: set[str] = set()
        slot_ids = {slot.id: slot for slot in slots}

        ordered = sorted(
            candidates,
            key=lambda c: (len(c.available_slot_ids & set(slot_ids)), c.id),
        )
        for candidate in ordered:
            options = sorted(candidate.available_slot_ids & (set(slot_ids) - taken))
            if not options:
                plan.unscheduled.append(candidate.id)
                continue
            chosen = options[0]
            taken.add(chosen)
            plan.bookings.append(
                InterviewBooking(
                    candidate_id=candidate.id,
                    slot_id=chosen,
                    interviewer=slot_ids[chosen].interviewer,
                )
            )

        plan.bookings.sort(key=lambda b: b.slot_id)
        plan.unused_slots = sorted(set(slot_ids) - taken)
        return plan
