"""Hiring: postings that survive an offline laptop, scores that explain themselves."""

from __future__ import annotations

from datetime import datetime, timedelta

from hr_scheduling_agent.domain import TimeWindow
from hr_scheduling_agent.hiring import (
    Candidate,
    InterviewAgent,
    InterviewSlot,
    Recommendation,
)
from hr_scheduling_agent.providers import StubProvider

BASE = datetime(2026, 8, 17, 10, 0)


def slot(identifier: str, offset_hours: int, interviewer: str = "mgr") -> InterviewSlot:
    start = BASE + timedelta(hours=offset_hours)
    return InterviewSlot(identifier, interviewer, TimeWindow(start, start + timedelta(minutes=45)))


class TestPostingDrafting:
    def test_template_is_used_without_a_provider(self):
        posting = InterviewAgent().draft_posting("cook", "store-01", headcount=2)
        assert posting.source == "template"
        assert "2 cooks" in posting.body
        assert "store-01" in posting.title

    def test_singular_wording_for_one_opening(self):
        posting = InterviewAgent().draft_posting("server", "store-01", headcount=1)
        assert "hiring 1 server at" in posting.body
        assert "opening)" in posting.title

    def test_must_haves_appear_in_the_body(self):
        posting = InterviewAgent().draft_posting(
            "rn", "ward-3", must_haves=("Valid RN licence", "BLS certification")
        )
        assert "Valid RN licence" in posting.body
        assert "BLS certification" in posting.body

    def test_llm_output_is_used_when_available(self):
        provider = StubProvider(default='{"title":"Great Cook Wanted","body":"Come cook."}')
        posting = InterviewAgent(provider).draft_posting("cook", "store-01")
        assert posting.title == "Great Cook Wanted"
        assert posting.source == "llm:stub"

    def test_falls_back_when_the_llm_returns_nothing_useful(self):
        posting = InterviewAgent(StubProvider(default="{}")).draft_posting("cook", "store-01")
        assert posting.source == "template"

    def test_falls_back_when_the_provider_raises(self):
        class Broken(StubProvider):
            def complete(self, *args, **kwargs):
                raise RuntimeError("down")

        posting = InterviewAgent(Broken()).draft_posting("cook", "store-01")
        assert posting.source == "template"


class TestScreeningQuestions:
    def test_includes_the_role_and_the_must_haves(self):
        questions = InterviewAgent().screening_questions("cook", ("Food hygiene level 2",))
        assert any("cook" in q for q in questions)
        assert any("Food hygiene level 2" in q for q in questions)


class TestScoring:
    def test_a_fully_qualified_candidate_advances(self):
        candidate = Candidate(
            "c1", "Robin", 5.0, frozenset({"rn"}), frozenset({"BLS", "ACLS"})
        )
        result = InterviewAgent().score_screening(
            candidate, frozenset({"rn"}), frozenset({"BLS", "ACLS"}), minimum_years=2.0
        )
        assert result.score == 100.0
        assert result.recommendation is Recommendation.ADVANCE
        assert result.gaps == ()

    def test_missing_credentials_lower_the_score_and_are_named(self):
        candidate = Candidate("c2", "Sam", 5.0, frozenset({"rn"}), frozenset())
        result = InterviewAgent().score_screening(
            candidate, frozenset({"rn"}), frozenset({"BLS"}), minimum_years=2.0
        )
        assert result.score < 100.0
        assert any("BLS" in gap for gap in result.gaps)

    def test_thin_experience_is_reported(self):
        candidate = Candidate("c3", "Wren", 1.0, frozenset({"cook"}))
        result = InterviewAgent().score_screening(
            candidate, frozenset({"cook"}), minimum_years=4.0
        )
        assert any("years experience" in gap for gap in result.gaps)

    def test_a_weak_candidate_is_held_not_auto_rejected(self):
        candidate = Candidate("c4", "Alex", 0.0, frozenset(), frozenset())
        result = InterviewAgent().score_screening(
            candidate, frozenset({"rn"}), frozenset({"BLS"}), minimum_years=5.0
        )
        assert result.recommendation is Recommendation.DECLINE
        assert result.score < 40.0

    def test_no_requirements_means_full_marks(self):
        result = InterviewAgent().score_screening(Candidate("c5", "Kit", 0.0))
        assert result.score == 100.0

    def test_ranking_is_stable_and_descending(self):
        agent = InterviewAgent()
        results = [
            agent.score_screening(Candidate("low", "L", 0.0), frozenset({"cook"})),
            agent.score_screening(Candidate("high", "H", 0.0, frozenset({"cook"})), frozenset({"cook"})),
        ]
        ranked = agent.rank(results)
        assert [r.candidate_id for r in ranked] == ["high", "low"]

    def test_describe_mentions_score_and_recommendation(self):
        result = InterviewAgent().score_screening(
            Candidate("c1", "Robin", 5.0, frozenset({"rn"})), frozenset({"rn"})
        )
        text = result.describe()
        assert "100" in text and "advance" in text


class TestInterviewScheduling:
    def test_books_each_candidate_into_a_slot_they_can_make(self):
        candidates = [
            Candidate("c1", "A", 1.0, available_slot_ids=frozenset({"s1", "s2"})),
            Candidate("c2", "B", 1.0, available_slot_ids=frozenset({"s2", "s3"})),
        ]
        plan = InterviewAgent().schedule_interviews(
            candidates, [slot("s1", 1), slot("s2", 2), slot("s3", 3)]
        )
        assert len(plan.bookings) == 2
        assert plan.unscheduled == []
        for booking in plan.bookings:
            candidate = next(c for c in candidates if c.id == booking.candidate_id)
            assert booking.slot_id in candidate.available_slot_ids

    def test_no_slot_is_double_booked(self):
        candidates = [
            Candidate("c1", "A", 1.0, available_slot_ids=frozenset({"s1"})),
            Candidate("c2", "B", 1.0, available_slot_ids=frozenset({"s1"})),
        ]
        plan = InterviewAgent().schedule_interviews(candidates, [slot("s1", 1)])
        assert len(plan.bookings) == 1
        assert plan.unscheduled == ["c2"]

    def test_the_least_flexible_candidate_is_placed_first(self):
        """Someone who can only make one slot must not be squeezed out."""
        flexible = Candidate("flex", "F", 1.0, available_slot_ids=frozenset({"s1", "s2"}))
        constrained = Candidate("tight", "T", 1.0, available_slot_ids=frozenset({"s1"}))
        plan = InterviewAgent().schedule_interviews(
            [flexible, constrained], [slot("s1", 1), slot("s2", 2)]
        )
        assert plan.unscheduled == []
        booked = {b.candidate_id: b.slot_id for b in plan.bookings}
        assert booked["tight"] == "s1"
        assert booked["flex"] == "s2"

    def test_candidates_with_no_workable_slot_are_reported(self):
        candidate = Candidate("c1", "A", 1.0, available_slot_ids=frozenset({"nope"}))
        plan = InterviewAgent().schedule_interviews([candidate], [slot("s1", 1)])
        assert plan.unscheduled == ["c1"]
        assert plan.unused_slots == ["s1"]

    def test_interviewer_is_carried_onto_the_booking(self):
        candidate = Candidate("c1", "A", 1.0, available_slot_ids=frozenset({"s1"}))
        plan = InterviewAgent().schedule_interviews(
            [candidate], [slot("s1", 1, interviewer="director")]
        )
        assert plan.bookings[0].interviewer == "director"

    def test_empty_inputs_are_handled(self):
        plan = InterviewAgent().schedule_interviews([], [])
        assert plan.bookings == [] and plan.unscheduled == []
