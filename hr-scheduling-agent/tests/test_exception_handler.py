"""Real-time cover: rank the options, exclude the absentee, escalate honestly."""

from __future__ import annotations

from datetime import date, datetime, time

from conftest import make_context, make_employee, make_shift
from hr_scheduling_agent.domain import AvailabilityWindow, Certification, ShiftPeriod
from hr_scheduling_agent.exception_handler import (
    ExceptionEvent,
    ExceptionHandler,
    ExceptionType,
)
from hr_scheduling_agent.optimizer import ScheduleOptimizer
from hr_scheduling_agent.scenarios import build_retail_context


def call_out(shift_id: str, employee_id: str) -> ExceptionEvent:
    return ExceptionEvent(
        type=ExceptionType.CALL_OUT,
        shift_id=shift_id,
        employee_id=employee_id,
        reported_at=datetime(2026, 8, 17, 8, 0),
        note="unwell",
    )


class TestResolution:
    def test_offers_a_replacement(self):
        context = make_context(
            [make_employee("working"), make_employee("spare")],
            [make_shift("s1")],
        )
        schedule = ScheduleOptimizer(context).solve()
        holder = schedule.assignments[0].employee_id
        other = "spare" if holder == "working" else "working"

        resolution = ExceptionHandler(context).resolve(schedule, call_out("s1", holder))
        assert resolution.is_coverable
        assert resolution.recommended.employee_id == other

    def test_absentee_is_never_offered_as_their_own_cover(self):
        context = make_context([make_employee("only")], [make_shift("s1")])
        schedule = ScheduleOptimizer(context).solve()

        resolution = ExceptionHandler(context).resolve(schedule, call_out("s1", "only"))
        assert not resolution.is_coverable
        assert "reported unavailable" in resolution.blockers["only"]

    def test_escalates_with_reasons_when_nobody_qualifies(self):
        context = make_context(
            [make_employee("holder", roles={"server"}), make_employee("cook", roles={"cook"})],
            [make_shift("s1", role="server")],
        )
        schedule = ScheduleOptimizer(context).solve()

        resolution = ExceptionHandler(context).resolve(schedule, call_out("s1", "holder"))
        assert not resolution.is_coverable
        summary = resolution.escalation_summary()
        assert "No legal coverage" in summary
        assert "not qualified" in summary

    def test_respects_hard_constraints_when_covering(self):
        """A replacement who would breach the rest rule is not offered."""
        holder = make_employee("holder")
        exhausted = make_employee("exhausted")
        late = make_shift("late", day_offset=0, start_hour=14, hours=8)  # ends 22:00
        early = make_shift("early", day_offset=1, start_hour=4, hours=6)  # 6h gap

        context = make_context([holder, exhausted], [late, early])
        schedule = ScheduleOptimizer(context).solve()
        early_holder = schedule.assignment_for_shift("early").employee_id
        late_holder = schedule.assignment_for_shift("late").employee_id

        resolution = ExceptionHandler(context).resolve(schedule, call_out("early", early_holder))
        offered = {option.employee_id for option in resolution.options}
        assert late_holder not in offered, "must not offer someone owed rest"

    def test_reports_how_long_triage_took(self):
        context = build_retail_context()
        schedule = ScheduleOptimizer(context).solve()
        target = schedule.assignments[0]
        resolution = ExceptionHandler(context).resolve(
            schedule, call_out(target.shift_id, target.employee_id)
        )
        assert resolution.resolved_in_ms >= 0.0

    def test_options_are_capped(self):
        context = build_retail_context()
        schedule = ScheduleOptimizer(context).solve()
        target = schedule.assignments[0]
        handler = ExceptionHandler(context, max_options=2)
        resolution = handler.resolve(schedule, call_out(target.shift_id, target.employee_id))
        assert len(resolution.options) <= 2

    def test_unknown_shift_is_rejected(self):
        import pytest

        context = build_retail_context()
        schedule = ScheduleOptimizer(context).solve()
        with pytest.raises(ValueError):
            ExceptionHandler(context).resolve(schedule, call_out("nope", "r01"))


class TestCostDelta:
    def test_cheaper_replacement_shows_a_saving(self):
        pricey = make_employee("pricey", rate=50.0)
        cheap = make_employee("cheap", rate=10.0)
        context = make_context([pricey, cheap], [make_shift("s1", hours=8)])
        # Pin the expensive person so the cheap one is the replacement.
        schedule = ScheduleOptimizer(context).solve(locked={"s1": "pricey"})

        resolution = ExceptionHandler(context).resolve(schedule, call_out("s1", "pricey"))
        assert resolution.recommended.employee_id == "cheap"
        assert resolution.recommended.cost_delta < 0


class TestApply:
    def test_apply_replaces_the_assignment(self):
        context = make_context(
            [make_employee("working"), make_employee("spare")], [make_shift("s1")]
        )
        schedule = ScheduleOptimizer(context).solve()
        holder = schedule.assignments[0].employee_id
        event = call_out("s1", holder)

        handler = ExceptionHandler(context)
        resolution = handler.resolve(schedule, event)
        handler.apply(schedule, event, resolution.recommended)

        assert len(schedule.assignments) == 1, "coverage count must be preserved"
        assignment = schedule.assignment_for_shift("s1")
        assert assignment.employee_id != holder
        assert "exception cover" in assignment.rationale

    def test_apply_clears_a_matching_gap(self):
        context = make_context(
            [make_employee("a"), make_employee("b")], [make_shift("s1")]
        )
        schedule = ScheduleOptimizer(context).solve()
        holder = schedule.assignments[0].employee_id
        event = call_out("s1", holder)
        handler = ExceptionHandler(context)
        resolution = handler.resolve(schedule, event)
        handler.apply(schedule, event, resolution.recommended)
        assert all(u.shift_id != "s1" for u in schedule.unfilled)

    def test_full_scenario_cover_keeps_schedule_legal(self):
        from test_optimizer import assert_schedule_is_legal

        context = build_retail_context()
        schedule = ScheduleOptimizer(context).solve()
        target = schedule.assignments[len(schedule.assignments) // 2]
        event = call_out(target.shift_id, target.employee_id)

        handler = ExceptionHandler(context)
        resolution = handler.resolve(schedule, event)
        assert resolution.is_coverable
        handler.apply(schedule, event, resolution.recommended)
        assert_schedule_is_legal(context, schedule)
