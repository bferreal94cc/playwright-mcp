"""Orchestrator routing and the full build → audit → approve → publish loop."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from hr_scheduling_agent.approvals import Decision, RequestState
from hr_scheduling_agent.cli import main
from hr_scheduling_agent.exception_handler import ExceptionEvent, ExceptionType
from hr_scheduling_agent.orchestrator import Orchestrator
from hr_scheduling_agent.providers import StubProvider
from hr_scheduling_agent.scenarios import (
    WEEK_START,
    build_healthcare_context,
    build_retail_context,
)


@pytest.fixture
def retail() -> Orchestrator:
    return Orchestrator(build_retail_context(), provider=StubProvider())


class TestRouting:
    def test_create_schedule_reaches_the_optimizer(self, retail):
        result = retail.handle("build the schedule for next week", "manager-01", WEEK_START)
        assert result.agent == "schedule_optimizer"
        assert result.ok
        assert retail.schedule is not None

    def test_post_job_reaches_the_interview_agent(self, retail):
        result = retail.handle("we need to hire two more cooks", "manager-01", WEEK_START)
        assert result.agent == "interview_agent"
        posting = result.data["posting"]
        assert posting.role == "cook"
        assert posting.headcount == 2

    def test_query_schedule_lists_that_person_only(self, retail):
        retail.build_schedule()
        result = retail.handle("when am I working?", "r01", WEEK_START)
        assert result.ok
        expected = len(retail.schedule.assignments_for("r01"))
        assert len(result.data["shifts"]) == expected

    def test_time_off_request_is_logged(self, retail):
        result = retail.handle("I need a day off on aug 20", "r01", WEEK_START)
        assert result.action == "time_off_request"
        assert result.data["dates"] == ["2026-08-20"]

    def test_unintelligible_input_asks_for_a_rephrase(self, retail):
        result = retail.handle("the quarterly figures", "r01", WEEK_START)
        assert not result.ok
        assert result.action == "clarify"

    def test_every_request_is_audited(self, retail):
        retail.handle("build the schedule", "manager-01", WEEK_START)
        received = [e for e in retail.audit.entries if e.action == "request_received"]
        assert len(received) == 1
        assert received[0].actor == "manager-01"

    def test_absence_routing_finds_the_right_shift(self, retail):
        retail.build_schedule()
        assignment = retail.schedule.assignments[0]
        shift = retail.context.shift_by_id(assignment.shift_id)
        day = shift.window.start.date()

        result = retail.handle(
            f"I'm sick, calling out for {day.isoformat()}", assignment.employee_id, WEEK_START
        )
        assert result.agent == "exception_handler"

    def test_absence_on_a_day_off_is_reported_honestly(self, retail):
        retail.build_schedule()
        result = retail.handle(
            "calling out for 2027-01-01", retail.context.employees[0].id, WEEK_START
        )
        assert not result.ok
        assert "No scheduled shift" in result.summary


class TestGuards:
    def test_actions_needing_a_schedule_fail_cleanly(self):
        orchestrator = Orchestrator(build_retail_context())
        for result in (
            orchestrator.audit_schedule(),
            orchestrator.submit_for_approval(),
            orchestrator.publish(),
            orchestrator.describe_schedule_for("r01"),
        ):
            assert not result.ok
            assert "No schedule has been built" in result.summary


class TestFullLoop:
    def test_build_audit_approve_publish(self, retail):
        built = retail.build_schedule()
        assert built.ok
        assert retail.schedule.unfilled == []

        audited = retail.audit_schedule()
        assert audited.ok, audited.summary

        t0 = datetime(2026, 8, 14, 9, 0)
        submitted = retail.submit_for_approval("manager-01", at=t0)
        request = submitted.data["request"]
        retail.approvals.decide(request, "manager-01", Decision.APPROVE, "ok", at=t0)
        retail.approvals.tick(request, t0 + timedelta(hours=9))
        assert request.state is RequestState.APPROVED

        published = retail.publish()
        assert published.ok
        assert published.data["hris_records"] == len(retail.schedule.assignments)
        assert published.data["notified"] > 0
        assert retail.audit.verify().valid

    def test_exception_cover_end_to_end(self, retail):
        retail.build_schedule()
        target = retail.schedule.assignments[0]
        event = ExceptionEvent(
            type=ExceptionType.CALL_OUT,
            shift_id=target.shift_id,
            employee_id=target.employee_id,
            reported_at=datetime.now(),
            note="unwell",
        )
        resolved = retail.resolve_exception(event)
        assert resolved.ok

        applied = retail.apply_cover(event, resolved.data["resolution"])
        assert applied.ok
        assert retail.schedule.assignment_for_shift(target.shift_id).employee_id != target.employee_id

        # Both parties hear about it, and the cover goes out urgently.
        recipients = {n.recipient_id for n in retail.notifications.sent}
        assert target.employee_id in recipients
        assert any(n.urgency.value == "urgent" for n in retail.notifications.sent)

    def test_cover_is_recorded_in_the_audit_trail(self, retail):
        retail.build_schedule()
        target = retail.schedule.assignments[0]
        event = ExceptionEvent(
            ExceptionType.CALL_OUT, target.shift_id, target.employee_id, datetime.now()
        )
        resolution = retail.resolve_exception(event).data["resolution"]
        retail.apply_cover(event, resolution)
        actions = [e.action for e in retail.audit.entries]
        assert "exception_triaged" in actions
        assert "cover_applied" in actions
        assert retail.audit.verify().valid

    def test_healthcare_loop_is_compliant(self):
        orchestrator = Orchestrator(build_healthcare_context())
        assert orchestrator.build_schedule().ok
        audited = orchestrator.audit_schedule()
        assert audited.ok, audited.summary
        assert audited.data["report"].violations == []

    def test_group_disparity_reporting_is_opt_in(self):
        orchestrator = Orchestrator(build_retail_context())
        orchestrator.build_schedule()
        without = orchestrator.audit_schedule()
        assert without.data["report"].group_disparities == []

        groups = {e.id: ("a" if i % 2 else "b") for i, e in enumerate(orchestrator.context.employees)}
        with_groups = orchestrator.audit_schedule(groups=groups)
        assert isinstance(with_groups.data["report"].group_disparities, list)


class TestCLI:
    def test_both_scenarios_run_clean(self):
        assert main(["--scenario", "both", "--days", "3"]) == 0

    def test_profit_mode_runs_clean(self):
        assert main(["--scenario", "retail", "--days", "2", "--profit-weight", "1.0"]) == 0

    def test_group_reporting_flag_runs(self):
        assert main(["--scenario", "healthcare", "--days", "2", "--groups"]) == 0
