"""The auditor's job is to catch what the optimizer could not have prevented."""

from __future__ import annotations

from datetime import date, time

from conftest import MONDAY, make_context, make_employee, make_shift
from hr_scheduling_agent.compliance import ComplianceAuditor, Severity
from hr_scheduling_agent.domain import (
    Assignment,
    AvailabilityWindow,
    Certification,
    CompanyPolicy,
    Schedule,
    UnfilledShift,
)
from hr_scheduling_agent.optimizer import ScheduleOptimizer
from hr_scheduling_agent.scenarios import build_healthcare_context, build_retail_context


class TestCleanSchedules:
    def test_optimizer_output_passes_its_own_audit(self):
        for context in (build_retail_context(), build_healthcare_context()):
            schedule = ScheduleOptimizer(context).solve()
            report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
            assert report.violations == []
            assert report.is_compliant


class TestViolationDetection:
    def test_catches_a_hand_edited_illegal_assignment(self):
        """A manager overriding the agent must not slip past the auditor."""
        context = make_context(
            [make_employee("server-only", roles={"server"})],
            [make_shift("needs-cook", role="cook")],
        )
        schedule = Schedule(
            assignments=[Assignment("needs-cook", "server-only", cost=100.0)]
        )
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        assert len(report.violations) == 1
        assert "not qualified" in report.violations[0].rule
        assert not report.is_compliant

    def test_catches_an_overlapping_double_booking(self):
        context = make_context(
            [make_employee("e1")],
            [make_shift("a", start_hour=9, hours=8), make_shift("b", start_hour=12, hours=4)],
        )
        schedule = Schedule(
            assignments=[
                Assignment("a", "e1", cost=0.0),
                Assignment("b", "e1", cost=0.0),
            ]
        )
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        assert any("overlapping" in v.rule for v in report.violations)

    def test_catches_a_weekly_cap_breach(self):
        context = make_context(
            [make_employee("e1")],
            [make_shift(f"s{i}", day_offset=i, hours=8) for i in range(6)],
            policy=CompanyPolicy(max_hours_per_week=40, max_consecutive_days=7),
        )
        schedule = Schedule(
            assignments=[Assignment(f"s{i}", "e1", cost=0.0) for i in range(6)]
        )
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        assert any("weekly cap" in v.rule for v in report.violations)

    def test_flags_an_unknown_employee(self):
        context = make_context([make_employee("e1")], [make_shift("s1")])
        schedule = Schedule(assignments=[Assignment("s1", "ghost", cost=0.0)])
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        assert any("unknown" in v.rule for v in report.violations)

    def test_an_assignment_is_not_compared_with_itself(self):
        """A single legal assignment must not be reported as self-overlapping."""
        context = make_context([make_employee("e1")], [make_shift("s1")])
        schedule = Schedule(assignments=[Assignment("s1", "e1", cost=0.0)])
        assert ComplianceAuditor(context).audit(schedule, as_of=MONDAY).violations == []


class TestCredentials:
    def _context_with_expiry(self, expires_on: date):
        employee = make_employee(
            "nurse", roles={"rn"}, certifications=(Certification("BLS", expires_on),)
        )
        return make_context([employee], [make_shift("s1", role="rn", certifications={"BLS"})])

    def test_flags_a_credential_expiring_within_the_horizon(self):
        context = self._context_with_expiry(date(2026, 9, 1))  # 15 days out
        schedule = ScheduleOptimizer(context).solve()
        report = ComplianceAuditor(context, credential_horizon_days=30).audit(
            schedule, as_of=MONDAY
        )
        assert len(report.expiring) == 1
        alert = report.expiring[0]
        assert alert.certification == "BLS"
        assert alert.days_remaining == 15
        assert alert.severity is Severity.WARNING
        assert report.is_compliant, "a future expiry is a warning, not a violation"

    def test_ignores_a_credential_beyond_the_horizon(self):
        context = self._context_with_expiry(date(2027, 1, 1))
        schedule = ScheduleOptimizer(context).solve()
        report = ComplianceAuditor(context, credential_horizon_days=30).audit(
            schedule, as_of=MONDAY
        )
        assert report.expiring == []

    def test_a_lapsed_credential_on_a_scheduled_worker_is_a_violation(self):
        context = self._context_with_expiry(date(2026, 8, 10))  # already expired
        # The optimizer would refuse this, so simulate a manual override.
        schedule = Schedule(assignments=[Assignment("s1", "nurse", cost=0.0)])
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        assert not report.is_compliant
        assert any(c.severity is Severity.VIOLATION for c in report.expiring)

    def test_unscheduled_staff_do_not_raise_credential_noise(self):
        working = make_employee(
            "working", roles={"rn"}, certifications=(Certification("BLS", date(2027, 1, 1)),)
        )
        benched = make_employee(
            "benched", roles={"rn"}, certifications=(Certification("BLS", date(2026, 8, 20)),)
        )
        context = make_context([working, benched], [make_shift("s1", role="rn")])
        schedule = Schedule(assignments=[Assignment("s1", "working", cost=0.0)])
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        assert report.expiring == []


class TestWorkloadOutliers:
    def test_flags_someone_carrying_far_more_hours(self):
        context = make_context(
            [make_employee("busy"), make_employee("idle")],
            [make_shift(f"s{i}", day_offset=i, hours=8) for i in range(4)],
        )
        schedule = Schedule(
            assignments=[Assignment(f"s{i}", "busy", cost=0.0) for i in range(4)]
        )
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        metrics = {f.metric for f in report.workload_outliers}
        assert "weekly_hours" in metrics
        assert any(f.employee_id == "busy" for f in report.workload_outliers)

    def test_even_distribution_raises_nothing(self):
        context = make_context(
            [make_employee("a"), make_employee("b")],
            [make_shift("s1", day_offset=0), make_shift("s2", day_offset=1)],
        )
        schedule = Schedule(
            assignments=[Assignment("s1", "a", cost=0.0), Assignment("s2", "b", cost=0.0)]
        )
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        assert [f for f in report.workload_outliers if f.metric == "weekly_hours"] == []


class TestGroupDisparity:
    def _night_heavy_schedule(self):
        employees = [make_employee(f"e{i}") for i in range(4)]
        # Two weekend shifts, both landing on the same group.
        shifts = [
            make_shift("wk1", day_offset=5, hours=6),
            make_shift("wk2", day_offset=6, hours=6),
            make_shift("wd1", day_offset=1, hours=6),
            make_shift("wd2", day_offset=2, hours=6),
        ]
        context = make_context(employees, shifts)
        schedule = Schedule(
            assignments=[
                Assignment("wk1", "e0", cost=0.0),
                Assignment("wk2", "e1", cost=0.0),
                Assignment("wd1", "e2", cost=0.0),
                Assignment("wd2", "e3", cost=0.0),
            ]
        )
        return context, schedule

    def test_reports_a_disparity_between_supplied_groups(self):
        context, schedule = self._night_heavy_schedule()
        groups = {"e0": "team-a", "e1": "team-a", "e2": "team-b", "e3": "team-b"}
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY, groups=groups)
        unsocial = [f for f in report.group_disparities if f.metric == "unsocial_shifts"]
        assert unsocial, "team-a took every weekend shift and that should be reported"
        assert {f.group for f in unsocial} == {"team-a", "team-b"}

    def test_no_groups_means_no_disparity_findings(self):
        """Group labels are never inferred, so without them there is nothing to compare."""
        context, schedule = self._night_heavy_schedule()
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        assert report.group_disparities == []

    def test_a_single_group_is_not_compared_with_itself(self):
        context, schedule = self._night_heavy_schedule()
        groups = {f"e{i}": "everyone" for i in range(4)}
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY, groups=groups)
        assert report.group_disparities == []


class TestReporting:
    def test_uncovered_shifts_are_carried_into_the_report(self):
        context = make_context([make_employee("e1")], [make_shift("s1")])
        schedule = Schedule(unfilled=[UnfilledShift("s1", {"e1": "outside availability"})])
        report = ComplianceAuditor(context).audit(schedule, as_of=MONDAY)
        assert report.uncovered_shifts == ["s1"]

    def test_summary_counts_everything(self):
        context = make_context([make_employee("e1")], [make_shift("s1")])
        schedule = Schedule(assignments=[Assignment("s1", "e1", cost=0.0)])
        summary = ComplianceAuditor(context).audit(schedule, as_of=MONDAY).summary()
        assert "violation" in summary and "credential" in summary
