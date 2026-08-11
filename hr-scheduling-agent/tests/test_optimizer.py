"""The optimizer must never produce an illegal schedule, and must be honest
about what it could not fill."""

from __future__ import annotations

from datetime import date, time

from conftest import make_context, make_employee, make_shift
from hr_scheduling_agent.constraints import ConstraintEngine
from hr_scheduling_agent.domain import AvailabilityWindow, Certification, CompanyPolicy, ShiftPeriod
from hr_scheduling_agent.optimizer import ScheduleOptimizer
from hr_scheduling_agent.scenarios import build_healthcare_context, build_retail_context
from hr_scheduling_agent.scoring import ObjectiveWeights, compute_metrics


def assert_schedule_is_legal(context, schedule):
    """Independently re-verify every assignment against the constraint engine."""
    engine = ConstraintEngine(context)
    held: dict[str, list] = {}
    for assignment in schedule.assignments:
        shift = context.shift_by_id(assignment.shift_id)
        held.setdefault(assignment.employee_id, []).append(shift)

    for assignment in schedule.assignments:
        employee = context.employee_by_id(assignment.employee_id)
        shift = context.shift_by_id(assignment.shift_id)
        others = [s for s in held[assignment.employee_id] if s.id != shift.id]
        result = engine.check(employee, shift, others)
        assert result.ok, f"illegal assignment {employee.id}->{shift.id}: {result.reason}"


class TestBasicSolving:
    def test_fills_what_it_can(self, simple_context):
        schedule = ScheduleOptimizer(simple_context).solve()
        assert len(schedule.assignments) == 2
        assert schedule.unfilled == []

    def test_never_double_books(self):
        employee = make_employee("only")
        overlapping = [
            make_shift("a", start_hour=9, hours=8),
            make_shift("b", start_hour=12, hours=4),
        ]
        context = make_context([employee], overlapping)
        schedule = ScheduleOptimizer(context).solve()
        assert len(schedule.assignments) == 1
        assert len(schedule.unfilled) == 1
        assert_schedule_is_legal(context, schedule)

    def test_unfilled_shift_explains_itself(self):
        context = make_context(
            [make_employee("cook-only", roles={"cook"})],
            [make_shift("needs-server", role="server")],
        )
        schedule = ScheduleOptimizer(context).solve()
        assert len(schedule.unfilled) == 1
        assert "not qualified" in schedule.unfilled[0].summary()

    def test_empty_roster_leaves_everything_unfilled(self):
        context = make_context([], [make_shift("s1")])
        schedule = ScheduleOptimizer(context).solve()
        assert schedule.assignments == []
        assert schedule.unfilled[0].summary() == "no candidates in the roster"

    def test_assignments_carry_a_rationale(self, simple_context):
        schedule = ScheduleOptimizer(simple_context).solve()
        for assignment in schedule.assignments:
            assert assignment.rationale, "every assignment must explain itself"
            assert "eligible" in assignment.rationale

    def test_is_deterministic(self):
        first = ScheduleOptimizer(build_retail_context()).solve()
        second = ScheduleOptimizer(build_retail_context()).solve()
        assert {(a.shift_id, a.employee_id) for a in first.assignments} == {
            (a.shift_id, a.employee_id) for a in second.assignments
        }


class TestScarcityOrdering:
    def test_scarce_specialist_is_not_stranded(self):
        """The one certified person must be saved for the shift needing them."""
        specialist = make_employee(
            "specialist",
            roles={"nurse"},
            rate=50.0,
            certifications=(Certification("ACLS", date(2027, 1, 1)),),
        )
        generalist = make_employee("generalist", roles={"nurse"}, rate=40.0)
        # Both shifts run at the same time, so one person can only take one.
        needs_cert = make_shift("critical", start_hour=9, role="nurse", certifications={"ACLS"})
        general = make_shift("general", start_hour=9, role="nurse")

        context = make_context([specialist, generalist], [general, needs_cert])
        schedule = ScheduleOptimizer(context).solve()

        assert len(schedule.assignments) == 2
        assert schedule.assignment_for_shift("critical").employee_id == "specialist"
        assert schedule.assignment_for_shift("general").employee_id == "generalist"


class TestObjectiveTradeoff:
    def test_profit_mode_prefers_the_cheaper_worker(self):
        cheap = make_employee("cheap", rate=15.0)
        pricey = make_employee("pricey", rate=45.0)
        context = make_context([cheap, pricey], [make_shift("s1")])
        schedule = ScheduleOptimizer(context, ObjectiveWeights.balanced(1.0)).solve()
        assert schedule.assignments[0].employee_id == "cheap"

    def test_wellbeing_mode_honours_a_stated_preference(self):
        """With cost off the table, the person who wants nights gets the night."""
        night_owl = make_employee("owl", rate=40.0, preferred={ShiftPeriod.NIGHT})
        early_bird = make_employee("bird", rate=20.0, preferred={ShiftPeriod.MORNING})
        context = make_context([night_owl, early_bird], [make_shift("night", start_hour=18, hours=4)])
        schedule = ScheduleOptimizer(context, ObjectiveWeights.balanced(0.0)).solve()
        assert schedule.assignments[0].employee_id == "owl"

    def test_dial_moves_cost_and_wellbeing_in_opposite_directions(self):
        profit_context = build_retail_context()
        profit = ScheduleOptimizer(profit_context, ObjectiveWeights.balanced(1.0)).solve()
        profit_metrics = compute_metrics(profit_context, profit)

        people_context = build_retail_context()
        people = ScheduleOptimizer(people_context, ObjectiveWeights.balanced(0.0)).solve()
        people_metrics = compute_metrics(people_context, people)

        assert profit_metrics.total_cost < people_metrics.total_cost
        assert people_metrics.preference_rate > profit_metrics.preference_rate
        assert people_metrics.hours_fairness > profit_metrics.hours_fairness

    def test_coverage_is_never_traded_away(self):
        """Even in pure-profit mode every fillable shift is still filled."""
        context = build_retail_context()
        schedule = ScheduleOptimizer(context, ObjectiveWeights.balanced(1.0)).solve()
        assert schedule.unfilled == []

    def test_rejects_out_of_range_dial(self):
        import pytest

        with pytest.raises(ValueError):
            ObjectiveWeights.balanced(1.5)


class TestLockedAssignments:
    def test_pinned_assignments_are_preserved(self):
        context = make_context(
            [make_employee("e1", rate=10.0), make_employee("e2", rate=90.0)],
            [make_shift("s1"), make_shift("s2", day_offset=1)],
        )
        # Pin the expensive person onto s1 even though they cost more.
        schedule = ScheduleOptimizer(context).solve(locked={"s1": "e2"})
        assert schedule.assignment_for_shift("s1").employee_id == "e2"
        assert len(schedule.assignments) == 2


class TestFullScenarios:
    def test_retail_is_fully_covered_and_legal(self):
        context = build_retail_context()
        schedule = ScheduleOptimizer(context).solve()
        assert schedule.unfilled == []
        assert_schedule_is_legal(context, schedule)

    def test_healthcare_is_fully_covered_and_legal(self):
        context = build_healthcare_context()
        schedule = ScheduleOptimizer(context).solve()
        assert schedule.unfilled == []
        assert_schedule_is_legal(context, schedule)

    def test_healthcare_respects_the_twelve_hour_rest_rule(self):
        context = build_healthcare_context()
        schedule = ScheduleOptimizer(context).solve()
        by_employee: dict[str, list] = {}
        for assignment in schedule.assignments:
            shift = context.shift_by_id(assignment.shift_id)
            by_employee.setdefault(assignment.employee_id, []).append(shift)

        minimum = context.policy.min_rest_hours_between_shifts
        for shifts in by_employee.values():
            ordered = sorted(shifts, key=lambda s: s.window.start)
            for first, second in zip(ordered, ordered[1:]):
                assert first.window.gap_hours_to(second.window) >= minimum

    def test_nobody_exceeds_the_weekly_cap(self):
        context = build_healthcare_context()
        schedule = ScheduleOptimizer(context).solve()
        for employee_id, hours in schedule.hours_by_employee(context).items():
            employee = context.employee_by_id(employee_id)
            assert hours <= context.max_hours_for(employee, "hospital-north") + 1e-9

    def test_lapsed_credential_keeps_that_nurse_off_the_floor(self):
        """The fixture's nurse 14 has an expired ACLS and must not be scheduled."""
        context = build_healthcare_context()
        schedule = ScheduleOptimizer(context).solve()
        rn_shifts = {s.id for s in context.shifts if s.role == "rn"}
        assigned_rn = {
            a.employee_id for a in schedule.assignments if a.shift_id in rn_shifts
        }
        assert "n14" not in assigned_rn
