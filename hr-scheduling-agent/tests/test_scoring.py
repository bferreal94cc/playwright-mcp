"""The objective function: metric maths and the profit/wellbeing weighting."""

from __future__ import annotations

from conftest import make_context, make_employee, make_shift
from hr_scheduling_agent.domain import Assignment, Schedule, ShiftPeriod, UnfilledShift
from hr_scheduling_agent.scoring import (
    ObjectiveWeights,
    _evenness,
    compute_metrics,
    marginal_cost,
    reference_cost,
    score_schedule,
)


class TestWeights:
    def test_pure_profit_zeroes_the_wellbeing_terms(self):
        weights = ObjectiveWeights.balanced(1.0)
        assert weights.cost > 0
        assert weights.preference == 0
        assert weights.fairness == 0
        assert weights.min_hours == 0

    def test_pure_wellbeing_zeroes_the_cost_term(self):
        weights = ObjectiveWeights.balanced(0.0)
        assert weights.cost == 0
        assert weights.preference > 0
        assert weights.fairness > 0

    def test_coverage_weight_is_constant_across_the_dial(self):
        assert ObjectiveWeights.balanced(0.0).coverage == ObjectiveWeights.balanced(1.0).coverage

    def test_coverage_outweighs_everything_else_combined(self):
        """Coverage must never be tradeable against cost or wellbeing."""
        for dial in (0.0, 0.5, 1.0):
            weights = ObjectiveWeights.balanced(dial)
            others = weights.cost + weights.preference + weights.fairness + weights.min_hours
            assert weights.coverage > others


class TestEvenness:
    def test_identical_values_are_perfectly_even(self):
        assert _evenness([10.0, 10.0, 10.0]) == 1.0

    def test_a_single_value_is_even_by_definition(self):
        assert _evenness([5.0]) == 1.0

    def test_all_zeroes_are_even(self):
        assert _evenness([0.0, 0.0]) == 1.0

    def test_spread_lowers_the_score(self):
        assert _evenness([0.0, 20.0]) < _evenness([9.0, 11.0])

    def test_result_stays_within_bounds(self):
        assert 0.0 <= _evenness([0.0, 0.0, 100.0]) <= 1.0


class TestCosting:
    def test_marginal_cost_is_rate_times_hours_below_the_threshold(self):
        context = make_context([make_employee("e1", rate=20.0)], [make_shift("s1", hours=8)])
        employee, shift = context.employees[0], context.shifts[0]
        assert marginal_cost(context, employee, shift, hours_so_far=0.0) == 160.0

    def test_marginal_cost_applies_overtime_past_the_threshold(self):
        context = make_context([make_employee("e1", rate=20.0)], [make_shift("s1", hours=8)])
        employee, shift = context.employees[0], context.shifts[0]
        # Already at 40h, so the whole shift is overtime at 1.5x.
        assert marginal_cost(context, employee, shift, hours_so_far=40.0) == 8 * 20 * 1.5

    def test_reference_cost_uses_the_priciest_rate(self):
        context = make_context(
            [make_employee("cheap", rate=10.0), make_employee("pricey", rate=30.0)],
            [make_shift("s1", hours=8)],
        )
        assert reference_cost(context, context.shifts[0]) == 8 * 30 * 1.5


class TestMetrics:
    def _context(self):
        return make_context(
            [make_employee("e1", rate=20.0), make_employee("e2", rate=20.0)],
            [make_shift("s1", day_offset=0, hours=8), make_shift("s2", day_offset=1, hours=8)],
        )

    def test_fill_rate_reflects_coverage(self):
        context = self._context()
        schedule = Schedule(
            assignments=[Assignment("s1", "e1", 160.0)],
            unfilled=[UnfilledShift("s2", {})],
        )
        assert compute_metrics(context, schedule).fill_rate == 0.5

    def test_empty_demand_counts_as_fully_covered(self):
        context = make_context([make_employee("e1")], [])
        assert compute_metrics(context, Schedule()).fill_rate == 1.0

    def test_cheaper_schedules_score_a_higher_cost_index(self):
        context = self._context()
        cheap = Schedule(assignments=[Assignment("s1", "e1", 100.0)])
        pricey = Schedule(assignments=[Assignment("s1", "e1", 300.0)])
        assert (
            compute_metrics(context, cheap).cost_index
            > compute_metrics(context, pricey).cost_index
        )

    def test_stating_no_preference_never_counts_as_disappointment(self):
        context = self._context()  # neither employee named a preference
        schedule = Schedule(assignments=[Assignment("s1", "e1", 160.0)])
        assert compute_metrics(context, schedule).preference_rate == 1.0

    def test_preference_rate_tracks_matched_periods(self):
        context = make_context(
            [make_employee("owl", preferred={ShiftPeriod.NIGHT})],
            [make_shift("morning", start_hour=9, hours=4)],
        )
        schedule = Schedule(assignments=[Assignment("morning", "owl", 80.0)])
        assert compute_metrics(context, schedule).preference_rate == 0.0

    def test_even_hours_score_higher_than_lopsided_ones(self):
        context = self._context()
        even = Schedule(
            assignments=[Assignment("s1", "e1", 160.0), Assignment("s2", "e2", 160.0)]
        )
        lopsided = Schedule(
            assignments=[Assignment("s1", "e1", 160.0), Assignment("s2", "e1", 160.0)]
        )
        assert (
            compute_metrics(context, even).hours_fairness
            > compute_metrics(context, lopsided).hours_fairness
        )

    def test_minimum_hours_are_tracked(self):
        context = make_context(
            [make_employee("wants-hours", min_hours=20.0)],
            [make_shift("s1", hours=8)],
        )
        schedule = Schedule(assignments=[Assignment("s1", "wants-hours", 160.0)])
        assert compute_metrics(context, schedule).min_hours_met == 0.0

    def test_nobody_asking_for_hours_means_the_target_is_met(self):
        context = self._context()
        assert compute_metrics(context, Schedule()).min_hours_met == 1.0

    def test_prior_hours_carry_into_the_totals(self):
        context = make_context(
            [make_employee("e1")], [make_shift("s1", hours=8)], prior_hours={"e1": 12.0}
        )
        schedule = Schedule(assignments=[Assignment("s1", "e1", 160.0)])
        assert schedule.hours_by_employee(context)["e1"] == 20.0


class TestScoreSchedule:
    def test_breakdown_carries_a_weighted_total(self):
        context = make_context([make_employee("e1")], [make_shift("s1")])
        schedule = Schedule(assignments=[Assignment("s1", "e1", 100.0)])
        total, breakdown = score_schedule(context, schedule, ObjectiveWeights.balanced())
        assert breakdown["weighted_total"] == round(total, 4)
        assert breakdown["fill_rate"] == 1.0

    def test_covered_schedules_outscore_uncovered_ones(self):
        context = make_context(
            [make_employee("e1")],
            [make_shift("s1", day_offset=0), make_shift("s2", day_offset=1)],
        )
        weights = ObjectiveWeights.balanced()
        covered = Schedule(
            assignments=[Assignment("s1", "e1", 100.0), Assignment("s2", "e1", 100.0)]
        )
        partial = Schedule(
            assignments=[Assignment("s1", "e1", 100.0)], unfilled=[UnfilledShift("s2", {})]
        )
        assert (
            score_schedule(context, covered, weights)[0]
            > score_schedule(context, partial, weights)[0]
        )
