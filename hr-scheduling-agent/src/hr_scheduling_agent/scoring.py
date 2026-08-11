"""The multi-objective function: company profit balanced against employee wellbeing.

Coverage is scored separately and weighted above both, because an unfilled
shift is an operational failure rather than a trade-off. Within the remainder,
``profit_weight`` slides between the two goals the product promises to hold
together:

* profit    -- labour cost, overtime avoidance, budget adherence
* wellbeing -- preferred shift periods, even hours, even share of night and
               weekend work, and hitting each person's minimum hours

Every number produced here is normalised to ``[0, 1]`` and higher is better, so
the weights stay comparable and a score breakdown is readable at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from .domain import Employee, SchedulingContext, Schedule, Shift


@dataclass(frozen=True)
class ObjectiveWeights:
    """Relative importance of each objective. Need not sum to one."""

    coverage: float = 4.0
    cost: float = 1.0
    preference: float = 0.6
    fairness: float = 0.8
    min_hours: float = 0.4

    @classmethod
    def balanced(cls, profit_weight: float = 0.5) -> "ObjectiveWeights":
        """Build weights from a single profit/wellbeing dial in ``[0, 1]``.

        ``profit_weight=1`` optimises purely for labour cost; ``0`` optimises
        purely for the people. Coverage is unaffected -- it is never traded.
        """
        if not 0.0 <= profit_weight <= 1.0:
            raise ValueError("profit_weight must be between 0 and 1")
        wellbeing = 1.0 - profit_weight
        return cls(
            coverage=4.0,
            cost=2.0 * profit_weight,
            preference=1.0 * wellbeing,
            fairness=1.2 * wellbeing,
            min_hours=0.6 * wellbeing,
        )


@dataclass(frozen=True)
class ScheduleMetrics:
    """Interpretable view of a schedule, independent of the weighting used."""

    fill_rate: float
    total_cost: float
    cost_index: float
    preference_rate: float
    hours_fairness: float
    load_fairness: float
    min_hours_met: float

    def as_dict(self) -> dict[str, float]:
        return {
            "fill_rate": round(self.fill_rate, 4),
            "total_cost": round(self.total_cost, 2),
            "cost_index": round(self.cost_index, 4),
            "preference_rate": round(self.preference_rate, 4),
            "hours_fairness": round(self.hours_fairness, 4),
            "load_fairness": round(self.load_fairness, 4),
            "min_hours_met": round(self.min_hours_met, 4),
        }


def marginal_cost(
    context: SchedulingContext, employee: Employee, shift: Shift, hours_so_far: float
) -> float:
    """Cost of adding ``shift`` to someone who has already worked ``hours_so_far``."""
    return context.policy.overtime_cost(hours_so_far, shift.hours, employee.hourly_rate)


def reference_cost(context: SchedulingContext, shift: Shift) -> float:
    """Worst plausible cost for one shift, used to normalise cost into ``[0, 1]``."""
    rates = [e.hourly_rate for e in context.employees] or [1.0]
    return shift.hours * max(rates) * context.policy.overtime_multiplier


def _evenness(values: list[float]) -> float:
    """1.0 when every value is identical, falling towards 0 as they spread out."""
    positive = [v for v in values]
    if len(positive) < 2:
        return 1.0
    average = mean(positive)
    if average <= 0:
        return 1.0
    coefficient = pstdev(positive) / average
    return max(0.0, 1.0 - min(1.0, coefficient))


def compute_metrics(context: SchedulingContext, schedule: Schedule) -> ScheduleMetrics:
    """Summarise a schedule against every objective the product cares about."""
    total_shifts = len(context.shifts)
    fill_rate = len(schedule.assignments) / total_shifts if total_shifts else 1.0

    total_cost = schedule.total_cost
    ceiling = sum(reference_cost(context, s) for s in context.shifts)
    cost_index = 1.0 - min(1.0, total_cost / ceiling) if ceiling > 0 else 1.0

    matched = 0
    for assignment in schedule.assignments:
        employee = context.employee_by_id(assignment.employee_id)
        shift = context.shift_by_id(assignment.shift_id)
        if employee is None or shift is None:
            continue
        # Someone who named no preference is never counted as disappointed.
        if not employee.preferred_periods or employee.prefers(shift.period):
            matched += 1
    preference_rate = matched / len(schedule.assignments) if schedule.assignments else 1.0

    hours = schedule.hours_by_employee(context)
    hours_fairness = _evenness([hours.get(e.id, 0.0) for e in context.employees])

    undesirable = schedule.undesirable_by_employee(context)
    load_fairness = _evenness([float(undesirable.get(e.id, 0)) for e in context.employees])

    wanting = [e for e in context.employees if e.min_hours_per_week > 0]
    if wanting:
        met = sum(1 for e in wanting if hours.get(e.id, 0.0) >= e.min_hours_per_week)
        min_hours_met = met / len(wanting)
    else:
        min_hours_met = 1.0

    return ScheduleMetrics(
        fill_rate=fill_rate,
        total_cost=total_cost,
        cost_index=cost_index,
        preference_rate=preference_rate,
        hours_fairness=hours_fairness,
        load_fairness=load_fairness,
        min_hours_met=min_hours_met,
    )


def score_schedule(
    context: SchedulingContext, schedule: Schedule, weights: ObjectiveWeights
) -> tuple[float, dict[str, float]]:
    """Total weighted score plus a breakdown suitable for showing a manager."""
    metrics = compute_metrics(context, schedule)
    fairness = (metrics.hours_fairness + metrics.load_fairness) / 2.0
    total = (
        weights.coverage * metrics.fill_rate
        + weights.cost * metrics.cost_index
        + weights.preference * metrics.preference_rate
        + weights.fairness * fairness
        + weights.min_hours * metrics.min_hours_met
    )
    breakdown = metrics.as_dict()
    breakdown["weighted_total"] = round(total, 4)
    return total, breakdown


def candidate_score(
    context: SchedulingContext,
    employee: Employee,
    shift: Shift,
    hours_so_far: float,
    undesirable_so_far: int,
    average_hours: float,
    average_undesirable: float,
    weights: ObjectiveWeights,
) -> float:
    """Score one candidate for one shift during greedy construction.

    Combines the same objectives as :func:`score_schedule` but locally, so the
    greedy pass builds a schedule that the local search then only has to
    polish rather than rescue.
    """
    cost = marginal_cost(context, employee, shift, hours_so_far)
    ceiling = reference_cost(context, shift)
    cost_term = 1.0 - min(1.0, cost / ceiling) if ceiling > 0 else 1.0

    preference_term = 1.0 if (not employee.preferred_periods or employee.prefers(shift.period)) else 0.0

    # Favour whoever is currently furthest below the roster average, so hours
    # and unpleasant shifts even out instead of landing on the same few people.
    hours_room = _relative_room(hours_so_far, average_hours)
    if shift.is_undesirable:
        load_room = _relative_room(float(undesirable_so_far), average_undesirable)
        fairness_term = (hours_room + load_room) / 2.0
    else:
        fairness_term = hours_room

    shortfall = max(0.0, employee.min_hours_per_week - hours_so_far)
    min_hours_term = min(1.0, shortfall / shift.hours) if shift.hours > 0 else 0.0

    return (
        weights.cost * cost_term
        + weights.preference * preference_term
        + weights.fairness * fairness_term
        + weights.min_hours * min_hours_term
    )


def _relative_room(current: float, average: float) -> float:
    """1.0 when someone is unloaded relative to the roster, 0.0 when overloaded."""
    if average <= 0:
        return 1.0 if current <= 0 else 0.0
    return max(0.0, min(1.0, (2.0 * average - current) / (2.0 * average)))
