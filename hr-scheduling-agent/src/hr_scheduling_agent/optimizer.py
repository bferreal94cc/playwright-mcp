"""The constraint optimization engine -- the core IP of the product.

Two phases:

1. **Greedy construction.** Shifts are placed hardest-first (fewest eligible
   people), each going to the candidate with the best local score. Scheduling
   the scarce shifts first is what stops a cheap early assignment from
   stranding the only certified nurse for the night shift.
2. **Local search.** Bounded hill-climbing over single-shift moves and pairwise
   swaps, accepting only changes that raise the whole-schedule score.

Both phases refuse to make an assignment the :class:`ConstraintEngine` vetoes,
so every schedule returned is legal by construction. The optimizer never trades
compliance for cost -- the trade-off dial only moves cost against wellbeing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constraints import ConstraintEngine
from .domain import (
    Assignment,
    Employee,
    Schedule,
    SchedulingContext,
    Shift,
    UnfilledShift,
    week_of,
)
from .scoring import (
    ObjectiveWeights,
    candidate_score,
    compute_metrics,
    marginal_cost,
    score_schedule,
)


@dataclass
class _Working:
    """Mutable bookkeeping carried through construction and local search."""

    shifts_by_employee: dict[str, list[Shift]]
    hours: dict[str, float]
    undesirable: dict[str, int]

    @classmethod
    def empty(cls, context: SchedulingContext) -> "_Working":
        return cls(
            shifts_by_employee={e.id: [] for e in context.employees},
            hours={e.id: context.prior_hours.get(e.id, 0.0) for e in context.employees},
            undesirable={
                e.id: context.prior_undesirable_counts.get(e.id, 0) for e in context.employees
            },
        )

    def add(self, employee_id: str, shift: Shift) -> None:
        self.shifts_by_employee[employee_id].append(shift)
        self.hours[employee_id] = self.hours.get(employee_id, 0.0) + shift.hours
        if shift.is_undesirable:
            self.undesirable[employee_id] = self.undesirable.get(employee_id, 0) + 1

    def remove(self, employee_id: str, shift: Shift) -> None:
        self.shifts_by_employee[employee_id] = [
            s for s in self.shifts_by_employee[employee_id] if s.id != shift.id
        ]
        self.hours[employee_id] = self.hours.get(employee_id, 0.0) - shift.hours
        if shift.is_undesirable:
            self.undesirable[employee_id] = max(0, self.undesirable.get(employee_id, 0) - 1)


class ScheduleOptimizer:
    """Builds a legal, well-scored schedule for a :class:`SchedulingContext`."""

    def __init__(
        self,
        context: SchedulingContext,
        weights: ObjectiveWeights | None = None,
        max_passes: int = 4,
    ) -> None:
        self.context = context
        self.weights = weights or ObjectiveWeights.balanced()
        self.max_passes = max_passes
        self.engine = ConstraintEngine(context)

    # -- public API -------------------------------------------------------

    def solve(self, locked: dict[str, str] | None = None) -> Schedule:
        """Produce a schedule.

        ``locked`` optionally pins ``{shift_id: employee_id}`` assignments that
        must be preserved -- used by the exception handler to re-solve only the
        shifts a call-out actually disturbed.
        """
        schedule, working = self._construct(locked or {})
        self._improve(schedule, working, locked or {})
        self._recost(schedule)
        _, breakdown = score_schedule(self.context, schedule, self.weights)
        schedule.score_breakdown = breakdown
        return schedule

    def metrics(self, schedule: Schedule):
        return compute_metrics(self.context, schedule)

    # -- phase 1: greedy construction -------------------------------------

    def _construct(self, locked: dict[str, str]) -> tuple[Schedule, _Working]:
        schedule = Schedule()
        working = _Working.empty(self.context)

        for shift_id, employee_id in locked.items():
            shift = self.context.shift_by_id(shift_id)
            if shift is None or employee_id not in working.shifts_by_employee:
                continue
            working.add(employee_id, shift)
            schedule.assignments.append(
                Assignment(shift_id=shift.id, employee_id=employee_id, cost=0.0, rationale="pinned")
            )

        pending = [s for s in self.context.shifts if s.id not in locked]
        for shift in self._hardest_first(pending, working):
            eligible, blockers = self.engine.eligible_employees(shift, working.shifts_by_employee)
            if not eligible:
                schedule.unfilled.append(UnfilledShift(shift_id=shift.id, blockers=blockers))
                continue
            best = self._best_candidate(shift, eligible, working)
            working.add(best.id, shift)
            schedule.assignments.append(
                Assignment(
                    shift_id=shift.id,
                    employee_id=best.id,
                    cost=0.0,
                    rationale=self._rationale(best, shift, working, len(eligible)),
                )
            )
        return schedule, working

    def _hardest_first(self, shifts: list[Shift], working: _Working) -> list[Shift]:
        """Order shifts by scarcity of qualified staff, then chronologically."""

        def scarcity(shift: Shift) -> tuple[int, str, str]:
            eligible, _ = self.engine.eligible_employees(shift, working.shifts_by_employee)
            return (len(eligible), shift.window.start.isoformat(), shift.id)

        return sorted(shifts, key=scarcity)

    def _best_candidate(
        self, shift: Shift, eligible: list[Employee], working: _Working
    ) -> Employee:
        average_hours = self._average(list(working.hours.values()))
        average_undesirable = self._average([float(v) for v in working.undesirable.values()])

        def key(employee: Employee) -> tuple[float, str]:
            score = candidate_score(
                self.context,
                employee,
                shift,
                working.hours.get(employee.id, 0.0),
                working.undesirable.get(employee.id, 0),
                average_hours,
                average_undesirable,
                self.weights,
            )
            # Negated score first so higher scores sort earlier; id breaks ties
            # deterministically, which keeps runs reproducible for tests.
            return (-score, employee.id)

        return sorted(eligible, key=key)[0]

    @staticmethod
    def _average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _rationale(
        self, employee: Employee, shift: Shift, working: _Working, pool_size: int
    ) -> str:
        parts = [f"selected from {pool_size} eligible"]
        if employee.preferred_periods and employee.prefers(shift.period):
            parts.append(f"matches preferred {shift.period.value} period")
        hours_before = working.hours.get(employee.id, 0.0) - shift.hours
        average = self._average(list(working.hours.values()))
        if hours_before < average:
            parts.append(f"{average - hours_before:.1f}h below roster average")
        if shift.required_certifications:
            parts.append("holds " + ", ".join(sorted(shift.required_certifications)))
        parts.append(f"${employee.hourly_rate:.2f}/h")
        return "; ".join(parts)

    # -- phase 2: local search --------------------------------------------

    def _improve(self, schedule: Schedule, working: _Working, locked: dict[str, str]) -> None:
        self._recost(schedule)
        best_score, _ = score_schedule(self.context, schedule, self.weights)

        for _ in range(self.max_passes):
            improved = False
            if self._try_fill_gaps(schedule, working):
                self._recost(schedule)
                best_score, _ = score_schedule(self.context, schedule, self.weights)
                improved = True
            gained, best_score = self._try_moves(schedule, working, locked, best_score)
            improved = improved or gained
            gained, best_score = self._try_swaps(schedule, working, locked, best_score)
            improved = improved or gained
            if not improved:
                break

    def _try_fill_gaps(self, schedule: Schedule, working: _Working) -> bool:
        """Re-check unfilled shifts -- later moves may have freed someone up."""
        filled_any = False
        for gap in list(schedule.unfilled):
            shift = self.context.shift_by_id(gap.shift_id)
            if shift is None:
                continue
            eligible, blockers = self.engine.eligible_employees(shift, working.shifts_by_employee)
            if not eligible:
                gap.blockers = blockers
                continue
            best = self._best_candidate(shift, eligible, working)
            working.add(best.id, shift)
            schedule.assignments.append(
                Assignment(
                    shift_id=shift.id,
                    employee_id=best.id,
                    cost=0.0,
                    rationale=self._rationale(best, shift, working, len(eligible)),
                )
            )
            schedule.unfilled.remove(gap)
            filled_any = True
        return filled_any

    def _try_moves(
        self, schedule: Schedule, working: _Working, locked: dict[str, str], best_score: float
    ) -> tuple[bool, float]:
        improved = False
        for assignment in list(schedule.assignments):
            if assignment.shift_id in locked:
                continue
            shift = self.context.shift_by_id(assignment.shift_id)
            if shift is None:
                continue

            original = assignment.employee_id
            working.remove(original, shift)
            accepted = None
            for candidate in self.context.employees:
                if candidate.id == original:
                    continue
                if not self.engine.check(
                    candidate, shift, working.shifts_by_employee[candidate.id]
                ).ok:
                    continue
                assignment.employee_id = candidate.id
                working.add(candidate.id, shift)
                self._recost(schedule)
                score, _ = score_schedule(self.context, schedule, self.weights)
                if score > best_score + 1e-9:
                    best_score = score
                    accepted = candidate.id
                    improved = True
                    break
                working.remove(candidate.id, shift)

            if accepted is None:
                assignment.employee_id = original
                working.add(original, shift)
        return improved, best_score

    def _try_swaps(
        self, schedule: Schedule, working: _Working, locked: dict[str, str], best_score: float
    ) -> tuple[bool, float]:
        improved = False
        assignments = [a for a in schedule.assignments if a.shift_id not in locked]
        for i, first in enumerate(assignments):
            for second in assignments[i + 1 :]:
                if first.employee_id == second.employee_id:
                    continue
                if not self._swap(first, second, working, apply=True):
                    continue
                self._recost(schedule)
                score, _ = score_schedule(self.context, schedule, self.weights)
                if score > best_score + 1e-9:
                    best_score = score
                    improved = True
                else:
                    self._swap(first, second, working, apply=True, force=True)
        return improved, best_score

    def _swap(
        self,
        first: Assignment,
        second: Assignment,
        working: _Working,
        apply: bool,
        force: bool = False,
    ) -> bool:
        """Exchange the employees on two assignments if both directions stay legal."""
        shift_a = self.context.shift_by_id(first.shift_id)
        shift_b = self.context.shift_by_id(second.shift_id)
        employee_a = self.context.employee_by_id(first.employee_id)
        employee_b = self.context.employee_by_id(second.employee_id)
        if not all((shift_a, shift_b, employee_a, employee_b)):
            return False
        assert shift_a and shift_b and employee_a and employee_b

        working.remove(employee_a.id, shift_a)
        working.remove(employee_b.id, shift_b)
        legal = force or (
            self.engine.check(employee_a, shift_b, working.shifts_by_employee[employee_a.id]).ok
            and self.engine.check(employee_b, shift_a, working.shifts_by_employee[employee_b.id]).ok
        )
        if not legal:
            working.add(employee_a.id, shift_a)
            working.add(employee_b.id, shift_b)
            return False

        working.add(employee_a.id, shift_b)
        working.add(employee_b.id, shift_a)
        if apply:
            first.employee_id = employee_b.id
            second.employee_id = employee_a.id
        return True

    # -- costing ----------------------------------------------------------

    def _recost(self, schedule: Schedule) -> None:
        """Recompute every assignment cost from scratch.

        Overtime depends on how many hours someone has already worked that
        week, so cost is only well-defined once the whole schedule is known.
        Replaying each person's week chronologically makes the total
        independent of the order assignments happened to be created in.
        """
        by_employee: dict[str, list[Assignment]] = {}
        for assignment in schedule.assignments:
            by_employee.setdefault(assignment.employee_id, []).append(assignment)

        for employee_id, assignments in by_employee.items():
            employee = self.context.employee_by_id(employee_id)
            if employee is None:
                continue
            resolved = []
            for assignment in assignments:
                shift = self.context.shift_by_id(assignment.shift_id)
                if shift is not None:
                    resolved.append((shift, assignment))
            resolved.sort(key=lambda pair: (pair[0].window.start, pair[0].id))

            prior = self.context.prior_hours.get(employee_id, 0.0)
            running: dict[object, float] = {}
            for shift, assignment in resolved:
                bucket = week_of(shift.window.start.date())
                so_far = running.get(bucket, prior)
                assignment.cost = marginal_cost(self.context, employee, shift, so_far)
                running[bucket] = so_far + shift.hours
