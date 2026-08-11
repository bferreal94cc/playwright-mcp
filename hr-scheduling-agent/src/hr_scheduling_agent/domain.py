"""Core domain models for the HR scheduling agent.

The system carries state in three tiers, mirroring the architecture design:

* Global tier -- :class:`CompanyPolicy`. Company-wide, compliance driven.
* Local tier  -- :class:`LocationRules`. Site/department scheduling rules.
* User tier   -- :class:`Employee` availability, preferences and history.

:class:`SchedulingContext` binds the three together and is the single object
passed into the optimizer, exception handler and compliance auditor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum


class ShiftPeriod(str, Enum):
    """Coarse time-of-day bucket, used for fairness and preference matching."""

    MORNING = "morning"
    AFTERNOON = "afternoon"
    NIGHT = "night"

    @classmethod
    def from_start(cls, start: datetime) -> "ShiftPeriod":
        if start.hour < 12:
            return cls.MORNING
        if start.hour < 18:
            return cls.AFTERNOON
        return cls.NIGHT


@dataclass(frozen=True)
class TimeWindow:
    """A concrete half-open interval ``[start, end)``."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"time window end {self.end} must be after start {self.start}")

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    def overlaps(self, other: "TimeWindow") -> bool:
        return self.start < other.end and other.start < self.end

    def gap_hours_to(self, other: "TimeWindow") -> float:
        """Rest hours between this window and ``other``. Negative when they overlap."""
        if self.overlaps(other):
            return -1.0
        if self.end <= other.start:
            return (other.start - self.end).total_seconds() / 3600.0
        return (self.start - other.end).total_seconds() / 3600.0


_MINUTES_PER_DAY = 24 * 60


def _minutes(value: time) -> int:
    """Minutes from midnight."""
    return value.hour * 60 + value.minute


@dataclass(frozen=True)
class AvailabilityWindow:
    """A recurring weekly window an employee has declared themselves free for.

    ``weekday`` follows :meth:`datetime.date.weekday` -- Monday is 0.
    """

    weekday: int
    start: time
    end: time

    def covers(self, window: TimeWindow) -> bool:
        """True when ``window`` falls entirely inside this recurring window.

        Both sides are compared as minutes from midnight on the shift's start
        day, so a shift running past midnight simply has an end past 1440 and
        needs a window that wraps to match it.

        An ``end`` at or before ``start`` wraps into the next day (``19:00`` to
        ``07:00`` is a night window). ``23:59`` is accepted as a synonym for
        end-of-day, since that is how availability pickers usually spell it.
        """
        if window.start.weekday() != self.weekday:
            return False

        shift_start = _minutes(window.start.time())
        shift_end = shift_start + round(window.hours * 60)

        available_start = _minutes(self.start)
        available_end = _minutes(self.end)
        if available_end <= available_start:
            available_end += _MINUTES_PER_DAY
        elif available_end == _MINUTES_PER_DAY - 1:
            available_end = _MINUTES_PER_DAY

        return available_start <= shift_start and shift_end <= available_end


@dataclass(frozen=True)
class Certification:
    """A credential with an expiry date, e.g. ``BLS`` for a nurse."""

    name: str
    expires_on: date

    def valid_on(self, day: date) -> bool:
        return day <= self.expires_on


@dataclass
class Employee:
    """User tier state: who someone is, when they can work, what they prefer."""

    id: str
    name: str
    roles: frozenset[str]
    hourly_rate: float
    availability: tuple[AvailabilityWindow, ...] = ()
    certifications: tuple[Certification, ...] = ()
    unavailable_dates: frozenset[date] = frozenset()
    preferred_periods: frozenset[ShiftPeriod] = frozenset()
    max_hours_per_week: float | None = None
    min_hours_per_week: float = 0.0
    seniority_months: int = 0

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def certification_valid(self, name: str, day: date) -> bool:
        return any(c.name == name and c.valid_on(day) for c in self.certifications)

    def is_available_for(self, window: TimeWindow) -> bool:
        if window.start.date() in self.unavailable_dates:
            return False
        if window.end.date() in self.unavailable_dates and window.end.time() != time(0, 0):
            return False
        return any(a.covers(window) for a in self.availability)

    def prefers(self, period: ShiftPeriod) -> bool:
        return period in self.preferred_periods


@dataclass(frozen=True)
class Shift:
    """A unit of demand: one seat, at one place, over one window."""

    id: str
    location_id: str
    department: str
    role: str
    window: TimeWindow
    required_certifications: frozenset[str] = frozenset()

    @property
    def period(self) -> ShiftPeriod:
        return ShiftPeriod.from_start(self.window.start)

    @property
    def hours(self) -> float:
        return self.window.hours

    @property
    def is_undesirable(self) -> bool:
        """Night and weekend shifts are the ones fairness needs to spread out."""
        return self.period is ShiftPeriod.NIGHT or self.window.start.weekday() >= 5


@dataclass(frozen=True)
class CompanyPolicy:
    """Global tier state: the compliance envelope no schedule may leave."""

    max_hours_per_week: float = 40.0
    max_shift_hours: float = 12.0
    min_rest_hours_between_shifts: float = 8.0
    max_consecutive_days: int = 6
    overtime_threshold_hours: float = 40.0
    overtime_multiplier: float = 1.5

    def overtime_cost(self, base_hours: float, extra_hours: float, rate: float) -> float:
        """Cost of adding ``extra_hours`` on top of ``base_hours`` already worked."""
        straight = max(0.0, min(base_hours + extra_hours, self.overtime_threshold_hours) - base_hours)
        overtime = extra_hours - straight
        return straight * rate + overtime * rate * self.overtime_multiplier


@dataclass(frozen=True)
class LocationRules:
    """Local tier state: per-site budget and coverage rules."""

    location_id: str
    weekly_labor_budget: float | None = None
    min_staff_per_shift: int = 1
    max_hours_per_week: float | None = None

    def effective_max_hours(self, policy: CompanyPolicy) -> float:
        """Local rules may tighten the global cap but never loosen it."""
        if self.max_hours_per_week is None:
            return policy.max_hours_per_week
        return min(self.max_hours_per_week, policy.max_hours_per_week)


@dataclass
class SchedulingContext:
    """The three tiers bound together, plus the demand being scheduled."""

    policy: CompanyPolicy
    employees: list[Employee]
    shifts: list[Shift]
    location_rules: dict[str, LocationRules] = field(default_factory=dict)
    prior_hours: dict[str, float] = field(default_factory=dict)
    prior_undesirable_counts: dict[str, int] = field(default_factory=dict)

    def rules_for(self, location_id: str) -> LocationRules:
        return self.location_rules.get(location_id) or LocationRules(location_id=location_id)

    def employee_by_id(self, employee_id: str) -> Employee | None:
        return next((e for e in self.employees if e.id == employee_id), None)

    def shift_by_id(self, shift_id: str) -> Shift | None:
        return next((s for s in self.shifts if s.id == shift_id), None)

    def max_hours_for(self, employee: Employee, location_id: str) -> float:
        """Tightest of company policy, location rules and the employee's own cap."""
        caps = [self.rules_for(location_id).effective_max_hours(self.policy)]
        if employee.max_hours_per_week is not None:
            caps.append(employee.max_hours_per_week)
        return min(caps)


@dataclass
class Assignment:
    """One employee placed on one shift, with the reasoning that put them there."""

    shift_id: str
    employee_id: str
    cost: float
    rationale: str = ""

    def __hash__(self) -> int:
        return hash((self.shift_id, self.employee_id))


@dataclass
class UnfilledShift:
    """A shift no one could legally take, with the reason each candidate failed."""

    shift_id: str
    blockers: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.blockers:
            return "no candidates in the roster"
        counts: dict[str, int] = {}
        for reason in self.blockers.values():
            counts[reason] = counts.get(reason, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        return "; ".join(f"{reason} ({count})" for reason, count in ranked)


@dataclass
class Schedule:
    """The optimizer's output: assignments, gaps, and how it scored."""

    assignments: list[Assignment] = field(default_factory=list)
    unfilled: list[UnfilledShift] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return sum(a.cost for a in self.assignments)

    def assignments_for(self, employee_id: str) -> list[Assignment]:
        return [a for a in self.assignments if a.employee_id == employee_id]

    def assignment_for_shift(self, shift_id: str) -> Assignment | None:
        return next((a for a in self.assignments if a.shift_id == shift_id), None)

    def hours_by_employee(self, context: SchedulingContext) -> dict[str, float]:
        hours = {e.id: context.prior_hours.get(e.id, 0.0) for e in context.employees}
        for assignment in self.assignments:
            shift = context.shift_by_id(assignment.shift_id)
            if shift is not None:
                hours[assignment.employee_id] = hours.get(assignment.employee_id, 0.0) + shift.hours
        return hours

    def undesirable_by_employee(self, context: SchedulingContext) -> dict[str, int]:
        counts = {e.id: context.prior_undesirable_counts.get(e.id, 0) for e in context.employees}
        for assignment in self.assignments:
            shift = context.shift_by_id(assignment.shift_id)
            if shift is not None and shift.is_undesirable:
                counts[assignment.employee_id] = counts.get(assignment.employee_id, 0) + 1
        return counts


def week_of(day: date) -> date:
    """Monday of the ISO week containing ``day``. Used to bucket weekly caps."""
    return day - timedelta(days=day.weekday())
