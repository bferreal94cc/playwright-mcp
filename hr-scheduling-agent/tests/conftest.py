"""Small builders so each test states only what it actually cares about."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from hr_scheduling_agent.domain import (
    AvailabilityWindow,
    Certification,
    CompanyPolicy,
    Employee,
    SchedulingContext,
    Shift,
    ShiftPeriod,
    TimeWindow,
)

MONDAY = date(2026, 8, 17)


def window(day_offset: int, start_hour: int, hours: float) -> TimeWindow:
    start = datetime.combine(MONDAY + timedelta(days=day_offset), time(start_hour, 0))
    return TimeWindow(start=start, end=start + timedelta(hours=hours))


def always_available(start_hour: int = 0, end_hour: int = 23) -> tuple[AvailabilityWindow, ...]:
    end = time(23, 59) if end_hour >= 24 else time(end_hour, 0)
    return tuple(
        AvailabilityWindow(weekday=d, start=time(start_hour, 0), end=end) for d in range(7)
    )


def make_employee(
    employee_id: str,
    roles: set[str] = frozenset({"server"}),
    rate: float = 20.0,
    availability=None,
    certifications: tuple[Certification, ...] = (),
    preferred: set[ShiftPeriod] = frozenset(),
    max_hours: float | None = None,
    min_hours: float = 0.0,
    unavailable: frozenset[date] = frozenset(),
) -> Employee:
    return Employee(
        id=employee_id,
        name=employee_id.upper(),
        roles=frozenset(roles),
        hourly_rate=rate,
        availability=availability if availability is not None else always_available(),
        certifications=certifications,
        unavailable_dates=unavailable,
        preferred_periods=frozenset(preferred),
        max_hours_per_week=max_hours,
        min_hours_per_week=min_hours,
    )


def make_shift(
    shift_id: str,
    day_offset: int = 0,
    start_hour: int = 9,
    hours: float = 8,
    role: str = "server",
    certifications: set[str] = frozenset(),
    location: str = "loc-1",
) -> Shift:
    return Shift(
        id=shift_id,
        location_id=location,
        department="ops",
        role=role,
        window=window(day_offset, start_hour, hours),
        required_certifications=frozenset(certifications),
    )


def make_context(
    employees: list[Employee],
    shifts: list[Shift],
    policy: CompanyPolicy | None = None,
    **kwargs,
) -> SchedulingContext:
    return SchedulingContext(
        policy=policy or CompanyPolicy(),
        employees=employees,
        shifts=shifts,
        **kwargs,
    )


@pytest.fixture
def simple_context() -> SchedulingContext:
    """Two people, two non-overlapping shifts -- everything should fill."""
    return make_context(
        employees=[make_employee("e1", rate=20.0), make_employee("e2", rate=25.0)],
        shifts=[make_shift("s1", day_offset=0), make_shift("s2", day_offset=1)],
    )
