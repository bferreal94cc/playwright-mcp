"""Reference scenarios for the two launch verticals.

These are the datasets behind the Figma prototype pages: a multi-location
restaurant group (cost- and preference-driven) and a hospital nursing unit
(compliance- and certification-driven). They double as realistic fixtures for
the test suite.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from .domain import (
    AvailabilityWindow,
    Certification,
    CompanyPolicy,
    Employee,
    LocationRules,
    SchedulingContext,
    ShiftPeriod,
    Shift,
    TimeWindow,
)

WEEK_START = date(2026, 8, 17)  # a Monday


def _all_week(start: time, end: time) -> tuple[AvailabilityWindow, ...]:
    return tuple(AvailabilityWindow(weekday=d, start=start, end=end) for d in range(7))


def _weekdays_only(start: time, end: time) -> tuple[AvailabilityWindow, ...]:
    return tuple(AvailabilityWindow(weekday=d, start=start, end=end) for d in range(5))


def _window(day_offset: int, start_hour: int, hours: int) -> TimeWindow:
    start = datetime.combine(WEEK_START + timedelta(days=day_offset), time(start_hour, 0))
    return TimeWindow(start=start, end=start + timedelta(hours=hours))


# -- retail ---------------------------------------------------------------


def build_retail_context(days: int = 7) -> SchedulingContext:
    """A 2-location restaurant group: cost pressure, preferences, mixed roles."""
    policy = CompanyPolicy(
        max_hours_per_week=40.0,
        max_shift_hours=10.0,
        min_rest_hours_between_shifts=8.0,
        max_consecutive_days=6,
        overtime_threshold_hours=40.0,
        overtime_multiplier=1.5,
    )

    # 14 staff against 392 hours of weekly demand: enough slack for the
    # optimizer to have real choices, tight enough that they matter.
    roster = [
        ("r01", "Avery", {"server"}, 17.0, ShiftPeriod.MORNING, 20.0),
        ("r02", "Bailey", {"server"}, 16.5, ShiftPeriod.AFTERNOON, 30.0),
        ("r03", "Casey", {"server", "cook"}, 19.0, ShiftPeriod.MORNING, 24.0),
        ("r04", "Devon", {"cook"}, 21.0, ShiftPeriod.AFTERNOON, 30.0),
        ("r05", "Ellis", {"cook"}, 20.0, ShiftPeriod.MORNING, 24.0),
        ("r06", "Finley", {"server"}, 16.0, ShiftPeriod.AFTERNOON, 16.0),
        ("r07", "Gray", {"server", "cook"}, 22.0, ShiftPeriod.MORNING, 30.0),
        ("r08", "Harper", {"cook"}, 18.5, ShiftPeriod.AFTERNOON, 20.0),
        ("r09", "Indigo", {"server"}, 16.0, ShiftPeriod.MORNING, 24.0),
        ("r10", "Jordan", {"cook"}, 19.5, ShiftPeriod.MORNING, 24.0),
        ("r11", "Kai", {"server", "cook"}, 18.0, ShiftPeriod.AFTERNOON, 30.0),
        ("r12", "Logan", {"server"}, 17.5, ShiftPeriod.AFTERNOON, 20.0),
        ("r13", "Marlow", {"cook"}, 20.5, ShiftPeriod.AFTERNOON, 24.0),
        ("r14", "Nova", {"server", "cook"}, 17.0, ShiftPeriod.MORNING, 30.0),
    ]

    employees = [
        Employee(
            id=eid,
            name=name,
            roles=frozenset(roles),
            hourly_rate=rate,
            availability=_all_week(time(6, 0), time(23, 0)),
            preferred_periods=frozenset({period}),
            min_hours_per_week=min_hours,
            seniority_months=12,
        )
        for eid, name, roles, rate, period, min_hours in roster
    ]

    shifts: list[Shift] = []
    for offset in range(days):
        for location in ("store-01", "store-02"):
            shifts.append(
                Shift(
                    id=f"{location}-d{offset}-open-server",
                    location_id=location,
                    department="front-of-house",
                    role="server",
                    window=_window(offset, 8, 8),
                )
            )
            shifts.append(
                Shift(
                    id=f"{location}-d{offset}-close-server",
                    location_id=location,
                    department="front-of-house",
                    role="server",
                    window=_window(offset, 16, 6),
                )
            )
            shifts.append(
                Shift(
                    id=f"{location}-d{offset}-open-cook",
                    location_id=location,
                    department="kitchen",
                    role="cook",
                    window=_window(offset, 8, 8),
                )
            )
            shifts.append(
                Shift(
                    id=f"{location}-d{offset}-close-cook",
                    location_id=location,
                    department="kitchen",
                    role="cook",
                    window=_window(offset, 16, 6),
                )
            )

    return SchedulingContext(
        policy=policy,
        employees=employees,
        shifts=shifts,
        location_rules={
            "store-01": LocationRules(location_id="store-01", weekly_labor_budget=9000.0),
            "store-02": LocationRules(location_id="store-02", weekly_labor_budget=9000.0),
        },
    )


# -- healthcare -----------------------------------------------------------


def build_healthcare_context(days: int = 7) -> SchedulingContext:
    """A hospital nursing unit: 12h shifts, licence checks, hard rest rules."""
    policy = CompanyPolicy(
        max_hours_per_week=36.0,
        max_shift_hours=12.0,
        min_rest_hours_between_shifts=10.0,
        max_consecutive_days=3,
        overtime_threshold_hours=36.0,
        overtime_multiplier=1.5,
    )

    day_availability = _all_week(time(7, 0), time(19, 0))
    night_availability = _all_week(time(19, 0), time(7, 0))
    both = day_availability + night_availability

    valid = Certification(name="BLS", expires_on=date(2027, 6, 30))
    acls = Certification(name="ACLS", expires_on=date(2027, 3, 31))
    expiring = Certification(name="ACLS", expires_on=date(2026, 8, 1))  # already lapsed

    # 14 RNs cover 336 RN-hours at a 36h cap; 5 LPNs cover the 84 LPN-hours.
    employees: list[Employee] = []
    for index in range(1, 20):
        eid = f"n{index:02d}"
        is_rn = index <= 14
        certifications = [valid]
        if is_rn:
            # One RN's ACLS has lapsed -- the compliance auditor must catch it.
            certifications.append(expiring if index == 14 else acls)
        employees.append(
            Employee(
                id=eid,
                name=f"Nurse {index:02d}",
                roles=frozenset({"rn"} if is_rn else {"lpn"}),
                hourly_rate=52.0 if is_rn else 34.0,
                availability=both,
                certifications=tuple(certifications),
                preferred_periods=frozenset(
                    {ShiftPeriod.NIGHT} if index % 3 == 0 else {ShiftPeriod.MORNING}
                ),
                min_hours_per_week=24.0,
                seniority_months=6 * index,
            )
        )

    shifts: list[Shift] = []
    for offset in range(days):
        for seat in range(2):
            shifts.append(
                Shift(
                    id=f"icu-d{offset}-day-rn{seat}",
                    location_id="hospital-north",
                    department="icu",
                    role="rn",
                    window=_window(offset, 7, 12),
                    required_certifications=frozenset({"BLS", "ACLS"}),
                )
            )
            shifts.append(
                Shift(
                    id=f"icu-d{offset}-night-rn{seat}",
                    location_id="hospital-north",
                    department="icu",
                    role="rn",
                    window=_window(offset, 19, 12),
                    required_certifications=frozenset({"BLS", "ACLS"}),
                )
            )
        shifts.append(
            Shift(
                id=f"icu-d{offset}-day-lpn",
                location_id="hospital-north",
                department="icu",
                role="lpn",
                window=_window(offset, 7, 12),
                required_certifications=frozenset({"BLS"}),
            )
        )

    return SchedulingContext(
        policy=policy,
        employees=employees,
        shifts=shifts,
        location_rules={
            "hospital-north": LocationRules(
                location_id="hospital-north", min_staff_per_shift=2, max_hours_per_week=36.0
            )
        },
    )
