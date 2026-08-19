"""Integration bridges: HRIS, calendar and notifications.

The competitive research put legacy integration at the top of the gap list.
Each bridge here is an abstract port with an in-memory reference adapter, so
the agents can be exercised end to end without a Workday tenant, and a real
adapter is a subclass rather than a rewrite.

Adapters are deliberately narrow. The HRIS is the system of record for people;
this product is the system of record for *when they work*, and pushes that back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .domain import Assignment, Employee, Schedule, SchedulingContext, Shift


# -- HRIS -----------------------------------------------------------------


class HRISConnector(ABC):
    """Read the roster from the system of record, write the schedule back."""

    system_name: str = "abstract"

    @abstractmethod
    def fetch_employees(self) -> list[Employee]:
        """Current active roster."""

    @abstractmethod
    def push_schedule(self, schedule: Schedule, context: SchedulingContext) -> int:
        """Publish assignments. Returns the number of records written."""


class InMemoryHRIS(HRISConnector):
    """Reference adapter backed by a list. Mirrors the shape of a real one."""

    system_name = "in-memory"

    def __init__(self, employees: list[Employee] | None = None) -> None:
        self._employees = list(employees or [])
        self.published: list[tuple[str, str, datetime]] = []

    def fetch_employees(self) -> list[Employee]:
        return list(self._employees)

    def push_schedule(self, schedule: Schedule, context: SchedulingContext) -> int:
        written = 0
        for assignment in schedule.assignments:
            shift = context.shift_by_id(assignment.shift_id)
            if shift is None:
                continue
            self.published.append((assignment.employee_id, shift.id, shift.window.start))
            written += 1
        return written


# -- calendar -------------------------------------------------------------


class CalendarConnector(ABC):
    """Push shifts into whatever calendar the employee actually looks at."""

    provider_name: str = "abstract"

    @abstractmethod
    def publish(self, employee: Employee, shift: Shift) -> str:
        """Create/update an event. Returns an event id."""

    @abstractmethod
    def withdraw(self, event_id: str) -> bool:
        """Remove an event. Returns whether anything was removed."""


class InMemoryCalendar(CalendarConnector):
    """Reference adapter; records events in a dict."""

    provider_name = "in-memory"

    def __init__(self) -> None:
        self.events: dict[str, tuple[str, str, datetime, datetime]] = {}
        self._counter = 0

    def publish(self, employee: Employee, shift: Shift) -> str:
        self._counter += 1
        event_id = f"evt-{self._counter:05d}"
        self.events[event_id] = (employee.id, shift.id, shift.window.start, shift.window.end)
        return event_id

    def withdraw(self, event_id: str) -> bool:
        return self.events.pop(event_id, None) is not None


# -- notifications --------------------------------------------------------


class Urgency(str, Enum):
    """Drives which channels a message goes out on."""

    ROUTINE = "routine"
    IMPORTANT = "important"
    URGENT = "urgent"


@dataclass(frozen=True)
class Notification:
    recipient_id: str
    subject: str
    body: str
    urgency: Urgency
    sent_at: datetime


class NotificationChannel(ABC):
    """One delivery mechanism: email, SMS, push, Slack."""

    channel_name: str = "abstract"

    @abstractmethod
    def send(self, notification: Notification) -> bool:
        """Deliver. Returns success."""


class InMemoryChannel(NotificationChannel):
    """Reference adapter that captures what would have been sent."""

    def __init__(self, channel_name: str) -> None:
        self.channel_name = channel_name
        self.outbox: list[Notification] = []

    def send(self, notification: Notification) -> bool:
        self.outbox.append(notification)
        return True


@dataclass
class NotificationEngine:
    """Routes a message to the right channels for its urgency.

    Routine schedule updates go to email and in-app. An urgent call-out cover
    also goes to SMS and push, because a shift starting in an hour cannot wait
    for someone to check their inbox.
    """

    channels: dict[str, NotificationChannel] = field(default_factory=dict)
    routing: dict[Urgency, tuple[str, ...]] = field(
        default_factory=lambda: {
            Urgency.ROUTINE: ("email", "in_app"),
            Urgency.IMPORTANT: ("email", "in_app", "push"),
            Urgency.URGENT: ("sms", "push", "in_app"),
        }
    )
    sent: list[Notification] = field(default_factory=list)

    @classmethod
    def in_memory(cls) -> "NotificationEngine":
        names = ("email", "sms", "push", "in_app")
        return cls(channels={name: InMemoryChannel(name) for name in names})

    def register(self, channel: NotificationChannel) -> None:
        self.channels[channel.channel_name] = channel

    def notify(
        self,
        recipient_id: str,
        subject: str,
        body: str,
        urgency: Urgency = Urgency.ROUTINE,
        at: datetime | None = None,
    ) -> list[str]:
        """Send one message. Returns the channels it actually went out on."""
        notification = Notification(
            recipient_id=recipient_id,
            subject=subject,
            body=body,
            urgency=urgency,
            sent_at=at or datetime.now(),
        )
        delivered: list[str] = []
        for name in self.routing.get(urgency, ()):
            channel = self.channels.get(name)
            if channel is not None and channel.send(notification):
                delivered.append(name)
        self.sent.append(notification)
        return delivered

    def broadcast_schedule(
        self, schedule: Schedule, context: SchedulingContext, at: datetime | None = None
    ) -> int:
        """Tell everyone what they are working. Returns recipients reached."""
        by_employee: dict[str, list[Assignment]] = {}
        for assignment in schedule.assignments:
            by_employee.setdefault(assignment.employee_id, []).append(assignment)

        reached = 0
        for employee_id, assignments in sorted(by_employee.items()):
            employee = context.employee_by_id(employee_id)
            if employee is None:
                continue
            lines = []
            for assignment in sorted(assignments, key=lambda a: a.shift_id):
                shift = context.shift_by_id(assignment.shift_id)
                if shift is None:
                    continue
                lines.append(
                    f"{shift.window.start:%a %d %b %H:%M}-{shift.window.end:%H:%M} "
                    f"{shift.role} @ {shift.location_id}"
                )
            self.notify(
                recipient_id=employee_id,
                subject=f"Your schedule: {len(lines)} shift(s)",
                body="\n".join(lines),
                urgency=Urgency.ROUTINE,
                at=at,
            )
            reached += 1
        return reached
