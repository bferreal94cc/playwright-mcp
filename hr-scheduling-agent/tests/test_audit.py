"""The audit chain must detect any edit to recorded history."""

from __future__ import annotations

import json
from datetime import datetime

from hr_scheduling_agent.audit import GENESIS_HASH, AuditLog

T0 = datetime(2026, 8, 17, 9, 0)


def populated(count: int = 4) -> AuditLog:
    log = AuditLog()
    for index in range(count):
        log.record(
            actor=f"actor-{index}",
            action="decision",
            subject="schedule",
            reasoning=f"reason {index}",
            metadata={"index": index},
            at=T0,
        )
    return log


class TestRecording:
    def test_first_entry_chains_to_genesis(self):
        log = AuditLog()
        entry = log.record("a", "act", "s", "why", at=T0)
        assert entry.sequence == 0
        assert entry.previous_hash == GENESIS_HASH

    def test_entries_chain_to_their_predecessor(self):
        log = populated(3)
        for previous, current in zip(log.entries, log.entries[1:]):
            assert current.previous_hash == previous.entry_hash

    def test_sequence_numbers_are_dense(self):
        log = populated(5)
        assert [e.sequence for e in log.entries] == [0, 1, 2, 3, 4]

    def test_metadata_is_copied_not_aliased(self):
        log = AuditLog()
        metadata = {"key": "value"}
        entry = log.record("a", "act", "s", "why", metadata=metadata, at=T0)
        metadata["key"] = "mutated"
        assert entry.metadata["key"] == "value"


class TestVerification:
    def test_clean_log_verifies(self):
        result = populated(6).verify()
        assert result.valid
        assert "6 entries verified" in result.detail

    def test_empty_log_verifies(self):
        assert AuditLog().verify().valid

    def test_detects_edited_content(self):
        log = populated(4)
        object.__setattr__(log.entries[2], "reasoning", "forged")
        result = log.verify()
        assert not result.valid
        assert result.broken_at == 2
        assert "altered" in result.detail

    def test_detects_a_deleted_entry(self):
        log = populated(4)
        del log.entries[1]
        result = log.verify()
        assert not result.valid
        assert result.broken_at == 1

    def test_detects_a_reordered_entry(self):
        log = populated(4)
        log.entries[1], log.entries[2] = log.entries[2], log.entries[1]
        assert not log.verify().valid

    def test_detects_a_forged_appended_entry(self):
        log = populated(3)
        forged = log.entries[-1]
        object.__setattr__(forged, "actor", "someone-else")
        assert not log.verify().valid


class TestQuerying:
    def test_filters_by_subject(self):
        log = AuditLog()
        log.record("a", "act", "schedule", "x", at=T0)
        log.record("a", "act", "shift-1", "y", at=T0)
        assert len(log.entries_for("schedule")) == 1

    def test_filters_by_actor(self):
        log = AuditLog()
        log.record("manager", "act", "s", "x", at=T0)
        log.record("agent", "act", "s", "y", at=T0)
        assert len(log.entries_by("manager")) == 1


class TestExport:
    def test_jsonl_round_trips(self):
        log = populated(3)
        lines = log.export_jsonl().splitlines()
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["actor"] == "actor-0"
        assert first["entry_hash"] == log.entries[0].entry_hash

    def test_empty_log_exports_empty_string(self):
        assert AuditLog().export_jsonl() == ""
