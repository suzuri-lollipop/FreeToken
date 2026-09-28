"""Durable accounting receipts (daemon/accounting.py).

Receipts are the daemon's side of the "allow stop only after the receipt is on
disk" contract: a silently dropped or renamed receipt shows up months later as
an accounting gap nobody can attribute. The expectations here are the contract
itself - one idempotent file per receiptId, deterministic pending order,
injection-safe ids - exercised through a tmp_path outbox.
"""

from __future__ import annotations

import json

import pytest

from freetoken.daemon.accounting import (
    AccountingOutbox,
    AccountingOutboxError,
    stable_receipt_id,
)


def _receipt(receipt_id: str, created_at: int) -> dict:
    return {"receiptId": receipt_id, "createdAt": created_at, "kind": "stop"}


def test_persist_writes_exactly_one_file_with_the_receipt(tmp_path):
    outbox = AccountingOutbox(str(tmp_path / "outbox"))
    persisted = outbox.persist(_receipt("abc123", 10))

    assert persisted == _receipt("abc123", 10)
    files = list((tmp_path / "outbox").iterdir())
    assert len(files) == 1 and files[0].name == "abc123.json"
    on_disk = json.loads(files[0].read_text())
    assert on_disk == _receipt("abc123", 10)


def test_persist_is_idempotent_and_reuses_the_first_document(tmp_path):
    outbox = AccountingOutbox(str(tmp_path / "outbox"))
    first = outbox.persist({**_receipt("id1", 1), "payload": "alpha"})
    second = outbox.persist({**_receipt("id1", 1), "payload": "beta"})

    # The retry must reuse the prior durable document, not create a second event.
    assert first["payload"] == "alpha"
    assert second == first
    assert len(list((tmp_path / "outbox").iterdir())) == 1


def test_pending_is_empty_when_the_outbox_dir_is_missing(tmp_path):
    outbox = AccountingOutbox(str(tmp_path / "missing"))
    assert outbox.pending() == []


def test_pending_orders_by_created_at_then_receipt_id(tmp_path):
    outbox = AccountingOutbox(str(tmp_path / "outbox"))
    outbox.persist(_receipt("r3", 20))
    outbox.persist(_receipt("r1", 10))
    outbox.persist(_receipt("r2", 10))

    ids = [doc["receiptId"] for doc in outbox.pending()]
    assert ids == ["r1", "r2", "r3"]


def test_ack_removes_the_receipt_and_is_idempotent(tmp_path):
    outbox = AccountingOutbox(str(tmp_path / "outbox"))
    outbox.persist(_receipt("abc123", 10))

    assert outbox.ack("abc123") == {"acked": True, "already": False, "receiptId": "abc123"}
    assert not (tmp_path / "outbox" / "abc123.json").exists()
    # a second ack (crash-then-retry) reports already-acked, never an error
    assert outbox.ack("abc123") == {"acked": True, "already": True, "receiptId": "abc123"}


@pytest.mark.parametrize(
    "receipt_id", ["../escape", "a/b", "x" * 201, "", "has space", 12, None]
)
def test_invalid_receipt_ids_are_rejected(tmp_path, receipt_id):
    outbox = AccountingOutbox(str(tmp_path / "outbox"))
    with pytest.raises(ValueError, match="invalid accounting receiptId"):
        outbox.persist({"receiptId": receipt_id, "kind": "stop"})
    assert outbox.pending() == []


def test_ack_rejects_invalid_ids_too(tmp_path):
    outbox = AccountingOutbox(str(tmp_path / "outbox"))
    with pytest.raises(ValueError, match="invalid accounting receiptId"):
        outbox.ack("a/b")


def test_corrupt_receipt_file_raises_and_is_reported_in_pending(tmp_path):
    outbox = AccountingOutbox(str(tmp_path / "outbox"))
    (tmp_path / "outbox").mkdir()
    (tmp_path / "outbox" / "bad.json").write_text("{not json")

    with pytest.raises(AccountingOutboxError, match="failed to read accounting receipt"):
        outbox.pending()


def test_receipt_whose_filename_does_not_match_its_id_raises(tmp_path):
    outbox = AccountingOutbox(str(tmp_path / "outbox"))
    (tmp_path / "outbox").mkdir()
    (tmp_path / "outbox" / "mismatch.json").write_text(
        json.dumps(_receipt("somethingelse", 10))
    )

    with pytest.raises(AccountingOutboxError, match="filename/id mismatch"):
        outbox.pending()


def test_replace_failure_raises_and_leaves_no_stale_tmp(tmp_path):
    def broken_replace(src, dst):
        raise OSError("disk full")

    outbox = AccountingOutbox(str(tmp_path / "outbox"), replace_fn=broken_replace)
    with pytest.raises(AccountingOutboxError, match="failed to persist"):
        outbox.persist(_receipt("abc123", 10))

    leftovers = [p for p in (tmp_path / "outbox").iterdir() if ".tmp" in p.name]
    assert leftovers == []
    assert not (tmp_path / "outbox" / "abc123.json").exists()


def test_stable_receipt_id_is_deterministic_and_distinct():
    a = stable_receipt_id("serve.machine:30000")
    b = stable_receipt_id("serve.machine:30000")
    other = stable_receipt_id("serve.machine:30001")
    assert a == b
    assert a != other
    assert all(c.isalnum() or c == "-" for c in a)