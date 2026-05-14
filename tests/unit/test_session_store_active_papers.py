import pytest

from api.in_memory_session_store import InMemorySessionStore, SessionNotFoundError


def test_add_active_paper_appends_to_list():
    store = InMemorySessionStore()
    session = store.create_session()

    updated = store.add_active_paper(session.id, "2310.06825")

    assert updated.active_paper_ids == ["2310.06825"]
    assert store.require_session(session.id).active_paper_ids == ["2310.06825"]


def test_add_active_paper_idempotent_on_duplicate():
    store = InMemorySessionStore()
    session = store.create_session()

    store.add_active_paper(session.id, "2310.06825")
    updated = store.add_active_paper(session.id, "2310.06825")

    assert updated.active_paper_ids == ["2310.06825"]


def test_add_active_paper_preserves_insertion_order():
    store = InMemorySessionStore()
    session = store.create_session()

    store.add_active_paper(session.id, "2310.06825")
    updated = store.add_active_paper(session.id, "2401.12345")

    assert updated.active_paper_ids == ["2310.06825", "2401.12345"]


def test_add_active_paper_returns_updated_session():
    store = InMemorySessionStore()
    session = store.create_session()

    updated = store.add_active_paper(session.id, "2310.06825")

    assert updated.id == session.id
    assert updated.active_paper_ids == ["2310.06825"]


def test_add_active_paper_raises_for_missing_session():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        store.add_active_paper("missing", "2310.06825")


def test_add_active_paper_does_not_mutate_returned_session_objects():
    store = InMemorySessionStore()
    session = store.create_session()

    updated = store.add_active_paper(session.id, "2310.06825")
    updated.active_paper_ids.append("mutated-outside")

    assert store.require_session(session.id).active_paper_ids == ["2310.06825"]
