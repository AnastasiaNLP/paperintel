import pytest

from api.in_memory_session_store import InMemorySessionStore, SessionNotFoundError


def test_store_returns_copies_not_mutable_internal_objects():
    store = InMemorySessionStore()
    session = store.create_session()

    session.phase = "failed"

    assert store.require_session(session.id).phase == "idle"


def test_store_sets_selected_candidate_ids_idempotently():
    store = InMemorySessionStore()
    session = store.create_session()

    updated = store.set_selected_candidate_ids(
        session.id,
        ["candidate-1", "candidate-2", "candidate-1"],
    )

    assert updated.selected_candidate_ids == ["candidate-1", "candidate-2"]
    assert store.require_session(session.id).selected_candidate_ids == [
        "candidate-1",
        "candidate-2",
    ]


def test_append_and_list_recent_turns():
    store = InMemorySessionStore()
    session = store.create_session()

    for index in range(5):
        store.append_turn(session.id, role="user", content=f"turn {index}")

    recent = store.list_recent_turns(session.id, limit=2)

    assert [turn.content for turn in recent] == ["turn 3", "turn 4"]


def test_store_raises_for_missing_session_on_turn_write():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        store.append_turn("missing", role="user", content="hello")
