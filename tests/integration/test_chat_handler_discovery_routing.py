from api.chat_handler import ChatHandler
from api.in_memory_session_store import InMemorySessionStore
from models.discovery import SelectionAdvice
from models.qa import IntentResolution
from services.selection_parser import SelectionHandler


class FakeRunner:
    def __init__(self, result=None):
        self.result = result or {}
        self.calls = []

    def invoke(self, input, config):
        self.calls.append({"input": input, "config": config})
        return self.result


class FakeSearchCandidateRepository:
    def list_latest_for_session(self, session_id):
        return []

    def update_status(self, candidate_id, status):
        return None


class FakeSearcher:
    pass


def _discovery_result():
    advice = SelectionAdvice(
        topic="agent memory",
        response_text="Choose papers 1 or 2.",
        recommended_candidate_ids=[],
        candidate_count=2,
    )
    return {
        "discovery_topic": "agent memory",
        "selection_advice": advice,
        "response_text": advice.response_text,
        "search_warnings": ["Search query failed (HTTP 429): agent memory"],
        "next_phase": "selection",
    }


def _handler(*, conversation_result=None, discovery_result=None, discovery_runner=True):
    store = InMemorySessionStore()
    conversation = FakeRunner(conversation_result or {"response_text": "conversation"})
    discovery = FakeRunner(discovery_result or _discovery_result()) if discovery_runner else None
    selection_handler = SelectionHandler(
        session_store=store,
        candidate_repository=FakeSearchCandidateRepository(),
    )
    handler = ChatHandler(
        store=store,
        conversation_runner=conversation,
        discovery_runner=discovery,
        searcher=FakeSearcher(),
        selection_handler=selection_handler,
    )
    return handler, store, conversation, discovery


def test_handler_routes_discovery_request_to_discovery_runner():
    handler, store, conversation, discovery = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "Find recent papers about agent memory")

    assert conversation.calls == []
    assert len(discovery.calls) == 1
    assert discovery.calls[0]["input"]["user_message"] == (
        "Find recent papers about agent memory"
    )
    assert discovery.calls[0]["input"]["discovery_turn_id"] == result.user_turn_id
    assert result.intent == "discover"
    assert result.response_text == "Choose papers 1 or 2."
    assert result.search_warnings == ["Search query failed (HTTP 429): agent memory"]


def test_handler_sets_phase_selection_after_discovery():
    handler, store, _, _ = _handler()
    session = handler.create_session()

    result = handler.handle_message(session.id, "Find papers about agent memory")

    assert result.phase == "selection"
    assert store.require_session(session.id).phase == "selection"


def test_handler_conversation_needs_discovery_reroutes_to_discovery():
    conversation_resolution = IntentResolution(
        intent="discover",
        referenced_paper_ids=[],
        ambiguous=False,
        confidence=0.8,
    )
    handler, _, conversation, discovery = _handler(
        conversation_result={
            "intent_resolution": conversation_resolution,
            "intent": "discover",
            "needs_discovery": True,
            "discovery_topic": "agent memory",
        }
    )
    session = handler.create_session()

    result = handler.handle_message(session.id, "Can you search for this topic?")

    assert len(conversation.calls) == 1
    assert len(discovery.calls) == 1
    assert discovery.calls[0]["input"]["discovery_topic"] == "agent memory"
    assert result.phase == "selection"


def test_handler_discovery_runner_missing_returns_controlled_response():
    handler, _, conversation, discovery = _handler(discovery_runner=False)
    session = handler.create_session()

    result = handler.handle_message(session.id, "Find papers about agent memory")

    assert conversation.calls == []
    assert discovery is None
    assert result.intent == "discover"
    assert result.needs_discovery is True
    assert "not configured" in result.response_text


def test_handler_config_includes_searcher_for_discovery():
    handler, _, _, discovery = _handler()
    session = handler.create_session()

    handler.handle_message(session.id, "Find papers about agent memory")

    config = discovery.calls[0]["config"]["configurable"]
    assert "searcher" in config
    assert isinstance(config["searcher"], FakeSearcher)


def test_handler_does_not_route_selection_phase_to_discovery():
    handler, store, conversation, discovery = _handler()
    session = handler.create_session()
    store.update_phase(session.id, "selection")

    result = handler.handle_message(session.id, "Find papers about agent memory")

    assert conversation.calls == []
    assert discovery.calls == []
    assert result.intent == "select_papers"
