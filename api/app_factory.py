from api.chat_handler import ChatHandler, ConversationRunner
from storage.db import make_engine, make_session_factory
from storage.repositories import PostgresAgentRunPersistence, PostgresSessionStore


def create_chat_handler(
    *,
    database_url: str,
    conversation_runner: ConversationRunner,
) -> ChatHandler:
    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    return ChatHandler(
        store=PostgresSessionStore(session_factory),
        conversation_runner=conversation_runner,
        agent_run_persistence=PostgresAgentRunPersistence(session_factory),
    )
