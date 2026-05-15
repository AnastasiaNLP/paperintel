from api.chat_handler import AnalysisRunner, ChatHandler, ConversationRunner
from services.retrieval_layer import RetrievalLayer
from storage.db import make_engine, make_session_factory
from storage.repositories import PostgresAgentRunPersistence, PostgresSessionStore


def create_chat_handler(
    *,
    database_url: str,
    conversation_runner: ConversationRunner,
    analysis_runner: AnalysisRunner | None = None,
    retrieval_layer: RetrievalLayer | None = None,
) -> ChatHandler:
    engine = make_engine(database_url)
    session_factory = make_session_factory(engine)
    return ChatHandler(
        store=PostgresSessionStore(session_factory),
        conversation_runner=conversation_runner,
        analysis_runner=analysis_runner,
        agent_run_persistence=PostgresAgentRunPersistence(session_factory),
        retrieval_layer=retrieval_layer,
    )
