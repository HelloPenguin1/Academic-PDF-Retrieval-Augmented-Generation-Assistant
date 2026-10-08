from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


class SessionManager:
    def __init__(self):
        self.store = {}  # In-memory store for chat sessions

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_all_sessions(self):
        """Get all active sessions"""
        return self.store

    def clear_session(self, session_id: str):
        """Clear history for a specific session"""
        if session_id in self.store:
            del self.store[session_id]

    def clear_all_sessions(self):
        """Clear all sessions"""
        self.store = {}
