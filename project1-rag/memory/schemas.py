from pydantic import BaseModel, Field
from typing import List


class UserProfile(BaseModel):
    user_id: str
    role: str = "unknown"
    team: str = "unknown"
    preferred_answer_style: str = "concise_with_citations"
    preferred_doc_types: List[str] = Field(default_factory=list)
    recent_topics: List[str] = Field(default_factory=list)
    frequent_docs: List[str] = Field(default_factory=list)
    session_context: List[str] = Field(default_factory=list)


class SessionMessage(BaseModel):
    user_id: str
    query: str
    answer: str
    sources: List[str] = Field(default_factory=list)


class FeedbackEvent(BaseModel):
    user_id: str
    chunk_id: str
    doc_id: str
    positive: bool
