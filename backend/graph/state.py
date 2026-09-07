"""Internal agent state definition for the LangGraph pipeline."""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    prompt: str
    history: List[Dict[str, str]]
    is_relevant: bool
    rejection_message: Optional[str]
    semantic_query: Optional[str]
    metadata_filters: Dict[str, Any]
    query_vector: Optional[List[float]]
    retrieved_context: List[Dict[str, Any]]
    excluded_track_ids: List[str]
    retry_count: int
    is_vetted: bool
    vetting_rationale: Optional[str]
    reasoning_analysis: Optional[str]
    document_rationales: Dict[str, str]
    final_response: str
    sources: List[Dict[str, Any]]
    extracted_keywords: Optional[List[str]]
    pulled_tracks: Optional[List[Dict[str, Any]]]
    vetting_decisions: Optional[List[Dict[str, Any]]]
    agent_trace: Optional[Dict[str, Any]]
