from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query_text: str
    search_model: Optional[str] = None
    relevance_model: Optional[str] = None
    extraction_model: Optional[str] = None
    answer_model: Optional[str] = None
    search_prompt: Optional[str] = None
    relevance_prompt: Optional[str] = None
    extraction_prompt: Optional[str] = None
    answer_prompt: Optional[str] = None
