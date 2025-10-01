from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .data import AD_TEMPLATES
from .pipeline import (
    translate_to_english_if_needed,
    build_template_embeddings,
    semantic_search,
)

# Build template embeddings at startup
TEMPLATE_EMBEDDINGS = build_template_embeddings(AD_TEMPLATES)

app = FastAPI(title="Multilingual Contextual Search")


class QueryInput(BaseModel):
    query: str
    language: Optional[str] = None  # 'en', 'es', 'auto', etc.
    limit: int = 5


@app.post("/search")
async def search(query_input: QueryInput):
    query_text = translate_to_english_if_needed(
        text=query_input.query,
        language_hint=query_input.language,
    )

    results = semantic_search(
        query_text=query_text,
        templates=AD_TEMPLATES,
        template_embeddings=TEMPLATE_EMBEDDINGS,
        limit=query_input.limit,
    )
    return {"results": results}
