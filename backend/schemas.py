from typing import Literal

from pydantic import BaseModel, Field


class AutocompleteRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=500)
    method: Literal["greedy", "temperature", "beam"] = "temperature"
    length: int = Field(default=120, ge=10, le=500)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    beam_width: int = Field(default=5, ge=1, le=8)


class AutocompleteCandidate(BaseModel):
    text: str
    generated: str
    score: float | None = None


class AutocompleteResponse(BaseModel):
    completions: list[AutocompleteCandidate]
    method_used: str


class TranslationRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=300)


class TranslationResponse(BaseModel):
    translation: str
    words: list[str]


class ClassificationRequest(BaseModel):
    input: str = Field(..., min_length=3, max_length=2000)


class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]

