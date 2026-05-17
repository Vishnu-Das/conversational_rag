from pydantic import BaseModel, Field


class LLMRouterOutput(BaseModel):

    strategy: str = Field(
        description="One of: hybrid, parent_child, fusion"
    )

    reason: str = Field(
        description="Short explanation for why this strategy was selected"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )