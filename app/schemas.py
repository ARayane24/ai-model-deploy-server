from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from pyparsing import Any, Union

# Input when creating a model
class AIModelCreate(BaseModel):
    name: str
    description: str
    tags: List[str]

# Output when returning model metadata
class AIModelOut(BaseModel):
    id: UUID
    name: str
    description: str
    tags: List[str]
    file_name: str
    file_size: float
    num_inputs: int
    num_outputs: int
    num_parameters: int
    params: Optional[dict] = {}
    created_at: datetime

    class Config:
        from_attributes = True

# Input to store a model result
class ResultInput(BaseModel):
    model_id: UUID
    output_vector: List[float]

# Output model result
class AIModelResultOut(BaseModel):
    id: UUID
    model_id: UUID
    output_vector: Any
    created_at: datetime

    class Config:
        from_attributes = True
