import uuid
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, ARRAY, Float, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AIModel(Base):
    __tablename__ = "ai_models"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    tags = Column(ARRAY(String))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    file_name = Column(String)
    file_size = Column(Float)  # in MB
    num_inputs = Column(Integer)
    num_outputs = Column(Integer)
    num_parameters = Column(Integer)
    params = Column(JSON)

    results = relationship("AIModelResult", back_populates="model", cascade="all, delete-orphan")


class AIModelResult(Base):
    __tablename__ = "ai_model_results"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey("ai_models.id"), nullable=False)
    output_vector = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    model = relationship("AIModel", back_populates="results")
