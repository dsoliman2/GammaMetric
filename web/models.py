"""
Database models for LeapfrogDose platform.
Uses SQLite locally; set DATABASE_URL env var for PostgreSQL on Railway.
"""

import os
import json
import platform
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship

_default_db = "sqlite:///./leapfrogdose.db" if platform.system() == "Windows" else "sqlite:////tmp/leapfrogdose.db"
DATABASE_URL = os.getenv("DATABASE_URL", _default_db)

# Railway gives postgres://, SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_kwargs = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True)
    email         = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    facility_name = Column(String, default="")
    created_at    = Column(DateTime, default=datetime.utcnow)

    analyses = relationship("AnalysisResult", back_populates="user",
                            order_by="desc(AnalysisResult.analyzed_at)")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    facility_name = Column(String)
    filename      = Column(String)
    analyzed_at   = Column(DateTime, default=datetime.utcnow)
    results_json  = Column(Text)   # full results dict as JSON

    user = relationship("User", back_populates="analyses")

    @property
    def results(self) -> dict:
        return json.loads(self.results_json) if self.results_json else {}


class DriftAlert(Base):
    __tablename__ = "drift_alerts"

    id               = Column(Integer, primary_key=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    analysis_id      = Column(Integer, ForeignKey("analysis_results.id"), nullable=False)
    region           = Column(String, nullable=False)
    old_status       = Column(String)
    new_status       = Column(String)
    old_p75          = Column(Float)
    new_p75          = Column(Float)
    created_at       = Column(DateTime, default=datetime.utcnow)
    acknowledged_at  = Column(DateTime, nullable=True)
    acknowledged_by  = Column(String, nullable=True)   # email
    note             = Column(Text, nullable=True)

    user     = relationship("User", backref="drift_alerts")
    analysis = relationship("AnalysisResult", backref="drift_alerts")

    @property
    def is_acknowledged(self):
        return self.acknowledged_at is not None


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
