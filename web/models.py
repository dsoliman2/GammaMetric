"""
Database models for GammaDose platform.
Uses SQLite locally; set DATABASE_URL env var for PostgreSQL on Railway.
"""

import os
import json
import platform
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Boolean, Float, text
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
    acknowledged_by  = Column(String, nullable=True)
    note             = Column(Text, nullable=True)

    user     = relationship("User", backref="drift_alerts")
    analysis = relationship("AnalysisResult", backref="drift_alerts")

    @property
    def is_acknowledged(self):
        return self.acknowledged_at is not None


class StudyResult(Base):
    """One row per DICOM study ingested via the Orthanc webhook."""
    __tablename__ = "study_results"

    id                    = Column(Integer, primary_key=True)
    study_instance_uid    = Column(String, unique=True, nullable=False, index=True)
    acquisition_date      = Column(String)
    scanner_model         = Column(String)
    model_version         = Column(String)
    slice_thickness_mm    = Column(Float)
    reconstruction_kernel = Column(String)
    ctdivol_mgy           = Column(Float)
    kvp                   = Column(Float)
    estimated_sensitivity = Column(Float)
    degradation_pp        = Column(Float)
    classification        = Column(String)
    out_of_distribution       = Column(Boolean, default=False)
    result_json               = Column(Text)
    ingested_at               = Column(DateTime, default=datetime.utcnow)
    alerted                   = Column(Boolean, default=False)
    iterative_recon_detected  = Column(Boolean, default=False)
    iterative_recon_level     = Column(String, nullable=True)
    iterative_recon_method    = Column(String, nullable=True)


class ReaderStudyParticipant(Base):
    __tablename__ = "reader_study_participants"

    token          = Column(String, primary_key=True)
    name           = Column(String, nullable=False)
    role           = Column(String, nullable=False)
    institution    = Column(String, default="")
    case_order     = Column(Text)   # JSON list of case indices (0-14)
    flag_assignment = Column(Text)  # JSON dict: case_idx -> bool (show_flag)
    created_at     = Column(DateTime, default=datetime.utcnow)
    completed_at   = Column(DateTime, nullable=True)

    responses = relationship("ReaderStudyResponse", back_populates="participant",
                             order_by="ReaderStudyResponse.case_idx")


class ReaderStudyResponse(Base):
    __tablename__ = "reader_study_responses"

    id                  = Column(Integer, primary_key=True)
    participant_token   = Column(String, ForeignKey("reader_study_participants.token"), nullable=False)
    case_idx            = Column(Integer, nullable=False)
    show_flag           = Column(Boolean, nullable=False)
    management_decision = Column(String, nullable=False)
    confidence_rating   = Column(Integer, nullable=True)
    response_time_sec   = Column(Float, nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow)

    participant = relationship("ReaderStudyParticipant", back_populates="responses")


def init_db():
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        for ddl in [
            "ALTER TABLE reader_study_responses ADD COLUMN confidence_rating INTEGER",
            "ALTER TABLE study_results ADD COLUMN iterative_recon_detected BOOLEAN DEFAULT FALSE",
            "ALTER TABLE study_results ADD COLUMN iterative_recon_level VARCHAR",
            "ALTER TABLE study_results ADD COLUMN iterative_recon_method VARCHAR",
        ]:
            try:
                conn.execute(text(ddl))
                conn.commit()
            except Exception:
                pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
