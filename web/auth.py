"""
Auth utilities: password hashing, signed session cookies.
"""

import os
from fastapi import Request, Response
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from passlib.context import CryptContext

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
RESET_MAX_AGE   = 60 * 60             # 1 hour

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_serializer = URLSafeTimedSerializer(SECRET_KEY)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def set_session(response: Response, user_id: int):
    token = _serializer.dumps(user_id, salt="session")
    response.set_cookie(
        "session", token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,   # set True when behind HTTPS in prod (Railway handles this)
    )


def clear_session(response: Response):
    response.delete_cookie("session")


def make_reset_token(email: str) -> str:
    return _serializer.dumps(email.lower(), salt="password-reset")


def verify_reset_token(token: str) -> str | None:
    try:
        return _serializer.loads(token, salt="password-reset", max_age=RESET_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def get_user_id(request: Request) -> int | None:
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        return _serializer.loads(token, salt="session", max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
