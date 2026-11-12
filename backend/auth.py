"""
Authentication for the IndicSent-M web app.

- Users are stored in a SQLite file (backend/users.db, created automatically).
- Passwords are hashed with scrypt (Python standard library) and a random salt per user.
- Login/registration return a signed JWT (HS256). The frontend sends it as
  "Authorization: Bearer <token>".
- Failed logins are throttled per email + IP address.

Optional environment variables:
    INDICSENT_SECRET        signing key. If unset, one is generated into backend/.secret_key
    INDICSENT_DB            path of the SQLite file (default: backend/users.db)
    INDICSENT_TOKEN_HOURS   token lifetime in hours (default: 168, i.e. 7 days)
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("INDICSENT_DB") or BASE_DIR / "users.db")
TOKEN_TTL_HOURS = int(os.environ.get("INDICSENT_TOKEN_HOURS", "168"))
JWT_ALGORITHM = "HS256"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1

MAX_FAILED_LOGINS = 8
FAILED_LOGIN_WINDOW_SECONDS = 600


# ------------------------------------------------------------------ secret key
def _load_secret() -> str:
    env_value = os.environ.get("INDICSENT_SECRET")
    if env_value:
        return env_value
    key_file = BASE_DIR / ".secret_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    key_file.write_text(key, encoding="utf-8")
    try:
        os.chmod(key_file, 0o600)
    except OSError:
        pass  # not supported on every platform (e.g. Windows)
    return key


SECRET_KEY = _load_secret()


# -------------------------------------------------------------------- database
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )
            """
        )


# ------------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


# Verified against when the email is unknown, so response time does not reveal
# which emails are registered.
_DUMMY_HASH = hash_password("not-a-real-password")


# ---------------------------------------------------------------------- tokens
def make_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    headers = {"WWW-Authenticate": "Bearer"}
    if creds is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.", headers=headers)
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Your session has expired. Please sign in again.",
            headers=headers,
        )
    with db() as conn:
        row = conn.execute(
            "SELECT id, name, email FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=401,
            detail="Your session has expired. Please sign in again.",
            headers=headers,
        )
    return dict(row)


# ------------------------------------------------------------ login throttling
_failures = {}
_failures_lock = threading.Lock()


def _recent_failures(key: str) -> list:
    now = time.time()
    hits = [t for t in _failures.get(key, []) if now - t < FAILED_LOGIN_WINDOW_SECONDS]
    _failures[key] = hits
    return hits


def _is_throttled(key: str) -> bool:
    with _failures_lock:
        return len(_recent_failures(key)) >= MAX_FAILED_LOGINS


def _record_failure(key: str) -> None:
    with _failures_lock:
        _recent_failures(key).append(time.time())


def _clear_failures(key: str) -> None:
    with _failures_lock:
        _failures.pop(key, None)


# ---------------------------------------------------------------------- routes
class RegisterIn(BaseModel):
    name: str
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


router = APIRouter(prefix="/api/auth", tags=["auth"])


def _auth_response(row_id: int, name: str, email: str) -> dict:
    return {"token": make_token(row_id), "user": {"id": row_id, "name": name, "email": email}}


@router.post("/register", status_code=201)
def register(body: RegisterIn):
    name = " ".join(body.name.split())
    email = body.email.strip().lower()

    if not 2 <= len(name) <= 60:
        raise HTTPException(status_code=400, detail="Name must be 2 to 60 characters.")
    if len(email) > 254 or not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if not 8 <= len(body.password) <= 128:
        raise HTTPException(status_code=400, detail="Password must be 8 to 128 characters.")

    password_hash = hash_password(body.password)
    try:
        with db() as conn:
            cursor = conn.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name, email, password_hash, datetime.now(timezone.utc).isoformat()),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    return _auth_response(user_id, name, email)


@router.post("/login")
def login(body: LoginIn, request: Request):
    email = body.email.strip().lower()
    client_ip = request.client.host if request.client else "unknown"
    throttle_key = f"{email}|{client_ip}"

    if _is_throttled(throttle_key):
        raise HTTPException(
            status_code=429, detail="Too many failed attempts. Please try again in a few minutes."
        )

    with db() as conn:
        row = conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()

    stored = row["password_hash"] if row else _DUMMY_HASH
    password_ok = len(body.password) <= 128 and verify_password(body.password, stored)
    if row is None or not password_ok:
        _record_failure(throttle_key)
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    _clear_failures(throttle_key)
    return _auth_response(row["id"], row["name"], row["email"])


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    return user
