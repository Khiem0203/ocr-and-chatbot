import os
import secrets

from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

load_dotenv()

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").strip().lower() in ("1", "true", "yes")
USERNAME = os.getenv("API_USERNAME", "")
PASSWORD = os.getenv("API_PASSWORD", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

security = HTTPBasic(auto_error=False)


def require_user(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    if not AUTH_ENABLED:
        return "anonymous"

    valid = (
        credentials is not None
        and bool(USERNAME)
        and bool(PASSWORD)
        and secrets.compare_digest(credentials.username.encode("utf-8"), USERNAME.encode("utf-8"))
        and secrets.compare_digest(credentials.password.encode("utf-8"), PASSWORD.encode("utf-8"))
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Sai tài khoản hoặc mật khẩu.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
