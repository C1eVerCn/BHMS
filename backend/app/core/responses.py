"""统一响应封装。"""

from __future__ import annotations

from typing import Any


def success_response(data: Any, message: str = "ok") -> dict[str, Any]:
    return {"success": True, "message": message, "data": data}


def error_response(message: str, code: str = "error") -> dict[str, Any]:
    return {"success": False, "message": message, "error_code": code}
