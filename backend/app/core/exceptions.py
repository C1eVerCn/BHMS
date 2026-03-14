"""应用级异常定义。"""

from __future__ import annotations


class BHMSException(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "bhms_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
