from typing import Any, Optional

from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    s3_path: str = Field(..., description="Đường dẫn S3 tới file cần OCR (PDF hoặc ảnh: jpg/png/bmp/tiff/webp/gif)")
    access_key: str = Field(..., description="S3-compatible access key")
    secret_key: str = Field(..., description="S3-compatible secret key")
    endpoint: str = Field(..., description="S3-compatible endpoint URL")
    document_id: Optional[int] = Field(None, description="ID tài liệu phía hệ thống gọi (tuỳ chọn)")


class SubmitResponse(BaseModel):
    error_code: int
    error_message: str
    request_id: str


class StatusResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[Any] = None
