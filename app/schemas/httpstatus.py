from enum import Enum
from typing import List, Optional

from pydantic import UUID4, BaseModel, EmailStr, Field

from app.http_utils import *


class model200(BaseModel):
    message: str = ""


class model400BadRequest(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.BAD_REQUEST.value)


class model400SFBadRequest(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.SF_BAD_REQUEST.value)


class model401UnOAuth(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.UNOAUTH.value)


class model401NotSFAdmin(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.NotSFAdmin.value)


class model402PaymentRequire(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.NotSFAdmin.value)


class model403RequestLimitExceeded(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.REQUEST_LIMIT_EXCEEDED.value)


class model403ForbiddenOperation(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.FORBIDDEN_OPERATION.value)


class model403SFForbiddenOperation(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.SF_FORBIDDEN_OPERATION.value)


class model403InactiveUser(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.INACTIVE_USER.value)


class model403InsufficientUser(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.INSUFFICIENT_USER.value)


class model404NotFound(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.NOT_FOUND.value)


class model404UserNotFound(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.USER_NOT_FOUND.value)


class model404WorkspaceNotFound(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.WORKSPACE_NOT_FOUND.value)


class model404SFResourceNotFound(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.SF_RESOURCE_NOT_FOUND.value)


class model404SFResourceNotFound(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.SF_RESOURCE_NOT_FOUND.value)


class model409Conflict(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.CONFLICT.value)


class model500SFInternalError(BaseModel):
    message: str
    code: str = Field(default=ErrorCode.SF_RESOURCE_NOT_FOUND.value)


class model307(BaseModel):
    message: str = "redirection"


class model404SyncStatusNotFound(BaseModel):
    message: str = "sync status not found"
