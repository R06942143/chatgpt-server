from enum import Enum


class ErrorCode(Enum):
    INVALID_VALUE = "INVALID_VALUE"  # 400
    BAD_REQUEST = "BAD_REQUEST"
    SF_BAD_REQUEST = "SF_BAD_REQUEST"
    INVALID_OPERATION = "INVALID_OPERATION"  # 400
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"  # 401
    EMAIL_ACCOUNT_EXISTS_ERROR = "EMAIL_ACCOUNT_EXISTS_ERROR"  # 401
    OAUTH_ACCOUNT_EXISTS_ERROR = "OAUTH_ACCOUNT_EXISTS_ERROR"  # 401
    UNOAUTH = "UNOAUTH"  # 401
    NotSFAdmin = "NotSFAdmin"  # 401
    PAYMENT_REQUIRED = "PAYMENT_REQUIRED"  # 402
    FORBIDDEN_OPERATION = "FORBIDDEN_OPERATION"  # 403
    INACTIVE_USER = "INACTIVE_USER"  # 403
    INSUFFICIENT_USER = "INSUFFICIENT_USER"  # 403
    REQUEST_LIMIT_EXCEEDED = "REQUEST_LIMIT_EXCEEDED"  # 403
    SF_FORBIDDEN_OPERATION = "SF_FORBIDDEN_OPERATION"  # 403
    OVER_LIMIT = "OVER_LIMIT"  # 403
    LINE_IMPORT_OVER_LIMIT = "LINE_IMPORT_OVER_LIMIT"  # 403
    SF_RESOURCE_NOT_FOUND = "SF_RESOURCE_NOT_FOUND"  # 404
    NOT_FOUND = "NOT_FOUND"  # 404
    USER_NOT_FOUND = "USER_NOT_FOUND"  # 404
    WORKSPACE_NOT_FOUND = "WORKSPACE_NOT_FOUND"  # 404
    CONFLICT = "CONFLICT"  # 409
    ETAG_NOT_MATCHED = "ETAG_NOT_MATCHED"  # 412

    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"  # 500
    SF_INTERNAL_ERROR = "SF_INTERNAL_ERROR"  # 500


class ABCError(Exception):
    """Base API exception"""

    message = "{class_name}: Unknown error occurred for user:{user_id} in workspace:{workspace_id}. Response content: {trace}"

    def __init__(
        self,
        class_name: str = "",
        trace: str = "",
        user_id: str = "",
        workspace_id: str = "",
        trace_id: str = "",
        extra_data: dict = {},
    ):
        """Initialize the exception
        Args:
            class_name: Where the error raise
            trace: the error trace
            user_id:  user id
            workspace_id: where the user belong
            trace_id: as same as the header
            extra_data: the other data that u want to put
        """

        self.class_name = class_name
        self.trace = trace
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.trace_id = trace_id
        self.extra_data = extra_data

    def __str__(self):
        return self.message.format(
            class_name=self.class_name,
            user_id=self.user_id,
            trace=self.trace,
            workspace_id=self.workspace_id,
            trace_id=self.trace_id,
            extra_data=self.extra_data,
        )


class InvalidDataFormat(ABCError):
    message = "{class_name}: InvalidDataFormat occurred for user: {user_id} in workspace: {workspace_id}. Response content: {trace}"


class InvalidPhoneFormat(ABCError):
    def __str__(self):
        return f"{self.class_name}: Invalid Format occurred for Phone: {self.extra_data['phone']}."


class UserNotFound(ABCError):
    def __str__(self):
        return f"cannot find user({self.user_id}) in server."


class UnConnected(ABCError):
    def __str__(self):
        return f"workspace_id : {self.workspace_id} haven't connected."


class UnOAuth(ABCError):
    def __str__(self):
        return f"UnOAuth User({self.user_id}). Please OAuth first."
