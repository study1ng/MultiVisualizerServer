class MvisError(Exception):
    def __init__(self, *message, payload: dict[str, object] | None = None):
        """MvisError is a base exception about this application
        message: a sequence of message send to client
        payload: a sequence of regarding data send to client
        """
        super().__init__(*message)
        self.payload = payload


class InternalError(MvisError):
    """Internal Error is a error which is not because of user manipulation but system implemention"""


class FileError(MvisError):
    """File Error is a error which occurs if a passed file path is inappropriate"""


class ServerError(Exception):
    """Server Error is an error which occurs if a bug was in server process and not related to client"""
