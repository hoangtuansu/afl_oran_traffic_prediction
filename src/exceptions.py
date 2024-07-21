class Error(Exception):
    """Base class for other exceptions"""
    pass


class NoDataError(BaseException):
    """Raised when there is no data available in database for a given measurment"""
    pass
