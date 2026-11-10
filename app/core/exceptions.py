"""Custom application exceptions with structured details."""
from __future__ import annotations


class CopilotException(Exception):
    """Base exception for all application errors."""
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class FileParseError(CopilotException):
    """Raised when a file cannot be parsed."""


class ValidationError(CopilotException):
    """Raised when input data fails validation."""


class ModelNotLoadedError(CopilotException):
    """Raised when an ML model is required but not loaded."""


class VectorStoreError(CopilotException):
    """Raised when vector store operations fail."""


class LLMError(CopilotException):
    """Raised when LLM calls fail."""


class DatasetNotFoundError(CopilotException):
    """Raised when a requested dataset does not exist."""


class RateLimitError(CopilotException):
    """Raised when rate limit is exceeded."""
