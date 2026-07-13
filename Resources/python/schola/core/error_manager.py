# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Exceptions for making gRPC errors more interpretable, and a context manager that automatically converts gRPC errors into our custom exceptions.
"""

import abc
from contextlib import ContextDecorator
from types import TracebackType

import grpc


class ScholaException(Exception):
    """
    Base class for all exceptions in Schola.
    """

    ...


class WrappedGrpcException(ScholaException, metaclass=abc.ABCMeta):
    """
    Base class for all Schola exceptions that wrap gRPC errors, to add more information.
    """

    @classmethod
    @abc.abstractmethod
    def comes_from(cls, exception: grpc.RpcError) -> bool:
        """
        Return whether ``exception`` should be wrapped by this Schola exception type.

        Parameters
        ----------
        exception : grpc.RpcError
            gRPC error raised from a Schola RPC call.

        Returns
        -------
        bool
            ``True`` if this class should replace ``exception`` when handling the error.
        """
        ...


class NoServerError(WrappedGrpcException):
    """
    Error raised when the Schola gRPC server is not reachable (Unreal not running or not in play).

    Parameters
    ----------
    exception : grpc.RpcError
        Original gRPC error; retained for exception chaining when re-raised.
    """

    def __init__(self, exception: grpc.RpcError):
        super().__init__()

    def __str__(self) -> str:
        return (
            "No Server detected. Is Unreal Running? If it is, have you hit begin play?"
        )

    @classmethod
    def comes_from(cls, exception: grpc.RpcError) -> bool:
        """
        Match ``UNAVAILABLE`` with a ``failed to connect to all addresses`` detail prefix.

        Parameters
        ----------
        exception : grpc.RpcError
            gRPC error to classify.

        Returns
        -------
        bool
            ``True`` if the error indicates no server at the configured address.
        """
        details = exception.details()
        return (
            exception.code() == grpc.StatusCode.UNAVAILABLE
            and details is not None
            and details.startswith("failed to connect to all addresses")
        )


class UnrealCrashedError(WrappedGrpcException):
    """
    Error raised when the Unreal session appears to have stopped or torn down the stream.

    Parameters
    ----------
    exception : grpc.RpcError
        Original gRPC error; retained for exception chaining when re-raised.
    """

    def __init__(self, exception: grpc.RpcError):
        super().__init__()

    def __str__(self) -> str:
        return "It looks like Unreal has stopped responding. Did you stop the running game?"

    @classmethod
    def comes_from(cls, exception: grpc.RpcError) -> bool:
        """
        Match cancelled/unavailable/unknown statuses that indicate a dead or removed stream.

        Parameters
        ----------
        exception : grpc.RpcError
            gRPC error to classify.

        Returns
        -------
        bool
            ``True`` if the error likely indicates Unreal stopped responding.
        """
        code_cancelled = exception.code() == grpc.StatusCode.CANCELLED
        details_cancelled = (
            exception.code() == grpc.StatusCode.UNAVAILABLE
            and exception.details() == "Cancelling all calls"
        )
        stream_cancelled = (
            exception.code() == grpc.StatusCode.UNKNOWN
            and exception.details() == "Stream removed"
        )
        return code_cancelled or details_cancelled or stream_cancelled


class MissingMethodError(WrappedGrpcException):
    """
    Error raised when the server reports the RPC or Schola endpoint is not implemented.

    Parameters
    ----------
    exception : grpc.RpcError
        Original gRPC error; retained for exception chaining when re-raised.
    """

    def __init__(self, exception: grpc.RpcError):
        super().__init__()

    def __str__(self) -> str:
        return "Expected an endpoint to exist in unreal but it doesn't. Check that your environment is configured correctly."

    @classmethod
    def comes_from(cls, exception: grpc.RpcError) -> bool:
        """
        Match gRPC ``UNIMPLEMENTED`` (method or service missing on server).

        Parameters
        ----------
        exception : grpc.RpcError
            gRPC error to classify.

        Returns
        -------
        bool
            ``True`` if the server does not implement the requested RPC.
        """
        return exception.code() == grpc.StatusCode.UNIMPLEMENTED


class EnvironmentException(ScholaException):
    """
    Exception for generic issues in a Gymnasium (or similar) wrapper around Schola.

    Parameters
    ----------
    message : str
        Explanation of the environment error.
    """

    def __init__(self, message: str):
        super().__init__(message)


ALL_EXCEPTIONS = [NoServerError, UnrealCrashedError, MissingMethodError]


class ScholaErrorContextManager(ContextDecorator):
    """
    Context manager / decorator that maps known :class:`grpc.RpcError` values to Schola exceptions.
    """

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """
        If ``exc_value`` is a :class:`grpc.RpcError` that matches a known Schola
        wrapper, replace it with that wrapper; otherwise propagate.

        Parameters
        ----------
        exc_type : type or None
            Exception type, or None when the block completed without error.
        exc_value : BaseException or None
            Exception instance raised in the block, if any.
        exc_tb : types.TracebackType or None
            Traceback for ``exc_value``.

        Returns
        -------
        bool or None
            ``False`` to re-raise after handling, ``None`` to propagate unchanged.
        """
        if isinstance(exc_value, grpc.RpcError):
            # check if it matches any of our current custom exceptions
            for exception_class in ALL_EXCEPTIONS:
                if exception_class.comes_from(exc_value):
                    raise exception_class(exc_value) from exc_value
            # re-raise the current exception with false
            return False
        # return None to let the exceptions propagate on their own
        return None


class NoAgentsException(ScholaException):
    """
    Exception raised when an environment in the Schola definition has no registered agents.

    This exception is raised when the connection to Unreal is successful, and the definition
    contained environment(s), but some environment(s) had no agents.

    Parameters
    ----------
    env_id : int
        Environment index in the Schola definition that had no agents.
    """

    def __init__(self, env_id: int):
        super().__init__()
        self.env_id: int = env_id

    def __str__(self) -> str:
        return f"Connected to Unreal successfully but Env:{self.env_id} has no agents. Please register at least one agent to each environment."


class NoEnvironmentsException(ScholaException):
    """
    Exception raised when Unreal returns no environment definitions after a successful connection.

    This exception is raised when the connection to Unreal is successful but
    no environment definitions are received, indicating that no environment
    objects are present in the loaded map.
    """

    def __init__(self):
        super().__init__()

    def __str__(self) -> str:
        return "Connected to Unreal successfully but received no Environment Definitions. Check that there is an environment object in your map."
