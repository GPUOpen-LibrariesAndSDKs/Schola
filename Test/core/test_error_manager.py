# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Tests for schola.core.error_manager module
"""

import grpc
import pytest
from typing_extensions import override
from unittest.mock import Mock
from schola.core.error_manager import (
    ScholaException,
    WrappedGrpcException,
    NoServerError,
    UnrealCrashedError,
    MissingMethodError,
    EnvironmentException,
    ScholaErrorContextManager,
    NoAgentsException,
    NoEnvironmentsException,
)


class MockGrpcError(grpc.RpcError):

    def __init__(self, status_code: grpc.StatusCode, details_msg: str = "") -> None:
        super().__init__()
        self._status_code: grpc.StatusCode = status_code
        self._details_msg: str = details_msg

    @override
    def code(self) -> grpc.StatusCode:
        return self._status_code

    @override
    def details(self) -> str:
        return self._details_msg


class TestScholaException:
    """Tests for ScholaException base class"""

    def test_is_exception(self):
        """Test that ScholaException can be raised"""
        with pytest.raises(ScholaException):
            raise ScholaException("Test error")

    def test_inheritance(self):
        """Test that ScholaException inherits from Exception"""
        assert issubclass(ScholaException, Exception)


class TestNoServerError:
    """Tests for NoServerError"""

    def test_error_message(self):
        """Test error message formatting"""
        mock_exception = Mock()
        error = NoServerError(mock_exception)

        message = str(error)
        assert "No Server detected" in message
        assert "Unreal Running" in message

    def test_comes_from_unavailable(self):
        """Test detection of UNAVAILABLE status with correct message"""
        mock_exception = MockGrpcError(
            grpc.StatusCode.UNAVAILABLE, "failed to connect to all addresses"
        )
        assert NoServerError.comes_from(mock_exception)

    def test_comes_from_wrong_code(self):
        """Test rejection of wrong status code"""
        mock_exception = MockGrpcError(
            grpc.StatusCode.CANCELLED, "failed to connect to all addresses"
        )
        assert not NoServerError.comes_from(mock_exception)

    def test_comes_from_wrong_details(self):
        """Test rejection of wrong details message"""
        mock_exception = MockGrpcError(grpc.StatusCode.UNAVAILABLE, "some other error")
        assert not NoServerError.comes_from(mock_exception)


class TestUnrealCrashedError:
    """Tests for UnrealCrashedError"""

    def test_error_message(self):
        """Test error message formatting"""
        mock_exception = Mock()
        error = UnrealCrashedError(mock_exception)

        message = str(error)
        assert "stopped responding" in message
        assert "Unreal" in message

    def test_comes_from_cancelled(self):
        """Test detection of CANCELLED status"""
        mock_exception = MockGrpcError(grpc.StatusCode.CANCELLED)
        assert UnrealCrashedError.comes_from(mock_exception)

    def test_comes_from_unavailable_with_cancel_message(self):
        """Test detection of UNAVAILABLE with 'Cancelling all calls'"""
        mock_exception = MockGrpcError(
            grpc.StatusCode.UNAVAILABLE, "Cancelling all calls"
        )
        assert UnrealCrashedError.comes_from(mock_exception)

    def test_comes_from_unknown_stream_removed(self):
        """Test detection of UNKNOWN with 'Stream removed'"""
        mock_exception = MockGrpcError(grpc.StatusCode.UNKNOWN, "Stream removed")
        assert UnrealCrashedError.comes_from(mock_exception)

    def test_comes_from_rejects_other_errors(self):
        """Test rejection of unrelated errors"""
        mock_exception = MockGrpcError(grpc.StatusCode.UNAVAILABLE, "different error")
        assert not UnrealCrashedError.comes_from(mock_exception)


class TestMissingMethodError:
    """Tests for MissingMethodError"""

    def test_error_message(self):
        """Test error message formatting"""
        mock_exception = Mock()
        error = MissingMethodError(mock_exception)

        message = str(error)
        assert "endpoint" in message
        assert "unreal" in message.lower()

    def test_comes_from_unimplemented(self):
        """Test detection of UNIMPLEMENTED status"""
        mock_exception = MockGrpcError(grpc.StatusCode.UNIMPLEMENTED)
        assert MissingMethodError.comes_from(mock_exception)

    def test_comes_from_rejects_other_codes(self):
        """Test rejection of other status codes"""
        mock_exception = MockGrpcError(grpc.StatusCode.CANCELLED)
        assert not MissingMethodError.comes_from(mock_exception)


class TestEnvironmentException:
    """Tests for EnvironmentException"""

    def test_with_message(self):
        """Test exception with custom message"""
        message = "Custom environment error"
        error = EnvironmentException(message)

        assert str(error) == message

    def test_is_schola_exception(self):
        """Test that EnvironmentException is a ScholaException"""
        assert issubclass(EnvironmentException, ScholaException)


class TestNoAgentsException:
    """Tests for NoAgentsException"""

    def test_error_message_with_env_id(self):
        """Test error message includes environment ID"""
        env_id = 5
        error = NoAgentsException(env_id)

        message = str(error)
        assert str(env_id) in message
        assert "no agents" in message.lower()

    def test_is_schola_exception(self):
        """Test that NoAgentsException is a ScholaException"""
        assert issubclass(NoAgentsException, ScholaException)

    def test_stores_env_id(self):
        """Test that env_id is stored as attribute"""
        env_id = 10
        error = NoAgentsException(env_id)

        assert error.env_id == env_id


class TestNoEnvironmentsException:
    """Tests for NoEnvironmentsException"""

    def test_error_message(self):
        """Test error message formatting"""
        error = NoEnvironmentsException()

        message = str(error)
        assert "no Environment Definitions" in message
        assert "Connected to Unreal successfully" in message

    def test_is_schola_exception(self):
        """Test that NoEnvironmentsException is a ScholaException"""
        assert issubclass(NoEnvironmentsException, ScholaException)


class TestScholaErrorContextManager:
    """Tests for ScholaErrorContextManager"""

    def test_context_manager_protocol(self):
        """Test that it implements context manager protocol"""
        manager = ScholaErrorContextManager()

        # Should be usable as context manager
        with manager:
            pass

    @pytest.mark.parametrize(
        "grpc_error, expected_schola_error",
        [
            (
                MockGrpcError(
                    grpc.StatusCode.UNAVAILABLE, "failed to connect to all addresses"
                ),
                NoServerError,
            ),
            (MockGrpcError(grpc.StatusCode.CANCELLED), UnrealCrashedError),
            (MockGrpcError(grpc.StatusCode.UNIMPLEMENTED), MissingMethodError),
        ],
    )
    def test_converts_grpc_error_to_schola_error(
        self,
        grpc_error: MockGrpcError,
        expected_schola_error: type[WrappedGrpcException],
    ) -> None:
        with pytest.raises(expected_schola_error):
            with ScholaErrorContextManager():
                raise grpc_error

    def test_passes_through_non_grpc_exceptions(self):
        """Test that non-gRPC exceptions are not converted"""
        with pytest.raises(ValueError):
            with ScholaErrorContextManager():
                raise ValueError("test")

    def test_passes_through_unknown_grpc_errors(self):
        """Test that unknown gRPC errors are not converted"""
        with pytest.raises(grpc.RpcError):
            with ScholaErrorContextManager():
                raise MockGrpcError(grpc.StatusCode.INTERNAL, "some internal error")

    def test_no_exception_case(self):
        """Test context manager when no exception occurs"""
        with ScholaErrorContextManager():
            pass

    def test_decorator_usage(self):
        """Test that context manager works as decorator"""

        @ScholaErrorContextManager()
        def function_that_raises():
            # Create a real grpc.RpcError-like exception
            raise MockGrpcError(
                grpc.StatusCode.UNAVAILABLE, "failed to connect to all addresses"
            )

        with pytest.raises(NoServerError):
            function_that_raises()


class TestWrappedGrpcException:
    """Tests for WrappedGrpcException base class"""

    def test_is_abstract(self):
        """Test that WrappedGrpcException requires comes_from implementation"""
        # WrappedGrpcException uses abc.ABCMeta but comes_from is a classmethod
        # We can verify it's abstract by checking the metaclass
        assert hasattr(WrappedGrpcException, "__abstractmethods__")
        assert "comes_from" in WrappedGrpcException.__abstractmethods__

    def test_is_schola_exception(self):
        """Test that WrappedGrpcException is a ScholaException"""
        assert issubclass(WrappedGrpcException, ScholaException)

    @pytest.mark.parametrize(
        "wrapped_class", [NoServerError, UnrealCrashedError, MissingMethodError]
    )
    def test_all_wrapped_exceptions_implement_comes_from(
        self, wrapped_class: type[WrappedGrpcException]
    ) -> None:
        """Test that all wrapped exception classes implement comes_from"""
        assert hasattr(
            wrapped_class, "comes_from"
        ), f"{wrapped_class} does not implement comes_from"
        assert callable(
            wrapped_class.comes_from
        ), f"{wrapped_class} comes_from is not callable"
