# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Base Class for Unreal Connections
"""

from typing import Any, Dict, List, Optional, Tuple
import socket
from .base_protocol import BaseProtocol, BaseProtocolMixin


class SocketProtocolMixin(BaseProtocolMixin):
    """
    TCP listen socket mixin for advertising the Python host to Unreal.

    Notes
    -----
    Used together with gRPC protocols so Unreal can open the reverse connection.
    """

    def __init__(self, url: str, port: Optional[int] = None, client_only: bool = False):
        self.url = url
        self.port = 0 if port is None else port
        self.tcp_socket = None
        self._client_only = client_only
        self._started = False

    def on_close(self) -> None:
        """
        Close the Unreal Connection. Method must be safe to call multiple times.
        """
        self._started = False
        if self.tcp_socket is not None:
            self.tcp_socket.close()

    def on_start(self) -> None:
        """
        Bind the tcp_socket to discover a free port (local mode only).

        When ``client_only`` is True (e.g. insecure/remote channels),
        no TCP socket is created — the protocol operates in client-only mode
        and ``port`` must already be set to a non-zero value.
        """
        if self._client_only:
            if self.port == 0:
                raise ValueError(
                    "An explicit port is required when using a remote/insecure "
                    "connection (client_only=True).  Set --port on the CLI "
                    "or the 'port' field in GrpcProtocolConfig."
                )
            self._started = True
            return

        if not self.has_socket:
            if socket.has_ipv6:
                self.tcp_socket = socket.socket(socket.AF_INET6)
            else:
                self.tcp_socket = socket.socket(socket.AF_INET)
            if self.port == 0:
                self.tcp_socket.bind((self.url, 0))
                self.port = self.tcp_socket.getsockname()[1]

        self._started = True

    @property
    def address(self) -> str:
        """
        Returns the address of the connection

        Returns
        -------
        str
            The address of the connection
        """
        return self.url + ":" + str(self.port)

    @property
    def has_socket(self) -> bool:
        """
        Returns whether the connection is active or not

        Returns
        -------
        bool
            Whether the connection is active or not
        """
        return self.tcp_socket is not None and self.tcp_socket.fileno() != -1

    @property
    def is_started(self) -> bool:
        """Whether ``on_start`` has been called (works for both socket and client-only modes)."""
        return self._started

    @property
    def mixin_properties(self) -> Dict[str, Any]:
        return {"Port": self.port}
