"""Socket-based transport for PD-separated inference.

Replaces torch.distributed with pure TCP sockets for robust inter-node
communication. No simultaneous startup required.

Protocol:
    Each message: 4-byte big-endian length prefix + torch.save serialized data.
    Request:  {"op": "op_name", "payload": {...}}
    Response: {"ok": True/False, "result": ..., "error": "..."}
"""

from __future__ import annotations

import io
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, Optional

import torch


# ── Low-level send/recv ──


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a socket."""
    buf = bytearray(n)
    view = memoryview(buf)
    while n > 0:
        nrecv = sock.recv_into(view, n)
        if nrecv == 0:
            raise ConnectionError("Connection closed by peer")
        view = view[nrecv:]
        n -= nrecv
    return bytes(buf)


def send_obj(sock: socket.socket, obj: Any) -> None:
    """Send a Python object over a socket.

    Serializes with torch.save, sends length prefix + data.
    """
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


# Progress bar chunk size: 64 KB
_PROGRESS_CHUNK = 64 * 1024


def send_obj_progress(
    sock: socket.socket,
    obj: Any,
    on_progress: "Callable[[int, int], None] | None" = None,
) -> int:
    """Send a Python object with optional progress callback.

    Args:
        sock: Socket to send on
        obj: Object to serialize and send
        on_progress: callback(bytes_sent, total_bytes) called after each chunk

    Returns:
        Total bytes sent (including 4-byte header)
    """
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    total = len(data)
    header = struct.pack("!I", total)
    sock.sendall(header)

    sent = 0
    while sent < total:
        end = min(sent + _PROGRESS_CHUNK, total)
        sock.sendall(data[sent:end])
        sent = end
        if on_progress is not None:
            on_progress(sent, total)

    return total + 4


def recv_obj(sock: socket.socket) -> Any:
    """Receive a Python object from a socket.

    Reads length prefix, then deserializes with torch.load.
    """
    header = _recv_exact(sock, 4)
    length = struct.unpack("!I", header)[0]
    data = _recv_exact(sock, length)
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=False)


# ── Socket Server ──


class SocketServer:
    """TCP server that handles requests from a single client.

    Listens on a port, accepts one connection at a time, and dispatches
    requests to registered handler functions.

    Usage:
        handlers = {"run_decode": my_decode_fn}
        server = SocketServer("0.0.0.0", 29501, handlers)
        server.serve_forever()
    """

    def __init__(self, host: str, port: int, handlers: Dict[str, Callable]):
        self.host = host
        self.port = port
        self.handlers = handlers
        self._sock: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._running = True

    def serve_forever(self) -> None:
        """Listen for and handle requests indefinitely.

        Accepts one client connection at a time. After the client
        disconnects, continues listening for new connections.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(1)
        self._sock.settimeout(1.0)
        print(f"[socket_server] Listening on {self.host}:{self.port}")

        try:
            while self._running:
                try:
                    self._client, addr = self._sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                # Client inherits timeout from listening socket — set to blocking
                self._client.settimeout(None)
                print(f"[socket_server] Client connected from {addr}")
                try:
                    self._handle_client()
                except ConnectionError:
                    print("[socket_server] Client disconnected")
                except Exception as e:
                    print(f"[socket_server] Client error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    try:
                        self._client.close()
                    except Exception:
                        pass
                    self._client = None
        finally:
            self._cleanup()

    def _handle_client(self) -> None:
        """Process requests from the connected client."""
        while self._running:
            try:
                msg = recv_obj(self._client)
            except (ConnectionError, struct.error):
                break
            except Exception as e:
                print(f"[socket_server] Receive error: {e}")
                break

            if not isinstance(msg, dict):
                self._send_error("invalid_message")
                continue

            op = msg.get("op")
            payload = msg.get("payload") or {}

            if op == "shutdown":
                self._send_ok("bye")
                self._running = False
                break

            handler = self.handlers.get(op)
            if handler is None:
                self._send_error(f"unknown_op:{op}")
                continue

            try:
                result = handler(**payload)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_error(str(e))
                continue

            # ── ABKT streaming mode ──
            if isinstance(result, dict) and result.get("_abkt_stream") is True:
                self._handle_abkt_stream(result)
                break  # stream complete, connection closes

            self._send_ok(result)

    def _send_ok(self, result: Any) -> None:
        send_obj(self._client, {"ok": True, "result": result})

    def _send_error(self, error: str) -> None:
        send_obj(self._client, {"ok": False, "error": error})

    def _cleanup(self) -> None:
        """Close all sockets."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def _handle_abkt_stream(self, ctx: dict) -> None:
        """Receive KV chunks in streaming mode, then trigger decode when complete.

        ctx contains the chunk assembler, assembly event, and decode params.
        Called by _handle_client after run_decode_abkt returns the stream marker.
        """
        assembler = ctx.get("_assembler")
        assembly_done = ctx.get("_assembly_done")
        assembled_kv = ctx.get("_assembled_kv", {})
        decode_fn = ctx.get("_decode_fn")
        request_id = ctx.get("_request_id", "")

        if assembler is None or assembly_done is None or decode_fn is None:
            self._send_error("Invalid ABKT stream context")
            return

        print(f"[socket_server] ABKT stream mode: awaiting chunks for {request_id}")
        chunk_count = 0
        try:
            while self._running:
                try:
                    msg = recv_obj(self._client)
                except (ConnectionError, struct.error):
                    print(f"[socket_server] Connection lost after {chunk_count} chunks")
                    break
                except Exception as e:
                    print(f"[socket_server] Stream receive error: {e}")
                    break

                if not isinstance(msg, dict):
                    continue

                op = msg.get("op")
                payload = msg.get("payload") or {}

                if op == "kv_chunk":
                    chunk_req_id = payload.get("request_id", "")
                    if chunk_req_id == request_id:
                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            print(f"[socket_server] Received {chunk_count} chunks...")
                        assembler.add_chunk(
                            request_id=chunk_req_id,
                            layer_idx=payload["layer_idx"],
                            chunk_start=payload["chunk_start"],
                            chunk_end=payload["chunk_end"],
                            total_seq_len=payload["total_seq_len"],
                            k=payload["k"],
                            v=payload["v"],
                            meta=payload.get("meta", {}),
                            is_last_chunk=payload.get("is_last_chunk", False),
                        )

                elif op == "decode_start":
                    print(f"[socket_server] decode_start received, "
                          f"waiting for assembly ({chunk_count} chunks so far)...")
                    if assembly_done.wait(timeout=120.0):
                        dequantized = assembled_kv.get(request_id)
                        if dequantized is None:
                            self._send_error("Assembly failed: no data")
                            return
                        print(f"[socket_server] Assembly done, running decode...")
                        result = decode_fn(dequantized)
                        print(f"[socket_server] Decode complete, sending result")
                        self._send_ok(result)
                    else:
                        self._send_error(f"Assembly timeout ({chunk_count} chunks received)")
                    return
                else:
                    print(f"[socket_server] Unknown op in stream: {op}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_error(str(e))
        finally:
            try:
                assembler.cancel(request_id)
            except Exception:
                pass

    def shutdown(self) -> None:
        """Request server shutdown."""
        self._running = False


# ── Socket Client ──


class SocketClient:
    """TCP client that connects to a server and sends requests.

    Usage:
        client = SocketClient("192.168.0.20", 29501)
        result = client.call("run_decode", kv_cache=..., input_ids=...)
        client.close()
    """

    def __init__(self, host: str, port: int, timeout: float = 300.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        """Establish connection to the server."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        print(f"[socket_client] Connected to {self.host}:{self.port}")

    def call(self, op: str, **payload) -> Any:
        """Send a request and wait for the response.

        Args:
            op: Operation name
            **payload: Operation arguments

        Returns:
            The operation result

        Raises:
            RuntimeError: If the call fails or returns an error
        """
        if self._sock is None:
            raise RuntimeError("Not connected. Call connect() first.")

        msg = {"op": op, "payload": payload}
        send_obj(self._sock, msg)

        resp = recv_obj(self._sock)
        if not isinstance(resp, dict):
            raise RuntimeError(f"Invalid response type: {type(resp)}")

        if resp.get("ok"):
            return resp.get("result")

        error = resp.get("error", "Unknown error")
        raise RuntimeError(f"Remote error: {error}")

    def send_raw(self, obj: Any) -> None:
        """Send an object without waiting for response (streaming mode).

        For ChunkedSender where multiple messages are sent before
        receiving a single final response.
        """
        if self._sock is None:
            raise RuntimeError("Not connected. Call connect() first.")
        msg = obj if isinstance(obj, dict) and "op" in obj else {"op": "raw", "payload": obj}
        send_obj(self._sock, msg)

    def send_with_progress(
        self, op: str, on_progress: "Callable[[int, int], None] | None" = None, **payload
    ) -> int:
        """Send a request with progress tracking (no response wait).

        Args:
            op: Operation name
            on_progress: callback(bytes_sent, total_bytes) for progress display
            **payload: Operation arguments

        Returns:
            Total bytes sent
        """
        if self._sock is None:
            raise RuntimeError("Not connected. Call connect() first.")
        msg = {"op": op, "payload": payload}
        return send_obj_progress(self._sock, msg, on_progress=on_progress)

    def recv_obj(self) -> Any:
        """Receive a single response from the socket (after streaming sends)."""
        return recv_obj(self._sock)

    def close(self) -> None:
        """Close the connection."""
        if self._sock:
            try:
                send_obj(self._sock, {"op": "shutdown", "payload": {}})
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
