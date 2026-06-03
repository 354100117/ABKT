"""RPC communication for distributed PD inference.

Uses torch.distributed send/recv for reliable point-to-point messaging.

Architecture:
    - Rank 0 (prefill node) orchestrates the pipeline
    - Rank 1..N-1 (decode nodes) respond to RPC requests
    - Messages are serialized via torch.save/torch.load
"""

from __future__ import annotations

import io
import os
import pickle
import queue
import threading
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist


# ── Message tags ──
_TAG_REQUEST = 100
_TAG_RESPONSE = 101


# ── Serialization ──


def _serialize(obj: Any) -> bytes:
    """Serialize an object to bytes, preferring torch.save."""
    buf = io.BytesIO()
    try:
        torch.save(obj, buf)
        return buf.getvalue()
    except Exception:
        return pickle.dumps(obj)


def _deserialize(blob: bytes) -> Any:
    """Deserialize bytes to object."""
    buf = io.BytesIO(blob)
    try:
        try:
            return torch.load(buf, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(buf, map_location="cpu")
    except Exception:
        return pickle.loads(blob)


def _bytes_to_tensor(blob: bytes) -> torch.Tensor:
    """Convert bytes to a torch.ByteTensor for distributed send."""
    storage = torch.ByteStorage.from_buffer(blob)
    return torch.ByteTensor(storage)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a torch.ByteTensor back to bytes."""
    try:
        return tensor.cpu().numpy().tobytes()
    except Exception:
        return bytes(bytearray(tensor.tolist()))


# ── Low-level send/recv ──


def send_obj(obj: Any, dst: int, tag: int = _TAG_REQUEST) -> None:
    """Send a Python object to another rank."""
    blob = _serialize(obj)
    length = torch.tensor([len(blob)], dtype=torch.int64)
    dist.send(length, dst=dst, tag=tag)
    if blob:
        payload = _bytes_to_tensor(blob)
        dist.send(payload, dst=dst, tag=tag)


def recv_obj(src: int, tag: int = _TAG_REQUEST) -> Any:
    """Receive a Python object from another rank."""
    length = torch.empty(1, dtype=torch.int64)
    dist.recv(length, src=src, tag=tag)
    size = int(length.item())
    if size <= 0:
        return None
    payload = torch.empty(size, dtype=torch.uint8)
    dist.recv(payload, src=src, tag=tag)
    blob = _tensor_to_bytes(payload)
    return _deserialize(blob)


# ── High-level RPC ──


class _PendingCall:
    """A pending RPC call, waiting for a response."""

    def __init__(self):
        self._event = threading.Event()
        self.response: Any = None

    def set(self, resp: Any) -> None:
        self.response = resp
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> Any:
        self._event.wait(timeout=timeout)
        return self.response


class RPCClient:
    """RPC client that sends requests and waits for responses.

    Caller (rank 0) sends RPC requests to workers (ranks 1..N-1).
    """

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self._lock = threading.Lock()
        self._next_id = 1
        self._pending: Dict[int, _PendingCall] = {}
        self._running = True

        # Response receiver threads (one per worker)
        self._receivers = []
        for src in range(world_size):
            if src == rank:
                continue
            t = threading.Thread(target=self._recv_loop, args=(src,), daemon=True)
            t.start()
            self._receivers.append(t)

    def _alloc_id(self) -> int:
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            return req_id

    def _recv_loop(self, src: int) -> None:
        """Continuously receive responses from a specific rank."""
        while self._running:
            try:
                resp = recv_obj(src=src, tag=_TAG_RESPONSE)
                self._deliver_response(resp)
            except Exception:
                if self._running:
                    import traceback
                    traceback.print_exc()

    def _deliver_response(self, resp: Any) -> None:
        """Match a response to its pending call and deliver it."""
        if not isinstance(resp, dict):
            return
        req_id = resp.get("req_id")
        if req_id is None:
            return
        with self._lock:
            pending = self._pending.pop(int(req_id), None)
        if pending:
            pending.set(resp)

    def call(self, dst: int, op: str, payload: Dict, timeout: Optional[float] = None) -> Any:
        """Send an RPC call and wait for the response.

        Args:
            dst: Destination rank
            op: Operation name
            payload: Operation arguments as a dict
            timeout: Optional timeout in seconds

        Returns:
            The operation result

        Raises:
            RuntimeError: If the call fails or returns an error
        """
        req_id = self._alloc_id()
        pending = _PendingCall()
        with self._lock:
            self._pending[req_id] = pending

        msg = {"op": op, "payload": payload, "req_id": req_id}
        try:
            send_obj(msg, dst=dst, tag=_TAG_REQUEST)
        except Exception as e:
            with self._lock:
                self._pending.pop(req_id, None)
            raise RuntimeError(f"RPC send to rank {dst} failed: {e}")

        resp = pending.wait(timeout=timeout)
        if resp is None:
            with self._lock:
                self._pending.pop(req_id, None)
            raise RuntimeError(f"RPC call to rank {dst} timed out")

        if isinstance(resp, dict) and resp.get("ok"):
            return resp.get("result")
        error = None
        if isinstance(resp, dict):
            error = resp.get("error")
        raise RuntimeError(error or f"RPC error from rank {dst}")

    def shutdown(self):
        """Stop all receiver threads."""
        self._running = False


class RPCServer:
    """RPC server that handles requests from a caller.

    Worker (rank != 0) runs this server to respond to RPC requests.
    """

    def __init__(self, handlers: Dict[str, Callable], caller_rank: int = 0):
        """
        Args:
            handlers: Dict mapping operation names to handler functions
            caller_rank: Rank of the caller (default: 0)
        """
        self.handlers = handlers
        self.caller_rank = caller_rank

    def serve_forever(self) -> None:
        """Listen for requests and handle them until shutdown."""
        while True:
            try:
                msg = recv_obj(src=self.caller_rank, tag=_TAG_REQUEST)
                if not isinstance(msg, dict):
                    self._send_error("invalid_message", None, self.caller_rank)
                    continue

                op = msg.get("op")
                payload = msg.get("payload") or {}
                req_id = msg.get("req_id")

                if op == "shutdown":
                    self._send_ok("bye", req_id, self.caller_rank)
                    break

                handler = self.handlers.get(op)
                if handler is None:
                    self._send_error(f"unknown_op:{op}", req_id, self.caller_rank)
                    continue

                try:
                    result = handler(**payload)
                    self._send_ok(result, req_id, self.caller_rank)
                except Exception as e:
                    self._send_error(str(e), req_id, self.caller_rank)

            except Exception:
                import traceback
                traceback.print_exc()
                if not self._check_alive():
                    break

    def _send_ok(self, result: Any, req_id: Optional[int], dst: int) -> None:
        send_obj({"ok": True, "result": result, "req_id": req_id}, dst=dst, tag=_TAG_RESPONSE)

    def _send_error(self, error: str, req_id: Optional[int], dst: int) -> None:
        send_obj({"ok": False, "error": error, "req_id": req_id}, dst=dst, tag=_TAG_RESPONSE)

    def _check_alive(self) -> bool:
        """Check if caller is still alive (best-effort)."""
        try:
            dist.barrier(timeout=torch.timedelta(seconds=1))
            return True
        except Exception:
            return False


# ── Utility: distributed init ──


def init_distributed(config) -> None:
    """Initialize torch.distributed process group.

    Args:
        config: PDConfig with master_addr, master_port, rank, world_size

    Sets:
        MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE env vars if missing.
        Then calls dist.init_process_group(backend="gloo").
    """
    os.environ.setdefault("MASTER_ADDR", config.master_addr)
    os.environ.setdefault("MASTER_PORT", str(config.master_port))
    os.environ.setdefault("RANK", str(config.rank))
    os.environ.setdefault("WORLD_SIZE", str(config.world_size))

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://")
        print(f"[dist] Initialized: rank={config.rank}/{config.world_size}, "
              f"master={config.master_addr}:{config.master_port}")


def destroy_distributed() -> None:
    """Destroy the distributed process group."""
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass
