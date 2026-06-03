"""Network probing, EWMA estimation, and calibration for ABKT.

Architecture:
  ProbeServer (decode node, port 9877)
      ← TCP — 64-byte RTT echo probes
      ← TCP — N-byte bandwidth probes (PROBE_BW)
  NetworkProbeClient (prefill node)
      → runs background thread for periodic probing
      → manages EWMA for RTT + bandwidth
      → calibration from real KV Cache transfer timing

Key design decisions (from research phase):
  - RTT measured over persistent connection (avoids TCP handshake pollution)
  - Bandwidth estimated primarily from calibration (real transfers) not small probes
  - EWMA update_clamped with ±20% cap prevents feedback-loop oscillation
  - Calibration computed from uncompressed-equivalent bandwidth
"""

from __future__ import annotations

import contextlib
import logging
import struct
import threading
import time
from collections import deque
from typing import Optional

import torch

from backend.ewma import EWMA
from backend.state_machine import NetworkState, NetworkStateMachine

logger = logging.getLogger(__name__)

# ── Probe protocol constants ──
PROBE_RTT = 0x01
PROBE_BW = 0x02

# ── Defaults (tuned for 1 GbE x86↔Jetson) ──
DEFAULT_PROBE_PORT = 9877
RTT_INTERVAL_SEC = 1.0
BW_ALPHA = 0.3            # bandwidth EWMA — calibration data is ground truth
RTT_ALPHA = 0.5           # RTT EWMA — faster response to congestion
CALIBRATION_ALPHA = 0.5   # calibration sample weight (higher = trust calibration more)
EWMA_MAX_CHANGE = 0.20    # ±20% cap per update
WARMUP_BYTES = 100 * 1024  # 100 KB TCP warmup payload
MIN_CALIB_BYTES = 1 * 1024 * 1024  # 1 MB — skip calibration for smaller transfers
MIN_CALIB_ELAPSED = 0.100  # 100ms — skip calibration if too fast
COLD_BW = 15e6            # conservative cold-start default: 15 MB/s (120 Mbps)

# ── Bandwidth estimation: sliding window + transfer EWMA ──
BW_WINDOW_SIZE = 10           # sliding window: last 10 probe/transfer samples
TRANSFER_BW_ALPHA = 0.5       # transfer EWMA — trust actual transfers more
DIVERGENCE_THRESHOLD = 2.5    # probe/transfer ratio to flag mismatch
COLD_START_MARGIN = 0.7       # use 70% of estimate when uncalibrated (was 50%)
MIN_SAFETY_MARGIN = 0.8       # floor: never use more than 80% of estimated bw (was 60%)
CONFIDENCE_TRANSFERS = 5      # full confidence after this many transfers
TARGET_TRANSFER_TIME = 2.0    # target transfer time in seconds


# ════════════════════════════════════════════════════════════════════
# ProbeServer — runs on decode node
# ════════════════════════════════════════════════════════════════════


class ProbeServer:
    """TCP probe server, runs as daemon thread on the decode node.

    Handles RTT echo probes and bandwidth probe transfers.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PROBE_PORT):
        self.host = host
        self.port = port
        self._sock: Optional[torch.socket.socket] = None  # noqa
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True, name="probe-server")
        self._thread.start()
        logger.info("ProbeServer started on %s:%d", self.host, self.port)

    def stop(self) -> None:
        self._running = False
        if self._sock:
            with contextlib.suppress(Exception):
                self._sock.close()

    def _serve(self) -> None:
        import socket as _socket
        self._sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        self._sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(8)
        self._sock.settimeout(1.0)

        while self._running:
            try:
                conn, addr = self._sock.accept()
                t = threading.Thread(target=self._handle, args=(conn,), daemon=True)
                t.start()
            except _socket.timeout:
                continue
            except Exception:
                if self._running:
                    logger.exception("ProbeServer accept error")

    def _handle(self, conn) -> None:
        import socket as _socket
        conn.settimeout(10.0)
        try:
            while self._running:
                header = conn.recv(1)
                if not header:
                    break
                ptype = header[0]

                if ptype == PROBE_RTT:
                    conn.sendall(bytes([PROBE_RTT]))
                elif ptype == PROBE_BW:
                    raw = conn.recv(4)
                    if len(raw) < 4:
                        break
                    data_len = struct.unpack("!I", raw)[0]
                    received = 0
                    while received < data_len:
                        chunk = conn.recv(min(data_len - received, 65536))
                        if not chunk:
                            break
                        received += len(chunk)
                    conn.sendall(bytes([PROBE_BW]))
        except _socket.timeout:
            pass
        except ConnectionResetError:
            pass
        except Exception:
            pass
        finally:
            conn.close()


# ════════════════════════════════════════════════════════════════════
# NetworkProbeClient — runs on prefill node
# ════════════════════════════════════════════════════════════════════


class NetworkProbeClient:
    """Network probe client that manages EWMA, state machine, and calibration.

    Usage:
        probe = NetworkProbeClient(target_host="192.168.0.20")
        probe.start()

        # In prefill pipeline:
        snapshot = probe.get_snapshot()
        budget = snapshot.budget_bytes

        # After transfer completes:
        probe.record_transfer(bytes_sent, elapsed_sec, compression_ratio)
    """

    def __init__(
        self,
        target_host: str,
        target_port: int = DEFAULT_PROBE_PORT,
        rtt_interval: float = RTT_INTERVAL_SEC,
    ):
        self.target_host = target_host
        self.target_port = target_port
        self.rtt_interval = rtt_interval

        # Sliding window for bandwidth (replaces EWMA for estimation)
        self._bw_window = deque(maxlen=BW_WINDOW_SIZE)   # recent BW samples
        # Transfer EWMA (ground truth from real transfers)
        self._bw_transfer_ewma = EWMA(alpha=TRANSFER_BW_ALPHA)
        self._rtt_ewma = EWMA(alpha=RTT_ALPHA)

        # State machine
        self._state_machine = NetworkStateMachine()

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Persistent TCP connection for RTT probes
        self._rtt_conn: Optional[object] = None

        # Calibration state
        self._calibrated = False
        self._bw_probe_counter = 0
        self._transfer_count = 0

        # Bandwidth probe pause (paused during KV cache transfer)
        self._bw_probes_paused = False

    # ── Lifecycle ──

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._probe_loop, daemon=True, name="probe-client")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._close_rtt_conn()
        self._rtt_conn = None

    # ── Public API ──

    def get_effective_bw(self) -> float:
        """Conservative bandwidth estimate using sliding window minimum.

        Uses the minimum of the sliding window (recent probes + transfers)
        with a confidence-scaled safety margin. The window minimum responds
        immediately to bandwidth drops while being naturally conservative
        on recovery — exactly the behavior we want for budget planning.
        """
        window_bw = min(self._bw_window) if self._bw_window else None
        transfer_bw = self._bw_transfer_ewma.value

        # Cold start: no data yet
        if window_bw is None and transfer_bw is None:
            return COLD_BW * COLD_START_MARGIN

        # Only probe data, no transfer calibration
        if transfer_bw is None:
            return (window_bw or COLD_BW) * COLD_START_MARGIN

        # Post-calibration: use window min as primary, cross-check with transfer
        if window_bw is not None:
            ratio = window_bw / max(transfer_bw, 1.0)
            if ratio > DIVERGENCE_THRESHOLD:
                # Window too optimistic — trust transfer only
                base = transfer_bw
            else:
                base = min(window_bw, transfer_bw)
        else:
            base = transfer_bw

        # Confidence-scaled safety margin: more data → less margin
        confidence = min(self._transfer_count / CONFIDENCE_TRANSFERS, 1.0)
        margin = COLD_START_MARGIN + confidence * (MIN_SAFETY_MARGIN - COLD_START_MARGIN)
        return base * margin

    def _get_bw_for_state_machine(self) -> float:
        """Smoothed bandwidth for state machine (avoids oscillation)."""
        if self._bw_window:
            # Use median of window for state machine (smoother than min)
            sorted_bw = sorted(self._bw_window)
            return sorted_bw[len(sorted_bw) // 2]
        return COLD_BW

    def get_snapshot(self, total_bytes: float = 0.0) -> "NetworkSnapshot":
        """Thread-safe snapshot of current network conditions."""
        with self._lock:
            bw = self.get_effective_bw()
            rtt = self._rtt_ewma.value if self._rtt_ewma.valid else 0.0
            state = self._state_machine.state
            # Budget: target transfer time × effective bandwidth
            budget = bw * TARGET_TRANSFER_TIME
            if total_bytes > 0:
                budget = min(budget, total_bytes)  # never exceed original size
        return NetworkSnapshot(
            timestamp=time.time(),
            bandwidth_bps=bw,
            bandwidth_ewma=bw,
            rtt_ms=rtt,
            rtt_ewma=rtt,
            state=state,
            budget_bytes=budget,
        )

    def calibrate_with_transfer(self, num_bytes: int, elapsed_sec: float) -> None:
        """Cold-start calibration from first real transfer."""
        if elapsed_sec <= 0 or num_bytes < MIN_CALIB_BYTES:
            return
        measured_bw = num_bytes / elapsed_sec
        with self._lock:
            self._bw_window.append(measured_bw)
            self._calibrated = True

    def record_transfer(self, compressed_bytes: int, elapsed_sec: float,
                        compression_ratio: float) -> None:
        """Record a real KV Cache transfer completion for calibration.

        Uses uncompressed-equivalent bandwidth to prevent feedback loop:
            measured_bw = compressed_bytes * compression_ratio / elapsed_sec
        This represents what the bandwidth *would be* for uncompressed data.

        Feeds both the sliding window (for estimation) and transfer EWMA.
        """
        if elapsed_sec <= MIN_CALIB_ELAPSED or compressed_bytes < MIN_CALIB_BYTES:
            return
        measured_bw = compressed_bytes * compression_ratio / elapsed_sec
        with self._lock:
            self._bw_window.append(measured_bw)
            self._bw_transfer_ewma.update(measured_bw)
            self._transfer_count += 1
            self._calibrated = True

    def warmup_connection(self) -> None:
        """Send a small warmup payload to open TCP congestion window.

        Call once before the first real transfer on a new connection.
        """
        import socket as _socket
        try:
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.target_host, self.target_port))
            data = b'\x00' * WARMUP_BYTES
            sock.sendall(bytes([PROBE_BW]) + struct.pack("!I", WARMUP_BYTES) + data)
            sock.recv(1)
            sock.close()
        except Exception:
            pass  # warmup is best-effort

    def pause_bw_probes(self) -> None:
        """Pause bandwidth probes during KV cache transfer (avoids contention)."""
        self._bw_probes_paused = True

    def resume_bw_probes(self) -> None:
        """Resume bandwidth probes after transfer."""
        self._bw_probes_paused = False

    def probe_now(self, data_size: int = 1 * 1024 * 1024) -> None:
        """Force an immediate bandwidth probe (used for cold-start calibration)."""
        self._probe_bandwidth(data_size=data_size)

    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def state_machine(self) -> NetworkStateMachine:
        return self._state_machine

    # ── Internal: probe loop ──

    def _probe_loop(self) -> None:
        rtt_count = 0
        while self._running:
            try:
                state = self._state_machine.state
                bw_interval = self._state_machine.probe_interval_bw
                bw_data_size = self._state_machine.probe_bw_data_size
                probes_per_bw = max(1, int(bw_interval / self.rtt_interval))

                if rtt_count % probes_per_bw == 0 and not self._bw_probes_paused:
                    self._probe_bandwidth(data_size=bw_data_size)
                else:
                    self._probe_rtt()

                rtt_count += 1
            except Exception:
                logger.exception("Probe error")
            time.sleep(self.rtt_interval)

    def _probe_rtt(self) -> None:
        """RTT probe over persistent connection."""
        import socket as _socket
        conn = self._get_rtt_conn()
        if conn is None:
            return
        try:
            t0 = time.time()
            conn.sendall(bytes([PROBE_RTT]))
            conn.recv(1)
            rtt = (time.time() - t0) * 1000  # ms
            with self._lock:
                self._rtt_ewma.update(rtt)
                self._state_machine.update(
                    self._get_bw_for_state_machine(),
                    self._rtt_ewma.value or rtt,
                )
        except Exception:
            self._close_rtt_conn()
            self._rtt_conn = None

    def _probe_bandwidth(self, data_size: int = 1 * 1024 * 1024) -> None:
        """Bandwidth probe — actual data transfer measurement."""
        import socket as _socket
        try:
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.target_host, self.target_port))

            data = b'\x00' * data_size
            t0 = time.time()
            sock.sendall(bytes([PROBE_BW]) + struct.pack("!I", data_size) + data)
            sock.recv(1)
            elapsed = time.time() - t0

            if elapsed > 0:
                bw_bps = data_size / elapsed
                with self._lock:
                    self._bw_window.append(bw_bps)
                    self._state_machine.update(
                        self._get_bw_for_state_machine(),
                        self._rtt_ewma.value or 0.0,
                    )
            sock.close()
        except Exception:
            pass

    def _get_rtt_conn(self):
        """Get or create persistent TCP connection for RTT probes."""
        if self._rtt_conn is not None:
            return self._rtt_conn
        import socket as _socket
        try:
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.target_host, self.target_port))
            self._rtt_conn = sock
            return sock
        except Exception:
            return None

    def _close_rtt_conn(self) -> None:
        if self._rtt_conn is not None:
            with contextlib.suppress(Exception):
                self._rtt_conn.close()


# ════════════════════════════════════════════════════════════════════
# NetworkSnapshot — value object consumed by PrecisionAllocator
# ════════════════════════════════════════════════════════════════════


class NetworkSnapshot:
    """Immutable snapshot of network conditions at a point in time."""

    def __init__(self, timestamp: float, bandwidth_bps: float, bandwidth_ewma: float,
                 rtt_ms: float, rtt_ewma: float, state: NetworkState,
                 budget_bytes: float):
        self.timestamp = timestamp
        self.bandwidth_bps = bandwidth_bps
        self.bandwidth_ewma = bandwidth_ewma
        self.rtt_ms = rtt_ms
        self.rtt_ewma = rtt_ewma
        self.state = state
        self.budget_bytes = budget_bytes

    def __repr__(self) -> str:
        return (f"NetworkSnapshot(state={self.state.value}, "
                f"bw_ewma={self.bandwidth_ewma / 1e6:.1f} MB/s, "
                f"rtt_ewma={self.rtt_ewma:.1f} ms, "
                f"budget={self.budget_bytes / 1e6:.1f} MB)")
