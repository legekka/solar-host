"""Process manager for solar-host with multi-backend support."""

import subprocess
import socket
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import deque
import asyncio
import threading

from app.models import (
    Instance,
    InstanceStatus,
    LogMessage,
    InstanceRuntimeState,
    InstanceStateEvent,
    GenerationMetrics,
    BackendType,
)
from app.config import settings, config_manager, parse_instance_config
from app.backends.base import BackendRunner
from app.backends.llamacpp import LlamaCppRunner
from app.backends.huggingface import HuggingFaceRunner


def get_runner_for_config(config) -> BackendRunner:
    """Get the appropriate backend runner for a config type."""
    backend_type = getattr(config, "backend_type", "llamacpp")

    if backend_type == BackendType.LLAMACPP or backend_type == "llamacpp":
        return LlamaCppRunner()
    elif backend_type in (
        BackendType.HUGGINGFACE_CAUSAL,
        BackendType.HUGGINGFACE_CLASSIFICATION,
        "huggingface_causal",
        "huggingface_classification",
    ):
        return HuggingFaceRunner()
    else:
        # Default to llama.cpp for backward compatibility
        return LlamaCppRunner()


class ProcessManager:
    """Manages model server processes across multiple backends."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_buffers: Dict[str, deque] = {}
        self.log_sequences: Dict[str, int] = {}
        self.log_threads: Dict[str, threading.Thread] = {}
        self.log_dir = Path(settings.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Runtime state streaming (ephemeral)
        self.state_buffers: Dict[str, deque] = {}
        self.state_sequences: Dict[str, int] = {}

        # Per-instance parsing context (managed by backend runners)
        self.instance_contexts: Dict[str, Dict[str, Any]] = {}

        # Per-instance runner reference
        self.instance_runners: Dict[str, BackendRunner] = {}

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available (not bound by any process)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False

    def _get_assigned_ports(self, exclude_instance_id: Optional[str] = None) -> set:
        """Get all ports currently assigned to instances.
        
        Args:
            exclude_instance_id: Optional instance ID to exclude from the check
                                 (used when reassigning port for that instance)
        """
        assigned = set()
        for instance in config_manager.get_all_instances():
            if instance.port is not None and instance.id != exclude_instance_id:
                assigned.add(instance.port)
        return assigned

    def _get_available_port(self, exclude_instance_id: Optional[str] = None) -> int:
        """Get the lowest available port starting from settings.start_port.
        
        Finds the first port (starting from start_port) that is:
        1. Not assigned to another running/stopped instance
        2. Not currently bound by any process
        
        Args:
            exclude_instance_id: Instance ID to exclude from assigned port check
                                 (allows reusing the same port if still free)
        """
        assigned_ports = self._get_assigned_ports(exclude_instance_id)
        port = settings.start_port
        
        while port in assigned_ports or not self._is_port_available(port):
            port += 1
        
        return port

    def _read_logs(
        self,
        instance_id: str,
        process: subprocess.Popen,
        log_file: Path,
        runner: BackendRunner,
    ):
        """Read logs from process and store in buffer."""
        try:
            if not process.stdout:
                return

            with open(log_file, "a") as f:
                for line in iter(process.stdout.readline, b""):
                    if not line:
                        break

                    decoded_line = line.decode("utf-8", errors="replace").rstrip()

                    # Write to file
                    f.write(decoded_line + "\n")
                    f.flush()

                    # Store in buffer
                    if instance_id not in self.log_buffers:
                        self.log_buffers[instance_id] = deque(
                            maxlen=settings.log_buffer_size
                        )
                        self.log_sequences[instance_id] = 0

                    seq = self.log_sequences[instance_id]
                    self.log_sequences[instance_id] += 1

                    log_msg = LogMessage(
                        seq=seq, timestamp=datetime.now().isoformat(), line=decoded_line
                    )
                    self.log_buffers[instance_id].append(log_msg)

                    # Parse log line using backend runner
                    try:
                        context = self.instance_contexts.get(instance_id, {})
                        state_update = runner.parse_log_line(
                            instance_id, decoded_line, context
                        )
                        if state_update:
                            self._emit_state_event(instance_id, state_update)
                    except Exception:
                        # Parsing errors should not break logging
                        pass
        except Exception as e:
            print(f"Error reading logs for {instance_id}: {e}")

    def _emit_state_event(self, instance_id: str, update):
        """Emit a state event from a RuntimeStateUpdate."""
        # Update in-memory instance runtime fields
        config_manager.update_instance_runtime(
            instance_id,
            busy=update.busy,
            prefill_progress=update.prefill_progress,
            active_slots=update.active_slots,
        )

        # Initialize state buffer/seq lazily
        if instance_id not in self.state_buffers:
            self.state_buffers[instance_id] = deque(maxlen=settings.log_buffer_size)
            self.state_sequences[instance_id] = 0

        seq = self.state_sequences[instance_id]
        self.state_sequences[instance_id] += 1

        now_ts = datetime.now(timezone.utc).isoformat()
        state = InstanceRuntimeState(
            instance_id=instance_id,
            busy=update.busy,
            phase=update.phase,
            prefill_progress=update.prefill_progress,
            active_slots=update.active_slots,
            slot_id=update.slot_id,
            task_id=update.task_id,
            prefill_prompt_tokens=update.prefill_prompt_tokens,
            generated_tokens=update.generated_tokens,
            decode_tps=update.decode_tps,
            decode_ms_per_token=update.decode_ms_per_token,
            checkpoint_index=update.checkpoint_index,
            checkpoint_total=update.checkpoint_total,
            timestamp=now_ts,
        )
        event = InstanceStateEvent(
            seq=seq,
            timestamp=now_ts,
            data=state,
        )
        self.state_buffers[instance_id].append(event)

    def get_last_generation(self, instance_id: str) -> Optional[GenerationMetrics]:
        """Get the last generation metrics for an instance."""
        runner = self.instance_runners.get(instance_id)
        context = self.instance_contexts.get(instance_id, {})

        if runner and hasattr(runner, "get_last_generation"):
            return runner.get_last_generation(context)
        return None

    async def start_instance(self, instance_id: str) -> bool:
        """Start a model server instance."""
        instance = config_manager.get_instance(instance_id)
        if not instance:
            return False

        # Check if already running
        if instance.status == InstanceStatus.RUNNING:
            return True

        # Always find an available port on start
        # (previous port may now be in use by another process)
        # Pass instance_id to allow reusing this instance's previous port if still free
        instance.port = self._get_available_port(exclude_instance_id=instance_id)

        # Get the appropriate runner for this backend type
        runner = get_runner_for_config(instance.config)
        self.instance_runners[instance_id] = runner

        # Initialize parsing context
        self.instance_contexts[instance_id] = runner.initialize_context()

        # Update status
        instance.status = InstanceStatus.STARTING
        instance.error_message = None

        # Set supported endpoints
        backend_type = getattr(instance.config, "backend_type", "llamacpp")
        if hasattr(runner, "get_supported_endpoints_for_type"):
            instance.supported_endpoints = runner.get_supported_endpoints_for_type(
                backend_type
            )
        else:
            instance.supported_endpoints = runner.get_supported_endpoints()

        config_manager.update_instance(instance_id, instance)

        try:
            # Build command using runner
            cmd = runner.build_command(instance)

            # Create log file
            alias_safe = instance.config.alias.replace(":", "-").replace("/", "-")
            log_file = self.log_dir / f"{alias_safe}_{int(time.time())}.log"

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,  # Unbuffered for real-time output
            )

            self.processes[instance_id] = process

            # Start log reading thread
            log_thread = threading.Thread(
                target=self._read_logs,
                args=(instance_id, process, log_file, runner),
                daemon=True,
            )
            log_thread.start()
            self.log_threads[instance_id] = log_thread

            # Wait a bit and check if process is still running
            await asyncio.sleep(2)

            if process.poll() is None:
                # Process is running
                instance.status = InstanceStatus.RUNNING
                instance.pid = process.pid
                instance.started_at = datetime.now(timezone.utc)
                instance.retry_count = 0
                config_manager.update_instance(instance_id, instance)

                # Initialize ephemeral runtime state
                self.state_buffers[instance_id] = deque(maxlen=settings.log_buffer_size)
                self.state_sequences[instance_id] = 0
                config_manager.update_instance_runtime(
                    instance_id, busy=False, prefill_progress=None, active_slots=0
                )

                # Notify runner that process started
                runner.on_process_started(
                    instance_id, self.instance_contexts[instance_id]
                )

                return True
            else:
                # Process failed
                instance.status = InstanceStatus.FAILED
                instance.error_message = "Process exited immediately"
                instance.retry_count += 1
                config_manager.update_instance(instance_id, instance)

                # Retry if under limit
                if instance.retry_count < settings.max_retries:
                    await asyncio.sleep(1)
                    return await self.start_instance(instance_id)

                return False

        except Exception as e:
            instance.status = InstanceStatus.FAILED
            instance.error_message = str(e)
            instance.retry_count += 1
            config_manager.update_instance(instance_id, instance)

            # Retry if under limit
            if instance.retry_count < settings.max_retries:
                await asyncio.sleep(1)
                return await self.start_instance(instance_id)

            return False

    async def stop_instance(self, instance_id: str) -> bool:
        """Stop a model server instance."""
        instance = config_manager.get_instance(instance_id)
        if not instance:
            return False

        if instance.status == InstanceStatus.STOPPED:
            return True

        instance.status = InstanceStatus.STOPPING
        config_manager.update_instance(instance_id, instance)

        try:
            if instance_id in self.processes:
                process = self.processes[instance_id]
                process.terminate()

                # Wait for process to terminate
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

                del self.processes[instance_id]

            # Notify runner that process stopped
            runner = self.instance_runners.get(instance_id)
            if runner:
                context = self.instance_contexts.get(instance_id, {})
                runner.on_process_stopped(instance_id, context)

            # Cleanup
            if instance_id in self.instance_runners:
                del self.instance_runners[instance_id]
            if instance_id in self.instance_contexts:
                del self.instance_contexts[instance_id]

            instance.status = InstanceStatus.STOPPED
            instance.pid = None
            instance.started_at = None
            config_manager.update_instance(instance_id, instance)

            # Clean up old log file for stopped instances
            await self._cleanup_old_logs(instance.config.alias)

            return True

        except Exception as e:
            instance.status = InstanceStatus.FAILED
            instance.error_message = f"Failed to stop: {str(e)}"
            config_manager.update_instance(instance_id, instance)
            return False

    async def _cleanup_old_logs(self, alias: str):
        """Clean up old log files for stopped instances."""
        try:
            alias_safe = alias.replace(":", "-").replace("/", "-")
            pattern = f"{alias_safe}_*.log"
            for log_file in self.log_dir.glob(pattern):
                # Keep only the most recent log
                if log_file.stat().st_mtime < time.time() - 300:  # 5 minutes old
                    log_file.unlink()
        except Exception as e:
            print(f"Error cleaning up logs: {e}")

    async def restart_instance(self, instance_id: str) -> bool:
        """Restart a model server instance."""
        await self.stop_instance(instance_id)
        await asyncio.sleep(1)
        return await self.start_instance(instance_id)

    def create_instance(self, config) -> Instance:
        """Create a new instance."""
        # Parse config if it's a dict (from FastAPI request body)
        if isinstance(config, dict):
            config = parse_instance_config(config)
        
        instance_id = str(uuid.uuid4())

        # Determine supported endpoints based on backend type
        runner = get_runner_for_config(config)
        backend_type = getattr(config, "backend_type", "llamacpp")

        if hasattr(runner, "get_supported_endpoints_for_type"):
            supported_endpoints = runner.get_supported_endpoints_for_type(backend_type)
        else:
            supported_endpoints = runner.get_supported_endpoints()

        instance = Instance(
            id=instance_id,
            config=config,
            status=InstanceStatus.STOPPED,
            supported_endpoints=supported_endpoints,
        )
        config_manager.add_instance(instance)
        return instance

    def get_log_buffer(self, instance_id: str) -> List[LogMessage]:
        """Get log buffer for an instance."""
        if instance_id in self.log_buffers:
            return list(self.log_buffers[instance_id])
        return []

    def get_next_sequence(self, instance_id: str) -> int:
        """Get next sequence number for an instance."""
        return self.log_sequences.get(instance_id, 0)

    def get_state_buffer(self, instance_id: str) -> List[InstanceStateEvent]:
        """Get state buffer for an instance."""
        if instance_id in self.state_buffers:
            return list(self.state_buffers[instance_id])
        return []

    def get_state_next_sequence(self, instance_id: str) -> int:
        """Get next state sequence number for an instance."""
        return self.state_sequences.get(instance_id, 0)

    async def auto_restart_running_instances(self):
        """Auto-restart instances that were running before shutdown."""
        for instance in config_manager.get_running_instances():
            print(f"Auto-restarting instance: {instance.id} ({instance.config.alias})")
            # Reset status first
            instance.status = InstanceStatus.STOPPED
            instance.pid = None
            config_manager.update_instance(instance.id, instance)
            # Start the instance
            await self.start_instance(instance.id)


# Global process manager instance
process_manager = ProcessManager()
