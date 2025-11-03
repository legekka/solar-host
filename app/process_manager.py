import subprocess
import socket
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List
from collections import deque
import asyncio
import threading
import re

from app.models import (
    Instance,
    InstanceConfig,
    InstanceStatus,
    LogMessage,
    InstanceRuntimeState,
    InstanceStateEvent,
)
from app.config import settings, config_manager


class ProcessManager:
    """Manages llama-server processes"""
    
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
        self.active_slot_ids: Dict[str, set] = {}
        self.last_runtime: Dict[str, Dict[str, object]] = {}

        # Compile regex patterns for parsing llama-server logs
        self._re_launch = re.compile(r"slot\s+launch_slot_:\s+id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|\s*processing task")
        self._re_progress = re.compile(r"prompt processing progress.*progress\s*=\s*([0-9.]+)")
        self._re_prompt_done = re.compile(r"\|\s*prompt done\b")
        self._re_release = re.compile(r"slot\s+release:\s+id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|\s*stop processing")
        self._re_all_idle = re.compile(r"srv\s+update_slots:\s+all slots are idle")
        
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return True
            except OSError:
                return False
    
    def _get_available_port(self) -> int:
        """Get an available port starting from settings.start_port"""
        port = settings.start_port
        while not self._is_port_available(port):
            port += 1
        return port
    
    def _build_command(self, instance: Instance) -> List[str]:
        """Build llama-server command from instance config"""
        config = instance.config
        cmd = [
            "llama-server",
            "--model", config.model,
            "--alias", config.alias,
            "--threads", str(config.threads),
            "--n_gpu_layers", str(config.n_gpu_layers),
            "--temp", str(config.temp),
            "--top_p", str(config.top_p),
            "--top_k", str(config.top_k),
            "--min_p", str(config.min_p),
            "--ctx-size", str(config.ctx_size),
            "--host", config.host,
            "--port", str(instance.port),
            "--api-key", config.api_key,
            "--no-warmup"
        ]
        
        if config.chat_template_file:
            cmd.extend(["--jinja", "--chat-template-file", config.chat_template_file])
        
        return cmd
    
    def _read_logs(self, instance_id: str, process: subprocess.Popen, log_file: Path):
        """Read logs from process and store in buffer"""
        try:
            if not process.stdout:
                return
            
            with open(log_file, 'a') as f:
                for line in iter(process.stdout.readline, b''):
                    if not line:
                        break
                    
                    decoded_line = line.decode('utf-8', errors='replace').rstrip()
                    
                    # Write to file
                    f.write(decoded_line + '\n')
                    f.flush()
                    
                    # Store in buffer
                    if instance_id not in self.log_buffers:
                        self.log_buffers[instance_id] = deque(maxlen=settings.log_buffer_size)
                        self.log_sequences[instance_id] = 0
                    
                    seq = self.log_sequences[instance_id]
                    self.log_sequences[instance_id] += 1
                    
                    log_msg = LogMessage(
                        seq=seq,
                        timestamp=datetime.now().isoformat(),
                        line=decoded_line
                    )
                    self.log_buffers[instance_id].append(log_msg)

                    # Update runtime state based on parsed log lines
                    try:
                        self._parse_and_update_runtime(instance_id, decoded_line)
                    except Exception:
                        # Parsing errors should not break logging
                        pass
        except Exception as e:
            print(f"Error reading logs for {instance_id}: {e}")

    def _emit_state_if_changed(self, instance_id: str, *, busy: bool, prefill_progress: object, active_slots: int):
        """Update in-memory runtime and enqueue a state event if any value changed."""
        # Normalize prefill_progress: ensure float or None
        pp: object
        if prefill_progress is None:
            pp = None
        else:
            try:
                f = float(prefill_progress)  # type: ignore[assignment]
            except Exception:
                f = None
            pp = f

        prev = self.last_runtime.get(instance_id)
        changed = (
            prev is None or
            prev.get("busy") != busy or
            prev.get("prefill_progress") != pp or
            prev.get("active_slots") != active_slots
        )

        # Always update in-memory instance runtime fields
        config_manager.update_instance_runtime(
            instance_id,
            busy=busy,
            prefill_progress=pp,
            active_slots=active_slots,
        )

        if not changed:
            return

        # Persist last runtime snapshot (ephemeral, in-memory only)
        self.last_runtime[instance_id] = {
            "busy": busy,
            "prefill_progress": pp,
            "active_slots": active_slots,
        }

        # Initialize state buffer/seq lazily
        if instance_id not in self.state_buffers:
            self.state_buffers[instance_id] = deque(maxlen=settings.log_buffer_size)
            self.state_sequences[instance_id] = 0

        seq = self.state_sequences[instance_id]
        self.state_sequences[instance_id] += 1

        now_ts = datetime.now(timezone.utc).isoformat()
        state = InstanceRuntimeState(
            instance_id=instance_id,
            busy=busy,
            prefill_progress=pp if isinstance(pp, float) or pp is None else None,
            active_slots=active_slots,
            timestamp=now_ts,
        )
        event = InstanceStateEvent(
            seq=seq,
            timestamp=now_ts,
            data=state,
        )
        self.state_buffers[instance_id].append(event)

    def _parse_and_update_runtime(self, instance_id: str, line: str):
        """Parse a single log line and update runtime state accordingly."""
        # Ensure active slot set exists
        slots = self.active_slot_ids.get(instance_id)
        if slots is None:
            slots = set()
            self.active_slot_ids[instance_id] = slots

        # slot launch → add slot, busy true
        m = self._re_launch.search(line)
        if m:
            try:
                slot_id = int(m.group(1))
            except Exception:
                slot_id = -1
            slots.add(slot_id)
            self._emit_state_if_changed(
                instance_id,
                busy=True,
                prefill_progress=self.last_runtime.get(instance_id, {}).get("prefill_progress"),
                active_slots=len(slots),
            )
            return

        # prompt processing progress → update progress
        m = self._re_progress.search(line)
        if m:
            try:
                progress = float(m.group(1))
            except Exception:
                progress = None
            self._emit_state_if_changed(
                instance_id,
                busy=True if len(slots) > 0 else (self.last_runtime.get(instance_id, {}).get("busy") is True),
                prefill_progress=progress,
                active_slots=len(slots),
            )
            return

        # prompt done → set progress to 1.0 (stays until decode or release)
        if self._re_prompt_done.search(line):
            self._emit_state_if_changed(
                instance_id,
                busy=True if len(slots) > 0 else (self.last_runtime.get(instance_id, {}).get("busy") is True),
                prefill_progress=1.0,
                active_slots=len(slots),
            )
            return

        # slot release → remove slot; if none remain, clear busy and progress
        m = self._re_release.search(line)
        if m:
            try:
                slot_id = int(m.group(1))
            except Exception:
                slot_id = -1
            if slot_id in slots:
                slots.discard(slot_id)
            if len(slots) == 0:
                self._emit_state_if_changed(
                    instance_id,
                    busy=False,
                    prefill_progress=None,
                    active_slots=0,
                )
            else:
                self._emit_state_if_changed(
                    instance_id,
                    busy=True,
                    prefill_progress=self.last_runtime.get(instance_id, {}).get("prefill_progress"),
                    active_slots=len(slots),
                )
            return

        # all slots idle → force clear
        if self._re_all_idle.search(line):
            if len(slots) > 0:
                slots.clear()
            self._emit_state_if_changed(
                instance_id,
                busy=False,
                prefill_progress=None,
                active_slots=0,
            )
            return
    
    async def start_instance(self, instance_id: str) -> bool:
        """Start a llama-server instance"""
        instance = config_manager.get_instance(instance_id)
        if not instance:
            return False
        
        # Check if already running
        if instance.status == InstanceStatus.RUNNING:
            return True
        
        # Assign port if not set
        if not instance.port:
            instance.port = self._get_available_port()
        
        # Update status
        instance.status = InstanceStatus.STARTING
        instance.error_message = None
        config_manager.update_instance(instance_id, instance)
        
        try:
            # Build command
            cmd = self._build_command(instance)
            
            # Create log file
            log_file = self.log_dir / f"{instance.config.alias.replace(':', '-')}_{int(time.time())}.log"
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0  # Unbuffered for real-time output
            )
            
            self.processes[instance_id] = process
            
            # Start log reading thread
            log_thread = threading.Thread(
                target=self._read_logs,
                args=(instance_id, process, log_file),
                daemon=True
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
                self.active_slot_ids[instance_id] = set()
                self.state_buffers[instance_id] = deque(maxlen=settings.log_buffer_size)
                self.state_sequences[instance_id] = 0
                self.last_runtime[instance_id] = {
                    "busy": False,
                    "prefill_progress": None,
                    "active_slots": 0,
                }
                config_manager.update_instance_runtime(instance_id, busy=False, prefill_progress=None, active_slots=0)
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
        """Stop a llama-server instance"""
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
        """Clean up old log files for stopped instances"""
        try:
            pattern = f"{alias.replace(':', '-')}_*.log"
            for log_file in self.log_dir.glob(pattern):
                # Keep only the most recent log
                if log_file.stat().st_mtime < time.time() - 300:  # 5 minutes old
                    log_file.unlink()
        except Exception as e:
            print(f"Error cleaning up logs: {e}")
    
    async def restart_instance(self, instance_id: str) -> bool:
        """Restart a llama-server instance"""
        await self.stop_instance(instance_id)
        await asyncio.sleep(1)
        return await self.start_instance(instance_id)
    
    def create_instance(self, config: InstanceConfig) -> Instance:
        """Create a new instance"""
        instance_id = str(uuid.uuid4())
        instance = Instance(
            id=instance_id,
            config=config,
            status=InstanceStatus.STOPPED
        )
        config_manager.add_instance(instance)
        return instance
    
    def get_log_buffer(self, instance_id: str) -> List[LogMessage]:
        """Get log buffer for an instance"""
        if instance_id in self.log_buffers:
            return list(self.log_buffers[instance_id])
        return []
    
    def get_next_sequence(self, instance_id: str) -> int:
        """Get next sequence number for an instance"""
        return self.log_sequences.get(instance_id, 0)

    # Runtime state buffer accessors
    def get_state_buffer(self, instance_id: str) -> List[InstanceStateEvent]:
        if instance_id in self.state_buffers:
            return list(self.state_buffers[instance_id])
        return []

    def get_state_next_sequence(self, instance_id: str) -> int:
        return self.state_sequences.get(instance_id, 0)
    
    async def auto_restart_running_instances(self):
        """Auto-restart instances that were running before shutdown"""
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

