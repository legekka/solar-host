import subprocess
import socket
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import deque
import asyncio
import threading

from app.models import Instance, InstanceConfig, InstanceStatus, LogMessage
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
        except Exception as e:
            print(f"Error reading logs for {instance_id}: {e}")
    
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
                instance.started_at = datetime.now()
                instance.retry_count = 0
                config_manager.update_instance(instance_id, instance)
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

