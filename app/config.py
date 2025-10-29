import json
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings
from app.models import Instance, InstanceStatus


class Settings(BaseSettings):
    """Application settings"""
    api_key: str = "change-me-please"
    host: str = "0.0.0.0"
    port: int = 8001
    config_file: str = "config.json"
    log_dir: str = "logs"
    start_port: int = 3500
    max_retries: int = 2
    log_buffer_size: int = 1000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


class ConfigManager:
    """Manages instance configurations and persistence"""
    
    def __init__(self, config_file: str = None):
        self.config_file = Path(config_file or settings.config_file)
        self.instances: Dict[str, Instance] = {}
        self.load()
    
    def load(self):
        """Load configuration from disk"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    for instance_data in data.get('instances', []):
                        instance = Instance(**instance_data)
                        self.instances[instance.id] = instance
            except Exception as e:
                print(f"Error loading config: {e}")
                self.instances = {}
        else:
            self.instances = {}
    
    def save(self):
        """Save configuration to disk"""
        try:
            data = {
                'instances': [
                    instance.model_dump(mode='json')
                    for instance in self.instances.values()
                ]
            }
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def add_instance(self, instance: Instance):
        """Add a new instance"""
        self.instances[instance.id] = instance
        self.save()
    
    def update_instance(self, instance_id: str, instance: Instance):
        """Update an existing instance"""
        if instance_id in self.instances:
            self.instances[instance_id] = instance
            self.save()
    
    def remove_instance(self, instance_id: str):
        """Remove an instance"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            self.save()
    
    def get_instance(self, instance_id: str) -> Instance:
        """Get an instance by ID"""
        return self.instances.get(instance_id)
    
    def get_all_instances(self) -> List[Instance]:
        """Get all instances"""
        return list(self.instances.values())
    
    def get_running_instances(self) -> List[Instance]:
        """Get all running instances"""
        return [
            instance for instance in self.instances.values()
            if instance.status == InstanceStatus.RUNNING
        ]


# Global config manager instance
config_manager = ConfigManager()

