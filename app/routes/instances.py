from fastapi import APIRouter, HTTPException
from typing import List

from app.models import (
    Instance, InstanceCreate, InstanceUpdate, 
    InstanceResponse, InstanceStatus, InstanceRuntimeState
)
from app.config import config_manager
from app.process_manager import process_manager


router = APIRouter(prefix="/instances", tags=["instances"])


@router.post("", response_model=InstanceResponse)
async def create_instance(data: InstanceCreate):
    """Create a new llama-server instance"""
    try:
        instance = process_manager.create_instance(data.config)
        return InstanceResponse(
            instance=instance,
            message=f"Instance {instance.id} created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[Instance])
async def list_instances():
    """List all instances"""
    return config_manager.get_all_instances()


@router.get("/{instance_id}", response_model=Instance)
async def get_instance(instance_id: str):
    """Get instance details"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    return instance


@router.put("/{instance_id}", response_model=InstanceResponse)
async def update_instance(instance_id: str, data: InstanceUpdate):
    """Update instance configuration (only when stopped)"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    if instance.status != InstanceStatus.STOPPED:
        raise HTTPException(
            status_code=400, 
            detail="Cannot update running instance. Stop it first."
        )
    
    try:
        instance.config = data.config
        config_manager.update_instance(instance_id, instance)
        return InstanceResponse(
            instance=instance,
            message=f"Instance {instance_id} updated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{instance_id}", response_model=InstanceResponse)
async def delete_instance(instance_id: str):
    """Delete an instance (must be stopped first)"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    if instance.status != InstanceStatus.STOPPED:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete running instance. Stop it first."
        )
    
    try:
        config_manager.remove_instance(instance_id)
        return InstanceResponse(
            instance=instance,
            message=f"Instance {instance_id} deleted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{instance_id}/start", response_model=InstanceResponse)
async def start_instance(instance_id: str):
    """Start an instance"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    success = await process_manager.start_instance(instance_id)
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found after start")
    
    if success:
        return InstanceResponse(
            instance=instance,
            message=f"Instance {instance_id} started successfully"
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start instance: {instance.error_message}"
        )


@router.post("/{instance_id}/stop", response_model=InstanceResponse)
async def stop_instance(instance_id: str):
    """Stop an instance"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    success = await process_manager.stop_instance(instance_id)
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found after stop")
    
    if success:
        return InstanceResponse(
            instance=instance,
            message=f"Instance {instance_id} stopped successfully"
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop instance: {instance.error_message}"
        )


@router.post("/{instance_id}/restart", response_model=InstanceResponse)
async def restart_instance(instance_id: str):
    """Restart an instance"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    success = await process_manager.restart_instance(instance_id)
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found after restart")
    
    if success:
        return InstanceResponse(
            instance=instance,
            message=f"Instance {instance_id} restarted successfully"
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart instance: {instance.error_message}"
        )


@router.get("/{instance_id}/state", response_model=InstanceRuntimeState)
async def get_instance_state(instance_id: str):
    """Get ephemeral runtime state for an instance"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    # Build current snapshot (ephemeral values default to safe values)
    now_iso = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    return InstanceRuntimeState(
        instance_id=instance_id,
        busy=getattr(instance, "busy", False),
        prefill_progress=getattr(instance, "prefill_progress", None),
        active_slots=getattr(instance, "active_slots", 0),
        timestamp=now_iso,
    )

