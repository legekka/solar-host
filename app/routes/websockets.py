from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio

from app.config import config_manager
from app.process_manager import process_manager


router = APIRouter(tags=["websockets"])


@router.websocket("/instances/{instance_id}/logs")
async def websocket_logs(websocket: WebSocket, instance_id: str):
    """Stream logs for an instance via WebSocket"""
    await websocket.accept()
    
    # Check if instance exists
    instance = config_manager.get_instance(instance_id)
    if not instance:
        await websocket.send_json({"error": "Instance not found"})
        await websocket.close()
        return
    
    try:
        # Send historical buffer (last N lines)
        buffer = process_manager.get_log_buffer(instance_id)
        for log_msg in buffer:
            await websocket.send_json(log_msg.model_dump())
        
        # Get current sequence number
        last_seq = process_manager.get_next_sequence(instance_id)
        
        # Stream new logs
        while True:
            # Check for new logs
            current_seq = process_manager.get_next_sequence(instance_id)
            if current_seq > last_seq:
                buffer = process_manager.get_log_buffer(instance_id)
                # Send only new logs
                for log_msg in buffer:
                    if log_msg.seq >= last_seq:
                        await websocket.send_json(log_msg.model_dump())
                last_seq = current_seq
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


@router.websocket("/instances/{instance_id}/state")
async def websocket_instance_state(websocket: WebSocket, instance_id: str):
    """Stream runtime state updates for an instance via WebSocket"""
    await websocket.accept()

    # Validate instance exists
    instance = config_manager.get_instance(instance_id)
    if not instance:
        await websocket.send_json({"error": "Instance not found"})
        await websocket.close()
        return

    try:
        # Send current snapshot immediately
        from datetime import datetime, timezone
        snapshot = {
            "type": "instance_state",
            "data": {
                "instance_id": instance_id,
                "busy": getattr(instance, "busy", False),
                "prefill_progress": getattr(instance, "prefill_progress", None),
                "active_slots": getattr(instance, "active_slots", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
        await websocket.send_json(snapshot)

        # Track sequence for incremental updates
        last_seq = process_manager.get_state_next_sequence(instance_id)

        while True:
            current_seq = process_manager.get_state_next_sequence(instance_id)
            if current_seq > last_seq:
                buffer = process_manager.get_state_buffer(instance_id)
                for event in buffer:
                    if event.seq >= last_seq:
                        await websocket.send_json(event.model_dump())
                last_seq = current_seq

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

