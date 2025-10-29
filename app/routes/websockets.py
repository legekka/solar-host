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

