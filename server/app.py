import copy
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import StartupDecisionEnv

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Startup Decision Simulator OpenEnv API")

class StepRequest(BaseModel):
    action: Dict[str, Any]

@app.get("/health")
async def health_check():
    """
    Mandatory endpoint to pass Hugging Face automated Space validations.
    """
    return {"status": "healthy"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main stateful WebSocket connection for the OpenEnv client interactions.
    Maintains environment state across the lifetime of the socket.
    """
    await websocket.accept()
    env = StartupDecisionEnv() # Fresh session instance per connection
    
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("type", "")
            
            if command == "reset":
                # Some configs might pass kwargs in data, but default reset is fine
                obs = env.reset()
                await websocket.send_json({
                    "observation": obs.model_dump(),
                    "reward": None,
                    "done": False,
                    "info": {}
                })
                
            elif command == "step":
                action_data = data.get("action", {})
                obs, reward, done, info = env.step(action_data)
                
                await websocket.send_json({
                    "observation": obs.model_dump(),
                    "reward": reward.model_dump() if reward else None,
                    "done": done,
                    "info": info
                })
                
            elif command == "state":
                await websocket.send_json({"state": env.state()})
                
            else:
                await websocket.send_json({"error": f"Unknown command: {command}"})
                
    except Exception as e:
        logger.info(f"WebSocket session closed or encountered error: {e}")
        try:
            await websocket.close()
        except:
            pass

# Optional stateless fallback routes for HTTP debug
@app.post("/reset")
async def http_reset():
    # Because of stateless HTTP, this doesn't maintain session, but fulfills ping endpoints if they exist
    env = StartupDecisionEnv()
    obs = env.reset()
    return {
        "observation": obs.model_dump(),
        "reward": None, 
        "done": False, 
        "info": {}
    }

@app.post("/step")
async def http_step(req: StepRequest):
    raise HTTPException(status_code=400, detail="Stateless HTTP mode not fully supported. Please use WebSocket (/ws) for stateful interaction.")

@app.get("/state")
async def http_state():
    return {"state": "HTTP endpoints active."}

# Additional openenv structural pings
@app.get("/docs")
async def get_docs():
    # Will be intercepted by FastAPI naturally, just explicit note here
    pass
