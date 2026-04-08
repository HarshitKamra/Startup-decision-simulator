import logging
from threading import Lock
from typing import Any, Dict

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import uvicorn

from env.environment import StartupDecisionEnv

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Startup Decision Simulator OpenEnv API")

_http_env = StartupDecisionEnv()
_http_env_lock = Lock()

class StepRequest(BaseModel):
    action: Dict[str, Any]

def _serialize_step_result(env: StartupDecisionEnv, action: Dict[str, Any]) -> Dict[str, Any]:
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
        "state": env.state(),
    }

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "startup-decision-simulator",
        "docs": {
            "health": "/health",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "websocket": "/ws",
        },
    }

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

# HTTP routes for validator compatibility
@app.post("/reset")
async def http_reset():
    with _http_env_lock:
        obs = _http_env.reset()
        return {
            "observation": obs.model_dump(),
            "reward": None,
            "done": False,
            "info": {},
            "state": _http_env.state(),
        }

@app.post("/step")
async def http_step(req: StepRequest):
    with _http_env_lock:
        return _serialize_step_result(_http_env, req.action)

@app.get("/state")
async def http_state():
    with _http_env_lock:
        return {"state": _http_env.state()}


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
