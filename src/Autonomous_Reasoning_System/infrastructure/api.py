from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import uuid
import asyncio
from typing import AsyncGenerator

from Autonomous_Reasoning_System.control.core_loop import CoreLoop
from Autonomous_Reasoning_System.infrastructure.observability import HealthServer, Metrics

app = FastAPI(title="Tyrone Agent API", version="1.0.0")
tyrone = CoreLoop()
health_server = HealthServer(port=8001)
health_server.start()


class TaskRequest(BaseModel):
    goal: str


# ------------------------------------------------------------------
# 1. Submit a new goal → starts in background
# ------------------------------------------------------------------
@app.post("/v1/task")
async def create_task(request: TaskRequest):
    plan_id = f"plan_{uuid.uuid4().hex[:12]}"
    tyrone.run_background(request.goal, plan_id)
    Metrics().increment("task_submitted")
    return {"plan_id": plan_id, "status": "queued"}


# ------------------------------------------------------------------
# 2. Poll status of a running / finished plan
# ------------------------------------------------------------------
@app.get("/v1/task/{plan_id}")
async def get_task_status(plan_id: str):
    status = tyrone.get_plan_status(plan_id)
    if not status:
        raise HTTPException(404, "Plan not found")
    return status


# ------------------------------------------------------------------
# 3. Real-time streaming of thoughts (SSE)
# ------------------------------------------------------------------
async def event_stream(plan_id: str) -> AsyncGenerator[str, None]:
    queue: asyncio.Queue = asyncio.Queue()
    tyrone.subscribe_stream(plan_id, queue)

    try:
        while True:
            line = await queue.get()
            if line is None:  # signals end
                break
            yield f"data: {line}\n\n"
            await asyncio.sleep(0.01)
    finally:
        tyrone.unsubscribe_stream(plan_id)


@app.get("/v1/stream/{plan_id}")
async def stream_task(plan_id: str):
    return StreamingResponse(event_stream(plan_id), media_type="text/event-stream")


# ------------------------------------------------------------------
# 4. Health
# ------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("Autonomous_Reasoning_System.infrastructure.api:app", host="0.0.0.0", port=8000, reload=False)
