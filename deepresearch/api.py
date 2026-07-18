"""Optional FastAPI surface for starting a research task and streaming its events."""

import queue
import threading
from uuid import uuid4

from .runtime import DeepResearchRuntime


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="DeepResearch Agent API")
    runs = {}

    @app.post("/research")
    def create_research(payload: dict):
        question = str(payload.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "question is required")
        events = queue.Queue()
        request_id = "api_" + uuid4().hex[:12]
        runs[request_id] = {"events": events}
        runtime = DeepResearchRuntime(use_live_tools=bool(payload.get("live_tools", True)), use_llm=bool(payload.get("llm", False)), fetch_content=bool(payload.get("fetch_content", False)))

        def execute():
            try:
                state = runtime.ask(question, emit=lambda event: events.put({"type": event.event_type, "message": event.message}))
                runs[request_id].update({"state": state, "task_state": runtime.current_task_state})
            except Exception as exc:
                events.put({"type": "failed", "message": str(exc)})
            finally:
                events.put(None)

        threading.Thread(target=execute, daemon=True).start()
        return {"task_id": request_id, "events": "/research/{}/events".format(request_id)}

    @app.get("/research/{task_id}/events")
    def research_events(task_id: str):
        item = runs.get(task_id)
        if not item:
            raise HTTPException(404, "task not found")
        def stream():
            while True:
                event = item["events"].get()
                if event is None:
                    yield "event: done\ndata: {}\n\n"
                    return
                yield "event: {}\ndata: {}\n\n".format(event["type"], event["message"])
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/research/{task_id}")
    def get_research(task_id: str):
        item = runs.get(task_id)
        if not item or not item.get("state"):
            raise HTTPException(404, "task not finished")
        return {"task_id": task_id, "report": item["state"].report_markdown}

    return app
