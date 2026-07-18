"""Optional FastAPI surface for starting a research task and streaming its events."""

import queue
import threading
from uuid import uuid4

from .repository import ResearchRepository
from .runtime import DeepResearchRuntime


def create_app():
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="DeepResearch Agent API")
    repository = ResearchRepository()
    runs = {}

    def current_user(x_api_key: str = ""):
        user = repository.user_for_token(x_api_key)
        if not user:
            raise HTTPException(401, "provide a valid X-API-Key")
        return user

    @app.get("/", response_class=HTMLResponse)
    def home():
        return """<!doctype html><html><head><meta charset='utf-8'><title>DeepResearch</title><style>body{max-width:900px;margin:40px auto;font:16px system-ui}textarea,input{width:100%;box-sizing:border-box;padding:10px;margin:8px 0}button{padding:9px 14px;margin-right:8px}pre{white-space:pre-wrap;background:#f5f5f5;padding:14px;min-height:140px}</style></head><body><h1>DeepResearch Agent</h1><input id='token' placeholder='API Token，先注册或粘贴已有 Token'><button onclick='register()'>注册账号</button><textarea id='q' rows='4' placeholder='输入研究问题'></textarea><button onclick='run()'>开始研究</button><pre id='out'></pre><script>const out=document.querySelector('#out'),token=document.querySelector('#token');token.value=localStorage.t||'';async function register(){let n=prompt('用户名（至少3位）');let r=await fetch('/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:n})});let d=await r.json();token.value=d.api_token;localStorage.t=d.api_token;out.textContent='注册成功，Token 已保存。'}async function run(){out.textContent='';let r=await fetch('/research',{method:'POST',headers:{'Content-Type':'application/json','X-API-Key':token.value},body:JSON.stringify({question:q.value,live_tools:true,fetch_content:true})});let d=await r.json();if(!r.ok){out.textContent=JSON.stringify(d);return}let e=await fetch(d.events,{headers:{'X-API-Key':token.value}}),reader=e.body.getReader(),dec=new TextDecoder();while(true){let x=await reader.read();if(x.done)break;out.textContent+=dec.decode(x.value)}};</script></body></html>"""

    @app.post("/auth/register")
    def register(payload: dict):
        username = str(payload.get("username") or "").strip()
        if len(username) < 3:
            raise HTTPException(400, "username must be at least 3 characters")
        try:
            return repository.create_user(username)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/research")
    def create_research(payload: dict, x_api_key: str = Header(default="")):
        user = current_user(x_api_key)
        question = str(payload.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "question is required")
        events = queue.Queue()
        request_id = "api_" + uuid4().hex[:12]
        runs[request_id] = {"events": events}
        repository.create_task(request_id, user["id"], question)
        runtime = DeepResearchRuntime(use_live_tools=bool(payload.get("live_tools", True)), use_llm=bool(payload.get("llm", False)), fetch_content=bool(payload.get("fetch_content", False)))

        def execute():
            try:
                def emit(event):
                    repository.update_task(request_id, "running", event.event_type, runtime.current_task_state.run_id if runtime.current_task_state else "")
                    events.put({"type": event.event_type, "message": event.message})
                state = runtime.ask(question, emit=emit)
                runs[request_id].update({"state": state, "task_state": runtime.current_task_state})
                repository.update_task(request_id, "completed", "finished", runtime.current_task_state.run_id)
                repository.save_report(request_id, state.report_markdown, {"task_id": state.task_id, "run_id": runtime.current_task_state.run_id})
            except Exception as exc:
                repository.update_task(request_id, "failed", "failed", error=str(exc)[:500])
                events.put({"type": "failed", "message": str(exc)})
            finally:
                events.put(None)

        threading.Thread(target=execute, daemon=True).start()
        return {"task_id": request_id, "events": "/research/{}/events".format(request_id)}

    @app.get("/research/{task_id}/events")
    def research_events(task_id: str, x_api_key: str = Header(default="")):
        user = current_user(x_api_key)
        if not repository.get_task(task_id, user["id"]):
            raise HTTPException(404, "task not found")
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
    def get_research(task_id: str, x_api_key: str = Header(default="")):
        user = current_user(x_api_key)
        task = repository.get_task(task_id, user["id"])
        if not task: raise HTTPException(404, "task not found")
        return task

    @app.get("/research")
    def list_research(x_api_key: str = Header(default="")):
        return repository.list_tasks(current_user(x_api_key)["id"])

    return app
