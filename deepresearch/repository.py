"""SQLite persistence for users, research tasks and final reports."""
import json
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Optional


class ResearchRepository:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("DEEPRESEARCH_DB_PATH", ".deepresearch/deepresearch.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        return sqlite3.connect(str(self.path))

    def _init(self):
        with self._connect() as conn:
            conn.executescript("""
            create table if not exists users (id integer primary key, username text unique not null, api_token text unique not null, created_at text default current_timestamp);
            create table if not exists research_tasks (id text primary key, user_id integer not null, question text not null, status text not null, phase text, run_id text, error text, created_at text default current_timestamp, updated_at text default current_timestamp);
            create table if not exists research_reports (task_id text primary key, markdown text not null, report_json text not null, created_at text default current_timestamp);
            """)

    def create_user(self, username):
        token = secrets.token_urlsafe(32)
        with self._connect() as conn:
            try:
                cursor = conn.execute("insert into users(username, api_token) values (?, ?)", (username, token))
            except sqlite3.IntegrityError:
                raise ValueError("username already exists")
        return {"id": cursor.lastrowid, "username": username, "api_token": token}

    def user_for_token(self, token):
        with self._connect() as conn:
            row = conn.execute("select id, username from users where api_token=?", (token,)).fetchone()
        return {"id": row[0], "username": row[1]} if row else None

    def create_task(self, task_id, user_id, question):
        with self._connect() as conn:
            conn.execute("insert into research_tasks(id,user_id,question,status,phase) values (?,?,?,?,?)", (task_id, user_id, question, "running", "created"))

    def update_task(self, task_id, status, phase="", run_id="", error=""):
        with self._connect() as conn:
            conn.execute("update research_tasks set status=?, phase=?, run_id=?, error=?, updated_at=current_timestamp where id=?", (status, phase, run_id, error, task_id))

    def save_report(self, task_id, markdown, report):
        with self._connect() as conn:
            conn.execute("insert or replace into research_reports(task_id,markdown,report_json) values (?,?,?)", (task_id, markdown, json.dumps(report, ensure_ascii=False)))

    def list_tasks(self, user_id):
        with self._connect() as conn:
            rows = conn.execute("select id,question,status,phase,created_at,updated_at from research_tasks where user_id=? order by created_at desc", (user_id,)).fetchall()
        return [dict(zip(("id","question","status","phase","created_at","updated_at"), row)) for row in rows]

    def get_task(self, task_id, user_id):
        with self._connect() as conn:
            row = conn.execute("select id,question,status,phase,run_id,error,created_at,updated_at from research_tasks where id=? and user_id=?", (task_id,user_id)).fetchone()
            report = conn.execute("select markdown,report_json from research_reports where task_id=?", (task_id,)).fetchone()
        if not row: return None
        result = dict(zip(("id","question","status","phase","run_id","error","created_at","updated_at"), row))
        if report: result.update({"report": report[0], "structured_report": json.loads(report[1])})
        return result
