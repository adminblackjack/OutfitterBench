from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from .models import ToolResult, WorldState


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, day INTEGER NOT NULL, scope TEXT NOT NULL, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS day_starts (day INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS tool_calls (
  call_id TEXT PRIMARY KEY, day INTEGER NOT NULL, name TEXT NOT NULL,
  arguments TEXT NOT NULL, result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, day INTEGER NOT NULL, request TEXT NOT NULL,
  response TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS model_message_links (
  model_call_id INTEGER PRIMARY KEY, assistant_seq INTEGER NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS model_tool_links (
  model_call_id INTEGER NOT NULL, position INTEGER NOT NULL,
  raw_id TEXT, executed_id TEXT UNIQUE,
  PRIMARY KEY(model_call_id, position)
);
CREATE TABLE IF NOT EXISTS tool_call_origins (
  call_id TEXT PRIMARY KEY, origin TEXT NOT NULL, controller_kind TEXT
);
"""


class RunStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.db_path = run_dir / "run.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @classmethod
    def create(cls, run_dir: Path, manifest: dict[str, Any], initial_state: WorldState) -> "RunStore":
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        store = cls(run_dir)
        with store._connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT INTO state(id,value) VALUES(1,?)",
                (json.dumps(initial_state.to_dict(), ensure_ascii=False, sort_keys=True),),
            )
        return store

    @classmethod
    def resume(cls, run_dir: str | Path) -> "RunStore":
        path = Path(run_dir).expanduser().resolve()
        if not (path / "manifest.json").is_file() or not (path / "run.sqlite3").is_file():
            raise FileNotFoundError(f"not an OutfitterBench run directory: {path}")
        return cls(path)

    def load_state(self) -> WorldState:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM state WHERE id=1").fetchone()
        if row is None:
            raise RuntimeError("run state is missing")
        return WorldState.from_dict(json.loads(row[0]))

    @staticmethod
    def _save(connection: sqlite3.Connection, state: WorldState) -> None:
        connection.execute(
            "UPDATE state SET value=? WHERE id=1",
            (json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True),),
        )

    def has_day_start(self, day: int) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1 FROM day_starts WHERE day=?", (day,)).fetchone() is not None

    def record_day_start(self, state: WorldState, message: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute("INSERT OR IGNORE INTO day_starts(day) VALUES(?)", (state.day,))
            connection.execute(
                "INSERT INTO messages(day,scope,value) VALUES(?,?,?)",
                (state.day, "controller", json.dumps(message, ensure_ascii=False)),
            )
            self._save(connection, state)

    def append_message(self, day: int, scope: str, message: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(day,scope,value) VALUES(?,?,?)",
                (day, scope, json.dumps(message, ensure_ascii=False)),
            )

    def messages(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT value FROM messages ORDER BY seq").fetchall()
        return [json.loads(row[0]) for row in rows]

    def context_messages_for_day(self, day: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT value FROM messages WHERE day=? AND scope='controller' ORDER BY seq", (day,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def record_model_response(
        self,
        state: WorldState,
        *,
        request: dict[str, Any],
        response: dict[str, Any],
        assistant_message: dict[str, Any],
        input_tokens: int,
        output_tokens: int,
        charge: Callable[[WorldState, int, int], None],
        request_tail_message: dict[str, Any] | None = None,
    ) -> WorldState:
        charge(state, input_tokens, output_tokens)
        with self._connect() as connection:
            model_cursor = connection.execute(
                "INSERT INTO model_calls(day,request,response,input_tokens,output_tokens) VALUES(?,?,?,?,?)",
                (
                    state.day,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                    input_tokens,
                    output_tokens,
                ),
            )
            # Keep the exact budget notice that the model saw, before its
            # answer, atomically with that answer. Subsequent requests append
            # new notices; retries of failed API calls persist nothing.
            if request_tail_message is not None:
                connection.execute(
                    "INSERT INTO messages(day,scope,value) VALUES(?,?,?)",
                    (state.day, "controller", json.dumps(request_tail_message, ensure_ascii=False)),
                )
            message_cursor = connection.execute(
                "INSERT INTO messages(day,scope,value) VALUES(?,?,?)",
                (state.day, "controller", json.dumps(assistant_message, ensure_ascii=False)),
            )
            from .call_identity import response_message
            connection.execute('INSERT INTO model_message_links VALUES(?,?)',
                               (model_cursor.lastrowid, message_cursor.lastrowid))
            raw_calls = response_message(response).get('tool_calls') or []
            kept = assistant_message.get('tool_calls') or []
            for position in range(max(len(raw_calls), len(kept))):
                raw_id = raw_calls[position].get('id') if position < len(raw_calls) else None
                executed_id = kept[position].get('id') if position < len(kept) else None
                connection.execute('INSERT INTO model_tool_links VALUES(?,?,?,?)',
                    (model_cursor.lastrowid, position, raw_id, executed_id))
            self._save(connection, state)
        return state

    def execute_tool(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        execute: Callable[[WorldState, str, dict[str, Any]], ToolResult],
        *, controller_kind: str | None = None,
    ) -> tuple[WorldState, ToolResult, bool]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT result FROM tool_calls WHERE call_id=?", (call_id,)
            ).fetchone()
            state_row = connection.execute("SELECT value FROM state WHERE id=1").fetchone()
            state = WorldState.from_dict(json.loads(state_row[0]))
            if existing is not None:
                value = json.loads(existing[0])
                return state, ToolResult(**value), True
            call_day = state.day
            result = execute(state, name, arguments)
            connection.execute(
                "INSERT INTO tool_calls(call_id,day,name,arguments,result) VALUES(?,?,?,?,?)",
                (
                    call_id,
                    call_day,
                    name,
                    json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                    json.dumps(
                        {"ok": result.ok, "data": result.data, "terminal_day": result.terminal_day},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            )
            self._save(connection, state)
            linked = connection.execute('SELECT 1 FROM model_tool_links WHERE executed_id=?', (call_id,)).fetchone()
            origin = 'automatic' if controller_kind in {'forced_close','runoff'} else 'agent' if linked else 'unknown'
            connection.execute('INSERT INTO tool_call_origins VALUES(?,?,?)', (call_id,origin,controller_kind))
            return state, result, False

    def record_tool_message(
        self, state: WorldState, *, tool_call_id: str, name: str, content: str
    ) -> None:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM messages WHERE json_extract(value,'$.tool_call_id')=?", (tool_call_id,)
            ).fetchone()
            if exists is None:
                tool_row = connection.execute(
                    "SELECT day FROM tool_calls WHERE call_id=?", (tool_call_id,)
                ).fetchone()
                message_day = int(tool_row[0]) if tool_row is not None else state.day
                connection.execute(
                    "INSERT INTO messages(day,scope,value) VALUES(?,?,?)",
                    (
                        message_day,
                        "controller",
                        json.dumps(
                            {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content},
                            ensure_ascii=False,
                        ),
                    ),
                )

    def tool_result_exists(self, call_id: str) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1 FROM tool_calls WHERE call_id=?", (call_id,)).fetchone() is not None

    def tool_call_id_taken(self, call_id: str) -> bool:
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM tool_calls WHERE call_id=?", (call_id,)).fetchone():
                return True
            return (
                connection.execute(
                    "SELECT 1 FROM messages WHERE json_extract(value,'$.tool_call_id')=?",
                    (call_id,),
                ).fetchone()
                is not None
            )

    def rewrite_last_assistant_tool_call_id(self, old_id: str, new_id: str) -> bool:
        with self._connect() as connection:
            rows = connection.execute("SELECT seq, value FROM messages ORDER BY seq DESC").fetchall()
            for seq, raw in rows:
                message = json.loads(raw)
                if message.get("role") != "assistant":
                    continue
                changed = False
                for call in message.get("tool_calls") or []:
                    if str(call.get("id")) == old_id:
                        call["id"] = new_id
                        changed = True
                if changed:
                    connection.execute(
                        "UPDATE messages SET value=? WHERE seq=?",
                        (json.dumps(message, ensure_ascii=False), seq),
                    )
                    connection.execute('UPDATE model_tool_links SET executed_id=? WHERE executed_id=? '
                        'AND model_call_id=(SELECT model_call_id FROM model_message_links WHERE assistant_seq=?)',
                        (new_id, old_id, seq))
                    return True
                return False
        return False

    def model_call_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT count(*) FROM model_calls").fetchone()[0])

    def model_calls_for_day(self, day: int) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT count(*) FROM model_calls WHERE day=?", (day,)).fetchone()[0])

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
