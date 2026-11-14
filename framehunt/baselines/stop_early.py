"""Stop-early baseline as a Harbor custom agent.

    harbor run ... --agent framehunt.baselines.stop_early:StopEarly   (repo root on PYTHONPATH)

Uploads stop_early_client.py (stdlib only) into the agent's container and runs it there,
so it reaches the MCP sidecar over the task network like any agent: `info`, one
low-res `view`, then `submit` of a well-formed guess seeded by the item id. Expected
reward is ~0: the answers are random strings from a large space.

This module runs inside Harbor's own Python, so it imports only Harbor and the stdlib.
"""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

CLIENT = Path(__file__).with_name("stop_early_client.py")
REMOTE = "/tmp/framehunt_stop_early_client.py"
DEFAULT_URL = "http://fh-mcp:8000/mcp"


class StopEarly(BaseAgent):
    # It talks to the task's MCP server, so it must declare MCP support.
    capabilities = AgentCapabilities(mcp_servers=True)

    @staticmethod
    def name() -> str:
        return "framehunt-stop-early"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.upload_file(CLIENT, REMOTE)

    def _item_id(self, instruction: str) -> str:
        # Harbor names trials "<task name>__<random>", and the task name is the item id.
        sid = self.session_id or ""
        if "__" in sid:
            return sid.split("__", 1)[0]
        return "q-" + hashlib.sha256(instruction.encode()).hexdigest()[:16]

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        url = next((s.url for s in self.mcp_servers if getattr(s, "url", None)), DEFAULT_URL)
        item_id = self._item_id(instruction)
        cmd = f"python3 {REMOTE} {shlex.quote(url)} {shlex.quote(item_id)}"
        res = await environment.exec(cmd, timeout_sec=300)
        out = (res.stdout or "") + (f"\n[stderr]\n{res.stderr}" if res.stderr else "")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "stop_early.txt").write_text(f"$ {cmd}\nexit={res.return_code}\n{out}")
        if res.return_code != 0:
            raise RuntimeError(f"stop-early client failed (exit {res.return_code}): {out[-500:]}")
