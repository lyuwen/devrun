"""HTTPExecutor — sends job requests to inference / API endpoints."""

from __future__ import annotations

import json

import requests

from devrun.executors.base import BaseExecutor
from devrun.models import ExecutorEntry, TaskSpec
from devrun.registry import register_executor


@register_executor("http")
class HTTPExecutor(BaseExecutor):
    """POST job payloads to a remote HTTP/REST endpoint."""

    def __init__(self, name: str, config: ExecutorEntry) -> None:
        super().__init__(name, config)
        if not config.endpoint:
            raise ValueError(f"HTTPExecutor '{name}' requires an 'endpoint' in config")
        self._endpoint = config.endpoint.rstrip("/")
        self._timeout = config.extra.get("timeout", 30)
        self._headers = {
            "Content-Type": "application/json",
            **config.extra.get("headers", {}),
        }

    def submit(self, task_spec: TaskSpec) -> str:
        payload = {
            "command": task_spec.command,
            "resources": task_spec.resources,
            "env": task_spec.env,
            "metadata": task_spec.metadata,
        }
        self.logger.info("HTTP POST %s/run_job", self._endpoint)
        resp = requests.post(
            f"{self._endpoint}/run_job",
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = str(data.get("job_id", data.get("id", "")))
        if not job_id:
            raise RuntimeError(f"No job_id in response: {data}")
        self.logger.info("HTTP job submitted: %s", job_id)
        return job_id

    def status(self, job_id: str) -> str:
        try:
            resp = requests.get(
                f"{self._endpoint}/job_status/{job_id}",
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("status", "unknown")
        except requests.RequestException as exc:
            self.logger.error("Failed to query status: %s", exc)
            return "unknown"

    def logs(self, job_id: str, log_path: str | None = None) -> str:
        try:
            resp = requests.get(
                f"{self._endpoint}/job_logs/{job_id}",
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("logs", resp.text)
        except requests.RequestException as exc:
            self.logger.error("Failed to retrieve logs: %s", exc)
            return f"(error fetching logs: {exc})"
