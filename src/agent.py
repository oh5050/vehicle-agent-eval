"""Agent: prompt assembly and Ollama /api/chat inference.

External network rule: ALL inference goes to Ollama at config["model"]["endpoint"].
No other network calls are made.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from src.registry import Registry

# ── System-prompt template ──────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
당신은 차량 내 음성 비서입니다. 아래 도구 목록과 현재 차량 상태를 참고하여 운전자의 발화에 응답하세요.

## 사용 가능한 도구
{tools_json}

## 현재 차량 상태
{vehicle_state_json}

## 출력 규약
반드시 아래 형태의 JSON **하나만** 출력하세요. 설명, 마크다운 코드 블록, 다른 텍스트는 절대 포함하지 마세요.

{{
  "kind": "<아래 4개 값 중 하나만 사용>",
  "calls": [{{"tool": "<도구명>", "args": {{...}}}}],
  "message": "<운전자에게 전달할 짧은 한국어 응답>"
}}

kind 값 정의 (이 4개 외에는 절대 사용 금지):
- "execute"  : 요청을 즉시 실행할 수 있을 때
- "clarify"  : 필수 정보가 부족해 되물어야 할 때
- "refuse"   : 지원하지 않는 기능이거나 실행 불가능한 요청일 때
- "confirm"  : 위험하거나 민감한 동작이라 운전자에게 재확인이 필요할 때

calls 규칙:
- execute/confirm 시: 도구 호출 목록을 반드시 포함하세요.
- clarify/refuse 시: 빈 배열 [] 을 사용하세요.
- 도구 이름은 위 목록에 있는 이름만 정확히 사용하세요.\
"""

_TOOL_KEYS = ("name", "description", "parameters", "required", "required_one_of")


def _tool_to_dict(tool: Any) -> dict[str, Any]:
    """Serialize a ToolSchema to a compact dict for the prompt."""
    d: dict[str, Any] = {"name": tool.name, "description": tool.description}
    if tool.parameters:
        params: dict[str, Any] = {}
        for pname, pschema in tool.parameters.items():
            entry: dict[str, Any] = {}
            if pschema.type is not None:
                entry["type"] = pschema.type
            if pschema.enum is not None:
                entry["enum"] = pschema.enum
            if pschema.minimum is not None:
                entry["minimum"] = pschema.minimum
            if pschema.maximum is not None:
                entry["maximum"] = pschema.maximum
            if pschema.description is not None:
                entry["description"] = pschema.description
            if pschema.items is not None and pschema.items.enum is not None:
                entry["items"] = {"enum": pschema.items.enum}
            params[pname] = entry
        d["parameters"] = params
    if tool.required:
        d["required"] = tool.required
    if tool.required_one_of:
        d["required_one_of"] = tool.required_one_of
    return d


def build_prompt(
    registry: "Registry",
    utterance: str,
    vehicle_state: dict[str, Any],
) -> list[dict[str, str]]:
    """Assemble the Ollama messages list for one inference call.

    Returns a two-element list: [system_message, user_message].
    """
    tools_list = [_tool_to_dict(t) for t in registry.tools]
    tools_json = json.dumps(tools_list, ensure_ascii=False, indent=2)
    vehicle_state_json = json.dumps(vehicle_state, ensure_ascii=False, indent=2)

    system_content = _SYSTEM_TEMPLATE.format(
        tools_json=tools_json,
        vehicle_state_json=vehicle_state_json,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": utterance},
    ]


def call_model(
    messages: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Call the Ollama /api/chat endpoint with streaming enabled.

    Measures:
    - ttft_ms  : time from request start to first content token (ms)
    - total_ms : total wall-clock time for the complete response (ms)

    Returns::

        {
            "raw_output": str,
            "ttft_ms":    float,
            "total_ms":   float,
            "model":      str,
        }

    Raises ``requests.HTTPError`` on non-2xx responses so the caller can log
    the failure.  The caller (runner) is responsible for catching and logging.
    """
    endpoint: str = config["model"]["endpoint"].rstrip("/")
    model_name: str = config["model"]["name"]
    seed: int = config["experiment"]["seed"]

    url = f"{endpoint}/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0,
            "seed": seed,
        },
    }

    t_start = time.monotonic()
    ttft_ms: float | None = None
    content_parts: list[str] = []

    with requests.post(url, json=payload, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            chunk: dict[str, Any] = json.loads(raw_line)
            token = chunk.get("message", {}).get("content", "")
            if token and ttft_ms is None:
                ttft_ms = (time.monotonic() - t_start) * 1000
            if token:
                content_parts.append(token)
            if chunk.get("done"):
                break

    total_ms = (time.monotonic() - t_start) * 1000

    return {
        "raw_output": "".join(content_parts),
        "ttft_ms": round(ttft_ms or total_ms, 1),
        "total_ms": round(total_ms, 1),
        "model": model_name,
    }
