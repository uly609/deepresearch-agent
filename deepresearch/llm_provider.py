"""Optional LLM provider layer.

The project can run without an API key. When `--llm` is enabled and an
OpenAI-compatible key is configured, agents can ask a real model for planning
and report synthesis. Otherwise the deterministic rule-based path is used.
"""

import json
import os
import urllib.error
import urllib.request
from typing import List


class LLMProvider:
    """空 LLM Provider。

    没有开启 `--llm` 或没有 API key 时使用它，让项目仍然可以规则化运行。
    """

    name = "disabled"

    def complete(self, system: str, user: str) -> str:
        """返回空字符串，表示没有模型输出。"""
        return ""

    def available(self) -> bool:
        """表示当前没有可用 LLM。"""
        return False


class OpenAICompatibleLLMProvider(LLMProvider):
    """OpenAI-compatible Chat Completions 客户端。

    DeepSeek 和 OpenAI-compatible 服务都可以通过这个类接入。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        name: str = "openai_compatible",
        timeout_seconds: int = 20,
    ) -> None:
        """保存模型服务配置。"""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name
        self.timeout_seconds = timeout_seconds

    def available(self) -> bool:
        """只要存在 API key，就认为该 provider 可用。"""
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> str:
        """调用 Chat Completions 接口，返回模型生成的文本。"""
        if not self.available():
            return ""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "deepresearch-agent",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return ""
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()


def build_llm_provider(enabled: bool = False) -> LLMProvider:
    """根据 CLI 开关和环境变量构建 LLM Provider。"""
    if not enabled:
        return LLMProvider()

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        return OpenAICompatibleLLMProvider(
            api_key=deepseek_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            name="deepseek",
        )

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        return OpenAICompatibleLLMProvider(
            api_key=openai_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            name="openai",
        )

    return LLMProvider()


def parse_json_list(text: str) -> List[str]:
    """把 LLM 返回的 JSON 列表解析成 Python 字符串列表。"""
    text = _extract_json(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        items = payload.get("sub_questions") or payload.get("items") or []
        return [str(item).strip() for item in items if str(item).strip()]
    return []


def _extract_json(text: str) -> str:
    """从 Markdown 或解释性文本中提取 JSON 片段。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    starts = [index for index in [cleaned.find("["), cleaned.find("{")] if index >= 0]
    if not starts:
        return cleaned
    start = min(starts)
    end = max(cleaned.rfind("]"), cleaned.rfind("}"))
    if end > start:
        return cleaned[start : end + 1]
    return cleaned
