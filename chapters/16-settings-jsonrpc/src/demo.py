"""第 16 章：JSON-RPC 请求读取配置并真实驱动模型。"""

from __future__ import annotations

import json
from typing import Any

from client import DeepSeekClient
from rpc import RpcDispatcher
from settings import Settings


def exchange(dispatcher: RpcDispatcher, request: dict[str, Any]) -> dict[str, Any]:
    wire = json.dumps(request, ensure_ascii=False)
    response = dispatcher.dispatch(wire)
    print(f"→ {wire}")
    print(f"← {json.dumps(response, ensure_ascii=False)}")
    return response


def main() -> None:
    settings = Settings(
        user_document={
            "agent": {
                "model": "deepseek-chat",
                "system_prompt": "你是简洁的 Python 教学助手。",
            }
        }
    )
    agent_settings = settings.register(
        "agent",
        defaults={"model": "deepseek-chat", "system_prompt": "你是编程助手。"},
        base={"language": "zh-CN"},
    )
    revision = agent_settings.revision
    agent_settings.update(
        {"system_prompt": "你是简洁的 Python 教学助手，只回答一个自然段。"},
        expected_revision=revision,
    )

    dispatcher = RpcDispatcher()
    dispatcher.register(
        "settings.get",
        lambda params: dict(settings.get(str(params.get("namespace", "")))),
    )

    def run_agent(params: dict[str, Any]) -> dict[str, Any]:
        prompt = params.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise TypeError("prompt 必须是非空字符串")
        config = agent_settings.get()
        model = config["model"]
        system_prompt = config["system_prompt"]
        if not isinstance(model, str) or not isinstance(system_prompt, str):
            raise TypeError("agent 配置中的 model 与 system_prompt 必须是字符串")
        content = DeepSeekClient(model).answer(system_prompt, prompt)
        return {
            "model": model,
            "settings_revision": agent_settings.revision,
            "content": content,
        }

    dispatcher.register("agent.run", run_agent)

    print("=== 外部请求先读取当前配置 ===")
    exchange(
        dispatcher,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "settings.get",
            "params": {"namespace": "agent"},
        },
    )
    print("\n=== JSON-RPC 真实驱动模型 ===")
    exchange(
        dispatcher,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "agent.run",
            "params": {"prompt": "用一句通俗的话解释 Python 生成器。"},
        },
    )
    print("\n=== 非法请求仍返回结构化错误 ===")
    exchange(
        dispatcher,
        {"jsonrpc": "2.0", "id": 3, "method": "agent.run", "params": {}},
    )


if __name__ == "__main__":
    main()
