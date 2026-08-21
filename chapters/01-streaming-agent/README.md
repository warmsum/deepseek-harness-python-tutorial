# 01｜流式输出与消息组装

> 预计时间：60 分钟 ｜ 前置要求：会运行 Python 文件、知道 API Key 是什么 ｜ 本章调用真实 DeepSeek 模型

在 ChatGPT 或 DeepSeek 网页版中提交问题后，回答通常会逐步显示，而不是等整段文字生成完再一次出现。这个体验背后的机制就是流式输出。

智能体不仅要把新内容展示给用户，还要把每轮对话保存到历史中，供后续请求继续使用。因此需要同时满足两个要求：新生成的内容应当及时显示，写入历史的内容则必须是一条完整、稳定的消息。本章将处理好这两件事。

读完这一章，你会写一个能调用 DeepSeek 模型、支持流式输出、并把分片组装成完整消息的 Python 客户端。后续章节会继续扩展这个客户端，让模型能够调用工具并管理会话。

## 学习目标

完成本章后，你将能够：

- 说明 OpenAI 兼容接口中请求、响应和消息角色的基本结构；
- 分别完成一次非流式调用和一次 SSE 流式调用；
- 理解 `async`、`await` 与异步生成器在流式读取中的作用；
- 将多个分片组装成一条不可变的完整消息，再写入对话历史。

## 认识要调用的接口

写代码之前，先了解模型服务的基本接口。后续章节都会在这套接口之上继续扩展。

### API 与 OpenAI 兼容接口

模型运行在服务商的服务器上，程序需要通过约定好的 API 与它通信：向指定地址发送特定格式的数据，再读取服务器返回的结果。许多模型服务采用相近的请求格式，这类接口通常称为 OpenAI 兼容接口。使用兼容接口时，更换模型服务商通常只需要调整地址、模型名和 API Key。DeepSeek 提供了这类接口，基础地址是 `https://api.deepseek.com`。

### 一次请求长什么样

调用模型等于向 `/chat/completions` 发送一个 HTTP POST 请求，请求体是 JSON，核心字段有三个：

```json
{
  "model": "deepseek-chat",
  "messages": [
    { "role": "system", "content": "你是一个简洁的助手。" },
    { "role": "user", "content": "什么是流式输出？" }
  ],
  "stream": false
}
```

- `model` 指定用哪个模型，`deepseek-chat` 是 DeepSeek 的通用对话模型。
- `messages` 是对话历史，一个消息数组。模型对世界的全部了解都来自这个数组，模型没有记忆，给它什么它看什么。
- `stream` 为 false 表示生成完整回答后再返回，为 true 表示边生成边返回。

### 三种角色

`messages` 里每条消息都有一个 `role`，三种角色各有分工：

| 角色 | 谁说的话 | 作用 |
|------|----------|------|
| `system` | 系统 | 放在最前面，设定模型的行为规则 |
| `user` | 用户 | 人提出的问题和要求 |
| `assistant` | 模型 | 模型自己的回答 |

`system` 存在的意义是给模型设定行为。模型的默认行为是热心回答一切，当需要它扮演特定角色时，把要求写进 `system` 最有效。官方 Harness 的系统提示词就承担这个职责，第 06 章会专门讲它。

### 一次响应长什么样

非流式请求的响应体同样是 JSON：

```json
{
  "choices": [
    {
      "message": { "role": "assistant", "content": "流式输出是……" }
    }
  ]
}
```

`choices` 是候选回答的数组，一般只有一项。`choices[0].message.content` 是模型的完整回答文本，这个取值路径贯穿全书，第 02 章的工具调用会在这个 message 里多出一个 `tool_calls` 字段。

## 环境准备

需要三样东西，五分钟能搞定。

1. **Python 3.11 及以上**，终端运行 `python --version` 确认。
2. **uv**，一个用 Rust 写的 Python 包管理器，终端运行 `uv --version` 确认，没有就去 [astral.sh/uv](https://astral.sh/uv) 按说明安装。项目依赖写在根目录 `pyproject.toml` 里，`uv run python 某文件.py` 会自动安装缺失的包。
3. **DeepSeek API Key**，到 DeepSeek 开放平台申请。然后在项目根目录创建 `.env` 文件，根目录已有 `.env.example` 模板，写入一行：

   ```bash
   DEEPSEEK_API_KEY=sk-你的key
   ```

   本项目已经在 `.gitignore` 中忽略 `.env`，正常执行 `git add` 时不会把它加入版本库。仍然不要在代码、文档或终端输出中公开真实 Key。

## 先运行示例

```bash
uv run python chapters/01-streaming-agent/src/demo.py
```

三次调用，每次都是真实模型回答：

```
============================================================
演示 1：非流式调用（一次拿回完整回答）
============================================================
完整回答：流式输出是边生成边传输数据，无需等待全部完成后才显示，像打字机一样逐字呈现结果。

============================================================
演示 2：流式调用（边生成边显示）
============================================================
逐分片输出：流式输出是指数据在生成过程中逐段实时传输给用户，而非等待全部完成后一次性返回。

============================================================
演示 3：流式 + 组装成完整消息（Agent 的标准做法）
============================================================
进入历史的消息：role='assistant', 长度=45 字
消息内容：流式输出是指数据或内容在生成过程中分块、连续地传送给接收方，而非等待全部完成后一次性输出。
```

回答的文字由真实模型生成，每次略有不同。演示 2 会在终端中逐段显示内容，让用户在完整回答生成之前就能开始阅读。

本章代码都在 `chapters/01-streaming-agent/src/` 里，共两个文件：

```
chapters/01-streaming-agent/src/
├── client.py   # 本章主角：从零实现的客户端
└── demo.py     # 把 client.py 跑起来看效果
```

接下来按组成部分实现这个客户端。

## 1.1 把 API Key 读进来

调用模型前先拿到 Key。Key 不能硬编码在代码里，否则一开源就泄露，标准做法是放进环境变量，代码只在运行时读取。本地学习时最方便的形式是 `.env` 文件。`load_api_key()` 按环境变量优先、`.env` 兜底的顺序读取：

```python
import os
from pathlib import Path

from dotenv import dotenv_values


def load_api_key() -> str:
    # 第一步：环境变量（部署到服务器时的标准做法）
    from_env = os.getenv("DEEPSEEK_API_KEY")
    if from_env:
        return from_env

    # 第二步：项目根目录的 .env 文件
    env_path = Path(__file__).resolve().parents[3] / ".env"
    from_file = dotenv_values(env_path).get("DEEPSEEK_API_KEY")
    if from_file:
        return from_file

    raise RuntimeError("找不到 DEEPSEEK_API_KEY：请参考 .env.example 创建 .env")
```

逐行看：

- `os.getenv("DEEPSEEK_API_KEY")` 读环境变量，没有就返回 `None`。
- `Path(__file__).resolve().parents[3]` 是定位 `.env` 的关键。`__file__` 是这段代码自己的文件路径，文件位于 `chapters/01-streaming-agent/src/`，`parents[0]` 是 `src/`，`parents[1]` 是 `01-streaming-agent/`，`parents[2]` 是 `chapters/`，`parents[3]` 才是项目根目录。`resolve()` 把相对路径变成绝对路径，无论从哪个目录启动程序都能找到 `.env`。
- `dotenv_values(env_path)` 使用 `python-dotenv` 解析 `.env`，引号、注释等常见语法交给成熟库处理；它返回配置字典，不会把文件里的其他值写入当前进程环境。
- `.get("DEEPSEEK_API_KEY")` 只取本项目需要的密钥，因此即使 `.env` 里误放了其他配置，也不会由这里注入程序。
- 最后 `raise RuntimeError(...)`：Key 缺失时立即报错并说明原因。问题应在最接近来源的位置暴露，而不是带着空 Key 继续运行，直到发送请求时才得到难以判断原因的 401。

## 1.2 先定义一条消息

在写客户端之前，先定义贯穿全书的数据结构，一条对话消息。Python 的 `dataclass` 是定义这类数据类的标准工具，它省掉手写 `__init__` 的样板代码。这里给它加一个关键修饰，`frozen=True`：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str     # "system"（规矩）/"user"（人）/"assistant"（模型）
    content: str  # 消息正文
```

`frozen=True` 做了什么？`dataclass` 默认生成的 `__init__` 用 `self.role = role` 这样的方式给属性赋值，而 `frozen=True` 会额外生成一个 `__setattr__` 方法，拒绝任何后续的属性写入。于是：

```python
m = Message(role="user", content="你好")
m.content = "篡改"   # 抛 FrozenInstanceError
```

写入历史的消息不应再被修改。对话历史会被反复读取，每一轮都要完整地发给模型，压缩、持久化和界面展示也会使用它。如果某处代码悄悄改动了旧消息，后续行为都会受到影响，而且很难排查。`frozen=True` 把这条约定变成 Python 会主动检查的限制。第 05 章的事件日志还会再次用到这个思想。

## 1.3 第一次调用：一次拿回完整回答

最朴素的调用方式：把历史发过去，等模型全部想完，一次性拿回完整回答。先把客户端类的骨架和 `chat()` 写出来：

```python
import httpx


class DeepSeekClient:
    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-chat"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or load_api_key()

    def chat(self, messages: list[Message]) -> str:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
```

逐段理解：

- `httpx` 是 Python 生态主流的 HTTP 客户端库，接口风格与经典库 `requests` 一致，同时支持同步和异步两种用法，本章两节会各用一次。`with httpx.Client(...)` 保证请求结束后连接被正确关闭。
- `Authorization` 头是 `Bearer <key>`，OpenAI 兼容接口的统一认证格式，DeepSeek 服务器读到它就知道你是谁、有没有额度。
- `Content-Type` 头告诉服务器请求体是 JSON 格式。
- `json={...}` 是 httpx 的便捷参数，自动把字典序列化成 JSON 并设置好格式头。请求体的三个字段正是前面讲的 `model`、`messages`、`stream`。
- `stream` 为 `False`，表示全部想完再一次性返回。
- `raise_for_status()` 在 HTTP 状态码非 2xx 时抛出带状态码的异常。401 表示 Key 错误，429 表示请求过频，502 表示服务端问题。不检查状态码是新手最常见的坑，请求失败时直接往下解析，会得到一个莫名其妙的 KeyError。
- 返回路径 `data["choices"][0]["message"]["content"]` 对应前面的响应结构，逐层取到回答文本。

调用它只有一行：

```python
client = DeepSeekClient()
answer = client.chat(HISTORY)   # HISTORY 是一个 list[Message]
```

运行效果（演示 1）：

```
完整回答：流式输出是边生成边传输数据，无需等待全部完成后才显示，像打字机一样逐字呈现结果。
```

到这里，客户端已经能完成一次完整的问答。但这行文字要等模型全部生成后才会出现，回答较长时，用户在等待期间看不到任何内容。下一节改用流式调用解决这个问题。

## 1.4 流式调用：边生成边显示

模型生成文字是逐字算出来的，一段几百字的回答可能要花 20 秒以上。非流式模式下，这 20 秒里用户什么都看不到。网页版聊天没有这个问题，答案一个字一个字往外蹦，因为网页版用的是流式接口。

使用流式接口时，请求中的 `stream` 设为 `true`。服务器不再一次返回完整回答，而是在生成过程中持续发送小片段，直到回答结束。这些小片段在接口中常称为 chunk，后文统一称为“分片”。一段 500 字的回答可能被拆成几十个分片，每个分片只包含少量新内容。

服务器持续推送数据用的是什么协议？答案是 SSE，Server-Sent Events，服务器推送事件。普通的 HTTP 响应是一问一答，服务器把完整响应体发完就关闭连接。SSE 的响应不同：服务器发完响应头后保持连接不关，之后持续不断地写入一条条格式如下的数据：

```
data: {"choices":[{"delta":{"content":"流式"}}]}

data: {"choices":[{"delta":{"content":"输出"}}]}

data: [DONE]
```

三条规律：

1. 每条推送以 `data: ` 开头，后面跟一段文本，空行分隔。
2. 内容是增量而不是全量，每块只带新生成的一小段文字，所以取值路径是 `choices[0].delta.content`，与 1.3 节的 `message.content` 只差一个字段名。
3. 结束信号是固定的一行 `data: [DONE]`，收到它表示模型说完了。

`httpx-sse` 库负责连接管理、按空行切分事件并移除 `data: ` 前缀，调用方直接读取 `event.data` 中的内容。

流式连接的大部分时间都在等待新数据。如果始终占用同步执行流，程序就难以同时处理其他任务。因此本节引入两个 Python 概念。这里只需要先理解它们的用途，后续章节还会继续使用。

async / await 是异步。同步程序的执行流是一条直线：调用函数，等它返回，继续。网络等待期间，程序整个停住。异步程序把等待和执行分开，遇到 `await` 时先挂起当前任务，事件循环去处理其他任务，数据到了再回来继续。本书不要求掌握事件循环的实现细节，只要记住写法：异步函数用 `async def` 定义，调用时用 `await`，运行入口用 `asyncio.run(...)`。

`yield` 用来构造生成器。函数执行到 `yield` 时，会把当前值交给调用方并暂停；调用方下一次迭代时，函数再从暂停处继续。这样，程序收到一个分片后就能立刻处理，例如打印到屏幕，而不用等待整个回答生成完毕。

下面是 `stream()` 的完整实现：

```python
import json

import httpx
from httpx_sse import aconnect_sse


class DeepSeekClient:
    # 上一节的字段和 chat() 略

    async def stream(self, messages: list[Message]):
        completed = False
        async with httpx.AsyncClient(timeout=60) as client:
            async with aconnect_sse(
                client,
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": True,
                },
            ) as event_source:
                async for event in event_source.aiter_sse():
                    if event.data == "[DONE]":
                        completed = True
                        break
                    payload = json.loads(event.data)
                    delta = payload["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece
        if not completed:
            raise RuntimeError("流式响应在 [DONE] 之前中断，拒绝保存不完整消息")
```

逐段理解：

- `async with httpx.AsyncClient(...)` 是异步版 HTTP 客户端。`async with` 与普通 `with` 作用相同，用完自动关闭，区别是进出块时可以 `await` 等待。
- `aconnect_sse(...)` 是 httpx-sse 提供的异步 SSE 连接入口。第一个参数是 http 客户端，然后是请求方法和地址，后面与 `chat()` 一样传 `headers` 和 `json`，注意 `stream` 为 `True`。
- `async for event in event_source.aiter_sse()` 每收到一条 SSE 事件迭代一次，`event.data` 就是去掉 `data: ` 前缀后的文本。
- `[DONE]` 是 DeepSeek 的正常结束标记。实现会单独记录它是否出现；SSE 连接断开但没有 `[DONE]` 时必须报错，不能把已经收到的不完整回答当作完整消息。
- `payload["choices"][0].get("delta", {})` 用 `get` 而不是下标，个别事件里可能没有 `delta` 字段，比如只带用量统计的事件，用 `get` 安全地给空字典。
- `if piece: yield piece` 只产出非空文本。有些 delta 的 `content` 是 `None`，比如流刚开始时的元信息，跳过它们。

调用方用 `async for` 消费生成器，配合 `flush=True` 让文字立刻上屏：

```python
async for piece in client.stream(HISTORY):
    print(piece, end="", flush=True)
```

`flush=True` 会立即刷新输出缓冲区。省略它时，文字可能暂存在缓冲区中，终端无法及时呈现逐步输出的效果。

## 1.5 组装：历史只存完整消息

流式输出缩短了等待第一段内容的时间，但也带来了历史记录问题。先比较两种直接处理分片的方式：

- 每收到一个 chunk 就往历史里塞一条消息。下一轮请求将带着几十条碎消息发给模型，token 浪费还在其次，模型看到的历史会异常混乱。
- 流在中途断掉。如果历史里已经塞了半句话，程序必须知道它是正常完成、提供方失败，还是用户主动取消；否则后续请求会把来源不明的半个回答当成完整历史。

解决办法是把“实时展示”和“写入历史”分开。每个流式分片到达时立即显示，同时暂存在组装器中；只有收到正常结束标记后，组装器才生成一条完整消息。教学版还没有用户取消入口，因此采用较保守的规则：只要缺少 `[DONE]`，就不生成可写入历史的消息。官方实现还会区分请求失败和用户主动取消，并在取消时按规则保留已经展示的有效内容，具体差异放在章末说明。

```python
class DeepSeekClient:
    # 前两节略

    async def stream_message(self, messages: list[Message]) -> Message:
        pieces: list[str] = []
        async for piece in self.stream(messages):
            pieces.append(piece)
        return Message(role="assistant", content="".join(pieces))
```

- `pieces` 列表暂存所有分片，`"".join(pieces)` 把它们按顺序拼成完整文本。
- 返回的是 `Message` 实例，`frozen=True` 保证这条消息从此不可篡改。
- 流中途抛异常，或连接结束前没有收到 `[DONE]` 时，函数都以异常结束，半成品消息根本不会产生。这就是历史只存完整消息的落地方式。

这段实现虽然很短，却建立了后续章节都会遵守的约束。从第 02 章起，历史中还会出现工具调用和工具结果，但“分片用于展示，完整消息才写入历史”的规则不会改变。

运行效果（演示 3）：

```
进入历史的消息：role='assistant', 长度=45 字
消息内容：流式输出是指数据或内容在生成过程中分块、连续地传送给接收方，而非等待全部完成后一次性输出。
```

要验证 `frozen` 的约束，可以打开 `demo.py`，恢复最后一段被注释掉的 `message.content = "篡改"`，再运行一次。程序会抛出 `FrozenInstanceError`。

## 本章小结

- `load_api_key()`：优先读取环境变量，再使用 `.env`，缺少密钥时明确报错
- `Message`：`frozen` 数据类，从语言层面保证历史消息不可篡改
- `DeepSeekClient.chat()`：非流式调用，发历史、等全量、解析 `choices[0]`
- `DeepSeekClient.stream()`：SSE 流式调用，异步生成器逐 chunk 产出增量文字
- `DeepSeekClient.stream_message()`：组装分片为完整 `Message`

## 对照官方

官方用 TypeScript 完成了同样的事，两处代码点开对照：

| 官方代码 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/llm/llm-deepseek/src/adapter.ts`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/llm/llm-deepseek/src/adapter.ts) | `DeepSeekClient.stream()` | DeepSeek 适配器解析 SSE，并把模型服务报告的正常结束、错误与中止转换成明确的结束原因 |
| [`packages/llm/llm/src/assembler.ts`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/llm/llm/src/assembler.ts) | `stream_message()` | `BlockAssembler` 正常结束时组装完整块；显式取消时只保留非空文本/思考前缀，工具调用不进入中断消息 |

教学版只处理文本。官方组装器还处理思考与工具调用分片，DeepSeek 适配器也能为支持图片的模型解析持久化附件并限制请求中的图片总量；这些多模态能力不在本章范围内。工具调用分片留到第 02 章展开，取消与错误的完整生命周期留到持续运行的智能体中处理。

## 练习

1. 流式调用让用户更早看到内容，却让消息状态和错误处理变得更复杂。请从首字延迟、完整性、取消和历史记录四个方面比较流式与非流式调用，并说明各自更适合什么场景。
2. 假设这套客户端用于代码生成或医疗问答。连接中断时，界面可以保留已经显示的内容，也可以全部撤回。请选择一种策略，并说明它如何避免用户把不完整回答误认为最终结论。
3. 为一次模型调用设计清晰的结束状态，至少区分正常完成、长度截断、用户取消、HTTP 错误和传输中断。哪些状态可以产生一条进入历史的 assistant 消息，哪些只能留下诊断信息？
4. 扩展本章 demo，让它在流式结束后报告结束原因，并只把符合你所定义完整性规则的消息写入历史。使用模拟分片或真实请求验证正常结束和一种异常结束，记录两条路径的可见输出与历史差异。
