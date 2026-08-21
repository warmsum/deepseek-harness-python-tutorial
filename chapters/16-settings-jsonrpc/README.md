# 16｜配置与 RPC

> 预计时间：55 分钟 ｜ 前置：完成第 15 章 ｜ 本章通过 JSON-RPC 调用真实 DeepSeek 模型

到第 15 章为止，智能体的运行流程和主要能力已经分别实现。要把它交给 IDE 插件、命令行工具或网页前端使用，还需要一个稳定的外部入口，让其他进程能够发送消息、查询状态并读取结果。这带来本章的两个主题：

1. 配置：外部入口和智能体怎样共享同一套设置，避免各自使用不同值？
2. RPC：两个进程怎样用统一格式发起调用、返回结果和报告错误？

RPC 是 Remote Procedure Call 的缩写，中文通常称为远程过程调用。它让一个进程能够像调用本地函数一样，请求另一个进程执行某个方法。本章使用结构简单、资料丰富的 JSON-RPC 2.0 讲解基本原理。官方采用的是 Typert RPC，两者并不兼容，具体差异放在章末说明。

## 学习目标

完成本章后，你将能够：

- 按命名空间注册配置，并按“程序默认值、当前组合的基础值、用户设置”依次合并；
- 使用不可变快照和版本号防止配置被误改或被旧数据覆盖；
- 读懂 JSON-RPC 2.0 的请求、成功响应和错误响应；
- 实现请求解析、方法注册与分发；
- 把不可信输入转换成结构化协议错误，而不是让服务进程退出；
- 让 `agent.run` 从当前配置读取模型和系统提示词，再返回真实模型结果。

## 16.1 配置为什么要分组和分层

完整的智能体由许多插件组成。模型插件可能有 `model`，搜索插件也可能有 `model`；如果所有配置项都放进一个大字典，很快就无法判断某个字段属于谁。为此，每个插件使用自己的命名空间 `namespace`，例如 `agent` 或 `web-search-deepseek`，再在这个分组内合并三层配置：

```
程序默认值  <  当前组合的基础值  <  用户设置
```

- 程序默认值由插件提供，使缺少用户配置时仍有合理行为；
- 基础值由当前智能体组合提供，用来统一同一运行方式下的设置；
- 用户设置保存在配置文件中，优先级最高。

三层配置会递归合并。用户只修改 `agent.model` 时，不会一并删除基础配置中的 `max_steps`。调用 `replace({})` 则会清空当前命名空间的用户设置，让最终结果重新继承基础值和默认值。

## 16.2 用 Settings 注册、读取和修改配置

```python
class Settings:
    def register(self, namespace, *, defaults=None, base=None, validate=None):
        if namespace in self._registrations:
            raise ValueError(f'namespace "{namespace}" 已注册')
        # 保存三层信息，并先验证当前解析值
        ...
        return SettingsScope(self, namespace)

    def get(self, namespace):
        return _freeze(self._resolved(namespace))

    def _resolved(self, namespace):
        registration = self._require(namespace)
        return _merge(
            _merge(registration.defaults, registration.base),
            self._user_section(namespace),
        )
```

注册时会拒绝格式错误或重复的命名空间；已有用户设置如果未通过校验，注册本身也会失败。`get()` 返回一份与内部状态分离且不可修改的快照：外部不能修改字典，嵌套列表也会转换成元组。

写入有两种形态：

- `update(patch)`：把需要变化的字段递归合并到用户设置中；
- `replace(section)`：整体替换这个命名空间下的用户设置。

每个命名空间有独立且只增不减的版本号 `revision`。假设配置界面先读到版本 4，准备保存时其他调用方已经写入版本 5，那么带 `expected_revision=4` 的更新会返回冲突错误，而不是覆盖新值。这与第 13 章保护目标更新的方法相同：先确认自己读取的版本仍然是最新版本，再提交修改。

写入前还会检查数据能否无损转换成 JSON，拒绝循环引用、非字符串键、`NaN`、无穷值、负零、过大的整数和其他 JSON 不支持的类型。`watch()` 用于监听配置变化，返回的取消函数可以安全地重复调用；最终配置真正发生变化时，回调会同时收到新值和旧值。

## 16.3 JSON-RPC 2.0：跨进程调用的最小语言

进程 A 调用进程 B 的方法时，进程间传输的是文本。JSON-RPC 2.0 使用请求与响应两类 JSON 消息描述这次调用：

```json
{ "jsonrpc": "2.0", "id": 1, "method": "settings.get", "params": {"namespace": "agent"} }
{ "jsonrpc": "2.0", "id": 1, "result": {"model": "deepseek-chat"} }
```

- 请求：`method` 表示要调用的方法，`params` 保存参数，`id` 是本次调用编号；响应会原样带回编号，使调用方能够对应请求与结果。
- 响应：成功时包含 `result`，失败时包含 `error`。调用失败也应返回符合协议的消息，而不是让服务进程直接退出。

JSON-RPC 为常见错误规定了编号：-32700 表示 JSON 解析失败，-32600 表示请求结构不合法，-32601 表示方法不存在，-32602 表示参数不合法。调用方可以根据编号分类处理，不需要分析自然语言错误文本。

`RpcError` 还带一个可选的 `request_id`。彻底无法解析、无法确认请求身份时响应 id 为 null；请求结构和 id 已经有效、只是 params 类型错误时，错误响应必须原样带回该 id，调用方才能把失败对应到正确请求。

## 16.4 用 RpcDispatcher 解析和分发请求

```python
def parse_request(text: str) -> RpcRequest | RpcError:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return RpcError(PARSE_ERROR, "Parse error: 不是合法 JSON")
    if not isinstance(data, dict):
        return RpcError(INVALID_REQUEST, "Invalid Request: 请求必须是 JSON 对象")
    if data.get("jsonrpc") != "2.0":
        return RpcError(INVALID_REQUEST, "Invalid Request: jsonrpc 必须为 2.0")
    method = data.get("method")
    if not isinstance(method, str) or not method:
        return RpcError(INVALID_REQUEST, "Invalid Request: method 缺失")
    params = data.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return RpcError(
            INVALID_PARAMS,
            "Invalid params: 必须是 JSON 对象",
            request_id=data.get("id"),
        )
    return RpcRequest(jsonrpc="2.0", id=data.get("id"), method=method, params=params)
```

本章遵循同一条规则：校验失败时返回结构化错误。RPC 是进程边界，收到的文本可能来自输入错误、异常客户端或恶意构造，解析层需要把这些情况都转换成合法响应。dispatch 的路由部分采用相同处理：

```python
    def dispatch(self, text: str) -> dict[str, Any]:
        parsed = parse_request(text)
        if isinstance(parsed, RpcError):
            return _error_response(parsed.request_id, parsed)
        handler = self._handlers.get(parsed.method)
        if handler is None:
            return _error_response(
                parsed.id, RpcError(METHOD_NOT_FOUND, f"Method not found: {parsed.method}")
            )
        try:
            result = handler(parsed.params)
        except TypeError as error:
            return _error_response(
                parsed.id, RpcError(INVALID_PARAMS, f"Invalid params: {error}")
            )
        except Exception as error:
            return _error_response(
                parsed.id, RpcError(INTERNAL_ERROR, f"Internal error: {error}")
            )
        return {"jsonrpc": "2.0", "id": parsed.id, "result": result}
```

这里区分解析失败、方法不存在、参数错误和处理器内部异常。所有路径都会返回 JSON-RPC 响应，使外部程序不会因为一次错误请求而失去连接。

## 16.5 让 RPC 方法读取配置并调用模型

只注册一个 `echo` 方法能够讲清分发格式，却看不到配置和智能体入口怎样真正连接。示例因此注册 `agent.run`：先校验外部传入的 `prompt`，再从 `agent` 命名空间读取模型与系统提示词，最后调用 DeepSeek。

```python
def run_agent(params: dict[str, Any]) -> dict[str, Any]:
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise TypeError("prompt 必须是非空字符串")
    config = agent_settings.get()
    model = config["model"]
    system_prompt = config["system_prompt"]
    content = DeepSeekClient(model).answer(system_prompt, prompt)
    return {
        "model": model,
        "settings_revision": agent_settings.revision,
        "content": content,
    }
```

这里有两条边界。第一，RPC 参数是不可信输入，缺少 `prompt` 时抛出的 `TypeError` 会由分发器转换成编号为 -32602 的参数错误。第二，模型和系统提示词来自已经解析的配置快照，响应同时返回模型名与配置版本，调用方能够知道这次回答依据哪一版设置。

`client.py` 只保留本章需要的最小非流式请求。它读取 `DEEPSEEK_API_KEY`，向 `/chat/completions` 发送系统消息和用户消息，并要求响应包含非空文本。模型请求失败时，分发器会返回内部错误；它不会把失败伪装成成功结果。

## 16.6 运行完整示例

```bash
uv run python chapters/16-settings-jsonrpc/src/demo.py
```

下面是一次真实运行。模型回答的具体措辞可能变化，JSON-RPC 外层结构保持不变：

```
=== 外部请求先读取当前配置 ===
→ {"jsonrpc": "2.0", "id": 1, "method": "settings.get", "params": {"namespace": "agent"}}
← {"jsonrpc": "2.0", "id": 1, "result": {"model": "deepseek-chat", "system_prompt": "你是简洁的 Python 教学助手，只回答一个自然段。", "language": "zh-CN"}}

=== JSON-RPC 真实驱动模型 ===
→ {"jsonrpc": "2.0", "id": 2, "method": "agent.run", "params": {"prompt": "用一句通俗的话解释 Python 生成器。"}}
← {"jsonrpc": "2.0", "id": 2, "result": {"model": "deepseek-chat", "settings_revision": 1, "content": "生成器就像个“懒人列表”——它不会一次性把所有数据都算好存起来，而是按需一个接一个地“现做现卖”，这样既省内存又省时间，特别适合处理海量数据或无限序列。"}}

=== 非法请求仍返回结构化错误 ===
→ {"jsonrpc": "2.0", "id": 3, "method": "agent.run", "params": {}}
← {"jsonrpc": "2.0", "id": 3, "error": {"code": -32602, "message": "Invalid params: prompt 必须是非空字符串"}}
```

第一条请求证明外部调用方读到的是三层配置合并后的结果。第二条请求使用其中的 `deepseek-chat` 和系统提示词完成真实模型调用，并在协议结果中带回配置版本。第三条请求缺少必要参数，分发器返回结构化错误，进程仍可继续处理后续请求。

## 本章小结

- `Settings`：按命名空间注册配置，并依次合并默认值、基础值和用户设置
- 安全写入：返回不可变快照，校验 JSON 数据，并用版本号发现并发冲突
- `parse_request`：检查 JSON-RPC 2.0 请求格式，并把失败转换成结构化错误
- `RpcDispatcher`：注册方法、查找处理函数并返回调用结果
- 标准错误码：让调用方稳定地区分不同失败原因
- `agent.run`：从配置快照读取模型参数，通过统一 RPC 响应返回真实模型结果

至此，各项能力和外部入口都已经准备好。第 17 章不再单独增加一种工具，而是把前 16 章的组件装进同一个命令行智能体。

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/api/gateway/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/api/gateway/README.zh.md) | `RpcDispatcher` | 官方的 Typert 接口会根据方法描述检查具名参数和返回值；教学版使用较小的 JSON-RPC 分发器讲解协议边界 |
| [`packages/settings/settings/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/settings/settings/README.zh.md) | `Settings` | 与官方一样使用命名空间和三层配置，返回不可修改的快照，并检查更新版本与 JSON 数据 |

官方还支持按路径修改配置、敏感字段脱敏、按顺序异步通知观察者、可替换的写入后端、文件热重载和安全卸载；教学版不实现这些工程能力。官方使用 Typert 而不是 JSON-RPC，两端共享方法描述，并据此校验参数和返回值。教学版的手写 JSON-RPC 只保留逐行消息、错误格式和输入校验，不与 Typert 协议兼容。

## 练习

1. 一项配置同时出现在程序默认值、当前组合的基础值和用户设置中时，最终值如何确定？请设计一个包含嵌套字段的例子，并说明 `update` 与“清空用户设置、重新继承默认值”为什么是不同操作。
2. IDE 和命令行同时读取同一版本后分别修改配置。设计一次冲突处理过程，说明版本检查为什么比后写覆盖更适合可观察的智能体设置。
3. 为一个外部控制界面设计最小 RPC API，至少覆盖提交任务、查询状态和读取结果。定义成功响应、参数错误、未知方法和内部失败分别应向客户端暴露什么。
4. stdio、HTTP 和 WebSocket 都能承载 RPC。比较它们在部署、并发、双向通知、认证和调试方面的取舍，并说明本章为什么使用逐行 stdio。
5. 选择一项对外有用的操作，为 `RpcDispatcher` 增加完整方法：校验不可信参数，调用领域逻辑，返回稳定结果，并把预期失败转换成结构化错误。为成功、参数错误和方法内部失败各写一个协议示例。
