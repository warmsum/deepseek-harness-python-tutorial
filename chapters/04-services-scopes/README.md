# 04｜服务与依赖

> 预计时间：70 分钟 ｜ 前置：完成第 03 章 ｜ 本章纯本地运行，不调用模型

第 03 章的插件系统已经能够安装和清理插件，但插件之间只能发送事件，还不能直接使用其他插件提供的能力。完整智能体中，模型连接和工具注册表可能由不同插件提供，运行循环却需要同时使用二者。本章解决插件怎样声明、等待和获取这些共享能力。

cordis 使用“依赖注入”解决这个问题。插件先声明自己需要哪些能力，这些可以被其他插件使用的能力称为服务。运行环境负责查找服务：所需服务尚未出现时，插件先等待；服务准备好后，插件再启动。本章将实现并验证三个行为：

1. 服务后到，插件自动醒来；
2. 提供者被卸载，依赖方自动卸载；
3. 使用服务前必须声明依赖，未声明时程序直接报错。

本章最后还会实现一种按顺序包裹操作的处理链，官方称为 waterfall。权限、超时和日志插件可以在这条链上处理同一次工具调用，而不必修改工具本身。

## 学习目标

完成本章后，你将能够：

- 使用 `provide` 注册服务，并让服务随提供者一起卸载；
- 使用 `inject` 声明依赖，理解插件的等待、启动和重新加载；
- 通过 `__getattr__` 阻止插件读取未声明的服务；
- 理解按顺序执行的处理链如何让多个插件依次包装同一次工具执行。

## 4.1 两种共享方式，为什么选依赖注入

插件 A 提供了模型连接，插件 B 要用它。最直白的做法是全局变量：

```python
llm_client = None  # 模块级全局变量

def plugin_a(ctx, _config):
    global llm_client
    llm_client = DeepSeekClient()

def plugin_b(ctx, _config):
    llm_client.chat(...)  # 直接用全局变量
```

这个写法能够运行，但存在三个问题：

1. 顺序耦合。B 必须比 A 晚安装，而且 B 隐含依赖“A 已经安装完成”。插件数量增加后，安装顺序很难维护。
2. 无法卸载。A 被卸载后，`llm_client` 该不该清空？B 还指着它呢。
3. 无法替换。测试时如果要给 B 换成模拟模型，只能修改共享的全局变量，也会影响其他使用者。

依赖注入把“寻找服务”和“使用服务”分开。B 不主动寻找 A，只声明自己需要名为 `llm` 的服务；运行环境在服务就绪后启动 B，并把服务交给它。B 不需要知道服务由哪个插件提供，也不需要依赖固定的安装顺序。若要更换实现，先卸载旧的提供者，再注册新的提供者；同名服务不会被悄悄覆盖。

为什么读取服务前也要检查声明？如果任何插件都能随意取得任何服务，真实的依赖关系就会散落在各处。卸载一个服务时，运行环境无法判断哪些插件会受到影响。强制声明以后，每个插件依赖什么都有明确记录，环境才能正确安排启动和卸载。

## 4.2 provide 与 get

先给环境加一张服务表。`provide` 注册服务，`get` 非严格查找：

```python
class Context:
    def __init__(self) -> None:
        # 第 03 章字段略
        self._services: dict[str, tuple[object, int, int]] = {}
        self._version = 0

    def provide(self, name: str, value: object) -> Disposer:
        if name in self._services:
            raise ValueError(f'服务 "{name}" 已被注册')
        self._version += 1
        provider_uid = self._current.uid if self._current is not None else 0
        registration = (value, provider_uid, self._version)
        self._services[name] = registration
        self._notify()

        def unregister() -> None:
            if self._services.get(name) is registration:
                del self._services[name]
                self._notify()

        if self._current is not None:
            self._current.collect(unregister)
        return unregister

    def get(self, name: str) -> object | None:
        impl = self._services.get(name)
        return impl[0] if impl is not None else None
```

服务表的值是一个三元组 `(value, provider_uid, version)`，后两项为 4.3 节的依赖重算准备，这里先记住含义：

- `provider_uid` 是谁提供的。第 03 章给每个句柄发过全局唯一的 uid。
- `version` 是第几次 provide。每次 `provide` 递增，同名服务被重新提供时 version 变化，依赖方据此知道自己手里的服务过期了。

`provide` 会把经过一次性包装的注销函数登记到当前插件名下。提供者被卸载时，服务也随之注销。注册时若名称已经存在会立即报错，调用方必须先卸载旧的提供者；注销时还会检查服务表里仍是自己的那条记录，避免误删后来注册的服务。

`_notify()` 遍历全部句柄重算依赖，是整套机制的中枢：

```python
    def _notify(self) -> None:
        for handle in list(self._handles):
            handle._recheck()
```

## 4.3 声明依赖并等待服务

现在给插件增加声明依赖的能力。实现使用函数属性 `inject` 保存服务名称：

```python
def agent(ctx, _config):
    print(f"启动！llm={ctx.llm} tools={ctx.tools}")

agent.inject = ["llm", "tools"]
```

句柄构造时读取这个声明，然后重算依赖：

```python
class PluginHandle:
    def __init__(self, ctx, plugin, config):
        # 第 03 章字段略
        self.inject = tuple(dict.fromkeys(getattr(plugin, "inject", ())))
        self._store: dict[str, object] = {}   # 依赖快照
        self._epoch: str | None = None        # 依赖签名
        # 其余同第 03 章
        self._recheck()  # 第 03 章这里是直接 _run()
```

`_recheck` 负责比较依赖状态并决定插件下一步行为：

```python
    def _recheck(self) -> None:
        if self.state == "disposed":
            return
        resolved: dict[str, object] = {}
        tokens: list[str] = []
        for name in self.inject:
            impl = self._ctx._services.get(name)
            if impl is not None:
                resolved[name] = impl[0]
                tokens.append(f"{impl[1]}:{impl[2]}")  # uid:version
            else:
                tokens.append("-")
        epoch = ",".join(tokens)
        if epoch == self._epoch:
            return
        self._epoch = epoch

        was_active = self.state == "active"
        if was_active:
            self._unload()

        missing = any(name not in resolved for name in self.inject)
        if missing:
            self.state = "pending"
        else:
            self._store = resolved
            self._run()
```

逐段推演这套逻辑：

第一步，计算依赖签名。每个服务用 `uid:version` 表示，服务缺失时记为 `-`，最后拼成 `_epoch`。这个字符串记录了插件当前使用的是哪些服务版本。

第二步，签名没变就返回。`_notify` 会在每次 provide 和注销时叫醒全部句柄，签名比较保证只有真正受影响的插件才会动作。没有这一步，每次 provide 都会导致全体插件重装。

第三步，签名变化后更新插件状态。依赖齐全时，先卸载旧状态，再保存新的服务并重新执行插件函数；依赖缺失时，插件回到 `pending` 等待。两条路径覆盖了本章开头的两个结果：

- 服务后到自动启动。agent 声明依赖 tools，tools 未提供时签名是 `"uid:ver,-"`，句柄保持 pending；tools 出现后签名变成 `"uid:ver,uid:ver"`，环境重新检查依赖并启动 agent。
- 提供者卸载自动退场。tools 提供者被卸载，第 03 章的 effect 机制自动注销服务，签名里的 tools 变回 `-`，agent 卸载自己回到 pending。

依赖快照 `_store` 保存环境为插件解析出的服务，4.4 节的严格访问会从这里读取。`inject` 使用去重后的 tuple，而不是 set：依赖顺序稳定，epoch 也就不会因哈希顺序变化而抖动。

## 4.4 __getattr__：读服务必须先声明

Python 对象访问不存在的属性时，解释器会调用 `__getattr__`。这给了我们一个实现依赖显式化的机会，与官方 cordis 用 Proxy 拦截属性读取异曲同工：

```python
    def __getattr__(self, name: str) -> Any:
        handle = self._owner or self._root_context()._current
        if handle is not None:
            if name in handle._store:
                return handle._store[name]
            if name in handle.inject:
                raise AttributeError(f'服务 "{name}" 已声明依赖但尚未就绪')
        if name in self._services:
            raise AttributeError(f'读取服务 "{name}" 前必须在 inject 里声明')
        raise AttributeError(f"Context 没有属性 {name!r}")
```

三种结局：

1. 声明过且已就绪，返回快照里的服务值，这是 `ctx.llm` 的常规路径；
2. 声明过但没就绪，报已声明依赖但尚未就绪，插件在 pending 期间误读服务时，这个错误能立刻指出问题所在；
3. 根本没声明，报必须先 inject，即使服务明明存在。

第三种情况说明“先声明再使用”不是一条需要开发者自觉遵守的约定，而是程序会主动检查的规则。示例中，`llm` 服务虽然已经存在，未声明依赖的代码仍然无法读取它。

事件回调在插件安装结束后仍可能读取服务，因此每个插件获得的 `Context` 视图会记住自己的身份。回调再次访问 `ctx.llm` 时，仍会使用该插件的 `inject` 声明和依赖快照，而不是依赖仅在安装期间有效的 `_current` 字段。

## 4.5 按顺序执行的处理链（waterfall）

第 03 章的 `emit` 只能广播通知，监听器不能阻止操作，也不能修改数据。工具执行则需要更强的机制：权限插件可以拒绝调用，超时插件可以限制时间，日志插件可以记录结果，而且它们都不应修改工具的核心代码。

cordis 把这种处理链称为 waterfall。多个监听器依次包裹最内层的执行器：请求从最外层进入，逐层到达真正执行操作的函数；函数返回后，结果再沿原路径逐层返回：

```mermaid
flowchart LR
    A[调用] --> L1[监听器1] --> L2[监听器2] --> C[核心执行器]
    C --> R2[监听器2 收尾] --> R1[监听器1 收尾] --> B[结果]
```

实现，核心只有十行：

```python
    def waterfall(self, event: str, *args: Any) -> Any:
        *call_args, next_fn = args
        listeners = list(self._listeners.get(event, []))

        def dispatch(index: int, *inner_args: Any) -> Any:
            if index >= len(listeners):
                return next_fn(*inner_args)
            listener = listeners[index]

            def deeper(*new_args: Any) -> Any:
                return dispatch(index + 1, *(new_args if new_args else inner_args))

            return listener(*inner_args, deeper)

        return dispatch(0, *call_args)
```

三个约定：

1. 最后一个参数 `next` 表示下一层处理函数，链的最里面是真正的执行器。
2. 监听器调用 `next()` 才会继续执行；不调用就能阻止这次操作，权限插件可以利用这一点拒绝越权请求。
3. `next()` 不带参数时继续传递原参数，带参数时则把修改后的参数交给下一层。

典型用法，一个超时策略插件，给所有工具执行加上日志，不改核心一行：

```python
def timeout_policy(c: Context, _config) -> None:
    def wrap(exec_: dict, next_: object) -> str:
        print(f"开始执行工具 {exec_['name']}")
        result = next_()
        print(f"工具 {exec_['name']} 完成")
        return result

    c.on("tools/execute", wrap)
```

执行时：

```python
result = ctx.waterfall("tools/execute", {"name": "calculator"}, core_executor)
```

外层 `wrap` 先打印开始，再调用 `next()` 进入 `core_executor`。核心执行器返回结果后，`wrap` 打印完成，并把结果原样交回调用方。这个模式对应官方 Harness 中 `tools/pre-execute → tools/execute → tools/post-execute` 执行管线的简化形态，第 05 章会继续扩展工具相关的数据结构。

## 4.6 同名服务怎样彼此隔离

有时多个智能体会使用同一个服务名，却需要各自独立的实例。例如，每个智能体都可以有名为 `tools` 的工具注册表，但其中包含的工具不同。官方 cordis 使用 `ctx.isolate(name)` 为这类服务划分作用域，使不同插件能够找到各自所属的实例。

教学版不实现同一插件树内的服务隔离，而是为每个独立运行单元创建新的 `Context`。这种做法功能较少，但边界更容易观察。第 14 章会使用相同思路，为每个子智能体创建独立会话。官方的完整作用域机制放在章末对照表中。

## 4.7 运行完整示例

```bash
uv run python chapters/04-services-scopes/src/demo.py
```

完整输出，本地确定性运行：

```
=== 时刻 1：服务后到，插件自动醒来 ===
  [llm-provider] 已提供 llm 服务
  [agent] 当前状态: pending   ← 依赖不齐，安静等待
  [agent] 启动！llm={'provider': 'deepseek', 'model': 'deepseek-chat'} tools={'calculator': 'safe-eval'}
  [tools-provider] 已提供 tools 服务
  [agent] 当前状态: active      ← 依赖齐了，自动启动！

=== 时刻 2：提供者被卸载，依赖方自动卸载 ===
  重名服务被拒绝: 服务 "tools" 已被注册
  卸载 tools v1 后 [agent] 状态: pending
  [agent] 启动！llm={'provider': 'deepseek', 'model': 'deepseek-chat'} tools={'calculator': 'v2'}
  [tools-provider-2] 已提供 tools v2
  注册 tools v2 后 [agent] 状态: active
  卸载 tools v2 后 [agent] 状态: pending   ← 级联卸载

=== 时刻 3：读服务必须 inject ===
  报错: 读取服务 "llm" 前必须在 inject 里声明
  ← 依赖显式化不是约定，是语法

=== 时刻 4：waterfall 瀑布 ===
  [timeout-policy] 开始执行工具 calculator
  [core] 真正执行 calculator……
  [timeout-policy] 工具 calculator 完成
  最终结果: 计算结果: 42
```

时刻 2 中，v2 不能直接覆盖 v1。先卸载 v1 后，agent 回到 `pending`；随后注册 v2，新的提供者编号和版本改变了依赖签名，agent 才使用新服务重新启动。这样，每个服务始终有明确的所有者，卸载旧提供者时也不会误删新服务。

## 本章小结

完整示例没有实现同一插件树内的作用域隔离、YAML 加载和热重载，但已经保留“定义服务、提供服务、使用服务”这条主要协作路径。

- `provide` 与 `get`：重名拒绝、精确注销、服务表与提供者生命周期联动
- `inject` 与 `_recheck`：记录依赖、等待缺失服务，并在服务变化后重新启动
- 绑定插件身份的 `Context` 视图：回调在安装结束后仍按自己的声明读取服务
- `__getattr__`：读服务必须先声明的语法级约束
- 按顺序执行的处理链 `waterfall`：可以在操作前后处理数据，返回值沿原路径传回
- 服务作用域：本章使用独立 `Context` 隔离运行单元，没有实现官方在同一插件树中的精细隔离

插件怎样组织能力的问题暂时解决了。从第 05 章起，课程回到智能体的运行过程，先处理对话和工具调用产生的历史记录。

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`vendor/cordis/src/reflect.ts`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/vendor/cordis/src/reflect.ts) | `provide` 与 `_notify` | 官方只通知名称和作用域受到影响的插件任务；教学版同样拒绝重名并记录所有者，但会重新检查全部依赖 |
| [`vendor/cordis/src/fiber.ts`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/vendor/cordis/src/fiber.ts) | `_recheck` | 官方在 `fiber.ts` 中解析依赖并比较依赖版本；教学版用签名变化表达同一判断 |
| [`vendor/cordis/src/context.ts`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/vendor/cordis/src/context.ts) | `Context` 视图与 `__getattr__` | 官方使用 `Proxy` 保留访问服务的插件身份；Python 版使用所有者视图达到相同目的 |
| [`vendor/cordis/src/events.ts`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/vendor/cordis/src/events.ts) | `waterfall` | 官方与教学版都让监听器按注册顺序包裹内层执行器 |

## 练习

1. 对比“插件直接导入具体实现”“从全局变量读取能力”和“声明服务依赖”三种协作方式。对于需要替换模型服务的智能体，哪一种更容易测试和卸载，为什么？
2. 设计一个可替换的存储服务，至少包含一个提供服务的插件和两个使用服务的插件。说明服务尚未出现、提供者被替换以及服务再次恢复时，各使用者应处于什么状态。
3. 权限、日志、超时和重试策略都可能通过 `waterfall` 处理同一次工具执行。请安排一个合理顺序，并分析顺序变化可能带来的重复日志、越权执行或无效重试。
4. 多个智能体需要使用同名服务，但每个智能体应看到独立实例。比较“每个智能体使用独立 `Context`”和“同一插件树内使用服务作用域”两种方案的优缺点，并说明教学版为什么选择前者。
5. 编写一个不修改核心执行器的工具策略插件，可以选择审计、权限或结果转换中的一种能力。验证插件安装时策略生效、卸载后执行路径恢复，并说明它依赖哪些服务或事件。
