# 17｜命令行智能体的完整组装

> 预计时间：90 分钟 ｜ 前置：完成第 01–16 章 ｜ 命令行任务调用真实 DeepSeek 模型

前 16 章分别实现了模型调用、工具、会话、文件操作、任务管理和子智能体等能力。本章把它们接到同一个程序中，观察一项任务怎样从输入开始，经过模型与工具的多轮协作，最终保存会话并返回结果。

最终示例采用命令行运行方式，官方称为 headless。它不启动网页、桌面界面或 HTTP 服务：普通模式从命令行接收一项任务，完成后把最后一条模型回复写到标准输出；`--rpc` 模式则通过标准输入和标准输出逐行收发 JSON-RPC，供其他进程调用。

## 学习目标

完成本章后，你将能够：

- 用 `Context`、服务和 `Bundle` 组装一个插件化智能体；
- 区分程序内部的插件能力与模型可以调用的工具；
- 说明第 10–16 章的能力分别接入运行过程中的哪个位置；
- 跟踪保存、重试、结果裁剪和外部存储在一个步骤中的先后顺序；
- 区分一次性任务出口与持续 JSON-RPC 入口；
- 识别 Python 教学实现与官方命令行运行组件之间的差异。

## 17.1 运行入口

安装依赖后，可以直接运行一次任务：

```bash
uv run mini-harness "查看当前项目并总结入口"
```

也可以从本章源码目录运行：

```bash
PYTHONPATH=chapters/17-headless-capstone/src \
  uv run python -m mini_harness "查看当前项目并总结入口"
```

程序把最终回答写到标准输出。最后一条 `turn/end` 的 `reason` 为 `completed` 时，进程退出码是 0，否则是 1；运行失败会把错误写到标准错误。`--help` 只显示帮助，不会启动 Agent 或读取 API Key；没有提供任务时打印用法并以退出码 1 结束。

JSON-RPC 使用另一条入口：

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"settings.get","params":{"namespace":"agent"}}' \
  | DEEPSEEK_API_KEY=... uv run mini-harness --rpc
```

`--rpc` 每读取一行请求就写出一行响应。目前提供 `settings.get`、`agent.run` 与 `plan.set` 三个方法，实现的是第 16 章分发器的标准输入输出版本，不包含完整的宿主服务或网络 API 代理。

## 17.2 一切皆插件

Cordis 内核本身不实现模型调用、文件读写或任务管理。它只负责安装插件、连接插件依赖、分发事件，以及在卸载时清理资源。模型、工具、会话、提示词、运行循环、文件系统、技能和任务调度都由插件提供或扩展。这就是“一切皆插件”：内核保持精简，具体能力通过明确的扩展位置组合起来。

插件不一定是模型能够调用的工具。`skill`、`ask_user_question` 和 `job_output` 会出现在模型的工具清单中；模型连接、会话、自动重试、关键节点保存和上下文裁剪等插件只在程序内部工作，不应出现在工具清单中。最终示例一共向模型注册 24 个工具，其他插件则负责支撑这些工具和运行循环。

一个能力通常分成三层。服务定义（Service Definition）说明调用方可以依赖哪些方法；服务提供者（Provider）给出可以替换的具体实现；服务使用者（Consumer）再把这项服务接到提示词、模型工具或运行事件上。

例如，`filesystem_provider` 提供文件操作服务，`filesystem_consumer` 把其中五项操作注册成模型工具，并加入相应的安全提示。卸载使用者后，这些工具说明和提示词会一起消失；替换服务提供者时，使用者会先停止，再使用新服务重新启动。

## 17.3 用 Context、Bundle 和 build_agent 完成组装

`build_agent()` 只做三件事：创建根 Context，挂载 Python Bundle，从 `agents` 注册表取得明确的 `main` Agent。

```python
ctx = Context()
ctx.plugin(
    headless_bundle,
    BundleConfig(
        settings_document=load_settings_document(),
        enable_console_questions=enable_console_questions,
        checkpoint_flush=checkpoint_flush,
    ),
)
agents = cast(AgentRegistry, ctx.require("agents"))
agent = agents.get("main")
if agent is None:
    raise RuntimeError("headless Bundle 未创建 main Agent")
return agent
```

`headless_bundle` 是一份明确的 Python 插件清单。它先安装配置、会话、提示词、工具注册表和模型连接，再安装文件、命令、技能、目标、计划、网络搜索、子智能体、后台任务、工作流和运行策略，最后创建 `main` Agent 并注册到 `AgentRegistry`。`agent_provider` 把用户问答、任务委派和 RPC 使用者作为子插件安装，并显式传入这个 Agent；Agent 被替换或卸载时，这些使用者会先清理。

`Agent` 的构造器不直接接收重试、持久化检查点、计划模式或上下文控制策略。运行循环只在“步骤开始前”“准备请求”“调用模型”“执行工具”和“工具返回后”等位置发布事件，相关插件监听自己关心的位置。以后新增策略时，不需要继续修改主循环。

默认 `rules` 提示词要求回答直接、客观并先给结论，说明实际结果和使用方法，只在限制影响结果判断时指出未完成或未验证内容。它同时要求避免重复免责声明和通过贬低其他方案论证当前选择。

注册工具、提示词片段和 RPC 方法时都会返回对应的取消函数。插件通过 `ctx.effect(...)` 统一登记这些函数；`agent.close()` 只需关闭根 `Context`，运行环境就会按相反顺序卸载整棵插件树。可继续子智能体、后台任务、工具、提示词、RPC 方法和服务注册都会沿这条路径清理。

## 17.4 第 10–16 章接在哪里

完整示例一共注册 24 个模型工具，包含 `calculator` 和下面 23 个第 10–16 章工具。

| 章节 | 工具或服务 | 怎样接入最终示例 |
|---|---|---|
| 10 | `read`、`write`、`edit`、`grep`、`glob` | 文件服务提供操作，使用者注册工具和安全提示 |
| 11 | `shell` | 命令服务提供执行能力，使用者注册工具；教学版没有内核沙箱 |
| 12 | `skill` | 技能目录提供内容，使用者注册按需加载工具和菜单 |
| 13 | `get_goal`、`create_goal`、`update_goal`、`todo_write` | 目标与任务清单共用当前会话保存状态 |
| 13 | `ask_user_question`、`exit_plan_mode` | 用户问答和计划模式在智能体服务就绪后接入 |
| 14 | `subagent`、`subagent_fork`、`send_message`、`interrupt_agent` | 子智能体服务负责委派、继续对话和中断 |
| 14 | `job_output`、`job_list`、`job_kill` | 后台任务按根智能体隔离 |
| 14 | `workflow` | 工作流限制并发数和总任务数 |
| 15 | `web_search`、`web_fetch` | 搜索与网页抓取分别注册成工具 |
| 16 | `Settings`、`RpcDispatcher` | 配置服务和 RPC 方法在智能体、计划服务就绪后接入 |

所有相关插件都启动后，模型工具清单共有 24 项。计划模式只改变 `plan:policy` 提示词片段；`exit_plan_mode` 始终保留在工具说明中，但在非计划模式下调用会返回错误。如果卸载某个注册工具的插件，它提供的工具会从下一次模型请求中消失。

## 17.5 一次步骤中发生了什么

运行循环负责推进步骤，各个插件在约定的位置完成自己的工作：

1. 开始新步骤前，先保存上一批事件，再让已经批准的计划模式切换生效。
2. 写入 `step/start`，把当前系统提示词记录为 `system/message`，再接收本步骤需要处理的用户消息。
3. 从会话表层重新生成消息，并组装工具说明，估算它们占用的上下文空间。
4. 压力达到 80% 时，裁剪历史中的大型工具结果，再重新生成模型消息。
5. 模型配置或工具 schema 变化时记录新的 `request/header`；表层替换会记录新的消息序列边界。系统提示词不再复制到请求头。保存成功后才调用模型，临时错误在当前步骤中等待并重试。
6. 模型请求工具时，先保存 `tool/call`，再执行工具。工具返回超大文本时，保存完整原文，只把预览写入 `tool/result`。
7. 当前步骤的消息和工具结果都处理完后，写入 `step/end`。

如果检查点保存失败，模型和工具都不会继续执行。这保证了“先记录操作意图，再执行动作”，但不能保证外部操作只发生一次：程序仍可能在远端操作成功、结果尚未保存时退出。

## 17.6 计划模式与用户问答

`plan.set` 在两个轮次之间可以立即切换计划模式。如果智能体正在执行一个尚未结束的轮次，新选择会先保持待生效状态，到下一步骤开始时再提交。

JSON-RPC 的 `plan.set` 要求 `active` 是真正的 JSON boolean。字符串 `"false"` 不会按 Python truthiness 误转成 true，而是返回 `INVALID_PARAMS`。

`exit_plan_mode` 要求计划以 `#` 一级标题开头，并通过 UserQuestionService 提交结构化评审。只有唯一答案项、唯一选择 `Approve`、且没有 custom 文本时才算批准。

批准结果返回后，日志中的计划模式仍会暂时保持开启。同一条模型回复中剩余的工具调用继续遵守计划模式，下一步骤开始时才写入 `plan/mode=false`。这样可以避免一批工具调用执行到一半时改变规则。

普通命令行模式会在标准错误中显示问题、计划和选项，再从标准输入读取编号、标签或自定义反馈。`--rpc` 模式不会启用这项终端问答，因为 JSON-RPC 和 `input()` 不能同时读取同一个标准输入，当前的最小 RPC 接口也没有单独的问答通道。只有当前任务的根智能体能够发起用户问答，子智能体不能阻塞等待终端输入。

`update_goal` 要求传入 `get_goal` 返回的精确 `goal_id` 与 `revision`。模型可以编辑、暂停、完成或阻塞当前目标，也可以恢复因会话重启而处于 `disarmed` 的 active 目标和 blocked 目标；持久状态为 `paused` 的目标只能由用户控制入口恢复。教学版没有自动 Goal Round 驱动器，因此也没有实现官方只对自动续行轮次生效的连续阻塞阈值；宿主可以通过 Python 领域方法恢复暂停目标，但没有 `/goal resume` 命令和 Web 控件。

## 17.7 重试、保存与大结果处理

`RetryPolicy` 默认只重试空响应、限流、服务端错误、超时和网络传输错误，最多五次。等待时间从 500 毫秒逐次增加到最多 10 秒，并加入少量随机偏移，避免多个请求同时再次访问服务。

服务端返回合理的 `Retry-After` 时，程序优先采用它给出的等待时间。每次等待前先写入并保存 `llm/retry`，等待结束后再写入 `llm/retry-started`。失败结果不会进入模型消息；同一步骤的重试复用已经组装好的请求，不会重复执行步骤开始前的策略。

`ToolResultPruner` 按字符数判断工具结果大小。`TokenMeter` 使用当前官方 DeepSeek 默认的 1,000,000 token 上下文容量，估算系统提示词、当前消息和工具说明的总长度；只有上下文压力达到 80% 时，裁剪器才处理超过阈值的旧结果。模型随后只看到保留的开头、省略标记和结尾，完整原事件仍保存在日志中。

`SpillPolicy` 按 UTF-8 字节数判断刚刚产生的工具结果。过大的纯文本先由 `LocalSpillStore` 保存，模型只收到预算内的首尾预览、文件位置和读取提示。`read` 工具支持从指定行开始、每次最多读取 2000 行，因此模型可以分段取回完整内容。保存失败或当前结果不适合外存时，程序保留原文，不会把一次成功的工具调用改成失败。

第 09 章的模型摘要压缩仍作为独立示例运行。本章已经接入长度估算、旧结果裁剪和新结果外存，但尚未接入自动摘要，也没有处理模型服务返回的输入过长错误。

`web_search` 和 `web_fetch` 的结果都会标记为外部不可信数据。抓取器只接受无内嵌凭据的公共 HTTP(S) 地址，不读取环境代理，逐跳检查解析结果，只跟随同源重定向，并限制响应类型、字节数和字符数。教学版未实现官方的连接地址固定和 DNS64 检测。

## 17.8 子智能体与后台任务

`subagent` 可以创建一次性或可继续对话的子智能体，并选择在前台等待或放到后台运行。`subagent_fork` 固定创建一次性子智能体，同样可以选择前台或后台；它使用父会话中已经完成的历史作为起点。

全新子智能体从空会话开始，只看到完整的任务说明。分支子智能体只复制父会话到最后一个完整 `turn/end` 为止，当前尚未结束的轮次不会进入子会话。

可继续子智能体保留自己的会话，并按先到先得顺序处理消息。后台创建时，首条任务进入队列后便返回 `child_id` 和接收确认；`send_message` 也只确认消息已经入队，不等待回答。`interrupt_agent` 用于请求中断当前轮次。其他子智能体或根智能体不能只凭编号访问不属于自己的会话。

一次性后台任务由 `LocalJobs` 管理。每个任务绑定所有者，并支持列举、读取、等待和取消。排队中、运行中和正在停止的任务都会占用并发名额；取消请求发出后，只有任务停止才会进入已取消状态。可继续子智能体不属于普通后台任务。教学版没有完成通知、父子双向消息、按消息编辑或删除队列内容、`list_agents` 和跨进程恢复；父智能体通过查询工具取得后台任务结果。官方当前版本已把可继续子智能体的 Queue、Steer、停止和发现接口扩展到宿主控制面。

`WorkflowEngine` 使用 Python 线程运行任务函数，支持并行执行和分阶段执行，并限制并发数与总任务数。线程不是安全隔离，不能执行不可信代码。模型调用 `workflow` 时只需提供简化后的 `tasks[]` 参数。官方参考版本使用 Worker Thread 执行受限 JavaScript 脚本，功能和隔离方式更完整。

## 17.9 配置项 Settings

默认配置位于 `agent` 命名空间，合并顺序仍是程序默认值、当前组合的基础值、用户设置，后者覆盖前者。

用户文档默认读取 `.mini-harness/settings.json`，例如：

```json
{
  "agent": {
    "sandbox_mode": "workspace-write",
    "shell_mode": "read-only",
    "approval_policy": "never",
    "retry_max_retries": 3,
    "spill_max_inline_bytes": 8192,
    "jobs_max_concurrency": 4,
    "workflow_max_agents": 16
  }
}
```

更新配置时会检查 `revision` 版本号，避免旧数据覆盖新修改。本章只在启动时读取一次配置，没有实现运行时自动重新加载、通过 RPC 保存配置或在多套配置之间动态切换。

## 17.10 会话文件与退出

普通任务开始时会创建唯一的 `.mini-harness/sessions/<uuid>.jsonl` 文件。运行器把同一个 `JsonlStore.save` 交给检查点插件，因此模型请求和工具调用意图会在运行过程中逐步写入磁盘。无论任务成功还是发生错误，关闭根 `Context` 前都会再保存一次，保证 `step/end` 和 `turn/end(error)` 不只停留在内存中。RPC 模式退出时也执行相同收尾。

超大工具结果默认保存在 `.mini-harness/spills/<session>/`。工具给出的建议名称只用于生成文件名，存储服务会清理不安全字符、避免重名覆盖，并返回绝对路径。

程序从会话中找到最后一条非空模型回复作为最终结果。完成状态只读取最后一条 `turn/end`，不会用较早轮次的成功覆盖后续错误。系统提示词、Inbox 插入与领取、模型消息和工具结果都保存在同一日志中。

## 对照官方 `dsh-v0.1.5-alpha.2`

本章的官方源码对照版本是 tag `dsh-v0.1.5-alpha.2`、commit `b2e3b2a0125854567a4a5fcba75782e42fe84901`。

如果要继续研究完整架构，可以对照官方文档的 [Cordis Primer](https://deepseek-harness.github.io/deepseek-harness/reference/cordis-primer)、[Capability Seams](https://deepseek-harness.github.io/deepseek-harness/reference/capability-seams) 与 [Extension Cookbook](https://deepseek-harness.github.io/deepseek-harness/reference/cookbook/extension-cookbook)。它们分别介绍插件内核、服务定义与使用关系，以及新增能力应接入扩展位置而不是不断修改智能体主循环的原则。

| 官方源码 | 教学版对应 | 边界 |
|---|---|---|
| `packages/bundle/base/cordis.patch.yml`、`packages/bundle/headless/cordis.patch.yml` | `cordis.py`、`bundle.py`、`build_agent` | Python 版使用明确的插件清单，不实现 YAML 配置加载、热重载和隔离作用域 |
| `vendor/cordis/src/context.ts`、`fiber.ts`、`reflect.ts`、`events.ts` | `Context`、`PluginHandle`、`depends`、`waterfall` | 保留插件生命周期、等待依赖、替换服务和自动清理；同步教学版不实现异步插件任务 |
| `packages/core/agent`、`packages/core/agent-loop/src/inbox.ts` | `AgentRegistry`、`Agent`、`Inbox` | 不使用 `ctx.agent`；Agent 显式传给使用者，Inbox 变更写入 Session；教学版不实现按消息标识编辑与删除 |
| `packages/core/session`、`packages/session/session-format-v2-to-v3` | `session.py`、`persistence.py` | 系统提示词进入表层消息，请求头不保存其副本；教学版使用自有 JSONL 版本，不读取官方 V3 文件 |
| `packages/plan/plan-mode`、`packages/interaction/user-questions` | `plan.py`、`user_questions.py` | 保留日志折叠、稳定工具、评审和边界提交 |
| `packages/session/session-checkpoint-policy` | `checkpoint.py` 与模型请求、工具执行、步骤开始和重试前的保存位置 | 前三个位置与官方一致；教学版没有后台批量保存，因此还会显式保存重试调度事件 |
| `packages/llm/llm-retry` | `retry.py` | 实现次数有限的重试；没有始终重试和异步取消信号 |
| `packages/compaction/compaction-tool-result-pruner` | `pruner.py` | 教学版只处理纯文本消息块 |
| `packages/spill/*` | `spill.py` | 使用本地存储服务和外存策略；不处理分发日志中的结果 |
| `packages/subagent/*`、`packages/jobs/jobs-local` | `subagent.py`、`jobs.py` | 使用当前进程中的线程；没有远程服务和跨进程恢复 |
| `packages/workflow/workflow-worker-thread` | `workflow.py` | 使用 Python 函数讲解工作流，线程不构成安全隔离 |
| `packages/web/web-fetch-http`、`packages/web/tool-web` | `web_tools.py` | 对齐公共地址、同源重定向、内容类型和大小限制；教学版不实现连接地址固定与 DNS64 检测 |
| `packages/api/gateway` | `rpc.py`、`--rpc` | 只实现最小 JSON-RPC 接口，不包含完整宿主服务和 API 代理 |

## 本章小结

- 第 10–16 章的能力已经通过服务和事件接入同一个完整示例；
- 智能体主循环不直接实现重试、保存、计划模式和上下文控制，`build_agent()` 只负责挂载插件集合并从 `agents` 注册表取得实例；
- 计划模式与用户问答相互协作，但模式只在步骤边界切换；
- 重试、关键节点保存、结果裁剪和外部存储各自处理不同问题；
- 分支子智能体、可继续子智能体、后台任务和工作流分别处理历史继承、持续对话、后台执行与批量编排；
- 命令行入口支持一次性任务和标准输入输出上的 JSON-RPC，但不包含完整宿主服务或平台级沙箱。

## 练习

1. 选择一个真实任务，例如代码审查、资料研究或项目维护，画出它需要的插件树。区分哪些组件提供服务、哪些组件注册模型工具、哪些策略通过事件或 `waterfall` 处理链接入。
2. 假设要增加模型切换、产物导出或新的存储后端。说明这项能力应负责提供服务、使用服务还是监听运行事件，以及怎样做到卸载后不残留提示词、工具或监听器。
3. 为一次包含模型重试、大型工具结果和会话保存的步骤画出执行时间线。分析保存、重试、结果裁剪与外部存储的顺序如果改变，可能造成哪些数据丢失、重复调用或上下文浪费。
4. 基于本章完成一个端到端小项目：从命令行或 RPC 接收任务，使用至少两类工具，保存并恢复会话，最后输出可以核对的结果。记录实际启用的插件、关键事件和一个失败分支，不要求实现课程范围外的完整平台能力。
5. 当前教学版没有完整的宿主界面、平台级沙箱、Session 代际迁移、多模态、热重载和 Agent Teams。请选择其中一项作为进一步完善的优先方向，说明它解决的实际问题、应接入哪个扩展位置，以及为什么其他能力可以暂缓。
