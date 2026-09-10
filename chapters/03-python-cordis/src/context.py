"""第 03 章：mini-cordis —— 一个迷你插件系统。

对应官方 DSH vendor 目录下的 cordis（DSH 的插件底座）。
本章实现它的核心三件事：
1. `Context`      —— 插件安装的环境（容器）
2. `PluginHandle` —— 一次安装的句柄：状态机 + 资源清单
3. `effect`       —— 创建资源时同步登记清理方式

第 04 章在此基础上加入服务与依赖注入。
"""

from __future__ import annotations

from typing import Any, Callable

# 清理函数：被调用时释放某份资源（解绑监听器、关文件、停任务……）
Disposer = Callable[[], None]

# 插件本体：收到 ctx 与 config，返回一个可选的清理函数
PluginFn = Callable[["Context", Any], Disposer | None]


def _once(disposer: Disposer) -> Disposer:
    """包装清理函数，使其最多执行一次。"""
    active = True

    def run() -> None:
        nonlocal active
        if not active:
            return
        active = False
        disposer()

    return run


class PluginHandle:
    """一次插件安装的句柄，记录这次安装的状态与全部资源。

    状态机（对应官方 Fiber 状态机的教学子集）：
        pending（已创建）→ active（安装成功）→ disposed（已卸载）
        安装失败 → failed
    """

    _next_uid = 1

    def __init__(self, ctx: "Context", plugin: PluginFn, config: Any) -> None:
        self.uid = PluginHandle._next_uid
        PluginHandle._next_uid += 1
        self.name = getattr(plugin, "__name__", f"plugin#{self.uid}")
        self.config = config
        self.state = "pending"

        self._ctx = ctx
        self._plugin = plugin
        self._disposers: list[Disposer] = []
        self._error: Exception | None = None

        # 1) 登记到环境：全局可见
        ctx._handles.append(self)

        # 2) 级联销毁的关键一步：如果此刻有插件正在安装（父插件），
        #    把「销毁自己」挂到父插件的清理清单上。父被卸载 → 子自动被卸载。
        if ctx._current is not None:
            ctx._current.collect(self.dispose)

        # 3) 立即安装
        self._run()

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """执行插件本体，把返回的清理函数收进清单。"""
        previous = self._ctx._current
        self._ctx._current = self  # 让 apply 期间注册的资源都记到本句柄名下
        try:
            result = self._plugin(self._ctx, self.config)
            if result is not None:
                self.collect(_once(result))
            self.state = "active"
        except Exception as error:
            self.state = "failed"
            self._error = error
            try:
                self._dispose_all()
            except Exception as cleanup_error:
                raise ExceptionGroup(
                    "插件安装失败，回滚清理也失败", [error, cleanup_error]
                ) from None
            raise
        finally:
            self._ctx._current = previous

    def collect(self, disposer: Disposer) -> None:
        """把一份清理函数记到本句柄名下（effect / on 内部都调它）。"""
        self._disposers.append(disposer)

    def dispose(self) -> None:
        """卸载：逆序执行全部清理函数（后注册的先清理）。"""
        if self.state == "disposed":
            return
        try:
            self._dispose_all()
        finally:
            self.state = "disposed"

    def _dispose_all(self) -> None:
        disposers = list(reversed(self._disposers))
        self._disposers.clear()
        errors: list[Exception] = []
        for disposer in disposers:
            try:
                disposer()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("插件资源清理失败", errors)


class Context:
    """插件安装的环境：管理句柄与事件监听器。"""

    def __init__(self) -> None:
        self._handles: list[PluginHandle] = []
        self._listeners: dict[str, list[Callable[..., Any]]] = {}
        self._current: PluginHandle | None = None

    # ------------------------------------------------------------------
    # 插件安装
    # ------------------------------------------------------------------

    def plugin(self, plugin: PluginFn, config: Any = None) -> PluginHandle:
        """安装一个插件，返回它的句柄。"""
        return PluginHandle(self, plugin, config)

    # ------------------------------------------------------------------
    # 副作用：effect 是唯一的入口
    # ------------------------------------------------------------------

    def effect(self, fn: Callable[[], Disposer | None]) -> Disposer:
        """立即执行 fn，把它返回的清理函数挂到当前插件名下。

        插件在 apply 期间创建监听器、文件或任务时，同时返回相应的清理
        函数；句柄在卸载时统一执行这些函数。
        """
        raw_disposer = fn()
        disposer = _once(raw_disposer) if raw_disposer is not None else None
        if disposer is not None and self._current is not None:
            self._current.collect(disposer)
        return disposer if disposer is not None else (lambda: None)

    # ------------------------------------------------------------------
    # 事件：on / emit
    # ------------------------------------------------------------------

    def on(self, event: str, listener: Callable[..., Any]) -> Disposer:
        """注册事件监听器，并把解绑函数登记到当前插件。"""
        self._listeners.setdefault(event, []).append(listener)

        def remove_listener() -> None:
            listeners = self._listeners.get(event)
            if listeners is not None and listener in listeners:
                listeners.remove(listener)

        remove = _once(remove_listener)

        if self._current is not None:
            self._current.collect(remove)
        return remove

    def emit(self, event: str, *args: Any) -> None:
        """同步广播事件给全部监听器（不等待返回值）。"""
        for listener in list(self._listeners.get(event, [])):
            listener(*args)
