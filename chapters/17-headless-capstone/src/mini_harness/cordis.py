"""第 17 章使用的 mini-Cordis 插件内核。

它保留官方 Cordis 的四个核心语义：插件有生命周期、服务有提供者、
依赖未就绪时消费者等待、所有副作用在卸载时逆序清理。业务能力不写进
内核，而是通过 service、event 和 effect 组合。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

Disposer = Callable[[], None]
PluginFn = Callable[..., Disposer | None]
Plugin = TypeVar("Plugin", bound=PluginFn)


def _once(disposer: Disposer) -> Disposer:
    active = True

    def run() -> None:
        nonlocal active
        if not active:
            return
        active = False
        disposer()

    return run


def depends(*services: str) -> Callable[[Plugin], Plugin]:
    """声明插件依赖；依赖缺失时 fiber 保持 pending。"""

    def decorate(plugin: Plugin) -> Plugin:
        cast(Any, plugin).inject = tuple(dict.fromkeys(services))
        return plugin

    return decorate


class PluginHandle:
    """一次插件安装对应一个 fiber 与一份可逆资源清单。"""

    _next_uid = 1

    def __init__(self, ctx: Context, plugin: PluginFn, config: Any) -> None:
        self.uid = PluginHandle._next_uid
        PluginHandle._next_uid += 1
        self.name = getattr(plugin, "__name__", f"plugin#{self.uid}")
        self.config = config
        self.state = "pending"
        self.inject = tuple(dict.fromkeys(getattr(plugin, "inject", ())))

        self._ctx = ctx._root_context()
        self._plugin = plugin
        self._store: dict[str, object] = {}
        self._epoch: str | None = None
        self._disposers: list[Disposer] = []

        self._ctx._handles.append(self)
        parent = ctx._owner or self._ctx._current
        if parent is not None:
            parent.collect(_once(self.dispose))
        self._recheck()

    def _recheck(self) -> None:
        if self.state in {"loading", "unloading", "disposed", "failed"}:
            return
        resolved: dict[str, object] = {}
        tokens: list[str] = []
        for name in self.inject:
            registration = self._ctx._services.get(name)
            if registration is None:
                tokens.append(f"{name}=-")
                continue
            resolved[name] = registration[0]
            tokens.append(f"{name}={registration[1]}:{registration[2]}")
        epoch = ",".join(tokens)
        if epoch == self._epoch:
            return
        self._epoch = epoch

        if self.state == "active":
            self._unload()
        if len(resolved) != len(self.inject):
            self.state = "pending"
            return
        self._store = resolved
        self._load()

    def _load(self) -> None:
        previous = self._ctx._current
        self._ctx._current = self
        self.state = "loading"
        try:
            result = self._plugin(self._ctx._view(self), self.config)
            if result is not None:
                self.collect(_once(result))
            self.state = "active"
        except Exception as error:
            self.state = "failed"
            try:
                self._dispose_all()
            except Exception as cleanup_error:  # noqa: BLE001 - 合并原始错误与清理错误
                raise ExceptionGroup(
                    "插件安装失败，回滚清理也失败", [error, cleanup_error]
                ) from None
            raise
        finally:
            self._ctx._current = previous

    def collect(self, disposer: Disposer) -> None:
        self._disposers.append(disposer)

    def _dispose_all(self) -> None:
        disposers = list(reversed(self._disposers))
        self._disposers.clear()
        errors: list[Exception] = []
        for disposer in disposers:
            try:
                disposer()
            except Exception as error:  # noqa: BLE001 - 清理必须尽量执行完全部 disposer
                errors.append(error)
        self._store.clear()
        if errors:
            raise ExceptionGroup("插件资源清理失败", errors)

    def _unload(self) -> None:
        self.state = "unloading"
        self._dispose_all()
        self.state = "pending"

    def dispose(self) -> None:
        if self.state == "disposed":
            return
        try:
            if self.state in {"active", "loading", "unloading", "failed"}:
                self.state = "unloading"
                self._dispose_all()
        finally:
            self.state = "disposed"
            self._ctx._notify()


class Context:
    """插件容器：只管理 fiber、service、event 与 effect。"""

    def __init__(
        self,
        _root: Context | None = None,
        _owner: PluginHandle | None = None,
    ) -> None:
        self._root = _root
        self._owner = _owner
        if _root is not None:
            return
        self._handles: list[PluginHandle] = []
        self._listeners: dict[str, list[Callable[..., Any]]] = {}
        self._services: dict[str, tuple[object, int, int]] = {}
        self._current: PluginHandle | None = None
        self._version = 0
        self._disposed = False

    def _root_context(self) -> Context:
        return self if self._root is None else self._root

    def _view(self, owner: PluginHandle) -> Context:
        return Context(self._root_context(), owner)

    def plugin(self, plugin: PluginFn, config: Any = None) -> PluginHandle:
        root = self._root_context()
        if root._disposed:
            raise RuntimeError("Context 已销毁，不能再安装插件")
        return PluginHandle(self, plugin, config)

    def provide(self, name: str, value: object) -> Disposer:
        root = self._root_context()
        if name in root._services:
            raise ValueError(f'服务 "{name}" 已被注册')
        root._version += 1
        owner = self._owner or root._current
        provider_uid = owner.uid if owner is not None else 0
        registration = (value, provider_uid, root._version)
        root._services[name] = registration

        def unregister() -> None:
            if root._services.get(name) is registration:
                del root._services[name]
                root._notify()

        disposer = _once(unregister)
        if owner is not None:
            owner.collect(disposer)
        try:
            root._notify()
        except Exception:
            disposer()
            raise
        return disposer

    def get(self, name: str) -> object | None:
        registration = self._root_context()._services.get(name)
        return registration[0] if registration is not None else None

    def require(self, name: str) -> object:
        value = self.get(name)
        if value is None:
            raise LookupError(f'服务 "{name}" 尚未提供')
        return value

    def __getattr__(self, name: str) -> Any:
        root = self._root_context()
        owner = self._owner or root._current
        if owner is not None:
            if name in owner._store:
                return owner._store[name]
            if name in owner.inject:
                raise AttributeError(f'服务 "{name}" 已声明依赖但尚未就绪')
        if name in root._services:
            raise AttributeError(f'读取服务 "{name}" 前必须在 inject 里声明')
        raise AttributeError(f"Context 没有属性 {name!r}")

    def _notify(self) -> None:
        for handle in list(self._root_context()._handles):
            handle._recheck()

    def effect(self, factory: Callable[[], Disposer | None]) -> Disposer:
        raw_disposer = factory()
        disposer = _once(raw_disposer) if raw_disposer is not None else None
        root = self._root_context()
        owner = self._owner or root._current
        if disposer is not None and owner is not None:
            owner.collect(disposer)
        return disposer if disposer is not None else (lambda: None)

    def on(self, event: str, listener: Callable[..., Any]) -> Disposer:
        root = self._root_context()
        root._listeners.setdefault(event, []).append(listener)

        def remove() -> None:
            listeners = root._listeners.get(event)
            if listeners is not None and listener in listeners:
                listeners.remove(listener)

        disposer = _once(remove)
        owner = self._owner or root._current
        if owner is not None:
            owner.collect(disposer)
        return disposer

    def emit(self, event: str, *args: Any) -> None:
        for listener in list(self._root_context()._listeners.get(event, [])):
            listener(*args)

    def serial(self, event: str, *args: Any) -> Any:
        for listener in list(self._root_context()._listeners.get(event, [])):
            result = listener(*args)
            if result is not None:
                return result
        return None

    def waterfall(self, event: str, *args: Any) -> Any:
        *call_args, terminal = args
        listeners = list(self._root_context()._listeners.get(event, []))

        def dispatch(index: int, *inner_args: Any) -> Any:
            if index >= len(listeners):
                return terminal(*inner_args)
            listener = listeners[index]

            def next_fn(*new_args: Any) -> Any:
                return dispatch(index + 1, *(new_args if new_args else inner_args))

            return listener(*inner_args, next_fn)

        return dispatch(0, *call_args)

    @property
    def handles(self) -> tuple[PluginHandle, ...]:
        return tuple(self._root_context()._handles)

    @property
    def service_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._root_context()._services))

    def dispose(self) -> None:
        root = self._root_context()
        if root._disposed:
            return
        root._disposed = True
        errors: list[Exception] = []
        for handle in reversed(root._handles):
            try:
                handle.dispose()
            except Exception as error:  # noqa: BLE001 - 根清理聚合全部插件失败
                errors.append(error)
        if errors:
            raise ExceptionGroup("Context 清理失败", errors)
