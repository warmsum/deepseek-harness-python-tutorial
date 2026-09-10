"""第 16 章：按 namespace 注册、分层解析并带 revision 的 Settings。"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

NAMESPACE = re.compile(r"^[a-z][a-z0-9-]*$")
Validator = Callable[[Mapping[str, Any]], None]
Watcher = Callable[[Mapping[str, Any], Mapping[str, Any]], None]


class SettingsConflictError(RuntimeError):
    """写入方持有的 revision 已经过期。"""

    code = "SETTINGS_CONFLICT"

    def __init__(self, namespace: str, expected: int, actual: int) -> None:
        super().__init__(
            f'settings namespace "{namespace}" 已变化'
            f"（期望 revision {expected}，当前 {actual}）"
        )
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class SettingsDescriptor:
    namespace: str
    value: Mapping[str, Any]
    revision: int
    base: Mapping[str, Any]
    user: Mapping[str, Any]


@dataclass
class _Registration:
    defaults: dict[str, Any]
    base: dict[str, Any]
    validate: Validator | None
    watchers: list[Watcher]


class SettingsScope:
    """一个插件拥有的 namespace 视图。"""

    def __init__(self, settings: Settings, namespace: str) -> None:
        self._settings = settings
        self.namespace = namespace

    @property
    def revision(self) -> int:
        return self._settings._revisions[self.namespace]

    def get(self) -> Mapping[str, Any]:
        return self._settings.get(self.namespace)

    def update(
        self, patch: Mapping[str, Any], *, expected_revision: int | None = None
    ) -> None:
        self._settings.update(
            self.namespace, patch, expected_revision=expected_revision
        )

    def replace(
        self, section: Mapping[str, Any], *, expected_revision: int | None = None
    ) -> None:
        self._settings.replace(
            self.namespace, section, expected_revision=expected_revision
        )

    def watch(self, callback: Watcher) -> Callable[[], None]:
        watchers = self._settings._registrations[self.namespace].watchers
        watchers.append(callback)
        active = True

        def dispose() -> None:
            nonlocal active
            if not active:
                return
            active = False
            watchers.remove(callback)

        return dispose


class Settings:
    """schema 默认值 < 组合 base < 用户 document 三层设置。"""

    def __init__(self, user_document: Mapping[str, Any] | None = None) -> None:
        raw = _clone_json(user_document or {}, "$")
        if not isinstance(raw, dict):
            raise TypeError("settings document 必须是对象")
        self._document: dict[str, Any] = raw
        self._registrations: dict[str, _Registration] = {}
        self._revisions: dict[str, int] = {}

    def register(
        self,
        namespace: str,
        *,
        defaults: Mapping[str, Any] | None = None,
        base: Mapping[str, Any] | None = None,
        validate: Validator | None = None,
    ) -> SettingsScope:
        if not NAMESPACE.fullmatch(namespace):
            raise ValueError(f"无效的 settings namespace: {namespace!r}")
        if namespace in self._registrations:
            raise ValueError(f'settings namespace "{namespace}" 已注册')
        registration = _Registration(
            defaults=_object_clone(defaults or {}, "$.defaults"),
            base=_object_clone(base or {}, "$.base"),
            validate=validate,
            watchers=[],
        )
        self._registrations[namespace] = registration
        self._revisions[namespace] = 0
        try:
            self._resolved(namespace)
        except Exception:
            del self._registrations[namespace]
            del self._revisions[namespace]
            raise
        return SettingsScope(self, namespace)

    def get(self, namespace: str) -> Mapping[str, Any]:
        frozen = _freeze(self._resolved(namespace))
        assert isinstance(frozen, Mapping)
        return frozen

    def describe(self) -> tuple[SettingsDescriptor, ...]:
        descriptors: list[SettingsDescriptor] = []
        for namespace in sorted(self._registrations):
            registration = self._registrations[namespace]
            descriptors.append(
                SettingsDescriptor(
                    namespace=namespace,
                    value=self.get(namespace),
                    revision=self._revisions[namespace],
                    base=_freeze(registration.base),
                    user=_freeze(self._user_section(namespace)),
                )
            )
        return tuple(descriptors)

    def update(
        self,
        namespace: str,
        patch: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> None:
        section = _merge(self._user_section(namespace), _object_clone(patch, "$.patch"))
        self._commit(namespace, section, expected_revision)

    def replace(
        self,
        namespace: str,
        section: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> None:
        self._commit(namespace, _object_clone(section, "$.section"), expected_revision)

    def _commit(
        self,
        namespace: str,
        section: dict[str, Any],
        expected_revision: int | None,
    ) -> None:
        registration = self._require(namespace)
        actual = self._revisions[namespace]
        if expected_revision is not None and (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 0
        ):
            raise TypeError("expected_revision 必须是非负整数或 None")
        if expected_revision is not None and expected_revision != actual:
            raise SettingsConflictError(namespace, expected_revision, actual)
        previous = self.get(namespace)
        candidate = _merge(_merge(registration.defaults, registration.base), section)
        if registration.validate is not None:
            registration.validate(_freeze(candidate))
        self._document[namespace] = section
        self._revisions[namespace] = actual + 1
        next_value = self.get(namespace)
        if next_value != previous:
            for watcher in list(registration.watchers):
                watcher(next_value, previous)

    def _resolved(self, namespace: str) -> dict[str, Any]:
        registration = self._require(namespace)
        value = _merge(
            _merge(registration.defaults, registration.base),
            self._user_section(namespace),
        )
        if registration.validate is not None:
            registration.validate(_freeze(value))
        return value

    def _require(self, namespace: str) -> _Registration:
        try:
            return self._registrations[namespace]
        except KeyError as error:
            raise KeyError(f'settings namespace "{namespace}" 未注册') from error

    def _user_section(self, namespace: str) -> dict[str, Any]:
        raw = self._document.get(namespace, {})
        if not isinstance(raw, dict):
            raise TypeError(f'settings namespace "{namespace}" 的用户层必须是对象')
        return deepcopy(raw)


def _merge(lower: Mapping[str, Any], upper: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(lower))
    for key, value in upper.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _object_clone(value: Mapping[str, Any], path: str) -> dict[str, Any]:
    cloned = _clone_json(value, path)
    if not isinstance(cloned, dict):
        raise TypeError(f"{path} 必须是对象")
    return cloned


def _clone_json(value: Any, path: str, visiting: set[int] | None = None) -> Any:
    if visiting is None:
        visiting = set()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise TypeError(f"{path} 的整数超出 JSON 安全范围")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or (value == 0 and math.copysign(1, value) < 0):
            raise TypeError(f"{path} 不能包含非有限数或负零")
        return value
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in visiting:
            raise TypeError(f"{path} 不能包含循环引用")
        visiting.add(marker)
        try:
            return [
                _clone_json(item, f"{path}[{index}]", visiting)
                for index, item in enumerate(value)
            ]
        finally:
            visiting.remove(marker)
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in visiting:
            raise TypeError(f"{path} 不能包含循环引用")
        visiting.add(marker)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} 的对象键必须是字符串")
                result[key] = _clone_json(item, f"{path}.{key}", visiting)
            return result
        finally:
            visiting.remove(marker)
    raise TypeError(f"{path} 不能包含 {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
