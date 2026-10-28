"""One adapter registry, shared by every stage under `models/`.

Each stage declares a `_REGISTRY_SPEC` mapping a short name to
`(module, class)`; this module turns that into a lazy importer and a getter
that drops kwargs the chosen class does not accept. A stage's `__init__.py`
keeps only its docstring, `_REGISTRY_SPEC`, its `base` re-exports, and any
stage-specific helper.

`_REGISTRY_SPEC` must stay a dict literal at module level: tooling and
`tests/test_lightasd_adapter.py` read it with `ast.parse` without importing
the module, so a spec assembled by a helper reads as zero adapters.

The error messages are pinned by `scripts/ab_refactor.py` -- change the
wording and that check fails.
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type


@dataclass(frozen=True)
class Registry:
    """The four callables a stage's `__init__.py` re-exports."""
    available: Callable[[], list]
    get: Callable[..., Any]
    load: Callable[[str], Type]
    accepted_kwargs: Callable[[Type], set]


def make_registry(
    *,
    package: str,
    spec: dict,
    noun: str,
    import_noun: str,
    base: Optional[Type] = None,
    import_detail: str = "is not installed.",
    pass_device: bool = True,
    verbose_warning: bool = False,
    get_doc: Optional[str] = None,
) -> Registry:
    """Build one stage's registry callables.

    `noun` goes in the ValueError for an unregistered name ("MT model",
    "face detector"). `import_noun` + `import_detail` build the ImportError
    for a name that is registered but will not import; `import_detail` may
    interpolate `{key}`, which `models.diarization` uses to name
    `.venv-<key>`.

    `pass_device=False` gives the getter no `device` parameter at all, so a
    caller passing one gets a TypeError (used by `models.faces.track`,
    whose trackers are pure geometry). `verbose_warning=True` appends the
    accepted-kwarg list.

    `base` is the stage's abstract class, used only to put the return
    annotation back on the getter so `inspect.signature` still reports it.
    """

    def _load(key: str) -> Type:
        module_path, cls_name = spec[key]
        module = importlib.import_module(module_path, package=package)
        return getattr(module, cls_name)

    def available() -> list:
        return sorted(spec)

    def _accepted_kwargs(cls: Type) -> set:
        sig = inspect.signature(cls.__init__)
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            return set()   # sentinel — anything is accepted
        return {
            p.name for p in sig.parameters.values()
            if p.name != "self"
            and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
        }

    def _resolve(name: str, kwargs: dict) -> tuple:
        """(class, filtered kwargs).

        Unknown kwargs are dropped with a warning, not propagated, so one
        call site can pass `hf_token=..., batch_size=...` without knowing
        which adapter consumes which.
        """
        key = name.strip().lower()
        if key not in spec:
            raise ValueError(
                f"Unknown {noun} {name!r}. Available: {', '.join(available())}"
            )
        try:
            cls = _load(key)
        except ImportError as e:
            raise ImportError(
                f"{import_noun} {key!r} {import_detail.format(key=key)} "
                f"Original error: {e}"
            ) from e
        accepted = _accepted_kwargs(cls)
        if accepted:
            unknown = set(kwargs) - accepted
            if unknown:
                import warnings
                msg = (f"{cls.__name__} does not accept kwargs "
                       f"{sorted(unknown)!r}; dropping.")
                if verbose_warning:
                    msg += f" Accepted: {sorted(accepted - {'device'})!r}."
                warnings.warn(msg)
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return cls, kwargs

    if pass_device:
        def get(name: str, *, device: Optional[str] = None, **kwargs):
            cls, kwargs = _resolve(name, kwargs)
            return cls(device=device, **kwargs)
    else:
        def get(name: str, **kwargs):                  # type: ignore[misc]
            cls, kwargs = _resolve(name, kwargs)
            return cls(**kwargs)

    get.__doc__ = get_doc or _resolve.__doc__
    if base is not None:
        get.__annotations__["return"] = base
    # `__name__` is left as "get": deriving it from `noun` would produce
    # "get_lip_sync_model" where the exported name is `get_lipsync`.

    return Registry(available=available, get=get, load=_load,
                    accepted_kwargs=_accepted_kwargs)
