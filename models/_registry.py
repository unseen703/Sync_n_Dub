"""One adapter registry, written once instead of twelve times.

Every stage under `models/` resolves its implementation the same way: a
`_REGISTRY_SPEC` mapping a short name to `(module, class)`, a lazy import so
an adapter nobody asked for costs nothing, and a getter that drops the kwargs
the chosen class does not accept. That shape was spelled out in full in
twelve `__init__.py` files — about 45 identical lines each, differing only in
the noun inside the error messages.

This is that code, once. A stage's `__init__.py` keeps only what is genuinely
its own: the docstring, `_REGISTRY_SPEC`, its `base` re-exports, and any
stage-specific helper (`models.tts.resolve_style`, `models.asr`'s
language-filtered lister).

**`_REGISTRY_SPEC` stays a dict LITERAL at module level in every stage.**
`scripts/repo_checkup.py::_registry` reads it with `ast.parse` and never
imports the module — building it through a helper, or assembling it from
parts, makes the five-hourly README regeneration quietly report zero adapters
for that stage.

The messages are reproduced here character for character, nouns included.
They are what an operator sees when a model name is wrong, and
`scripts/ab_refactor.py` pins every one of them so this consolidation can be
proved to have changed nothing.
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
    "face detector"); `import_noun` + `import_detail` make the ImportError
    for a name that IS registered but will not import. They are separate
    because the two sentences start differently — "Unknown aligner …" but
    "Aligner 'mms_fa' is not installed …" — and because the install hint is
    per-stage. `import_detail` may interpolate `{key}`, which is what
    `models.diarization` uses to name `.venv-<key>` in its own message.

    `pass_device=False` is `models.faces.track`, whose trackers are pure
    geometry and take no device; its getter has no `device` parameter at all,
    so a caller passing one gets a TypeError rather than silence.
    `verbose_warning=True` appends the accepted-kwarg list, which three
    stages do.

    `base` is the stage's abstract class, and is only used to put the return
    annotation back on the getter. Each hand-written getter declared
    `-> BaseAligner` / `-> BaseMT` / …, and a generic closure loses that:
    `inspect.signature(get_mt)` stopped reporting a return type, which is
    what every editor and type-checker reads. It costs one line to keep.
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
        """(class, filtered kwargs). Unknown kwargs are dropped with a
        warning, NOT propagated — so one call site (CLI, pipeline) can pass
        `hf_token=…, batch_size=…` without knowing which adapter will
        consume which."""
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
    # `__name__` is deliberately left as "get". Deriving it from `noun` would
    # produce "get_lip_sync_model" where the exported name is `get_lipsync`,
    # and a wrong name in a repr is worse than a generic one. Tracebacks read
    # the code object, not this, so they are unaffected either way.

    return Registry(available=available, get=get, load=_load,
                    accepted_kwargs=_accepted_kwargs)
