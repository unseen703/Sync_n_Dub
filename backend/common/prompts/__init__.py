"""Central prompt registry — every LLM system prompt lives here.

Import the dict and look up by key:

    from backend.common.prompts import PROMPTS
    system = PROMPTS["retranslate"]

Prompts with ``{{placeholders}}`` are templates; the caller substitutes
at call time (e.g. ``PROMPTS["edit_planner"].replace("{{tone_styles}}", …)``).
"""
from .code_mix import PROMPT as _code_mix
from .condense import PROMPT as _condense
from .edit_planner import PROMPT as _edit_planner
from .merge_cues import PROMPT as _merge_cues
from .register_classifier import PROMPT as _register_classifier
from .retranslate import PROMPT as _retranslate
from .term_translation import PROMPT as _term_translation
from .tone_rewrite import PROMPT as _tone_rewrite

PROMPTS = {
    "edit_planner": _edit_planner,
    "condense": _condense,
    "term_translation": _term_translation,
    "retranslate": _retranslate,
    "tone_rewrite": _tone_rewrite,
    "code_mix": _code_mix,
    "merge_cues": _merge_cues,
    "register_classifier": _register_classifier,
}
