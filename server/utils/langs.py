"""Which languages this install can actually dub into.

Built from the voice catalog at call time rather than a hand-kept list, so a
language is offered only when the chosen TTS engine can speak it — otherwise
a job runs for minutes and then cannot synthesise a word.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("server.langs")

#: Display names, English first with the native name beside it. `script`
#: drives which font the UI loads.
LANGUAGES: Dict[str, Dict[str, str]] = {
    "as":  {"english": "Assamese",  "native": "অসমীয়া",      "script": "beng"},
    "bn":  {"english": "Bengali",   "native": "বাংলা",        "script": "beng"},
    "brx": {"english": "Bodo",      "native": "बड़ो",          "script": "deva"},
    "doi": {"english": "Dogri",     "native": "डोगरी",        "script": "deva"},
    "en":  {"english": "English",   "native": "English",     "script": "latn"},
    "gu":  {"english": "Gujarati",  "native": "ગુજરાતી",       "script": "gujr"},
    "hi":  {"english": "Hindi",     "native": "हिन्दी",        "script": "deva"},
    "kn":  {"english": "Kannada",   "native": "ಕನ್ನಡ",        "script": "knda"},
    "kok": {"english": "Konkani",   "native": "कोंकणी",       "script": "deva"},
    "ks":  {"english": "Kashmiri",  "native": "کٲشُر",        "script": "arab"},
    "mai": {"english": "Maithili",  "native": "मैथिली",       "script": "deva"},
    "ml":  {"english": "Malayalam", "native": "മലയാളം",      "script": "mlym"},
    "mni": {"english": "Manipuri",  "native": "ꯃꯤꯇꯩꯂꯣꯟ",     "script": "mtei"},
    "mr":  {"english": "Marathi",   "native": "मराठी",        "script": "deva"},
    "ne":  {"english": "Nepali",    "native": "नेपाली",       "script": "deva"},
    "or":  {"english": "Odia",      "native": "ଓଡ଼ିଆ",         "script": "orya"},
    "pa":  {"english": "Punjabi",   "native": "ਪੰਜਾਬੀ",        "script": "guru"},
    "sa":  {"english": "Sanskrit",  "native": "संस्कृतम्",     "script": "deva"},
    "sat": {"english": "Santali",   "native": "ᱥᱟᱱᱛᱟᱲᱤ",      "script": "olck"},
    "sd":  {"english": "Sindhi",    "native": "سنڌي",        "script": "arab"},
    "ta":  {"english": "Tamil",     "native": "தமிழ்",        "script": "taml"},
    "te":  {"english": "Telugu",    "native": "తెలుగు",       "script": "telu"},
    "ur":  {"english": "Urdu",      "native": "اردو",         "script": "arab"},
}

#: Font family per script.
SCRIPT_FONTS: Dict[str, str] = {
    "deva": "Noto Sans Devanagari",
    "beng": "Noto Sans Bengali",
    "gujr": "Noto Sans Gujarati",
    "guru": "Noto Sans Gurmukhi",
    "knda": "Noto Sans Kannada",
    "mlym": "Noto Sans Malayalam",
    "orya": "Noto Sans Oriya",
    "taml": "Noto Sans Tamil",
    "telu": "Noto Sans Telugu",
    "arab": "Noto Naskh Arabic",
    "olck": "Noto Sans Ol Chiki",
    "mtei": "Noto Sans Meetei Mayek",
    "latn": "",
}


def _catalog_languages(engine: str) -> List[str]:
    try:
        from backend.tts.voices import load_voice_catalog
    except Exception as e:                       # pragma: no cover
        log.warning("voice catalog unavailable: %s", e)
        return []
    langs = {(v.language or "").lower()
             for v in load_voice_catalog(engine) if v.language}
    return sorted(langs)


def languages_for(tts: Optional[str]) -> List[str]:
    """ISO codes the given TTS engine can speak, best-effort."""
    eng = (tts or "").strip().lower()
    if eng in ("bodhan", "indic-speak", ""):
        return _catalog_languages("indic-speak")
    # An engine with no catalog to narrow by: offer everything rather than an
    # empty dropdown.
    return _catalog_languages(eng) or sorted(LANGUAGES)


def options(tts: Optional[str] = None) -> List[Dict[str, Any]]:
    """Dropdown rows for the new-job form."""
    out = []
    for code in languages_for(tts):
        meta = LANGUAGES.get(code)
        if meta is None:
            # An adapter knows a language this table does not. Show the code
            # rather than hide a capability.
            out.append({"code": code, "english": code.upper(), "native": "",
                        "script": "latn", "font": ""})
            continue
        out.append({"code": code, "english": meta["english"],
                    "native": meta["native"], "script": meta["script"],
                    "font": SCRIPT_FONTS.get(meta["script"], "")})
    out.sort(key=lambda r: r["english"])
    return out


def describe(code: Optional[str]) -> str:
    """`hi` -> `Hindi (हिन्दी)`, for the spec panel."""
    if not code:
        return "—"
    meta = LANGUAGES.get(code.split("-")[0].lower())
    if meta is None:
        return code
    return f"{meta['english']} ({meta['native']})" if meta["native"] else meta["english"]


def english_name(code: Optional[str],
                 default: str = "the source language") -> str:
    """`hi` -> `Hindi`, for a prompt that has to name the language in words.

    An unknown code is returned as it stands rather than replaced by the
    default: a prompt naming `yue` is wrong in a way someone can see, one
    silently saying "the source language" is not.
    """
    if not code:
        return default
    meta = LANGUAGES.get(code.split("-")[0].lower())
    return meta["english"] if meta else code


def script_of(code: Optional[str]) -> str:
    meta = LANGUAGES.get((code or "").split("-")[0].lower())
    return meta["script"] if meta else "latn"
