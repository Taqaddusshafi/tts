"""Language normalization, validation, and routing helpers."""

import re

from app.core.errors import ValidationAppError

LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$")

INDIC_LANGUAGES: frozenset[str] = frozenset(
    {
        "as",
        "bn",
        "brx",
        "doi",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    }
)

SUPPORTED_NON_INDIC_LANGUAGES: frozenset[str] = frozenset(
    {
        "ar",
        "de",
        "en",
        "es",
        "fr",
        "it",
        "ja",
        "ko",
        "nl",
        "pl",
        "pt",
        "ru",
        "tr",
        "zh",
    }
)

SUPPORTED_LANGUAGES: frozenset[str] = INDIC_LANGUAGES | SUPPORTED_NON_INDIC_LANGUAGES


def normalize_language(language: str) -> str:
    """Normalize a language tag to its primary lowercase code."""

    normalized = language.strip().replace("_", "-").lower()
    primary = normalized.split("-", maxsplit=1)[0]

    if not LANGUAGE_RE.match(normalized):
        raise ValidationAppError(
            "Invalid language code.",
            {"language": language, "expected": "BCP-47 style code such as hi or en"},
        )

    if primary not in SUPPORTED_LANGUAGES:
        raise ValidationAppError(
            "Unsupported language code.",
            {"language": language, "supported": sorted(SUPPORTED_LANGUAGES)},
        )

    return primary


def is_indic(language: str) -> bool:
    """Return true when the language should route to Indic-Parler-TTS."""

    return normalize_language(language) in INDIC_LANGUAGES

