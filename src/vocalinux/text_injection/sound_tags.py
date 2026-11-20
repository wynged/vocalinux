"""Strip whisper's sound-effect tags from dictated text.

Whisper describes non-speech in brackets: "(keyboard clicking)", "(sighs)",
"[BLANK_AUDIO]", "[Music]". The engine's own ``_filter_non_speech`` drops a
segment that is nothing but such a tag, but one that sits among words goes
straight through, and the tag lands wherever the text is pasted. Nobody
dictates parentheses, so every (...) and [...] span is removed, nested ones
included, before a segment reaches the compose window or the focused app.
"""

import re

_PAREN = re.compile(r"\([^()]*\)")
_BRACKET = re.compile(r"\[[^\[\]]*\]")
_STARTS_WITH_TAG = re.compile(r"^\s*[(\[]")
_LEADING_ORPHAN = re.compile(r"^[\s,;:.!?]+")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?])")
_RUNS_OF_SPACE = re.compile(r"[ \t]{2,}")
_WORD = re.compile(r"\w")


def strip_sound_tags(text: str) -> str:
    """Return ``text`` with every (...) and [...] span removed.

    Innermost spans go first and the pass repeats, so "(a (b) c)" and a
    bracket inside parentheses both vanish whole. An unmatched "(" or "["
    is left as it is: there is no telling where it was meant to close.

    The removal's leftovers are tidied: doubled spaces collapse to one, a
    space left before punctuation goes, and punctuation stranded at the
    start by a leading tag ("(laughs), yes" -> "yes") goes too. If nothing
    but punctuation is left, the result is "" so the caller injects nothing.
    Text with no tag in it comes back unchanged, byte for byte.
    """
    if not text or ("(" not in text and "[" not in text):
        return text

    out = text
    while True:
        stripped = _BRACKET.sub(" ", _PAREN.sub(" ", out))
        if stripped == out:
            break
        out = stripped

    if out == text:
        return text

    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _RUNS_OF_SPACE.sub(" ", out)
    if _STARTS_WITH_TAG.match(text):
        out = _LEADING_ORPHAN.sub("", out)
    out = out.strip()
    if not _WORD.search(out):
        return ""
    return out
