"""Tests for strip_sound_tags: whisper's (sound effect) tags never get pasted."""

import pytest

from vocalinux.text_injection.sound_tags import strip_sound_tags


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Mid-sentence, with the doubled space the removal leaves.
        ("So I think (keyboard clicking) we should go.", "So I think we should go."),
        # Before punctuation: no stray space before the period or comma.
        ("Hello there (coughs).", "Hello there."),
        ("Well (sighs), fine.", "Well, fine."),
        # At the start, with the punctuation it stranded.
        ("(laughs), yes, that one.", "yes, that one."),
        ("(keyboard clicking) Send it.", "Send it."),
        # At the end.
        ("Send it (mouse click)", "Send it"),
        # Whisper's bracket tags are the same class.
        ("[BLANK_AUDIO] okay then", "okay then"),
        ("okay [Music] then", "okay then"),
        # Nested spans go whole, and mixed kinds.
        ("a (outer (inner) still outer) b", "a b"),
        ("a ([Music] and (claps)) b", "a b"),
        # Several tags in one segment.
        ("(coughs) one (sniffs) two [noise] three", "one two three"),
    ],
)
def test_tags_are_removed_and_the_words_read_cleanly(raw, expected):
    assert strip_sound_tags(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["(keyboard clicking)", "[BLANK_AUDIO]", "(coughs) (sighs)", "(laughs).", " [Music] , "],
)
def test_a_segment_of_only_tags_injects_nothing(raw):
    assert strip_sound_tags(raw) == ""


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "plain words, no tags.",
        "...and then it stopped.",
        "Double  spaces stay when nothing was removed.",
        "Ends in a newline\n",
    ],
)
def test_text_without_tags_is_untouched(raw):
    assert strip_sound_tags(raw) == raw


def test_an_unmatched_bracket_is_left_alone():
    assert strip_sound_tags("one (two three") == "one (two three"
    assert strip_sound_tags("one ] two") == "one ] two"


def test_a_leading_ellipsis_survives_when_no_tag_led():
    assert strip_sound_tags("...so (sighs) anyway") == "...so anyway"
