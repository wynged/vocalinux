"""Tests for reading an X selection request as a paste, or as background noise.

The X-level plumbing (owning CLIPBOARD, answering conversion requests) needs a
real server and is exercised by hand; what is worth pinning down here is the
judgement made on top of it, because that is what decides whether the user is
told their dictation went nowhere.
"""

import unittest

from vocalinux.text_injection.clipboard_owner import (
    CLIENT_ID_MASK,
    ClipboardOwner,
    client_of,
)

FOCUSED_WINDOW = 0x28000EE  # a Chrome window
FOCUSED_HELPER = 0x2800000  # its hidden "Chromium clipboard" window
CLIPBOARD_MANAGER = 0x1200001  # greenclip, polling once a second


class TestClientOf(unittest.TestCase):
    def test_windows_of_one_client_share_a_base(self):
        """Xorg hands each client a 2**21 block of resource ids."""
        self.assertEqual(client_of(FOCUSED_WINDOW), client_of(FOCUSED_HELPER))
        self.assertNotEqual(client_of(FOCUSED_WINDOW), client_of(CLIPBOARD_MANAGER))

    def test_no_focused_window_has_no_client(self):
        self.assertIsNone(client_of(None))
        self.assertIsNone(client_of(0))

    def test_mask_keeps_everything_above_the_low_21_bits(self):
        self.assertEqual(FOCUSED_WINDOW & CLIENT_ID_MASK, 0x2800000)


class TestReadClassification(unittest.TestCase):
    """The owner's read log, without an X server behind it."""

    def setUp(self):
        self.owner = ClipboardOwner()

    def test_a_read_from_the_focused_client_is_a_paste(self):
        with self.owner.watch() as watch:
            self.owner._record_read(FOCUSED_HELPER)
            self.assertTrue(watch.landed(client_of(FOCUSED_WINDOW), 0.0))

    def test_a_clipboard_manager_polling_is_not_a_paste(self):
        """greenclip converts the selection about once a second regardless."""
        with self.owner.watch() as watch:
            self.owner._record_read(CLIPBOARD_MANAGER)
            self.assertFalse(watch.landed(client_of(FOCUSED_WINDOW), 0.0))

    def test_the_read_log_does_not_grow_forever(self):
        for _ in range(1200):
            self.owner._record_read(CLIPBOARD_MANAGER)
        self.assertLess(len(self.owner._reads), 1200)

    def test_nothing_read_it_at_all(self):
        with self.owner.watch() as watch:
            self.assertFalse(watch.landed(client_of(FOCUSED_WINDOW), 0.0))

    def test_a_read_by_anyone_else_is_not_an_answer(self):
        """Only the focused app taking the text says the paste landed.

        Trusting an unrecognised requestor meant the clipboard manager's
        once-a-second poll could pass for a paste, which it did on the first
        dictation of a session, before the manager had been recognised.
        """
        with self.owner.watch() as watch:
            self.owner._record_read(0x9900002)
            self.assertFalse(watch.landed(client_of(FOCUSED_WINDOW), 0.0))

    def test_with_no_focused_window_nothing_can_have_taken_it(self):
        with self.owner.watch() as watch:
            self.owner._record_read(0x9900002)
            self.assertFalse(watch.landed(None, 0.0))

    def test_reads_before_the_watch_opened_do_not_count(self):
        self.owner._record_read(0x9900002)
        with self.owner.watch() as watch:
            self.assertFalse(watch.landed(client_of(FOCUSED_WINDOW), 0.0))


class TestAvailability(unittest.TestCase):
    def test_no_display_means_no_owner(self):
        import os
        from unittest.mock import patch

        owner = ClipboardOwner()
        with patch.dict(os.environ, {"DISPLAY": ""}):
            self.assertFalse(owner.start())

    def test_stop_before_start_is_harmless(self):
        ClipboardOwner().stop()


if __name__ == "__main__":
    unittest.main()
