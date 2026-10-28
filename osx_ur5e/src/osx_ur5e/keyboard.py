"""Non-blocking detection of exit keys (SPACE, ESC) typed in the controlling terminal."""

import os
import select
import sys
import termios
import tty

SPACE = 0x20
ESC = 0x1B
ESCAPE_SEQUENCE_PREFIXES = b"[O"  # ESC [ ... (CSI) and ESC O ... (SS3): arrow and function keys


def contains_exit_key(data: bytes) -> bool:
    """True if `data` holds a SPACE or a lone ESC; ESC that starts an escape sequence does not count."""
    for i, byte in enumerate(data):
        if byte == SPACE:
            return True
        if byte == ESC and (i + 1 == len(data) or data[i + 1] not in ESCAPE_SEQUENCE_PREFIXES):
            return True
    return False


class ExitKeys:
    """Puts a terminal in cbreak mode and reports SPACE or ESC presses; Ctrl-C keeps working.

    Use as a context manager so the terminal settings are restored. When `fd` is not a terminal
    the detector is inactive and pressed() always returns False.
    """

    def __init__(self, fd=None):
        self.fd = sys.stdin.fileno() if fd is None else fd
        self.active = os.isatty(self.fd)
        self._saved = None

    def __enter__(self):
        if self.active:
            self._saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc_info):
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def pressed(self) -> bool:
        """Consume pending input; True if it contained an exit key."""
        if not self.active:
            return False
        typed = b""
        while select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 64)
            if not chunk:
                break
            typed += chunk
        return contains_exit_key(typed)
