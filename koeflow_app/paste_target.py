from __future__ import annotations

import time

import keyboard


def paste_text_to_active_window(text: str) -> None:
    if not text.strip():
        return
    # Give the target window a brief moment after the hotkey
    time.sleep(0.05)
    # Type the text directly instead of going through the clipboard
    keyboard.write(text, delay=0)
