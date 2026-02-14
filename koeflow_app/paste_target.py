from __future__ import annotations

import time

import keyboard
import pyperclip


def paste_text_to_active_window(text: str) -> None:
    if not text.strip():
        return
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")
