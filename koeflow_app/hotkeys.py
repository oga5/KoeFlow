from __future__ import annotations

import threading
import time

import keyboard


def _release_then_call(hotkey_str: str, callback) -> None:  # noqa: ANN001
    """Wait until all keys in the hotkey combo are released, then invoke callback."""
    parts = keyboard.parse_hotkey(hotkey_str)
    scan_codes: set[int] = set()
    for step in parts:
        for codes in step:
            scan_codes.update(codes)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not any(keyboard.is_pressed(sc) for sc in scan_codes):
            break
        time.sleep(0.02)
    callback()


class HotkeyManager:
    def __init__(
        self,
        toggle_hotkey: str,
        confirm_hotkey: str,
        on_toggle,
        on_confirm,
        switch_model_hotkey: str | None = None,
        on_switch_model=None,
        clear_buffer_hotkey: str | None = None,
        on_clear_buffer=None,
    ) -> None:  # noqa: ANN001
        self.toggle_hotkey = toggle_hotkey
        self.confirm_hotkey = confirm_hotkey
        self.switch_model_hotkey = switch_model_hotkey
        self.clear_buffer_hotkey = clear_buffer_hotkey
        self._on_toggle = on_toggle
        self._on_confirm = on_confirm
        self._on_switch_model = on_switch_model
        self._on_clear_buffer = on_clear_buffer
        self._toggle_handle = None
        self._confirm_handle = None
        self._switch_model_handle = None
        self._clear_buffer_handle = None

    @staticmethod
    def _wrap(hotkey_str: str, callback) -> callable:  # noqa: ANN001
        def _handler() -> None:
            threading.Thread(
                target=_release_then_call, args=(hotkey_str, callback), daemon=True,
            ).start()
        return _handler

    def register(self) -> None:
        self._toggle_handle = keyboard.add_hotkey(
            self.toggle_hotkey, self._wrap(self.toggle_hotkey, self._on_toggle),
        )
        self._confirm_handle = keyboard.add_hotkey(
            self.confirm_hotkey, self._wrap(self.confirm_hotkey, self._on_confirm),
        )
        if self.switch_model_hotkey and self._on_switch_model is not None:
            self._switch_model_handle = keyboard.add_hotkey(
                self.switch_model_hotkey, self._wrap(self.switch_model_hotkey, self._on_switch_model),
            )
        if self.clear_buffer_hotkey and self._on_clear_buffer is not None:
            self._clear_buffer_handle = keyboard.add_hotkey(
                self.clear_buffer_hotkey, self._wrap(self.clear_buffer_hotkey, self._on_clear_buffer),
            )

    def unregister(self) -> None:
        if self._toggle_handle is not None:
            keyboard.remove_hotkey(self._toggle_handle)
        if self._confirm_handle is not None:
            keyboard.remove_hotkey(self._confirm_handle)
        if self._switch_model_handle is not None:
            keyboard.remove_hotkey(self._switch_model_handle)
        if self._clear_buffer_handle is not None:
            keyboard.remove_hotkey(self._clear_buffer_handle)
