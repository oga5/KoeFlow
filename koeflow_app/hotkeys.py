from __future__ import annotations

import keyboard


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

    def register(self) -> None:
        self._toggle_handle = keyboard.add_hotkey(self.toggle_hotkey, self._on_toggle)
        self._confirm_handle = keyboard.add_hotkey(self.confirm_hotkey, self._on_confirm)
        if self.switch_model_hotkey and self._on_switch_model is not None:
            self._switch_model_handle = keyboard.add_hotkey(self.switch_model_hotkey, self._on_switch_model)
        if self.clear_buffer_hotkey and self._on_clear_buffer is not None:
            self._clear_buffer_handle = keyboard.add_hotkey(self.clear_buffer_hotkey, self._on_clear_buffer)

    def unregister(self) -> None:
        if self._toggle_handle is not None:
            keyboard.remove_hotkey(self._toggle_handle)
        if self._confirm_handle is not None:
            keyboard.remove_hotkey(self._confirm_handle)
        if self._switch_model_handle is not None:
            keyboard.remove_hotkey(self._switch_model_handle)
        if self._clear_buffer_handle is not None:
            keyboard.remove_hotkey(self._clear_buffer_handle)
