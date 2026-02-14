from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class OverlayUI:
    def __init__(
        self,
        title: str,
        toggle_hotkey: str,
        confirm_hotkey: str,
        switch_model_hotkey: str,
        clear_buffer_hotkey: str,
    ) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("560x260+40+40")
        self.root.attributes("-topmost", True)

        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="待機中")
        status_label = ttk.Label(frame, textvariable=self.status_var)
        status_label.pack(anchor="w", pady=(0, 8))

        self.model_var = tk.StringVar(value="モデル: -")
        model_label = ttk.Label(frame, textvariable=self.model_var)
        model_label.pack(anchor="w", pady=(0, 8))

        self.text = tk.Text(frame, height=10, wrap="word")
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        hint = ttk.Label(
            frame,
            text=(
                f"{toggle_hotkey}: 録音トグル / "
                f"{confirm_hotkey}: 貼り付け / "
                f"{switch_model_hotkey}: モデル切替 / "
                f"{clear_buffer_hotkey}: バッファクリア"
            ),
        )
        hint.pack(anchor="w", pady=(8, 0))

    def set_status(self, status: str) -> None:
        self.status_var.set(status)

    def set_model(self, model_name: str) -> None:
        self.model_var.set(f"モデル: {model_name}")

    def set_text(self, value: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", value)
        self.text.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()
