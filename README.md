# KoeFlow

ローカルCPUで動く音声入力アプリです。  
- `Ctrl+Alt+V`: 録音開始/停止（トグル）
- `Enter`: 現在の文字起こしをアクティブウィンドウに貼り付け
- `Ctrl+Alt+M`: モデル切り替え（`MODEL_PRESETS`を順番に循環）
- `Esc`: 録音中バッファをクリア（プレビューと未処理キューを破棄）

## 1. セットアップ

```powershell
D:\Tools\Python3.11.9\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. 実行

```powershell
.\.venv\Scripts\python.exe main.py
```

## 3. 設定

`.env` を作成して必要に応じて調整します（` .env.example `をコピー）。

主要設定:
- `PRIMARY_MODEL_ID`: 優先モデル（既定: `mistralai/Voxtral-Mini-4B-Realtime-2602`）
- `MODEL_PRESETS`: 切替対象モデル一覧（カンマ区切り、例: `openai/whisper-small,sherpa-onnx,.models/voxtral_xet`）
  - `sherpa-onnx` を含めると、`Ctrl+Alt+M` の切替対象に Sherpa-ONNX バックエンドを追加できます
- `FALLBACK_MODEL_ID`: CPU向けフォールバック（既定: `Systran/faster-whisper-small`）
- `TRANSCRIBER_BACKEND`: 認識バックエンド（`auto` / `faster-whisper` / `sherpa-onnx`）
- `SHERPA_ONNX_MODEL_DIR`: Sherpa-ONNXモデルディレクトリ（既定: `.models/reazonspeech-v2-sherpa-onnx`）
- `SHERPA_ONNX_MODEL_TYPE`: Sherpa-ONNXモデル種別（`paraformer` / `zipformer-transducer`）
- `MODEL_CACHE_DIR`: モデル保存先（既定: `.models`）
- `TOGGLE_HOTKEY`: 録音トグル（既定: `ctrl+alt+v`）
- `CONFIRM_HOTKEY`: 貼り付け確定（既定: `enter`）
- `SWITCH_MODEL_HOTKEY`: モデル切替（既定: `ctrl+alt+m`）
- `CLEAR_BUFFER_HOTKEY`: 録音中バッファクリア（既定: `esc`）

### ReazonSpeech (v2) + Sherpa-ONNX を試す

1. `sherpa-onnx` をインストール
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. ReazonSpeech v2 の Sherpa-ONNX 形式モデルを `SHERPA_ONNX_MODEL_DIR` に配置
   - 必須ファイル例: `tokens.txt` と `model.int8.onnx`（または `model.onnx` / `paraformer.onnx`）
3. `.env` で以下を設定
   ```env
   TRANSCRIBER_BACKEND=sherpa-onnx
   SHERPA_ONNX_MODEL_DIR=.models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01
   SHERPA_ONNX_MODEL_TYPE=zipformer-transducer
   ```

読み込み失敗時は自動で既存の `faster-whisper` フォールバックへ切り替わります。

## 4. 注意

- 初回起動はモデルダウンロードで時間がかかります。
- HFモデルの利用条件同意が必要な場合は、事前にHugging Faceで承認し `huggingface-cli login` してください。
- `keyboard` は環境によって管理者権限が必要です。
- 4BモデルはCPUでは遅延が大きい場合があります。自動でフォールバックを使用します。
