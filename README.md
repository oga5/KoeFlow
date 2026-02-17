# KoeFlow

ローカルCPUで動く音声入力アプリです。  
- `Ctrl+Alt+V`: 録音開始/停止（トグル）
- `Enter`: （KoeFlowウィンドウがアクティブなときのみ）現在の文字起こしを貼り付け
- `Ctrl+Alt+M`: モデル切り替え（`MODEL_PRESETS`を順番に循環）
- `Esc`: （KoeFlowウィンドウがアクティブなときのみ）バッファをクリア（録音中/待機中どちらでも可）

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
- `PRIMARY_MODEL_ID`: 優先モデル（既定: `Systran/faster-whisper-medium`）
- `MODEL_PRESETS`: 切替対象モデル一覧（カンマ区切り、例: `Systran/faster-whisper-medium,sherpa-onnx,sapi`）
  - `sherpa-onnx` を含めると、`Ctrl+Alt+M` の切替対象に Sherpa-ONNX バックエンドを追加できます
- `MODEL_PRESETS`: `sapi` を含めると、`Ctrl+Alt+M` の切替対象に Windows Speech Recognition (SAPI) を追加できます
- `FALLBACK_MODEL_ID`: CPU向けフォールバック（既定: `Systran/faster-whisper-medium`）
- `TRANSCRIBER_BACKEND`: 認識バックエンド（`auto` / `faster-whisper` / `sherpa-onnx` / `sapi`）
- `SHERPA_ONNX_MODEL_DIR`: Sherpa-ONNXモデルディレクトリ（既定: `.models/reazonspeech-v2-sherpa-onnx`）
- `SHERPA_ONNX_MODEL_TYPE`: Sherpa-ONNXモデル種別（`paraformer` / `zipformer-transducer`）
- `MODEL_CACHE_DIR`: モデル保存先（既定: `.models`）
- `SAMPLE_RATE`: 録音サンプルレート（既定: `48000`、デバイスが対応するなら `16000` 推奨）
- `AUDIO_CHUNK_SECONDS`: リアルタイム認識のチャンク秒数（既定: `0.6`、精度寄りは `0.8`〜`1.2`）
- `AUDIO_MAX_CHUNK_LATENCY_SECONDS`: 途中確定でチャンクを先送りする最大待ち時間（既定: `0.6`、レスポンス重視なら `0.35`〜`0.5`）
- `UI_PUMP_INTERVAL_MS`: UIポーリング間隔（既定: `120`、レスポンス重視なら `60`〜`100`）
- `REALTIME_MAX_CHARS_PER_SECOND`: リアルタイムで不自然に長い出力を破棄する上限（既定: `14`）
- `REALTIME_MERGE_MAX_OVERLAP_CHARS`: チャンク連結時の重複除去文字数（既定: `12`）
- `REALTIME_BEAM_SIZE` / `REALTIME_BEST_OF`: リアルタイム推論の探索幅（既定: `1` / `1`）
- `FINALIZE_BEAM_SIZE` / `FINALIZE_BEST_OF`: 確定時推論の探索幅（既定: `5` / `5`）
- `WHISPER_NO_SPEECH_THRESHOLD`: 無音判定しきい値（既定: `0.75`）
- `USE_LIGHT_VAD_FOR_WHISPER`: 軽量VADで非音声を削る（既定: `1`）
- `REALTIME_MIN_RMS` / `FINALIZE_MIN_RMS`: RMS下限（既定: `0.010` / `0.003`）
- `REALTIME_CONDITION_ON_PREVIOUS_TEXT`: リアルタイム推論で前文脈を使うか（既定: `0`）
- `FINALIZE_CONDITION_ON_PREVIOUS_TEXT`: 確定時推論で前文脈を使うか（既定: `1`）
- `ENABLE_FINALIZE_PASS`: 貼り付け前に全音声を再推論するか（既定: `1`）
- `MIC_MONITOR_LOG_ENABLED`: マイク監視ログを出すか（既定: `1`）
- `MIC_MONITOR_LOG_INTERVAL_SECONDS`: 監視ログ出力間隔（既定: `2.0` 秒）
- `MIC_MONITOR_RMS_THRESHOLD`: `speech=yes/no` 判定のRMSしきい値（既定: `0.008`）
- `TOGGLE_HOTKEY`: 録音トグル（既定: `ctrl+alt+v`）
- `CONFIRM_HOTKEY`: 貼り付け確定（既定: `enter`）
- `SWITCH_MODEL_HOTKEY`: モデル切替（既定: `ctrl+alt+m`）
- `CLEAR_BUFFER_HOTKEY`: バッファクリア（KoeFlowウィンドウがアクティブなときのみ、既定: `esc`）

### 日本語精度を優先する推奨 `.env`（CPU）

```env
MODEL_PRESETS=Systran/faster-whisper-medium,sherpa-onnx,sapi
TRANSCRIBER_BACKEND=faster-whisper

SAMPLE_RATE=16000
AUDIO_CHUNK_SECONDS=0.6
AUDIO_MAX_CHUNK_LATENCY_SECONDS=0.45
UI_PUMP_INTERVAL_MS=80

REALTIME_BEAM_SIZE=1
REALTIME_BEST_OF=1
FINALIZE_BEAM_SIZE=5
FINALIZE_BEST_OF=5
WHISPER_NO_SPEECH_THRESHOLD=0.75
USE_LIGHT_VAD_FOR_WHISPER=1
REALTIME_MIN_RMS=0.010
FINALIZE_MIN_RMS=0.003
REALTIME_MAX_CHARS_PER_SECOND=14
REALTIME_CONDITION_ON_PREVIOUS_TEXT=0
FINALIZE_CONDITION_ON_PREVIOUS_TEXT=1
REALTIME_MERGE_MAX_OVERLAP_CHARS=12
ENABLE_FINALIZE_PASS=1
MIC_MONITOR_LOG_ENABLED=1
MIC_MONITOR_LOG_INTERVAL_SECONDS=2.0
MIC_MONITOR_RMS_THRESHOLD=0.008
```

起動ログの `MIC monitor` 行でマイク入力有無（`speech=yes/no`）を確認でき、`STT realtime` / `STT finalize` 行でモデル・推論時間・RTF・ドロップ率を比較できます。

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

### Whisper と SAPI を切り替えて比較する

`.env` 例:

```env
MODEL_PRESETS=Systran/faster-whisper-medium,sapi
TRANSCRIBER_BACKEND=auto
```

`Ctrl+Alt+M` で Whisper / SAPI を順番に切り替えられます。

## 4. 注意

- 初回起動はモデルダウンロードで時間がかかります。
- HFモデルの利用条件同意が必要な場合は、事前にHugging Faceで承認し `huggingface-cli login` してください。
- `keyboard` は環境によって管理者権限が必要です。
- 4BモデルはCPUでは遅延が大きい場合があります。自動でフォールバックを使用します。
