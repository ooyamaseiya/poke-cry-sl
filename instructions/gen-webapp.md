# 実装指示書：ポケモン鳴き声推定Webアプリ（FastAPI + ONNX）

## 1. プロジェクトの目的

すでに学習・生成済みの推論用ONNXモデル（`pokemon_model.onnx`）と設定ファイル（`labels.json`, `config.json`）を活用し、ユーザーがアップロードした音声ファイル（WAV, OGG）から隠れているポケモン（上位3候補）を推定するWebアプリを構築する。

---

## 2. 技術スタック・環境制約

- **バックエンド (APIサーバー):** Python 3.12+, FastAPI, Uvicorn, python-multipart
- **機械学習・音声処理:** `onnxruntime`, `librosa`, `soundfile`, `numpy`
- **フロントエンド:** HTML5, CSS3, Vanilla JavaScript（React等のフレームワークは不要、シンプルな構成とする）
- **前提条件:** 以下の学習成果物が `models/` ディレクトリに存在すること。
  - `models/pokemon_model.onnx`（推論モデル）
  - `models/labels.json`（クラスIDと名前の対応表）
  - `models/config.json`（音声前処理パラメータ）

---

## 3. ディレクトリ構成

以下の構成でWebアプリを実装すること。

    .
    ├── models/                 # (事前準備済みファイル)
    │   ├── pokemon_model.onnx
    │   ├── labels.json
    │   └── config.json
    ├── src/
    │   └── audio_processor.py  # 音声前処理モジュール
    ├── static/
    │   ├── index.html          # WebアプリのUI
    │   ├── style.css           # UIのスタイリング
    │   └── app.js              # API通信と結果の画面描画
    ├── main.py                 # FastAPIアプリケーション本体
    └── requirements.txt        # 必要なライブラリ一覧

---

## 4. バックエンド実装仕様（`main.py`, `src/audio_processor.py`）

1. **APIルーティング (`main.py`):**
   - FastAPIの `StaticFiles` を用いて、`static/` フォルダをマウントする。
   - `GET /`: `static/index.html` を返す（FileResponseを使用）。
   - `POST /predict`: `UploadFile` で音声ファイル（`WAV` または `OGG`）を受け取り、推論結果をJSONで返す。
2. **前処理ロジック（`src/audio_processor.py`）:**
   - サーバー起動時に `models/config.json` を読み込み、パラメータをメモリに保持。
   - アップロードされた音声をメモリ上（`io.BytesIO` 等）で読み込み、以下の前処理を実行。
     - 16kHz モノラル変換
     - 無音トリミング（前後0.15秒のマージン）
     - 2.5秒固定長（40,000サンプル）へのパディングまたはクロップ
     - 音量正規化（ピーク振幅 1.0）
     - ログメルスペクトログラム変換（標準化適用）
     - モデル入力テンソル形状 `[1, 1, 64, 251]` に整形して `numpy.ndarray` (float32) を返す。
3. **推論ロジック (`main.py` に組み込み):**
   - サーバー起動時のライフサイクル（`@asynccontextmanager` または起動時スクリプト）で `onnxruntime.InferenceSession` を用いてモデルを一度だけロードし、使い回す。
   - 前処理済みテンソルをONNXモデルに渡し、`logits` を取得。
   - `logits` にSoftmax関数を適用して確率（0.0〜1.0）に変換。
   - `models/labels.json` を参照し、確率が高い上位3候補（Top-3）を抽出。
4. **エラーハンドリング:**
   - ファイルが音声形式でない場合や、処理中にエラーが起きた場合は、適切なHTTPステータスコード（400等）とエラーメッセージを返すこと。
5. **APIレスポンス仕様（正常時）:**
   ```json
   {
     "success": true,
     "predictions": [
       { "rank": 1, "name": "01_rayquaza", "probability": 85.4 },
       { "rank": 2, "name": "00_other", "probability": 10.2 },
       { "rank": 3, "name": "02_pikachu", "probability": 4.4 }
     ]
   }
   ```

---

## 5. フロントエンド実装仕様（`static/`）

1. **UIデザイン (`index.html`, `style.css`):**
   - シンプルでスマートフォンでも操作しやすいレスポンシブデザイン。
   - 音声ファイル（.wav, .ogg）を選択する `<input type="file" accept="audio/*">` 要素。
   - 推論を実行する「判定する」ボタン。
   - 処理中であることを示すローディングインジケータ（CSSスピナー等）。
   - 結果を表示するエリア。
2. **JavaScriptロジック (`app.js`):**
   - ファイルが選択されていない場合はアラートを出す。
   - `FormData` を使って音声を `POST /predict` に送信。通信中はボタンを無効化（二重送信防止）し、ローディングを表示する。
   - レスポンスを受け取り、Top-3のポケモン名と確率（%）をプログレスバー（視覚的なゲージ）付きでリスト表示する。
   - エラーが発生した場合（通信エラー、非対応フォーマットなど）は、赤字等でエラーメッセージを画面に表示する。

---

## 6. 完了条件

1. 上記の要件を満たすファイル一式（`main.py`, `src/audio_processor.py`, `static/*`, `requirements.txt`）を生成する。
2. `uvicorn main:app --reload` でローカルサーバーを起動し、ブラウザ（`http://localhost:8000`）からWAVまたはOGGファイルをアップロードした際、エラーなく推論結果がTop-3まで画面に表示されること。
