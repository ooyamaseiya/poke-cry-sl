# 実装指示書：ポケモン鳴き声推定モデル（教師あり学習 → ONNX生成）

## 1. プロジェクトの目的

アーケードゲーム（ポケモンフレンダ等）の騒音環境下において、録音した鳴き声（WAV）から隠れているポケモンの種類（上位候補 Top-3）を推定する音声分類モデルを構築し、推論用の `pokemon_model.onnx` を生成する。

※ 強化学習（RL）は使用せず、対象ポケモン＋「その他（雑音）」を1つのモデルで同時に分類する**教師ありマルチクラス分類（ディープラーニング）**で実装すること。

---

## 2. 技術スタック・環境制約

- **言語・FW:** Python 3.12+ / PyTorch（CPU環境での学習を前提に軽量設計にすること）
- **音声処理:** `soundfile`, `librosa`, `numpy`（※Python 3.13非互換の `pydub` は使用しないこと）
- **ONNX関連:** `onnx`, `onnxruntime`
- **基本方針:** CPUで高速に試行錯誤できるよう、以下の通りスクリプトを機能分割すること。
  1. `src/config.py`: 音声変換パラメータ・パス設定の一元管理
  2. `src/prepare_data.py`: 前処理・データ拡張・ログメルスペクトログラム化・保存
  3. `src/train.py`: 学習・精度評価・ONNXエクスポート
  4. `src/predict_onnx.py`: 生成したONNXファイルを使った推論テスト
  5. `src/create_dummy_data.py`: 実データ投入前に動作確認を行うためのテスト用WAV生成スクリプト

---

## 3. ディレクトリ構成

以下の構成でプロジェクトを作成すること。

    .
    ├── data/
    │   ├── raw/                  # 学習用原音WAV（フォルダ名＝クラス名とする）
    │   │   ├── 00_other/         # 雑音・BGM・話し声・無音など（必須の背景クラス）
    │   │   ├── 01_rayquaza/      # 対象ポケモン1
    │   │   └── 02_pikachu/       # 対象ポケモン2（以降同様にフォルダ追加で自動認識）
    │   ├── noise/                # データ拡張用の環境音・雑音WAV（任意の複数ファイル）
    │   └── processed/            # prepare_data.py が出力する学習用テンソルデータ
    ├── models/                   # 学習成果物の出力先
    │   ├── pokemon_model.onnx    # 推論用ONNXモデル
    │   ├── labels.json           # クラスIDとポケモン名の対応表
    │   └── config.json           # 推論時に一致させるべき前処理パラメータ
    ├── src/
    │   ├── config.py
    │   ├── create_dummy_data.py
    │   ├── prepare_data.py
    │   ├── train.py
    │   └── predict_onnx.py
    └── requirements.txt

---

## 4. 音声前処理・音響特徴量の仕様（`config.py` / `prepare_data.py`）

学習時と推論時の計算ズレを防ぐため、全スクリプトで共通の関数・定数を使用すること。

1. **音声読み込み・基本整形**
   - サンプリングレート: `16000 Hz`（モノラル変換）
   - 無音トリミング: `librosa.effects.trim(top_db=30)` で前後の無音を除去し、前後 `0.15秒` の余白（マージン）を残す。
   - 固定長統一: **`2.5秒`（40,000サンプル）** に揃える。
     - 2.5秒未満の場合：後ろにゼロパディング（無音追加）。
     - 2.5秒超過の場合：振幅（エネルギー）が最大のピーク位置を中心に2.5秒分を切り出す。
   - 音量正規化: ピーク振幅が `1.0` になるよう正規化（無音の場合はゼロのまま）。

2. **ログメルスペクトログラム変換**
   - `sample_rate = 16000`
   - `n_fft = 1024`
   - `hop_length = 160`（10ms間隔）
   - `n_mels = 64`
   - `f_min = 50.0`, `f_max = 8000.0`
   - パワースペクトログラムをデシベル変換（`librosa.power_to_db(ref=np.max)`）し、標準化（平均0・分散1）を行う。
   - モデル入力テンソル形状: `(Batch, 1, 64, 251)` に固定すること。

---

## 5. データ拡張・データ分割の仕様（`prepare_data.py`）

ゲームセンターの騒音下でも判定できるよう、以下の拡張処理を実装すること。

- **データリーク防止（重要）:**
  - データ拡張（水増し）を行う**前**に、原音ファイル単位で `Train（80%）` と `Val（20%）` に分割すること（原音が1ファイルしかないクラスは、水増し後にTrain/Valへ分割するフォールバック処理を入れること）。
- **水増し処理（Trainデータは1原音あたり40倍、Valデータは10倍に拡張）:**
  1. **背景ノイズ合成:** `data/noise/` 内のWAV（なければ自動生成したホワイトノイズ／ピンクノイズ）をランダムな区間切り出し、**SNR 0dB 〜 20dB** の範囲でランダムに合成。
  2. **時間軸シフト:** 前後に最大 `±0.2秒` ランダムにロールシフト。
  3. **音量変動:** `0.6倍 〜 1.2倍` のランダムゲイン適用後に再正規化。
  4. **SpecAugment（スペクトログラム変換後に適用）:** 時間マスク（最大20フレーム）と周波数マスク（最大8ビン）をランダムに適用。

---

## 6. モデル学習・ONNX出力の仕様（`train.py`）

- **モデル構造:** CPUで数分以内に学習が完了し、過学習しにくい軽量2D CNN（Conv2d + BatchNorm2d + ReLU + MaxPool2d を3〜4層重ね、Global Average Pooling + Dropout(0.3) + Linear でクラス数分を出力）を実装すること。
- **学習設定:**
  - 損失関数: `nn.CrossEntropyLoss()`
  - 最適化: `torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)`
  - エポック数: `30`（検証データ `Val` の正解率が最高になった時点のモデル重みを採用）
- **成果物の保存（`models/` 配下）:**
  1. `models/pokemon_model.onnx`:
     - `model.eval()` 状態でエクスポート。
     - 入力名: `["spectrogram"]`（形状: `[batch_size, 1, 64, 251]`、float32）
     - 出力名: `["logits"]`（形状: `[batch_size, num_classes]`、float32）
     - `dynamic_axes` でバッチサイズ次元を可変に設定。
  2. `models/labels.json`: `{"0": "00_other", "1": "01_rayquaza", ...}` の形式で保存。
  3. `models/config.json`: `sample_rate`, `duration`, `n_fft`, `hop_length`, `n_mels`, `f_min`, `f_max` を保存。

---

## 7. 推論テストスクリプト（`predict_onnx.py`）と完了条件

- コマンドライン引数で指定した任意のWAVファイル（例: `python src/predict_onnx.py path/to/test.wav`）を読み込み、`models/config.json` に基づいて前処理・スペクトログラム変換を実行。
- `onnxruntime` で `models/pokemon_model.onnx` をロードして推論し、Softmaxで確率（%）に変換した上で **上位3候補（Top-3）のクラス名と確率** をコンソールに表示すること。
- **完了条件:**
  1. 上記すべてのファイルと `requirements.txt` を作成する。
  2. `create_dummy_data.py` を実行してダミーの鳴き声（周波数の異なる電子音）と雑音を生成する。
  3. `prepare_data.py` → `train.py` → `predict_onnx.py` を実際に実行し、エラーなく `pokemon_model.onnx` が生成され、テスト音声に対して正しく1位の予測確率が返ることを確認する。
