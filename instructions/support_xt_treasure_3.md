# 実装指示書：ポケモン鳴き声推定モデル（多クラス拡張版）

## 1. プロジェクトの目的

アーケードゲーム（ポケモンフレンダ等）の騒音環境下において、数十〜数百種類のポケモンの鳴き声を高精度に識別できるようモデルを拡張する。
学習対象のポケモンをコマンドライン引数ではなく `targets.csv` で一元管理し、多クラス分類に耐えうる学習ループ（Early Stopping、学習率スケジューラ）を導入した上で、単一の推論用 `pokemon_model.onnx` を生成する。

---

## 2. 技術スタック・基本方針

- **言語・FW:** Python 3.12+ / PyTorch（CPU環境で動作する軽量設計）
- **処理の完全共通化:** 音声前処理およびスペクトログラム変換は、必ず `src/audio_processor.py` の `AudioProcessor` クラスに一元化し、学習（`prepare_data.py`）・推論（`predict_onnx.py`, Webアプリ）のすべてで共通利用すること（標準化のズレ等の再発防止）。

---

## 3. ディレクトリ構成

以下の構成に拡張すること。

    .
    ├── targets.csv               # NEW: 学習対象のポケモンリスト（図鑑番号,英語名）
    ├── data/
    │   ├── raw/                  # 実データ格納ディレクトリ
    │   │   ├── 00_other/         # 背景音・ノイズクラス
    │   │   ├── 01_rayquaza/      # クラス名は「連番_名前」
    │   │   └── ...
    │   ├── noise/                # 合成用環境ノイズ
    │   └── processed/
    ├── models/
    │   ├── pokemon_model.onnx    # 単一ファイル出力
    │   ├── labels.json
    │   └── config.json
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── setup_real_data.py    # targets.csv を読み込んでPokeAPIから一括取得
    │   ├── audio_processor.py    # 前処理共通クラス
    │   ├── prepare_data.py
    │   ├── train.py              # Early Stopping / スケジューラ追加
    │   └── predict_onnx.py
    └── requirements.txt

---

## 4. 学習対象リスト（`targets.csv`）と実データ取得（`src/setup_real_data.py`）

多クラス化に対応するため、データの取得元をCSV管理に変更する。

1. **`targets.csv` の仕様:**
   - プロジェクトルートに作成。ヘッダーなし、カンマ区切りで `図鑑番号,英語名` を記述。
   - 例:
     ```csv
     384,rayquaza
     25,pikachu
     6,charizard
     150,mewtwo
     # 以降、必要なだけ追記可能
     ```
2. **`setup_real_data.py` の改修:**
   - コマンドライン引数ではなく `targets.csv` を読み込み、対象ポケモンの `latest` と `legacy` の `.ogg` をPokeAPIから一括ダウンロードする。
   - ダウンロード先のフォルダ名は `01_rayquaza`, `02_pikachu` のようにCSVの行順（1からの連番）で採番すること（`00_other` は固定）。
   - 既存ダミーデータの削除、および `00_other` 向け多様な背景音・ノイズ生成処理は維持すること。

---

## 5. 音声前処理の厳密な制約（`src/audio_processor.py`）※維持・厳守

- **16000Hz モノラル**、**`librosa.effects.trim(top_db=30)`**（前後0.15秒マージン）。
- 固定長 **2.5秒（40,000サンプル）**。超過時はピーク振幅中心で切り出し。
- ピーク振幅 `1.0` 正規化。
- ログメルスペクトログラム変換後、必ず**サンプル単位の標準化 `(S_db - mean) / (std + 1e-8)`** を適用し、テンソル形状を `[Batch, 1, 64, 251]` に固定すること。

---

## 6. データ拡張・分割の制約（`src/prepare_data.py`）※維持・厳守

1. **クラスごとの安全な Train / Val 分割:**
   - 原音が1〜2ファイルしかないクラスは、水増し（約50パターン）後に各原音由来データを `Train 80% / Val 20%` に分割すること。
2. **水増し処理（データリーク・破損防止）:**
   - クリーン原音（軽微な時間シフトのみ）も必ずTrain/Val両方に含めること。
   - `np.roll` は禁止（時間シフトはゼロパディングで行う）。
   - 音量変動（0.6〜1.2倍）、ノイズ合成（SNR 0〜20dB）、SpecAugment（Trainのみ）を実施。
   - Valデータにも固定乱数シードで軽〜中度のノイズやシフトを付与し、「同じ画像の単純コピー」による甘い評価を防ぐこと。
3. **メモリ最適化:**
   - クラス数増加に伴うRAM不足を防ぐため、リストを `.pkl` で一括保存する際は、変数の使い回しや適宜ガベージコレクション(`gc.collect()`)を行うこと。

---

## 7. 多クラス対応のモデル学習・ONNX出力（`src/train.py`）

クラス数増加に対応するため、学習ループを強化する。

1. **モデル構造:**
   - Conv2d(4層) + ReLU + BatchNorm2d + MaxPool2d -> Global Average Pooling -> Dropout(0.3) -> Linear の軽量CNN構造を維持。
2. **学習最適化（多クラス向け強化）:**
   - エポック数: 最大 `50`
   - 学習率スケジューラ: `torch.optim.lr_scheduler.ReduceLROnPlateau` を導入し、Val Lossが下がらなくなった場合に学習率を下げること（factor=0.5, patience=3）。
   - **Early Stopping（早期終了）:** Val Acc が `7` エポック連続で改善しなかった場合、無駄な学習を打ち切ること。
   - ベストモデルの保存: Val Acc が最高になった時点の重みを保存すること。
3. **単一ファイルONNXの出力:**
   - `model.eval()` 状態で `models/pokemon_model.onnx` に出力。
   - `load_external_data=True` 等を用いて、外部データ(`.data`)に分離させず**必ず単一の `.onnx` ファイルに統合保存**すること。

---

## 8. 完了条件

1. プロジェクトルートに `targets.csv`（10〜20種類程度の適当なポケモンを記載）を作成する。
2. `python -m src.setup_real_data` を実行し、すべての音声が取得・配置されることを確認。
3. `python -m src.prepare_data` → `python -m src.train` を実行し、多クラスでもエラーなく学習が進み、Early Stopping または最大エポック到達後に単一の `pokemon_model.onnx` が出力されることを確認する。
