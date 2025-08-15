# streamlit_image_toolkit

Streamlit製画像ユーティリティ：ブラウザでリサイズ・カラー数計測・減色を手軽に

---

## 概要

このツールは、画像のリサイズ・カラー数計測・減色・指定パレットへの色寄せなど、画像処理をブラウザ上で手軽に行えるStreamlitアプリです。インストール後、コマンド一つでWebアプリとして起動できます。

## 主な機能

- 画像のリサイズ（最近傍補間）
- RGBAユニークカラー数のカウント
- 減色処理（K-Means, Pillow median-cut, 特徴色抽出）
- 任意のカラーリストへの色寄せ（RGB/Lab空間対応）
- アルファ値の2値化
- オリジナル画像と処理後画像の並列表示
- 結果画像のダウンロード

## インストール

1. リポジトリをクローンまたはダウンロード
2. 必要なパッケージをインストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
streamlit run app.py
```

ブラウザが自動で開かない場合は、表示されたURL（例: http://localhost:8501）を開いてください。

## 機能詳細

### 画像リサイズ
- パーセント指定・幅指定・高さ指定でリサイズ可能
- アルファ値の2値化オプションあり

### カラー減色
- 目標カラー数・アルゴリズム選択（auto/kmeans/pillow/accent）
- リサイズ後画像を減色対象に選択可能
- 減色後のカラーサンプル一覧表示
- **autoアルゴリズムについて**: `auto`を選択すると、scikit-learnがインストールされていればkmeans法、なければPillow median-cut法が自動的に使われます。
- `accent`はscikit-imageが必要です。

### 指定カラーに寄せる
- RGBAリストを指定し、最も近い色に置換
- RGB/Lab色空間で距離計算を選択可能
- 変換後のユニークカラー数表示
- 使用されたカラーサンプル一覧表示
- **パレット指定方法の例**:
  - 下記のように1行ずつ `RGB(R, G, B, A)` 形式で入力してください。
  - 例：
    ```
    RGB(0, 0, 0, 0)
    RGB(255, 0, 0, 255)
    RGB(0, 255, 0, 255)
    RGB(0, 0, 255, 255)
    ```
  - JSONやCSV形式は直接サポートしていません。

## 必要パッケージ

- streamlit
- pillow
- numpy
- matplotlib（並列表示用、任意）
- scikit-learn（KMeans減色用、任意）
- scikit-image（accent/色寄せLab空間用、任意）

`requirements.txt` で一括インストールできます。

## ライセンス

MIT License

---

## Author

shimamura
