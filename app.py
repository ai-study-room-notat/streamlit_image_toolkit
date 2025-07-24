# =============================================================
# 画像処理ユーティリティ – Streamlit アプリ (app.py)
# =============================================================
#   * 画像リサイズ（最近傍補間）
#   * RGBA のユニークカラー数をカウント
#   * K‑Means または Pillow median‑cut による減色
# -------------------------------------------------------------
#  必須パッケージ:
#    - streamlit
#    - pillow
#    - numpy
#    - matplotlib (サイドバイサイド表示を有効にする場合のみ)
#    - scikit‑learn (KMeans を使う場合のみ)
# 
#  実行方法:
#      $ pip install -r requirements.txt
#      $ streamlit run app.py
# =============================================================

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ---- ローカルユーティリティ ----------------------------------
# リポジトリ内 utils/image_ops.py に配置された関数群をインポート
from utils.image_ops import (
    resize_image,
    count_colors,
    reduce_colors,
)

# -------------------------------------------------------------
# ページ設定
# -------------------------------------------------------------
st.set_page_config(
    page_title="画像処理ユーティリティ",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ 画像処理ユーティリティ")
st.markdown(
    """
ブラウザ上で **画像のリサイズ・カラー解析・減色処理** を気軽に行える Streamlit アプリです。
    """
)

# -------------------------------------------------------------
# サイドバー – 操作選択
# -------------------------------------------------------------
operation = st.sidebar.selectbox(
    "操作を選択してください",
    (
        "画像をリサイズ",
        "カラーを減色",
        "指定カラーに寄せる",  # 新モード追加
    ),
)

# -------------------------------------------------------------
# 共通 – 画像アップロード
# -------------------------------------------------------------
# -------------------------------------------------------------
# アルファ値2値化オプション
# -------------------------------------------------------------
binarize_alpha = st.checkbox("アルファ値を0か255に2値化する", value=True)
if binarize_alpha:
    alpha_threshold = st.slider("アルファ値の閾値", min_value=0, max_value=255, value=128)

uploaded_file = st.file_uploader(
    "PNG または JPEG 画像をアップロードしてください", type=["png", "jpg", "jpeg"], key="uploader"
)

if uploaded_file is None:
    st.info("👆 画像をアップロードすると操作を開始できます。")
    st.stop()

# PIL Image インスタンス
pil_image = Image.open(uploaded_file)

# --- アルファ値2値化処理 ---
def binarize_alpha_channel(img: Image.Image, threshold: int) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[..., 3]
    arr[..., 3] = np.where(alpha >= threshold, 255, 0)
    return Image.fromarray(arr)  # Pillow 13以降も警告なし

if binarize_alpha:
    pil_image = binarize_alpha_channel(pil_image, alpha_threshold)

# 共通 – カラー数カウント
with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    unique = count_colors(Path(tmp.name))

# -------------------------------------------------------------
# 画像リサイズ
# -------------------------------------------------------------
if operation == "画像をリサイズ":
    resize_mode = st.radio(
        "リサイズ方法を選択",
        ("パーセント指定", "幅を指定", "高さを指定"),
        horizontal=True
    )
    orig_w, orig_h = pil_image.size
    if resize_mode == "パーセント指定":
        percent = st.slider("リサイズ率（%）", min_value=1, max_value=400, value=100)
        target_w = int(orig_w * percent / 100)
        target_h = int(orig_h * percent / 100)
        st.write(f"新しいサイズ: {target_w}×{target_h} px")
    elif resize_mode == "幅を指定":
        target_w = st.number_input("幅 (px)", min_value=1, value=orig_w, step=1)
        target_h = int(orig_h * (target_w / orig_w))
        st.write(f"新しいサイズ: {target_w}×{target_h} px")
    else:  # 高さを指定
        target_h = st.number_input("高さ (px)", min_value=1, value=orig_h, step=1)
        target_w = int(orig_w * (target_h / orig_h))
        st.write(f"新しいサイズ: {target_w}×{target_h} px")

    st.metric("ユニーク RGBA カラー数", unique)

    if st.button("リサイズ実行"):
        with tempfile.NamedTemporaryFile(suffix=".png") as src, tempfile.NamedTemporaryFile(suffix=".png") as dst:
            src.write(uploaded_file.getvalue())
            src.flush()
            resize_image(src.name, dst.name, size=(int(target_w), int(target_h)))
            resized = Image.open(dst.name)

            st.image(resized, caption=f"リサイズ結果 {target_w}×{target_h}", use_container_width=True)

            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            st.download_button(
                label="リサイズ後の PNG をダウンロード",
                data=buf.getvalue(),
                file_name=f"resized_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
            )
            # --- 追加: リサイズ画像をセッションに保存 ---
            st.session_state["resized_img"] = resized
            st.session_state["use_resized_for_reduce"] = False

# -------------------------------------------------------------
# カラー減色
# -------------------------------------------------------------
elif operation == "カラーを減色":
    st.metric("ユニーク RGBA カラー数", unique)
    n_col = st.slider("目標カラー数", min_value=2, max_value=256, value=16)
    method = st.selectbox(
        "減色アルゴリズム",
        options=["auto", "kmeans", "pillow", "accent"],
        help="auto → scikit-learn があれば kmeans、無ければ Pillow median-cut。accent → 頻度+彩度で特徴色抽出 (scikit-image 必須)"
    )

    # --- 追加: リサイズ画像利用切替UI ---
    target_img = pil_image
    if "resized_img" in st.session_state:
        # 追加: リサイズ画像のユニークカラー数を表示
        resized_img = st.session_state["resized_img"]
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            resized_img.save(tmp.name, format="PNG")
            resized_unique = count_colors(Path(tmp.name))
        st.metric("リサイズ後画像のユニーク RGBA カラー数", resized_unique)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("リサイズ後の画像を使う"):
                st.session_state["use_resized_for_reduce"] = True
        with col_btn2:
            if st.button("元画像に戻す"):
                st.session_state["use_resized_for_reduce"] = False
        if st.session_state.get("use_resized_for_reduce"):
            st.info("減色対象: リサイズ後の画像")
            target_img = st.session_state["resized_img"]
        else:
            st.info("減色対象: 元画像")
    # --- ここまで追加 ---

    if st.button("減色実行"):
        reduced_img = reduce_colors(target_img, n_col, method=method)
        st.session_state["reduced_img"] = reduced_img

    if "reduced_img" in st.session_state:
        reduced_img = st.session_state["reduced_img"]
        if st.checkbox("オリジナルと並べて表示"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(target_img, caption="オリジナル", use_container_width=True)
            with col2:
                st.image(reduced_img, caption=f"減色後 ({n_col} colours)", use_container_width=True)
        else:
            st.image(reduced_img, caption=f"減色後 ({n_col} colours)", use_container_width=True)

        # ダウンロード
        out_buf = io.BytesIO()
        reduced_img.save(out_buf, format="PNG")
        st.download_button(
            label="減色後の PNG をダウンロード",
            data=out_buf.getvalue(),
            file_name=f"reduced_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
        )
        # --- 追加: 減色後カラーサンプル一覧 ---
        arr = np.array(reduced_img)
        # RGBA 形式の場合
        if arr.shape[-1] == 4:
            colors = np.unique(arr.reshape(-1, 4), axis=0)
        else:
            colors = np.unique(arr.reshape(-1, 3), axis=0)
        st.markdown("### 減色後のカラーサンプル一覧")
        for c in colors:
            rgb = tuple(int(x) for x in c[:3])
            rgba_str = f"({c[0]}, {c[1]}, {c[2]})" if len(c) == 3 else f"({c[0]}, {c[1]}, {c[2]}, {c[3]})"
            st.markdown(f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                        f"<div style='width:24px;height:24px;background:rgb{rgb};border:1px solid #ccc;margin-right:8px;'></div>"
                        f"<span style='font-size:0.95rem;'>RGB{rgba_str}</span>"
                        f"</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 指定カラーに寄せる
# -------------------------------------------------------------
elif operation == "指定カラーに寄せる":
    st.metric("ユニーク RGBA カラー数", unique)
    st.markdown("#### 寄せたいカラーリスト (RGBA, 1行1色)")
    color_text = st.text_area(
        "RGBAリストを貼り付け・入力してください (例: RGB(0,0,0,0))",
        height=200,
        value="RGB(0, 0, 0, 0)\nRGB(0, 0, 1, 255)\nRGB(44, 37, 19, 255)\nRGB(85, 52, 4, 255)\nRGB(166, 103, 37, 255)\nRGB(211, 165, 83, 255)\nRGB(228, 141, 46, 255)\nRGB(233, 206, 143, 255)\nRGB(247, 209, 81, 255)"
    )
    use_lab = st.checkbox("Lab色空間で距離計算する (より人間の感覚に近い色寄せ)", value=False)
    def parse_rgba_list(text):
        import re
        pattern = r"RGB\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
        return [tuple(map(int, m)) for m in re.findall(pattern, text)]
    palette = parse_rgba_list(color_text)
    if st.button("指定カラーに寄せる実行"):
        from utils.image_ops import map_to_palette
        mapped_img = map_to_palette(pil_image, palette, use_lab=use_lab)
        st.session_state["mapped_img"] = mapped_img
        # 実行後ユニークカラー数
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            mapped_img.save(tmp.name, format="PNG")
            mapped_unique = count_colors(Path(tmp.name))
        st.session_state["mapped_unique"] = mapped_unique
    if "mapped_img" in st.session_state:
        mapped_img = st.session_state["mapped_img"]
        mapped_unique = st.session_state.get("mapped_unique", None)
        if mapped_unique is not None:
            st.metric("変換後ユニーク RGBA カラー数", mapped_unique)
        # 並べて表示
        if st.checkbox("オリジナルと並べて表示"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_image, caption="オリジナル", use_container_width=True)
            with col2:
                st.image(mapped_img, caption="指定カラーに寄せた画像", use_container_width=True)
        else:
            st.image(mapped_img, caption="指定カラーに寄せた画像", use_container_width=True)
        buf = io.BytesIO()
        mapped_img.save(buf, format="PNG")
        st.download_button(
            label="変換後の PNG をダウンロード",
            data=buf.getvalue(),
            file_name=f"mapped_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
        )
        # カラーサンプル一覧
        arr = np.array(mapped_img)
        colors = np.unique(arr.reshape(-1, 4), axis=0)
        st.markdown("### 使用されたカラーサンプル一覧")
        for c in colors:
            rgb = tuple(int(x) for x in c[:3])
            rgba_str = f"({c[0]}, {c[1]}, {c[2]}, {c[3]})"
            st.markdown(f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                        f"<div style='width:24px;height:24px;background:rgb{rgb};border:1px solid #ccc;margin-right:8px;'></div>"
                        f"<span style='font-size:0.95rem;'>RGB{rgba_str}</span>"
                        f"</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# フッター
# -------------------------------------------------------------
st.markdown(
    """
<hr/>
<div style="text-align:center; font-size:0.9rem;">
    MIT License で公開 • Built with Streamlit
</div>
""",
    unsafe_allow_html=True,
)
