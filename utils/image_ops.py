from __future__ import annotations

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Literal, Union

# --- 既存インポート ---
try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# --- 新規インポート (scikit‑image) ---
try:
    from skimage.color import rgb2hsv, rgb2lab
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

def resize_image(input_path, output_path, size=(32, 32)):
    """
    画像を指定したサイズ(デフォルトは32x32)にリサイズして保存する。

    :param input_path: 入力画像のファイルパス
    :param output_path: 出力画像のファイルパス
    :param size: リサイズ後の画像サイズ（デフォルトは32x32）
    """
    with Image.open(input_path) as img:
        img = img.resize(size, Image.Resampling.NEAREST)  # 最近傍補間でリサイズ
        img.save(output_path)

def count_colors(png_path: Union[str, Path]) -> int:
    """PNG ファイルに含まれるユニークカラー数を返す"""
    img = Image.open(png_path).convert("RGBA")          # 透過情報も含めて RGBA に統一
    # getcolors(maxcolors) は [(count, (R,G,B,A)) ...] のリストを返す
    # maxcolors に総ピクセル数を渡せばフルスキャンしてくれる
    colors = img.getcolors(maxcolors=img.width * img.height)
    if colors is None:
        raise ValueError("カラー数が非常に多くて取得できませんでした。")
    return len(colors)

def _binarize_alpha(alpha_arr: np.ndarray, *, threshold: int = 128) -> np.ndarray:
    """
    α配列を二値化して返す (0 or 255)
      * α < threshold → 0   (完全透明)
      * α ≥ threshold → 255 (完全不透明)
    デフォルト threshold=128 なので「α=0のみ透明、それ以外は不透明化」
    """
    return np.where(alpha_arr >= threshold, 255, 0).astype(np.uint8)

# ====================================================
# メイン API
# ====================================================

def reduce_colors(
    img: Image.Image,
    n_colors: int,
    *,
    method: Literal["auto", "kmeans", "pillow", "accent"] = "auto",
    max_iter: int = 100,
) -> Image.Image:
    """
    * α を 0/255 の二値に丸めてから
    * 不透明画素だけ n_colors 色に減色する

    Parameters
    ----------
    img : PIL.Image
        入力画像 (どんな mode でも OK)
    n_colors : int
        出力時に不透明画素が持つ RGB 色数
    method : "auto" | "kmeans" | "pillow" | "accent"
        - "auto"    : scikit-learn が入っていれば kmeans、なければ pillow
        - "kmeans"  : k-means クラスタリング (scikit-learn 必須)
        - "pillow"  : Pillow median-cut
        - "accent"  : 頻度 + 彩度で「特徴色」を残す (scikit-image 必須)
    max_iter : int
        k-means の最大イテレーション
    """
    # ---------- ① 入力 RGBA ＆ α 二値化 ----------
    img_rgba = img.convert("RGBA")
    arr      = np.asarray(img_rgba, dtype=np.uint8).copy()        # H×W×4
    h, w, _  = arr.shape

    alpha_bin          = _binarize_alpha(arr[..., 3])
    arr[..., 3]        = alpha_bin                                # α 上書き
    opaque_mask        = alpha_bin == 255                         # True/False
    opaque_pixels_rgb  = arr[..., :3][opaque_mask]                # N×3

    if opaque_pixels_rgb.size == 0:
        # 完全透過画像の場合は α だけ二値化して返す
        return Image.fromarray(arr, mode="RGBA")

    # ---------- ② 減色 ----------
    if method == "auto":
        method = "kmeans" if SKLEARN_OK else "pillow"

    if method == "kmeans":
        if not SKLEARN_OK:
            raise RuntimeError("scikit-learn が見つかりません")
        palette = _palette_via_kmeans(opaque_pixels_rgb, n_colors, max_iter)
        mapped  = _apply_palette(opaque_pixels_rgb, palette)      # N×3
    elif method == "pillow":
        palette = _palette_via_pillow(opaque_pixels_rgb, n_colors)
        mapped  = _apply_palette(opaque_pixels_rgb, palette)
    elif method == "accent":
        palette = _palette_via_accent(opaque_pixels_rgb, n_colors)
        mapped  = _apply_palette(opaque_pixels_rgb, palette)
    else:
        raise ValueError(f"unknown method: {method}")

    # ---------- ③ マッピング結果を元画像に戻す ----------
    arr[..., :3][opaque_mask] = mapped

    # 透明ピクセルのRGB値は必ず0にする（念のため明示的に保証）
    arr[..., :3][~opaque_mask] = 0
    arr[..., 3][~opaque_mask] = 0  # α=0も明示

    return Image.fromarray(arr, mode="RGBA")

def map_to_palette(img: Image.Image, palette: list[tuple[int, int, int, int]], use_lab: bool = False) -> Image.Image:
    """
    画像の各ピクセルを、指定されたRGBAパレットの中で最も近い色（RGBまたはLab空間で判定）に置き換える。
    透明部分はRGB=0, α=0で維持。
    :param img: 入力画像 (PIL.Image)
    :param palette: RGBAカラーのリスト [(R,G,B,A), ...]
    :param use_lab: Lab色空間で距離計算する場合True
    :return: 変換後画像 (PIL.Image)
    """
    arr = np.array(img.convert("RGBA"))
    h, w, c = arr.shape
    arr_flat = arr.reshape(-1, 4)
    palette_np = np.array(palette, dtype=np.float32)
    palette_rgb = palette_np[:, :3]
    if use_lab:
        from skimage.color import rgb2lab
        palette_lab = rgb2lab(palette_rgb[np.newaxis, :, :] / 255.0)[0]
    mapped = np.empty_like(arr_flat)
    for i, px in enumerate(arr_flat):
        if px[3] == 0:
            mapped[i] = (0, 0, 0, 0)
        else:
            src_rgb = px[:3].astype(np.float32)
            if use_lab:
                from skimage.color import rgb2lab
                src_lab = rgb2lab(src_rgb[np.newaxis, np.newaxis, :] / 255.0)[0,0]
                dists = np.sum((palette_lab - src_lab) ** 2, axis=1)
            else:
                dists = np.sum((palette_rgb - src_rgb) ** 2, axis=1)
            best_idx = np.argmin(dists)
            best = palette_np[best_idx]
            mapped[i, :3] = best[:3]
            mapped[i, 3] = px[3]
    mapped = mapped.reshape(h, w, 4)
    return Image.fromarray(mapped.astype(np.uint8), mode="RGBA")

# ====================================================
# 内部ユーティリティ
# ====================================================

def _palette_via_kmeans(data: np.ndarray, k: int, max_iter: int) -> np.ndarray:
    """N×3 RGB → k×3 palette (uint8)"""
    km = KMeans(
        n_clusters=k,
        n_init="auto",
        max_iter=max_iter,
        random_state=0,
    ).fit(data.astype(np.float32))
    return km.cluster_centers_.astype(np.uint8)

def _palette_via_pillow(data: np.ndarray, k: int) -> np.ndarray:
    """
    Pillow の median-cut でパレット生成。
    data は N×3 RGB (uint8)
    """
    # 1行画像にして quantize すると Pillow がパレットを作ってくれる
    line_img = Image.fromarray(data.reshape(1, -1, 3), mode="RGB")
    pal_img  = line_img.quantize(colors=k, method=Image.MEDIANCUT, dither=Image.NONE)
    # パレットは 768byte (256×RGB) の配列。先頭 k 色を取り出す。
    full_pal = pal_img.getpalette()[: k * 3]
    return np.array(full_pal, dtype=np.uint8).reshape(-1, 3)

def _palette_via_accent(
    pixels_rgb: np.ndarray,
    n_colors: int,
    *,
    min_lab_dist: float = 20.0,
) -> np.ndarray:
    """
    頻度と彩度スコアで「特徴色」を抽出してパレットを返す。

    - スコア = 0.7 * frequency_norm + 0.3 * saturation
      * frequency_norm : その色の画素数 / 最大画素数
      * saturation     : HSV の S 成分 (0〜1)
    - スコア順に Lab 色差が一定以上離れた色を n_colors 色選択
    """
    if not SKIMAGE_OK:
        raise RuntimeError("scikit-image が見つかりません (pip install scikit-image)")

    # --- unique 色と頻度を取得 ---
    colors, counts = np.unique(pixels_rgb.reshape(-1, 3), axis=0, return_counts=True)
    counts_norm = counts / counts.max()

    # --- 彩度を計算 ---
    hsv = rgb2hsv(colors.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    sat = hsv[:, 1]                                     # 0〜1

    # --- スコアリング & ソート ---
    score = 0.7 * counts_norm + 0.3 * sat
    sort_idx = np.argsort(score)[::-1]                  # 高スコア順

    # --- Lab 空間で類似色を間引きつつ選択 ---
    selected = []
    lab_selected = None

    for idx in sort_idx:
        c = colors[idx]
        if lab_selected is None:
            selected.append(c)
            lab_selected = rgb2lab(c[np.newaxis, np.newaxis] / 255.0).reshape(1, 3)
        else:
            lab_c = rgb2lab(c[np.newaxis, np.newaxis] / 255.0).reshape(1, 3)
            dist = np.linalg.norm(lab_selected - lab_c, axis=1).min()
            if dist >= min_lab_dist:
                selected.append(c)
                lab_selected = np.vstack([lab_selected, lab_c])
        if len(selected) >= n_colors:
            break

    # 足りない場合はスコア順に追加
    if len(selected) < n_colors:
        for idx in sort_idx:
            c = colors[idx]
            if not any((c == np.array(selected)).all(axis=1)):
                selected.append(c)
            if len(selected) >= n_colors:
                break

    return np.array(selected[:n_colors], dtype=np.uint8)

def _apply_palette(data: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    data  : N×3 RGB (uint8)
    palette: k×3 RGB
    それぞれ最近傍のパレット色に置き換えて返す (N×3 uint8)
    """
    # 距離計算をベクトル化
    diff   = data[:, None, :].astype(int) - palette[None, :, :].astype(int)  # N×k×3
    dist2  = (diff ** 2).sum(axis=2)                                         # N×k
    idx    = dist2.argmin(axis=1)                                            # N
    return palette[idx]

# ====================================================
# 可視化補助
# ====================================================

def show_side_by_side(original: Image.Image, reduced: Image.Image) -> None:
    """オリジナルと減色後を並べて表示"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(reduced);  ax[1].set_title("Reduced");  ax[1].axis("off")
    plt.tight_layout()