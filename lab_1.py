import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

def imread_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No se puede cargar la imagen: {path}")
    if len(img.shape) == 3:
        R = img[:, :, 2].astype(np.float32)
        G = img[:, :, 1].astype(np.float32)
        B = img[:, :, 0].astype(np.float32)
        gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    else:
        gray = img.astype(np.float32)
    gmin, gmax = gray.min(), gray.max()
    if gmax > gmin:
        norm = (gray - gmin) * (255.0 / (gmax - gmin))
    else:
        norm = np.zeros_like(gray, dtype=np.float32)
    gray = np.clip(norm, 0, 255).astype(np.uint8)
    return gray

def save_image_with_title(img, title, out_path):
    h, w = img.shape[:2]
    
    banner_h = max(30, min(80, int(w * 0.05)))
    
    font_scale = max(0.3, min(1.5, w / 1000.0))
    
    thickness = max(1, int(font_scale * 2))
    
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    
    banner = np.full((banner_h, w, 3), 255, dtype=np.uint8)
    vis = np.vstack([banner, vis])
    
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = max(8, (w - text_size[0]) // 2)
    text_y = banner_h - max(8, (banner_h - text_size[1]) // 2)
    
    cv2.putText(vis, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)

def histogram(img):
    img_u8 = img.astype(np.uint8)
    counts = np.bincount(img_u8.flatten(), minlength=256)
    return counts

def save_histogram_plot(img, title, out_path):
    counts = histogram(img)
    plt.figure(figsize=(6,4))
    plt.bar(np.arange(256), counts, width=1.0)
    plt.title(title)
    plt.xlabel("Nivel de gris")
    plt.ylabel("Frecuencia de pixeles")
    plt.xlim(0,255)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def pair_save(img, header_title, base_name, out_dir):
    ih, iw = img.shape[:2]
    title = f"{header_title} | Dim: {iw}x{ih}"
    img_path  = out_dir / f"{base_name}.png"
    hist_path = out_dir / f"{base_name}_hist.png"
    save_image_with_title(img, title, img_path)
    save_histogram_plot(img, f"Histograma: {header_title}", hist_path)
    return img_path, hist_path

def rotate_image(img, angle_deg):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return rotated

def clip_u8(x): return np.clip(x, 0, 255).astype(np.uint8)

def downscale(img, k):
    h, w = img.shape[:2]
    new_h, new_w = max(1, h // k), max(1, w // k)
    
    result = np.zeros((new_h, new_w), dtype=img.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            src_i = i * k
            src_j = j * k
            
            src_i = min(src_i, h - 1)
            src_j = min(src_j, w - 1)
            
            result[i, j] = img[src_i, src_j]
    
    return result

def upscale(img, k):
    h, w = img.shape[:2]
    new_h, new_w = h * k, w * k
    
    result = np.zeros((new_h, new_w), dtype=img.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            src_i = i // k
            src_j = j // k
            
            src_i = min(src_i, h - 1)
            src_j = min(src_j, w - 1)
            
            result[i, j] = img[src_i, src_j]
    
    return result

def downscale_bilinear(img, k):
    h, w = img.shape[:2]
    new_h, new_w = max(1, h // k), max(1, w // k)
    
    result = np.zeros((new_h, new_w), dtype=np.float32)
    
    for i in range(new_h):
        for j in range(new_w):
            src_i = (i + 0.5) * k
            src_j = (j + 0.5) * k
            
            i1, j1 = int(src_i), int(src_j)
            i2, j2 = min(i1 + 1, h - 1), min(j1 + 1, w - 1)
            
            wi = src_i - i1
            wj = src_j - j1
            
            p11 = img[i1, j1].astype(np.float32)
            p12 = img[i1, j2].astype(np.float32)
            p21 = img[i2, j1].astype(np.float32)
            p22 = img[i2, j2].astype(np.float32)
            
            result[i, j] = (1 - wi) * (1 - wj) * p11 + \
                          (1 - wi) * wj * p12 + \
                          wi * (1 - wj) * p21 + \
                          wi * wj * p22
    
    return clip_u8(result)

def upscale_bilinear(img, k):
    h, w = img.shape[:2]
    new_h, new_w = h * k, w * k
    
    result = np.zeros((new_h, new_w), dtype=np.float32)
    
    for i in range(new_h):
        for j in range(new_w):
            src_i = i / k
            src_j = j / k
            
            i1, j1 = int(src_i), int(src_j)
            i2, j2 = min(i1 + 1, h - 1), min(j1 + 1, w - 1)
            
            wi = src_i - i1
            wj = src_j - j1
            
            p11 = img[i1, j1].astype(np.float32)
            p12 = img[i1, j2].astype(np.float32)
            p21 = img[i2, j1].astype(np.float32)
            p22 = img[i2, j2].astype(np.float32)
            
            result[i, j] = (1 - wi) * (1 - wj) * p11 + \
                          (1 - wi) * wj * p12 + \
                          wi * (1 - wj) * p21 + \
                          wi * wj * p22
    
    return clip_u8(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagen", required=True, help="Ruta a la imagen de entrada")
    parser.add_argument("--salida", default="./lab1_outputs", help="Carpeta de salida")
    parser.add_argument("--gamma", type=float, default=2.2, help="Valor de gamma para la correccion")
    parser.add_argument("--brillo", type=int, default=40, help="Delta para aumento/disminucion de brillo")
    parser.add_argument("--alpha_exp", type=float, default=2.0, help="Alfa para contraste exponencial")
    args = parser.parse_args()

    out_dir = Path(args.salida)
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = imread_gray(args.imagen)
    pair_save(gray, "Escala de grises (Base)", "00_grayscale_base", out_dir)

    for k in [2,3,4]:
        down = downscale_bilinear(gray, k)
        pair_save(down, f"Reduccion K={k} (Bilineal)", f"01_down_k{k}", out_dir)

    for k in [10,12,16]:
        up = upscale_bilinear(gray, k)
        pair_save(up, f"Aumento K={k} (Bilineal)", f"02_up_k{k}", out_dir)

    for ang in [90, 75, 47, 135]:
        rot = rotate_image(gray, ang)
        pair_save(rot, f"Rotacion {ang} Grados", f"03_rot_{ang}", out_dir)

    neg = clip_u8(255 - gray)
    pair_save(neg, "Negativo", "10_negativo", out_dir)

    bright = clip_u8(gray.astype(np.int16) + args.brillo)
    pair_save(bright, f"Aumento de brillo (+{args.brillo})", "11_aumento_brillo", out_dir)

    gamma_img = ((gray/255.0) ** (1.0/args.gamma)) * 255.0
    gamma_img = clip_u8(gamma_img)
    pair_save(gamma_img, f"Correccion gamma (Gamma={args.gamma})", "12_gamma", out_dir)

    log_img = 255.0 * (np.log1p(gray.astype(np.float64)) / np.log1p(255.0))
    log_img = clip_u8(log_img)
    pair_save(log_img, "Contraste logaritmico", "13_log", out_dir)

    dark = clip_u8(gray.astype(np.int16) - args.brillo)
    pair_save(dark, f"Disminucion de brillo (-{args.brillo})", "14_disminucion_brillo", out_dir)

    x = gray.astype(np.float64) / 255.0
    exp_img = (np.expm1(args.alpha_exp * x) / np.expm1(args.alpha_exp)) * 255.0
    exp_img = clip_u8(exp_img)
    pair_save(exp_img, f"Contraste exponencial (alpha={args.alpha_exp})", "15_exponencial", out_dir)

    gmin, gmax = gray.min(), gray.max()
    if gmax > gmin:
        norm_0_100 = ((gray.astype(np.float64) - gmin) * (100.0 / (gmax - gmin)))
    else:
        norm_0_100 = np.zeros_like(gray, dtype=np.float64)
    vis_0_100 = clip_u8((norm_0_100 / 100.0) * 255.0)
    pair_save(vis_0_100, "Normalizada [0,100] (visual)", "20_normalizada_0_100", out_dir)
    np.save(out_dir / "20_normalizada_0_100_values.npy", norm_0_100)

    if gmax > gmin:
        norm_m1_1 = ((gray.astype(np.float64) - gmin) * (2.0 / (gmax - gmin))) - 1.0
    else:
        norm_m1_1 = np.zeros_like(gray, dtype=np.float64) - 1.0
    vis_m1_1 = clip_u8(((norm_m1_1 + 1.0) / 2.0) * 255.0)
    pair_save(vis_m1_1, "Normalizada [-1,1] (visual)", "21_normalizada_-1_1", out_dir)
    np.save(out_dir / "21_normalizada_-1_1_values.npy", norm_m1_1)

    print(f"Listo. Archivos guardados en: {out_dir}")

if __name__ == "__main__":
    main()
