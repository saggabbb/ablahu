import cv2
import os
import numpy as np

# ==================================================
# KONFIGURASI
# ==================================================

INPUT_FOLDER  = 'dataset/REAL'            # folder gambar asli
OUTPUT_FOLDER = 'dataset/PREPROCESSED'    # folder hasil preprocessing

TARGET_SIZE   = (512, 512)                # ukuran standar semua gambar

# Brightness & Contrast
ALPHA = 1.15    # kontras  (1.0 = normal, >1 = lebih tajam)
BETA  = 10      # brightness offset (-127..+127, 0 = normal)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP  = 2.0       # batas clip CLAHE
CLAHE_GRID  = (8, 8)    # ukuran tile grid

# Gaussian Blur untuk noise reduction
BLUR_KERNEL = (3, 3)    # kernel kecil supaya detail tetap tajam

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==================================================
# FUNGSI-FUNGSI PREPROCESSING
# ==================================================

def resize_image(img, size=TARGET_SIZE):
    """
    Resize gambar ke ukuran standar.
    Semua gambar harus sama ukurannya supaya koordinat
    source & target copy-move konsisten.
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def convert_grayscale(img):
    """
    Konversi ke grayscale.
    SIFT bekerja pada intensitas (grayscale), jadi konversi
    di tahap awal supaya semua proses berikutnya konsisten.
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def adjust_brightness_contrast(img, alpha=ALPHA, beta=BETA):
    """
    Normalisasi brightness & kontras.
    - alpha (gain) : mengatur kontras
    - beta  (bias) : mengatur brightness
    
    Rumus: output = alpha * pixel + beta
    Diclip ke [0, 255] otomatis oleh convertScaleAbs.
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_clahe(img, clip_limit=CLAHE_CLIP, tile_grid=CLAHE_GRID):
    """
    CLAHE — Adaptive Histogram Equalization.
    Lebih baik dari histogram equalization biasa karena
    bekerja per-tile, sehingga kontras lokal merata tanpa
    over-amplifikasi noise.

    Ini SANGAT membantu SIFT karena:
    - Keypoint lebih banyak terdeteksi di area gelap/terang
    - Descriptor lebih diskriminatif
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def reduce_noise(img, kernel=BLUR_KERNEL):
    """
    Gaussian Blur ringan untuk mengurangi noise.
    Kernel kecil (3×3) cukup untuk menekan noise tanpa
    menghilangkan detail penting yang dibutuhkan SIFT.
    """
    return cv2.GaussianBlur(img, kernel, 0)


def normalize_intensity(img):
    """
    Min-Max Normalization ke range [0, 255].
    Memastikan seluruh dynamic range dipakai, sehingga
    SIFT bisa mendeteksi gradien dengan lebih optimal.
    """
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm


# ==================================================
# PIPELINE PREPROCESSING
# ==================================================

def preprocess_pipeline(img):
    """
    Pipeline lengkap preprocessing:
    
    1. Resize          → standarisasi ukuran
    2. Grayscale       → SIFT butuh 1 channel
    3. Noise Reduction → kurangi noise sebelum enhance
    4. CLAHE           → pemerataan kontras adaptif
    5. Brightness/Contrast → fine-tune global
    6. Normalize       → pastikan full dynamic range
    
    Urutan ini penting:
    - Noise reduction SEBELUM CLAHE supaya noise tidak
      ikut di-amplifikasi
    - CLAHE SEBELUM brightness/contrast supaya hasilnya
      lebih natural
    - Normalize di akhir supaya range pixel konsisten
    """
    # Step 1: Resize
    img = resize_image(img)
    
    # Step 2: Grayscale
    gray = convert_grayscale(img)
    
    # Step 3: Noise Reduction
    denoised = reduce_noise(gray)
    
    # Step 4: CLAHE
    enhanced = apply_clahe(denoised)
    
    # Step 5: Brightness & Contrast
    adjusted = adjust_brightness_contrast(enhanced)
    
    # Step 6: Normalize Intensity
    normalized = normalize_intensity(adjusted)
    
    return normalized


# ==================================================
# MAIN: PROSES SEMUA GAMBAR
# ==================================================

if __name__ == '__main__':

    print("=" * 55)
    print("   IMAGE PREPROCESSING untuk SIFT Copy-Move Detection")
    print("=" * 55)
    print(f"\n[FOLDER] Input  : {INPUT_FOLDER}")
    print(f"[FOLDER] Output : {OUTPUT_FOLDER}")
    print(f"[SIZE]   Target Size  : {TARGET_SIZE}")
    print(f"[ALPHA]  Contrast     : {ALPHA}")
    print(f"[BETA]   Brightness   : {BETA}")
    print(f"[CLAHE]  Clip Limit   : {CLAHE_CLIP}")
    print(f"[CLAHE]  Grid Size    : {CLAHE_GRID}")
    print(f"[BLUR]   Kernel       : {BLUR_KERNEL}")
    print("-" * 55)

    files = sorted(os.listdir(INPUT_FOLDER))
    total = len(files)
    success = 0
    fail = 0

    for i, file in enumerate(files, 1):

        img_path = os.path.join(INPUT_FOLDER, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  [FAIL] [{i}/{total}] Gagal membaca: {file}")
            fail += 1
            continue

        # --- Info ukuran asli ---
        h_orig, w_orig = img.shape[:2]

        # --- Jalankan pipeline ---
        result = preprocess_pipeline(img)

        # --- Simpan ---
        save_path = os.path.join(OUTPUT_FOLDER, file)
        cv2.imwrite(save_path, result)

        print(f"  [OK]   [{i}/{total}] {file}  ({w_orig}x{h_orig} -> {TARGET_SIZE[0]}x{TARGET_SIZE[1]})")
        success += 1

    # ==================================================
    # RINGKASAN
    # ==================================================

    print("\n" + "=" * 55)
    print("   PREPROCESSING SELESAI")
    print("=" * 55)
    print(f"   Total gambar  : {total}")
    print(f"   Berhasil      : {success}")
    print(f"   Gagal         : {fail}")
    print(f"\n[FOLDER] Hasil tersimpan di: {OUTPUT_FOLDER}/")
    print("=" * 55)

    # ==================================================
    # SAMPLE COMPARISON (sebelum vs sesudah)
    # ==================================================

    sample_file = files[0] if files else None
    if sample_file:
        orig = cv2.imread(os.path.join(INPUT_FOLDER, sample_file))
        prep = cv2.imread(os.path.join(OUTPUT_FOLDER, sample_file), cv2.IMREAD_GRAYSCALE)

        if orig is not None and prep is not None:
            orig_resized = cv2.resize(orig, TARGET_SIZE)
            orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)

            # Hitung statistik sebelum & sesudah
            print(f"\n[STATS] SAMPLE COMPARISON -- {sample_file}")
            print(f"   {'':20s} {'BEFORE':>10s}  {'AFTER':>10s}")
            print(f"   {'Mean Intensity':20s} {orig_gray.mean():10.2f}  {prep.mean():10.2f}")
            print(f"   {'Std Deviation':20s} {orig_gray.std():10.2f}  {prep.std():10.2f}")
            print(f"   {'Min Pixel':20s} {orig_gray.min():10d}  {prep.min():10d}")
            print(f"   {'Max Pixel':20s} {orig_gray.max():10d}  {prep.max():10d}")

            # Hitung jumlah keypoint SIFT sebelum & sesudah
            sift = cv2.SIFT_create()
            kp_before, _ = sift.detectAndCompute(orig_gray, None)
            kp_after, _ = sift.detectAndCompute(prep, None)
            print(f"\n[SIFT] Keypoints Comparison")
            print(f"   Before preprocessing : {len(kp_before)}")
            print(f"   After  preprocessing : {len(kp_after)}")
            improvement = ((len(kp_after) - len(kp_before)) / len(kp_before) * 100) if len(kp_before) > 0 else 0
            print(f"   Improvement          : {improvement:+.1f}%")
