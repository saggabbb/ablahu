import cv2
import os
import numpy as np
from preprocess import preprocess_pipeline

# ==================================================
# FOLDER
# ==================================================

real_folder  = 'dataset/REAL'
fake_folder  = 'dataset/FAKE'
output_folder = 'dataset/OUTPUT'

os.makedirs(fake_folder,   exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# ==================================================
# KOORDINAT SOURCE & TARGET
# ==================================================

source_x1, source_y1 = 140, 170
source_x2, source_y2 = 320, 350

target_x1, target_y1 = 260, 280
target_x2, target_y2 = 440, 460

# ==================================================
# KONFIGURASI METODE MATCHING
# 'knn_bf'  → Metode UTAMA  : KNN BFMatcher + Lowe's Ratio Test
# 'flann'   → Metode CADANGAN: FLANN Matcher + Lowe's Ratio Test
# ==================================================

MATCHING_METHOD = 'knn_bf'   # ganti ke 'flann' untuk metode cadangan
LOWE_RATIO      = 0.75        # threshold Lowe's ratio test (standar: 0.75)



# ==================================================
# MATCHER FACTORY
# ==================================================

def build_matcher(method='knn_bf'):
    """
    Mengembalikan objek matcher sesuai metode yang dipilih.

    knn_bf : BFMatcher dengan NORM_L2
             – cocok untuk SIFT (descriptor float)
             – matching via knnMatch + Lowe's ratio test

    flann  : FLANN-based Matcher
             – KD-Tree index (index=1), lebih cepat untuk
               dataset besar
             – matching via knnMatch + Lowe's ratio test
    """
    if method == 'knn_bf':
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        return matcher, 'KNN BFMatcher (Lowe Ratio Test)'

    elif method == 'flann':
        FLANN_INDEX_KDTREE = 1
        index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return matcher, 'FLANN Matcher (Lowe Ratio Test)'

    else:
        raise ValueError(f"Metode tidak dikenal: {method}. Pilih 'knn_bf' atau 'flann'.")


def apply_lowe_ratio(knn_matches, ratio=0.75):
    """
    Lowe's Ratio Test (Lowe, 2004):
    Sebuah match diterima jika jarak match terbaik (m)
    secara signifikan lebih kecil dari match terbaik kedua (n).
    Syarat: m.distance < ratio * n.distance
    """
    good = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


# ==================================================
# GENERATE COPY-MOVE IMAGE
# ==================================================

print("\n=== GENERATING COPY-MOVE IMAGES ===\n")

files = os.listdir(real_folder)

for i, file in enumerate(files):

    img_path = os.path.join(real_folder, file)
    img = cv2.imread(img_path)

    if img is None:
        print(f'❌ Gagal membaca {file}')
        continue

    img = cv2.resize(img, (512, 512))

    # ── Copy-Move: ambil source, paste ke target ──────────────────────
    copy_area = img[source_y1:source_y2, source_x1:source_x2]
    img[target_y1:target_y2, target_x1:target_x2] = copy_area

    # ── Visualisasi kotak SOURCE & TARGET ─────────────────────────────
    cv2.rectangle(img, (source_x1, source_y1), (source_x2, source_y2), (0, 255, 0), 2)
    cv2.rectangle(img, (target_x1, target_y1), (target_x2, target_y2), (0, 0, 255), 2)

    cv2.putText(img, 'SOURCE',    (source_x1, source_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(img, 'COPY-MOVE', (target_x1, target_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # ── Header overlay ────────────────────────────────────────────────
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 95), (15, 15, 15), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    cv2.putText(img, 'COPY-MOVE MANIPULATION',    (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(img, 'Local Feature Duplication', (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, 'Fake Image Generated',       (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # ── Simpan ────────────────────────────────────────────────────────
    fake_name = f'fake_{i+1}.jpg'
    save_path = os.path.join(fake_folder, fake_name)
    cv2.imwrite(save_path, img)
    print(f'✅ {fake_name} berhasil dibuat')

print("\nSEMUA FOTO FAKE BERHASIL DIBUAT!\n")

# ==================================================
# DETEKSI COPY-MOVE
# ==================================================

print(f"\n=== SIFT COPY-MOVE DETECTION ===")
print(f"    Metode Matching : {MATCHING_METHOD.upper()}")
print(f"    Lowe Ratio      : {LOWE_RATIO}")
print(f"    Pre-processing  : Resize → Grayscale → Noise Reduction → CLAHE → Brightness/Contrast → Normalize\n")

matcher, method_label = build_matcher(MATCHING_METHOD)
sift = cv2.SIFT_create()

fake_files = os.listdir(fake_folder)

for file in fake_files:

    img_path = os.path.join(fake_folder, file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # ──────────────────────────────────────────────────────────────────
    # PRE-PROCESSING (menggunakan pipeline dari preprocess.py)
    # Resize → Grayscale → Noise Reduction → CLAHE → Brightness/Contrast → Normalize
    # ──────────────────────────────────────────────────────────────────
    preprocessed = preprocess_pipeline(img)

    # ── Crop region source & target dari hasil pre-processing ─────────
    source_crop = preprocessed[source_y1:source_y2, source_x1:source_x2]
    target_crop = preprocessed[target_y1:target_y2, target_x1:target_x2]

    # ──────────────────────────────────────────────────────────────────
    # SIFT: Deteksi Keypoint & Ekstraksi Descriptor
    # ──────────────────────────────────────────────────────────────────
    kp1, des1 = sift.detectAndCompute(source_crop, None)
    kp2, des2 = sift.detectAndCompute(target_crop, None)

    if des1 is None or des2 is None:
        print(f'❌ Descriptor gagal pada {file}')
        continue

    # ──────────────────────────────────────────────────────────────────
    # MATCHING
    # knnMatch(k=2) → ambil 2 kandidat match terbaik per descriptor
    # Lalu terapkan Lowe's Ratio Test
    # ──────────────────────────────────────────────────────────────────
    knn_matches  = matcher.knnMatch(des1, des2, k=2)
    good_matches = apply_lowe_ratio(knn_matches, ratio=LOWE_RATIO)

    # ──────────────────────────────────────────────────────────────────
    # MATCHING PERCENTAGE
    # Pembagi = min(kp_source, kp_target) agar proporsional
    # ──────────────────────────────────────────────────────────────────
    min_kp = min(len(kp1), len(kp2))
    matching_percentage = (len(good_matches) / min_kp * 100) if min_kp > 0 else 0

    # ──────────────────────────────────────────────────────────────────
    # VISUALISASI KEYPOINT
    # ──────────────────────────────────────────────────────────────────
    source_vis = cv2.cvtColor(source_crop, cv2.COLOR_GRAY2BGR)
    target_vis = cv2.cvtColor(target_crop, cv2.COLOR_GRAY2BGR)

    source_vis = cv2.drawKeypoints(source_vis, kp1, None, color=(0, 255, 0))
    target_vis = cv2.drawKeypoints(target_vis, kp2, None, color=(0, 0, 255))

    cv2.putText(source_vis, 'SOURCE REGION', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(target_vis, 'TARGET REGION', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ──────────────────────────────────────────────────────────────────
    # DRAW MATCHES
    # ──────────────────────────────────────────────────────────────────
    result = cv2.drawMatches(
        source_vis, kp1,
        target_vis, kp2,
        good_matches[:80],
        None,
        matchColor=(0, 220, 120),
        singlePointColor=(120, 120, 120),
        flags=2
    )
    result = cv2.resize(result, (1024, 300))

    # ──────────────────────────────────────────────────────────────────
    # OVERLAY ANALISIS
    # ──────────────────────────────────────────────────────────────────
    overlay = result.copy()
    cv2.rectangle(overlay, (10, 10), (510, 175), (10, 10, 10), -1)
    result = cv2.addWeighted(overlay, 0.72, result, 0.28, 0)

    cv2.putText(result, 'SIFT COPY-MOVE ANALYSIS', (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    cv2.putText(result, f'Pre-processing     : Full Pipeline (preprocess.py)', (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)
    cv2.putText(result, f'Metode Matching    : {method_label}', (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)
    cv2.putText(result, f'Lowe Ratio         : {LOWE_RATIO}', (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.putText(result, f'Keypoints Source   : {len(kp1)}', (15, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(result, f'Keypoints Target   : {len(kp2)}', (15, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(result, f'Good Matches       : {len(good_matches)}', (15, 149),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(result, f'Matching           : {matching_percentage:.2f}%', (15, 166),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ──────────────────────────────────────────────────────────────────
    # STATUS DETEKSI
    # ──────────────────────────────────────────────────────────────────
    if matching_percentage > 20:
        status = 'COPY-MOVE FORGERY DETECTED'
        color  = (0, 0, 255)
    else:
        status = 'LOW MATCH - CHECK REGION'
        color  = (0, 200, 255)

    cv2.putText(result, status, (515, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(result, 'SIFT menemukan kemiripan descriptor', (515, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(result, 'dan keypoint tinggi antara source',   (515, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(result, 'dan target region — terindikasi',     (515, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(result, 'copy-move forgery.',                  (515, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # ──────────────────────────────────────────────────────────────────
    # PRINT TERMINAL
    # ──────────────────────────────────────────────────────────────────
    print(f'\n📸 {file}')
    print(f'   Pre-processing       : Resize → Grayscale → Noise Reduction → CLAHE → Brightness/Contrast → Normalize')
    print(f'   Metode Matching      : {method_label}')
    print(f'   Lowe Ratio           : {LOWE_RATIO}')
    print(f'   Keypoints Source     : {len(kp1)}')
    print(f'   Keypoints Target     : {len(kp2)}')
    print(f'   Good Matches         : {len(good_matches)}')
    print(f'   Matching Percentage  : {matching_percentage:.2f}%')
    print(f'   Status               : {status}')
    print('\n   Rumus Matching:')
    print('   matching = (good_matches / min(kp_source, kp_target)) x 100%')
    if matching_percentage > 20:
        print('\n   ✅ SIFT menemukan kemiripan descriptor dan keypoint yang tinggi')
        print('      antara source region dan target region sehingga terindikasi')
        print('      adanya copy-move forgery.')

    # ──────────────────────────────────────────────────────────────────
    # SIMPAN OUTPUT
    # ──────────────────────────────────────────────────────────────────
    output_path = os.path.join(output_folder, f'result_{file}')
    cv2.imwrite(output_path, result)

print("\n🎉 ANALISIS SELESAI!")
print("📂 Cek folder dataset/OUTPUT")