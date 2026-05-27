import cv2
import numpy as np
import os
import glob

# ====================================================================
# 1. KONFIGURASI DIRECTORY
# ====================================================================
INPUT_FOLDER = 'dataset'
OUTPUT_FOLDER = 'output'

def process_image(input_path, output_path=None):
    """
    Memproses gambar tunggal untuk deteksi copy-move (SIFT + FLANN + RANSAC).

    Kontrak untuk frontend (tanpa ubah frontend):
    - Jika gagal: kembalikan {"error": "..."} (boleh ada field lain)
    - Jika sukses: wajib ada keypoints (int), matches (int), is_forged (bool)
    """
    img = cv2.imread(input_path)
    if img is None:
        return {"error": "Gagal membaca gambar."}

    # ----------------------------------------------------------------
    # TAHAP PRE-PROCESSING
    # ----------------------------------------------------------------
    max_width = 800
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=10)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # ====================================================================
    # EKSTRAKSI FITUR SIFT
    # ====================================================================
    sift = cv2.SIFT_create(nfeatures=5000)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) < 3:
        if output_path is not None:
            cv2.imwrite(output_path, img)
        return {
            "error": "Fitur SIFT terlalu sedikit untuk dianalisis.",
            "keypoints": len(keypoints) if keypoints else 0,
            "matches": 0,
            "is_forged": False,
        }

    # ====================================================================
    # MATCHING (FLANN) + FILTER COPY-MOVE
    # ====================================================================
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors, descriptors, k=3)
    good_matches = []

    for match_tuple in matches:
        if len(match_tuple) < 3:
            continue

        m, n, o = match_tuple
        if n.distance < 0.72 * o.distance:
            pt_asal = np.array(keypoints[n.queryIdx].pt)
            pt_salinan = np.array(keypoints[n.trainIdx].pt)

            spatial_dist = np.linalg.norm(pt_asal - pt_salinan)
            if 40 < spatial_dist < 500:
                good_matches.append(n)

    # ====================================================================
    # RANSAC HOMOGRAPHY (verifikasi spasial)
    # ====================================================================
    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            mask = mask.ravel()
            good_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

    # ====================================================================
    # VISUALISASI (gambar output)
    # ====================================================================
    img_out = img.copy()
    for match in good_matches:
        pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
        pt2 = tuple(map(int, keypoints[match.trainIdx].pt))

        cv2.circle(img_out, pt1, 4, (0, 0, 255), -1)
        cv2.circle(img_out, pt2, 4, (0, 0, 255), -1)
        cv2.line(img_out, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(
        img_out,
        f"Keypoints: {len(keypoints)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    match_color = (0, 255, 0) if len(good_matches) > 0 else (0, 0, 255)
    cv2.putText(
        img_out,
        f"Matches: {len(good_matches)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        match_color,
        2,
        cv2.LINE_AA,
    )

    if output_path is not None:
        cv2.imwrite(output_path, img_out)

    # ====================================================================
    # OUTPUT JSON (untuk frontend)
    # ====================================================================
    match_percentage = (len(good_matches) / len(keypoints) * 100) if len(keypoints) > 0 else 0
    is_forged = len(good_matches) >= 4

    return {
        "success": True,
        "width": w,
        "height": h,
        "keypoints": len(keypoints),
        "matches": len(good_matches),
        "match_percentage": round(match_percentage, 2),
        "is_forged": is_forged,
    }


def main():
    # Membuat folder output jika belum ada
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"[*] Folder '{OUTPUT_FOLDER}' berhasil dibuat.")

    # Membaca semua file gambar di dalam folder dataset
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    print(f"[*] Menemukan {len(image_paths)} gambar di dalam folder '{INPUT_FOLDER}'.")
    print("[*] Memulai pemrosesan batch + Pre-processing, mohon tunggu...\n" + "=" * 60)

    for index, img_path in enumerate(image_paths, 1):
        file_name = os.path.basename(img_path)
        print(f"[{index}/{len(image_paths)}] Sedang Memproses: {file_name}")

        out_path = os.path.join(OUTPUT_FOLDER, file_name)
        result = process_image(img_path, output_path=out_path)

        if "error" in result:
            print(f"    [!] {result['error']} Skip.")
            continue

        print(f"    [+] Selesai. Hasil deteksi: {result['matches']} matches.")

    print("\n" + "=" * 60)
    print("[+] SELESAI! Silakan cek folder output.")


if __name__ == "__main__":
    main()