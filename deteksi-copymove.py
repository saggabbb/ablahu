import cv2
import numpy as np
import os
import glob

# ====================================================================
# 1. KONFIGURASI DIRECTORY
# ====================================================================
INPUT_FOLDER = 'dataset'
OUTPUT_FOLDER = 'output'

# Membuat folder output jika belum ada
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"[*] Folder '{OUTPUT_FOLDER}' berhasil dibuat.")

# Membaca semua file gambar di dalam folder dataset
extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

print(f"[*] Menemukan {len(image_paths)} gambar di dalam folder '{INPUT_FOLDER}'.")
print("[*] Memulai pemrosesan batch + Pre-processing, mohon tunggu...\n" + "="*60)

# ====================================================================
# 2. PROSES BATCH PROCESSING (LOOP 100 GAMBAR)
# ====================================================================
for index, img_path in enumerate(image_paths, 1):
    file_name = os.path.basename(img_path)
    print(f"[{index}/{len(image_paths)}] Sedang Memproses: {file_name}")
    
    img = cv2.imread(img_path)

    if img is None:
        print(f"    [!] Gagal membaca {file_name}. Skip.")
        continue

    # ----------------------------------------------------------------
    # TAHAP PRE-PROCESSING
    # ----------------------------------------------------------------
    max_width = 800

    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (kurangi noise sebelum CLAHE)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 3. CLAHE (Pemerataan kontras adaptif)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 4. Brightness & Contrast (Global adjustment)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=10)
    
    # 5. Min-Max Normalization (Maksimalkan dynamic range)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


    # ====================================================================
    # 3. EKSTRAKSI FITUR SIFT
    # ====================================================================

    sift = cv2.SIFT_create(nfeatures=5000)

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    if descriptors is None or len(descriptors) < 3:

        print(f"    [!] Keypoint terlalu sedikit")

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER,
                file_name
            ),
            img
        )

        continue


    # ====================================================================
    # 4. PENCUTOKAN TITIK
    # ====================================================================

    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )
    
    matches = flann.knnMatch(
        descriptors,
        descriptors,
        k=3
    )

    good_matches=[]


    # ====================================================================
    # 5. FILTER COPY-MOVE
    # ====================================================================

    for match_tuple in matches:

        if len(match_tuple) < 3:
            continue
            
        m,n,o = match_tuple
        
        if n.distance < 0.72 * o.distance:
            
            pt_asal = np.array(
                keypoints[n.queryIdx].pt
            )

            pt_salinan = np.array(
                keypoints[n.trainIdx].pt
            )
            
            spatial_dist = np.linalg.norm(
                pt_asal - pt_salinan
            )
            
            if 40 < spatial_dist < 500:

                good_matches.append(n)


    # ====================================================================
    # TAMBAHAN RANSAC
    # ====================================================================

    if len(good_matches) >= 4:

        src_pts=np.float32(
            [keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1,1,2)

        dst_pts=np.float32(
            [keypoints[m.trainIdx].pt for m in good_matches]
        ).reshape(-1,1,2)


        H,mask=cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )


        if mask is not None:

            mask=mask.ravel()

            good_matches=[
                good_matches[i]
                for i in range(len(good_matches))
                if mask[i]
            ]


    # ====================================================================
    # 6. VISUALISASI MANUAL LINE DI ATAS CITRA
    # ====================================================================

    img_out = img.copy()

    for match in good_matches:

        pt1 = tuple(map(int,keypoints[match.queryIdx].pt))
        pt2 = tuple(map(int,keypoints[match.trainIdx].pt))
        
        cv2.circle(img_out, pt1, 4, (0,0,255), -1)
        cv2.circle(img_out, pt2, 4, (0,0,255), -1)
        
        cv2.line(
            img_out,
            pt1,
            pt2,
            (0,255,0),
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        img_out,
        f'Keypoints: {len(keypoints)}',
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2,
        cv2.LINE_AA
    )
    
    match_color=(0,255,0) if len(good_matches)>0 else (0,0,255)

    cv2.putText(
        img_out,
        f'Matches: {len(good_matches)}',
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        match_color,
        2,
        cv2.LINE_AA
    )


    # ====================================================================
    # 7. MENYIMPAN HASIL
    # ====================================================================

    output_path=os.path.join(
        OUTPUT_FOLDER,
        file_name
    )

    cv2.imwrite(output_path,img_out)

    print(f"    [+] Selesai. Hasil deteksi: {len(good_matches)} matches.")


print("\n"+"="*60)
print("[+] SELESAI! Silakan cek folder output.")