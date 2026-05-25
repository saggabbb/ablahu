import cv2
import numpy as np
import os

def process_image(input_path, output_path):
    """
    Memproses gambar tunggal untuk deteksi copy-move menggunakan SIFT.
    Mengembalikan dictionary berisi data analisis.
    """
    img = cv2.imread(input_path)
    if img is None:
        return {"error": "Gagal membaca gambar."}

    # ====================================================================
    # 1. PRE-PROCESSING
    # ====================================================================
    max_width = 800
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    
    # Simpan ukuran asli setelah resize (untuk info web)
    h, w = img.shape[:2]
    
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
    # 2. EKSTRAKSI FITUR SIFT
    # ====================================================================
    sift = cv2.SIFT_create(nfeatures=5000)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) < 3:
        cv2.imwrite(output_path, img)
        return {
            "error": "Fitur SIFT terlalu sedikit untuk dianalisis.",
            "keypoints": len(keypoints) if keypoints else 0,
            "matches": 0,
            "status": "Aman"
        }

    # ====================================================================
    # 3. MATCHING & FILTERING
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
        
        # Lowe's ratio test (membandingkan match ke-2 dan ke-3 karena self-matching)
        if n.distance < 0.72 * o.distance:
            pt_asal = np.array(keypoints[n.queryIdx].pt)
            pt_salinan = np.array(keypoints[n.trainIdx].pt)
            
            spatial_dist = np.linalg.norm(pt_asal - pt_salinan)
            
            # Spatial distance filter (menghapus max 500 seperti yang direkomendasikan)
            if spatial_dist > 40:
                good_matches.append(n)

    # ====================================================================
    # 4. RANSAC HOMOGRAPHY
    # ====================================================================
    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if mask is not None:
            mask = mask.ravel()
            good_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

    # ====================================================================
    # 5. VISUALISASI
    # ====================================================================
    img_out = img.copy()

    for match in good_matches:
        pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
        pt2 = tuple(map(int, keypoints[match.trainIdx].pt))
        
        cv2.circle(img_out, pt1, 4, (0, 0, 255), -1)
        cv2.circle(img_out, pt2, 4, (0, 0, 255), -1)
        cv2.line(img_out, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    # Status text on image removed so we can handle it purely in UI, 
    # but let's keep basic stats on the image like the original code did.
    cv2.putText(img_out, f'Keypoints: {len(keypoints)}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    match_color = (0, 255, 0) if len(good_matches) > 0 else (0, 0, 255)
    cv2.putText(img_out, f'Matches: {len(good_matches)}', (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, img_out)

    # ====================================================================
    # 6. ANALISIS HASIL
    # ====================================================================
    # Tentukan persentase keypoints yang menjadi good matches
    match_percentage = (len(good_matches) / len(keypoints) * 100) if len(keypoints) > 0 else 0
    
    is_forged = len(good_matches) >= 4
    status_text = "TERINDIKASI COPY-MOVE FORGERY" if is_forged else "TIDAK TERDETEKSI MANIPULASI"

    return {
        "success": True,
        "width": w,
        "height": h,
        "keypoints": len(keypoints),
        "matches": len(good_matches),
        "match_percentage": round(match_percentage, 2),
        "status": status_text,
        "is_forged": is_forged
    }
