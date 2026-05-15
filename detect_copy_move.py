import cv2
import os
import numpy as np

# ==================================================
# FOLDER
# ==================================================

real_folder = 'dataset/REAL'
fake_folder = 'dataset/FAKE'
output_folder = 'dataset/OUTPUT'

os.makedirs(fake_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# ==================================================
# KOORDINAT SOURCE & TARGET
# (didefinisikan di luar supaya bisa dipakai ulang
#  di bagian deteksi)
# ==================================================

source_x1, source_y1 = 140, 170
source_x2, source_y2 = 320, 350

target_x1, target_y1 = 260, 280
target_x2, target_y2 = 440, 460

# ==================================================
# GENERATE COPY-MOVE IMAGE
# ==================================================

print("\n=== GENERATING COPY-MOVE IMAGES ===\n")

files = os.listdir(real_folder)

for i, file in enumerate(files):

    img_path = os.path.join(real_folder, file)
    img = cv2.imread(img_path)

    if img is None:
        print(f'❌ gagal membaca {file}')
        continue

    img = cv2.resize(img, (512, 512))

    # ==================================================
    # COPY-MOVE: ambil source, paste ke target
    # ==================================================

    copy_area = img[source_y1:source_y2, source_x1:source_x2]
    img[target_y1:target_y2, target_x1:target_x2] = copy_area

    # ==================================================
    # VISUAL SOURCE & TARGET
    # ==================================================

    cv2.rectangle(img, (source_x1, source_y1), (source_x2, source_y2), (0, 255, 0), 2)
    cv2.rectangle(img, (target_x1, target_y1), (target_x2, target_y2), (0, 0, 255), 2)

    cv2.putText(img, 'SOURCE', (source_x1, source_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(img, 'COPY-MOVE', (target_x1, target_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # ==================================================
    # HEADER OVERLAY
    # ==================================================

    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 95), (15, 15, 15), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    cv2.putText(img, 'COPY-MOVE MANIPULATION', (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(img, 'Local Feature Duplication', (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, 'Fake Image Generated', (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # ==================================================
    # SAVE FAKE IMAGE
    # ==================================================

    fake_name = f'fake_{i+1}.jpg'
    save_path = os.path.join(fake_folder, fake_name)
    cv2.imwrite(save_path, img)
    print(f'✅ {fake_name} berhasil dibuat')

print("\nSEMUA FOTO FAKE BERHASIL DIBUAT!\n")

# ==================================================
# DETEKSI COPY-MOVE: SOURCE REGION vs TARGET REGION
# ==================================================

print("\n=== SIFT COPY-MOVE DETECTION ===\n")

fake_files = os.listdir(fake_folder)

for file in fake_files:

    img_path = os.path.join(fake_folder, file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==================================================
    # CROP SOURCE & TARGET
    # Kita tahu persis koordinatnya karena kita yang
    # bikin manipulasinya — jadi langsung crop.
    # ==================================================

    source_crop = gray[source_y1:source_y2, source_x1:source_x2]
    target_crop = gray[target_y1:target_y2, target_x1:target_x2]

    # ==================================================
    # SIFT PER REGION
    # ==================================================

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(source_crop, None)
    kp2, des2 = sift.detectAndCompute(target_crop, None)

    if des1 is None or des2 is None:
        print(f'❌ descriptor gagal pada {file}')
        continue

    # ==================================================
    # BF MATCHER: source descriptor vs target descriptor
    # crossCheck=True → hanya match yang saling setuju
    # ==================================================

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # ==================================================
    # FILTER: ambil yang descriptor distance < 120
    # lebih ketat, descriptor jelek tidak masuk
    # ==================================================

    good_matches = [m for m in matches if m.distance < 120]

    # ==================================================
    # MATCHING PERCENTAGE
    # Pembaginya: minimum keypoint antara source & target
    # karena yang dibandingkan memang dua region terpisah
    # ==================================================

    min_kp = min(len(kp1), len(kp2))
    matching_percentage = (len(good_matches) / min_kp * 100) if min_kp > 0 else 0

    # ==================================================
    # VISUAL: drawMatches source crop vs target crop
    # ==================================================

    source_vis = cv2.cvtColor(source_crop, cv2.COLOR_GRAY2BGR)
    target_vis = cv2.cvtColor(target_crop, cv2.COLOR_GRAY2BGR)

    # ==================================================
    # KEYPOINT VISUAL
    # hijau = source, merah = target
    # ==================================================

    source_vis = cv2.drawKeypoints(
        source_vis, kp1, None, color=(0, 255, 0)
    )

    target_vis = cv2.drawKeypoints(
        target_vis, kp2, None, color=(0, 0, 255)
    )

    cv2.putText(source_vis, 'SOURCE REGION', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(target_vis, 'TARGET REGION', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    result = cv2.drawMatches(
        source_vis, kp1,
        target_vis, kp2,
        good_matches[:80],
        None,
        matchColor=(0, 220, 120),
        singlePointColor=(120, 120, 120),
        flags=2
    )

    # Resize result biar lebih enak dilihat
    result = cv2.resize(result, (1024, 300))

    # ==================================================
    # OVERLAY ANALISIS
    # ==================================================

    overlay = result.copy()
    cv2.rectangle(overlay, (10, 10), (500, 158), (10, 10, 10), -1)
    result = cv2.addWeighted(overlay, 0.72, result, 0.28, 0)

    cv2.putText(result, 'SIFT COPY-MOVE ANALYSIS', (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    cv2.putText(result, f'Keypoints Source   : {len(kp1)}', (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(result, f'Keypoints Target   : {len(kp2)}', (15, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(result, f'Good Matches       : {len(good_matches)}', (15, 101),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(result, f'Matching           : {matching_percentage:.2f}%', (15, 124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(result, 'Distance Threshold : 120', (15, 147),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ==================================================
    # STATUS
    # ==================================================

    if matching_percentage > 20:
        status = 'COPY-MOVE FORGERY DETECTED'
        color = (0, 0, 255)
    else:
        status = 'LOW MATCH - CHECK REGION'
        color = (0, 200, 255)

    cv2.putText(result, status, (515, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # deskripsi ilmiah di bawah status
    cv2.putText(result, 'SIFT menemukan kemiripan descriptor', (515, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(result, 'dan keypoint tinggi antara source', (515, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(result, 'dan target region — terindikasi', (515, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(result, 'copy-move forgery.', (515, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # ==================================================
    # PRINT TERMINAL
    # ==================================================

    print(f'\n📸 {file}')
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

    # ==================================================
    # SAVE OUTPUT
    # ==================================================

    output_path = os.path.join(output_folder, f'result_{file}')
    cv2.imwrite(output_path, result)

print("\n🎉 ANALISIS SELESAI!")
print("📂 Cek folder dataset/OUTPUT")