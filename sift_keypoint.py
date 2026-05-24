import cv2
import os

# ==================================================
# FOLDER
# ==================================================

INPUT_FOLDER = 'dataset/FAKE'
OUTPUT_FOLDER = 'dataset/KEYPOINT'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==================================================
# LOAD FILE
# ==================================================

files = os.listdir(INPUT_FOLDER)

print("=" * 50)
print("         SIFT KEYPOINT DETECTION")
print("=" * 50)

for file in files:

    img_path = os.path.join(INPUT_FOLDER, file)

    # baca gambar grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Gagal membaca {file}")
        continue

    # ==================================================
    # SIFT
    # ==================================================

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(img, None)

    # ==================================================
    # DRAW KEYPOINTS
    # ==================================================

    output = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # ==================================================
    # SAVE OUTPUT
    # ==================================================

    filename = os.path.splitext(file)[0]

    save_path = os.path.join(
        OUTPUT_FOLDER,
        f'{filename}_keypoints.jpg'
    )

    cv2.imwrite(save_path, output)

    # ==================================================
    # INFO
    # ==================================================

    print(f"\n📷 File : {file}")
    print(f"✅ Keypoints terdeteksi : {len(keypoints)}")
    print(f"✅ Descriptor shape     : {descriptors.shape}")

print("\n" + "=" * 50)
print("         SELESAI")
print("=" * 50)