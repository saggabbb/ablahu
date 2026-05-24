import cv2
import os
import random

# ==================================================
# FOLDER
# ==================================================

REAL_FOLDER = 'dataset/PREPROCESSED'
FAKE_FOLDER = 'dataset/FAKE'

os.makedirs(FAKE_FOLDER, exist_ok=True)

# ==================================================
# KONFIGURASI COPY-MOVE
# ==================================================

TARGET_SIZE = (512, 512)

PATCH_SIZE = 80     # ukuran area yang dicopy
OFFSET_MIN = 40     # jarak minimum perpindahan
OFFSET_MAX = 120    # jarak maksimum perpindahan

# ==================================================
# GENERATE COPY-MOVE
# ==================================================

print("=" * 55)
print("      GENERATOR COPY-MOVE FORGERY")
print("=" * 55)

files = os.listdir(REAL_FOLDER)

success = 0
fail = 0

for file in files:

    img_path = os.path.join(REAL_FOLDER, file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Gagal membaca {file}")
        fail += 1
        continue

    # ==================================================
    # RESIZE
    # ==================================================

    img = cv2.resize(img, TARGET_SIZE)

    h, w = img.shape[:2]

    # ==================================================
    # RANDOM SOURCE AREA
    # ==================================================

    source_x = random.randint(140, 260)
    source_y = random.randint(140, 260)

    # ==================================================
    # RANDOM TARGET AREA
    # ==================================================

    target_x = source_x + random.randint(20, 40)
    target_y = source_y + random.randint(20, 40)

    # ==================================================
    # COPY PATCH
    # ==================================================

    patch = img[
        source_y:source_y + PATCH_SIZE,
        source_x:source_x + PATCH_SIZE
    ].copy()

    # ==================================================
    # PASTE PATCH (COPY-MOVE)
    # ==================================================

    img[
        target_y:target_y + PATCH_SIZE,
        target_x:target_x + PATCH_SIZE
    ] = patch


    # ==================================================
    # SAVE
    # ==================================================

    filename = os.path.splitext(file)[0]
    save_name = f'{filename}_fake.jpg'

    save_path = os.path.join(FAKE_FOLDER, save_name)

    cv2.imwrite(save_path, img)

    print(f"✅ {save_name} berhasil dibuat")
    success += 1

# ==================================================
# RINGKASAN
# ==================================================

print("\n" + "=" * 55)
print("         GENERATE COPY-MOVE SELESAI")
print("=" * 55)

print(f"Berhasil : {success}")
print(f"Gagal    : {fail}")

print(f"\n📂 Hasil tersimpan di folder:")
print(f"{FAKE_FOLDER}/")

print("=" * 55)