import cv2
import numpy as np
import os

# ==================================================
# FOLDER
# ==================================================

INPUT_FOLDER = 'dataset/FAKE' #nanti diganti kl udah yg matching
OUTPUT_FOLDER = 'dataset/RANSAC'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==================================================
# LOAD FILE
# ==================================================

files = os.listdir(INPUT_FOLDER)

print("=" * 50)
print("            RANSAC FILTER")
print("=" * 50)

for file in files:

    img_path = os.path.join(
        INPUT_FOLDER,
        file
    )

    img = cv2.imread(
        img_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        print(f"❌ Gagal membaca {file}")
        continue


    # ==================================================
    # SIFT
    # ==================================================

    sift = cv2.SIFT_create(
        nfeatures=5000
    )

    keypoints, descriptors = sift.detectAndCompute(
        img,
        None
    )

    if descriptors is None:
        continue


    # ==================================================
    # FLANN MATCHER
    # ==================================================

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


    # ==================================================
    # LOWE RATIO
    # ==================================================

    for match_tuple in matches:

        if len(match_tuple)<3:
            continue

        m,n,o = match_tuple

        if n.distance < 0.72*o.distance:

            good_matches.append(n)



    # ==================================================
    # RANSAC
    # ==================================================

    ransac_matches=[]

    if len(good_matches)>=4:

        src_pts=np.float32(
            [keypoints[m.queryIdx].pt
            for m in good_matches]
        ).reshape(-1,1,2)


        dst_pts=np.float32(
            [keypoints[m.trainIdx].pt
            for m in good_matches]
        ).reshape(-1,1,2)


        H,mask=cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )


        if mask is not None:

            mask=mask.ravel()

            ransac_matches=[
                good_matches[i]
                for i in range(len(good_matches))
                if mask[i]
            ]


    # ==================================================
    # DRAW RANSAC
    # ==================================================

    output=cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    for match in ransac_matches:

        pt1=tuple(
            map(
                int,
                keypoints[match.queryIdx].pt
            )
        )

        pt2=tuple(
            map(
                int,
                keypoints[match.trainIdx].pt
            )
        )

        cv2.circle(
            output,
            pt1,
            4,
            (0,0,255),
            -1
        )

        cv2.circle(
            output,
            pt2,
            4,
            (0,0,255),
            -1
        )

        cv2.line(
            output,
            pt1,
            pt2,
            (0,255,0),
            1
        )


    # ==================================================
    # SAVE
    # ==================================================

    filename=os.path.splitext(file)[0]

    save_path=os.path.join(
        OUTPUT_FOLDER,
        f'{filename}_ransac.jpg'
    )

    cv2.imwrite(
        save_path,
        output
    )


    # ==================================================
    # INFO
    # ==================================================

    print(f"\n📷 File : {file}")
    print(f"✅ Sebelum RANSAC : {len(good_matches)}")
    print(f"✅ Setelah RANSAC : {len(ransac_matches)}")


print("\n"+"="*50)
print("          SELESAI")
print("="*50)