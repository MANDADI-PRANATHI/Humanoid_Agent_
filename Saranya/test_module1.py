# test_module1.py
"""
Simple test / demo for module1.py
Usage:
    python test_module1.py --image test.jpg --model_folder /path/to/openpose/models
"""

import argparse
import cv2
import numpy as np
from module import PoseExtractor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--model_folder", "-m", default="./models/", help="OpenPose models folder")
    parser.add_argument("--out", "-o", default="debug_skeleton.jpg", help="Output debug image path")
    args = parser.parse_args()

    # create extractor
    extractor = PoseExtractor(model_folder=args.model_folder)

    # run pipeline and get drawn image
    res = extractor.get_initial_pose(args.image, return_drawn=True)

    print("Selected person index:", res.selected_idx)
    print("BBox:", res.bbox)
    print("Confidence sum:", res.confidence_sum)
    print("Estimated angles (radians):")
    for k, v in res.theta_dict.items():
        print(f"  {k}: {v:.4f} rad  ({v * 180.0 / np.pi:.1f} deg)")

    if res.theta_init_vector is not None:
        print("Theta init vector 3D(placeholder):", res.theta_init_vector_3d)

    # save debug image
    cv2.imwrite(args.out, res.image)
    print("Saved debug image to", args.out)

    # try show (optional; may fail on headless)
    try:
        img = cv2.imread(args.out)
        cv2.imshow("Skeleton", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    main()

