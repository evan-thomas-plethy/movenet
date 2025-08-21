import cv2
import numpy as np
import os

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

if __name__ == "__main__":
    # Load image
    img_path = "../images/test/HEEL_SLIDES_11_frame_0112.jpg"  # Change to your image path
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image at {img_path}")

    height, width = img.shape[:2]

    # Parameters
    center = np.array([width / 2, height / 2], dtype=np.float32)
    scale = max(height, width) * 1.0
    rot = 0  # rotation in degrees
    output_size = (256, 256)  # width, height

    # Get affine transform matrix
    trans = get_affine_transform(center, scale, rot, output_size)

    # Warp image
    warped_img = cv2.warpAffine(img, trans, output_size, flags=cv2.INTER_LINEAR)

    # Ensure output directory exists
    os.makedirs("../images", exist_ok=True)

    # Save the warped (resized) image
    save_path = "../images/resized.jpg"
    cv2.imwrite(save_path, warped_img)

    print(f"Saved resized image to {save_path}")