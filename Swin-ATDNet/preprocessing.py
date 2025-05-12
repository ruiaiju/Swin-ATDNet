import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_retinal_pipeline(image_path):
    """生成与参考图匹配的预处理流程图（修正版）"""
    # 1. 读取原始图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像加载失败: {image_path}")

    # 2. 转换为RGB并裁剪ROI（移除黑色边框）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    img_cropped = img_rgb[y:y + h, x:x + w]

    # 3. 生成五种预处理结果
    ## (a) 原始图像
    original = img_cropped

    ## (b) 灰度化
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)

    ## (c) 标准化（修正实现）
    mean, std = cv2.meanStdDev(gray)
    standardized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # 改用MINMAX归一化

    ## (d) 自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_hist = clahe.apply(gray)

    ## (e) 伽马变换（γ=1.5）
    gamma =0.68
    lookup_table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_trans = cv2.LUT(gray, lookup_table)

    # 4. 创建可视化面板
    plt.figure(figsize=(18, 6))

    # (a) Original image
    plt.subplot(151)
    plt.imshow(original)
    plt.title('(a) Original image', fontsize=12, pad=10)
    plt.axis('off')

    # (b) Graying
    plt.subplot(152)
    plt.imshow(gray, cmap='gray')
    plt.title('(b) Graying', fontsize=12, pad=10)
    plt.axis('off')

    # (c) Standardization
    plt.subplot(153)
    plt.imshow(standardized, cmap='gray')
    plt.title('(c) Standardization', fontsize=12, pad=10)
    plt.axis('off')

    # (d) Adaptive histogram
    plt.subplot(154)
    plt.imshow(adaptive_hist, cmap='gray')
    plt.title('(d) Adaptive histogram', fontsize=12, pad=10)
    plt.axis('off')

    # (e) Gamma transform
    plt.subplot(155)
    plt.imshow(gamma_trans, cmap='gray')
    plt.title('(e) Gamma transform', fontsize=12, pad=10)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return original, gray, standardized, adaptive_hist, gamma_trans


# 使用示例
if __name__ == "__main__":
    image_path = "F:/project/2Retinal-Vessel-Segmentation-using-variants-of-UNET-main/3.png"  # 替换为您的图像路径
    preprocess_retinal_pipeline(image_path)