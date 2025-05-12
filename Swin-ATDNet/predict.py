import os
import cv2
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from tensorflow.keras.optimizers import Adam
from evaluation_metrics import IoU_coef, IoU_loss
from sklearn.metrics import jaccard_score, confusion_matrix
from datetime import datetime
import random
from DS import unetmodel

np.random.seed(0)

# CLAHE
def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized


patch_size = 512
IMG_HEIGHT = patch_size
IMG_WIDTH = patch_size
IMG_CHANNELS = 1
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Load model and weights
model = unetmodel(input_shape)  # Choose your model (unet, residualunet, etc.)
model.compile(optimizer=Adam(learning_rate=1e-3), loss=IoU_loss, metrics=['accuracy', IoU_coef])

# Ensure correct model loading path
model.load_weights('F:/project/2Retinal-Vessel-Segmentation-using-variants-of-UNET-main/retina_Unet_150epochs.hdf5')

path1 = 'content/drive/MyDrive/training/images'  # Test dataset images directory
path2 = 'content/drive/MyDrive/training/masks'  # Test dataset mask directory

testimg = []
ground_truth = []
prediction = []
global_IoU = []
global_accuracy = []

testimages = sorted(os.listdir(path1))
testmasks = sorted(os.listdir(path2))

for idx, image_name in enumerate(testimages):
    if image_name.endswith(".png"):
        predicted_patches = []
        test_img = skimage.io.imread(path1 + "/" + image_name)

        # Ensure we are working with the green channel for segmentation
        test = test_img[:, :, 1]  # Selecting the green channel
        test = clahe_equalized(test)  # Applying CLAHE

        # Ensure correct image size calculation
        SIZE_X = (test_img.shape[1] // patch_size) * patch_size
        SIZE_Y = (test_img.shape[0] // patch_size) * patch_size
        testimg.append(test)
        test = np.array(test)

        patches = patchify(test, (patch_size, patch_size), step=patch_size)  # Create patches

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, :, :]
                single_patch_norm = (single_patch.astype('float32')) / 255.
                single_patch_norm = np.expand_dims(single_patch_norm, axis=-1)
                single_patch_input = np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(
                    np.uint8)  # Predicat on single patch
                predicted_patches.append(single_patch_prediction)

        predicted_patches = np.array(predicted_patches)
        predicted_patches_reshaped = np.reshape(predicted_patches,
                                                (patches.shape[0], patches.shape[1], patch_size, patch_size))
        reconstructed_image = unpatchify(predicted_patches_reshaped, test.shape)  # Rejoin patches to form whole image
        prediction.append(reconstructed_image)

        # 读取并处理 groundtruth 图像
        groundtruth = skimage.io.imread(os.path.join(path2, testmasks[idx]), as_gray=True)  # 强制灰度读入
        groundtruth = cv2.resize(groundtruth, (reconstructed_image.shape[1], reconstructed_image.shape[0]))  # 尺寸匹配预测图
        groundtruth = (groundtruth > 0.5).astype(np.uint8)  # 二值化：确保只有 0 和 1
        ground_truth.append(groundtruth)

        # 同样确保预测图为 0/1 的二值图
        reconstructed_image_bin = (reconstructed_image > 0.5).astype(np.uint8)

        # 计算 IoU（Jaccard）
        y_true = groundtruth.flatten()
        y_pred = reconstructed_image_bin.flatten()
        IoU = jaccard_score(y_true, y_pred, average='binary')  # binary 二分类最适合你的图像
        global_IoU.append(IoU)

        # Compute Accuracy
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1])
        accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])  # Accuracy of single image
        global_accuracy.append(accuracy)

# Calculate average metrics
avg_acc = np.mean(global_accuracy)
mean_IoU = np.mean(global_IoU)

print('Average accuracy is', avg_acc)
print('Mean IoU is', mean_IoU)

# Checking segmentation results
test_img_number = random.randint(0, len(testimg) - 1)
plt.figure(figsize=(20, 18))
plt.subplot(231)
plt.title('Test Image')
plt.xticks([]); plt.yticks([]);
plt.imshow(testimg[test_img_number])
plt.subplot(232)
plt.title('Ground Truth')
plt.xticks([]); plt.yticks([]);
plt.imshow(ground_truth[test_img_number], cmap='gray')
plt.subplot(233)
plt.title('Prediction')
plt.xticks([]); plt.yticks([]);
plt.imshow(prediction[test_img_number], cmap='gray')
plt.show()

# Prediction on a single image
reconstructed_image = []
test_img = skimage.io.imread('content1/drive/MyDrive/training/images/01_manual1.png')  # Test image
predicted_patches = []
start = datetime.now()

test = test_img[:, :, 1]  # Selecting the green channel
test = clahe_equalized(test)  # Applying CLAHE
SIZE_X = (test_img.shape[1] // patch_size) * patch_size
SIZE_Y = (test_img.shape[0] // patch_size) * patch_size
test = cv2.resize(test, (SIZE_X, SIZE_Y))
test = np.array(test)

patches = patchify(test, (patch_size, patch_size), step=patch_size)  # Create patches

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        single_patch = patches[i, j, :, :]
        single_patch_norm = (single_patch.astype('float32')) / 255.
        single_patch_norm = np.expand_dims(single_patch_norm, axis=-1)
        single_patch_input = np.expand_dims(single_patch_norm, 0)
        single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(
            np.uint8)  # Predict on single patch
        predicted_patches.append(single_patch_prediction)

predicted_patches = np.array(predicted_patches)
predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size, patch_size))
reconstructed_image = unpatchify(predicted_patches_reshaped, test.shape)  # Rejoin patches to form whole image

stop = datetime.now()
print('Execution time: ', (stop - start))  # Computation time

plt.subplot(121)
plt.title('Test Image')
plt.xticks([]); plt.yticks([]);
plt.imshow(test_img)
plt.subplot(122)
plt.title('Prediction')
plt.xticks([]); plt.yticks([]);
plt.imshow(reconstructed_image, cmap='gray')
plt.show()