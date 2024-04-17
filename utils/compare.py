import cv2
from PIL import Image

# 使用OpenCV读取灰度图像
gray_image_opencv = cv2.imread('../experiments/slurryData/train/image1.jpg', cv2.IMREAD_GRAYSCALE)

# 使用PIL读取灰度图像
gray_image_pil = Image.open('../experiments/slurryData/train/image1.jpg').convert('L')

# 比较读取结果是否一致
# 这里可以使用像素级别的比较来确保结果一致
# 如果图像分辨率较大，可以考虑使用一些相似性度量方法
cv2.imshow('Grayscale Image', gray_image_opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()
