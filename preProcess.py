
import cv2
import numpy as np

def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)
    output_img = np.array(output_img,np.uint8)  # 这句一定要加上
    return output_img

def custom_blur_demo(image):
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
  dst = cv2.filter2D(image, -1, kernel=kernel)
  return dst

def transform_num(image):
	row,col=image.shape
	if row<col:
		print("尺寸错误")
		return image
	delta=int((row-col)*0.5)
	new_image=cv2.copyMakeBorder(image,0,0,delta,delta,cv2.BORDER_CONSTANT,value=0)
	new_image=cv2.resize(new_image,(16,16))
	new_image=cv2.copyMakeBorder(new_image,6,6,6,6,cv2.BORDER_CONSTANT,value=0)
	return new_image
