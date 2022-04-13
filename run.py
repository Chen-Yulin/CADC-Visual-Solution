import cv2
import detect
import preProcess

video=cv2.VideoCapture('./sample1.mp4')

if video.isOpened():
	Open, frame=video.read()
else:
	Open=False

while Open:
	ret, frame=video.read()
	if frame is None:
		break
	if ret is True:
		rect_list,ROI=detect.detect_target(frame)
		#cv2.imshow('cadc',target_img)
		detect.get_numberShot(frame,rect_list,ROI)	




		if cv2.waitKey(100) & 0xFF == 27:
			break

video.release()
cv2.destroyAllWindows()
