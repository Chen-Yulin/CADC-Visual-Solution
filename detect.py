import numpy as np
import argparse
import cv2
import preProcess

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from PIL import Image
from torchvision import transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(threshold=5)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2),
                                   nn.ReLU(), nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120), nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))
        # super(CNN,self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1,16,5,1,2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16,32,5,1,2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        # self.out = nn.Linear(32 * 7 * 7,10) #10分类的问题
 
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x




def my_hstack(img1,img2):
	row1,col1=img1.shape
	row2,col2=img2.shape
	#print(row1,row2)
	if row1>row2:
		img2=cv2.copyMakeBorder(img2,0,row1-row2,0,0,cv2.BORDER_CONSTANT,value=0)
	else:
		img1=cv2.copyMakeBorder(img1,0,row2-row1,0,0,cv2.BORDER_CONSTANT,value=0)
	stack_img=np.hstack((img1,img2))
	return stack_img


def detect_target(image):
	height,width,channels=image.shape
	image_cut=image[50:height-50,50:width-50]

	image_hsv=cv2.cvtColor(image_cut,cv2.COLOR_BGR2HSV)
	red1=np.array([0,100,100]) 
	red2=np.array([10,255,255])
	red3=np.array([160,100,100])
	red4=np.array([179,255,255])
	mask1=cv2.inRange(image_hsv,red1,red2)
	mask2=cv2.inRange(image_hsv,red3,red4)
	mask=cv2.bitwise_or(mask1,mask2)
	after_mask=cv2.add(image_cut, np.zeros(np.shape(image_cut), dtype=np.uint8), mask=mask)
	
	
	ret,binary=cv2.threshold(after_mask[:,:,2],10,255,cv2.THRESH_BINARY)
	#cv2.imshow('Open',binary)
	binary=cv2.copyMakeBorder(binary,50,50,50,50,cv2.BORDER_CONSTANT,value=[0,0,0])
	#cv2.imshow('binary',binary)
	kernel=np.ones([3,3],np.uint8)
	close=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,iterations=3)
	Open=cv2.morphologyEx(close,cv2.MORPH_OPEN,kernel,iterations=1)
	
	contours, hier=cv2.findContours(Open,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	res_img=image.copy()
	ROI=[]
	rect_list=[]
	for i in range(len(contours)):
		if cv2.contourArea(contours[i])<100:
			continue
		epsl=0.01*cv2.arcLength(contours[i],True)
		approx=cv2.approxPolyDP(contours[i],epsl,True)
		x,y,w,h=cv2.boundingRect(approx)
		if (float(w/h)<2) and (float(w/h)>0.5):
			res_img=cv2.rectangle(res_img,(x,y),(x+w,y+h),(0,255,0),1)
			ROI.append([x,y,w,h])
			rect = cv2.minAreaRect(approx)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			rect_list.append(rect)
			res_img=cv2.drawContours(res_img, [box], 0, (255, 0, 0), 2)
	cv2.imshow('res_img',res_img)
	return rect_list,ROI


def get_numberShot(img,rect_list,ROI):
	if len(rect_list)==0:
		#cv2.destroyAllWindows()
		return
	Total_numlist=[]
	print('In this frame, total '+str(len(rect_list))+' target')
	for target_index in range(len(rect_list)):
		x,y,w,h=ROI[target_index][0],ROI[target_index][1],ROI[target_index][2],ROI[target_index][3]	
		target_img=img[y:y+h,x:x+w]
	
		target_center=np.array([rect_list[target_index][0][0]-x,rect_list[target_index][0][1]-y],dtype='int8')
		target_draw=target_img.copy()
		#cv2.circle(target_draw,target_center,1,[0,255,0], thickness=2)
		#cv2.imshow('target'+str(target_index),target_draw)

		target_hsv=cv2.cvtColor(target_img,cv2.COLOR_BGR2HSV)
		white1=np.array([0,0,200]) 
		white2=np.array([180,60,255])
		mask=cv2.inRange(target_hsv,white1,white2)
		
		after_mask=cv2.add(target_img, np.zeros(np.shape(target_img), dtype=np.uint8), mask=mask)
		#cv2.imshow('mas',after_mask)
		kernel=np.ones((3,3),np.uint8)
		ret,binary=cv2.threshold(after_mask[:,:,2],10,255,cv2.THRESH_BINARY)
		binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,1)
		binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel,1)
		#cv2.imshow('number'+str(target_index),binary)
		contours,hier=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

		#获取面积最大的contour的索引值
		max_index=0
		if len(contours)==0:
			continue
		elif len(contours)>1:
			area=[]
			for i in range(len(contours)):
				area.append(abs(cv2.contourArea(contours[i],True)))
				
			max_index=area.index(max(area))

		#处理输出图像
		num_rect=cv2.minAreaRect(contours[max_index])
		#print(num_rect)
		cv2.drawContours(target_draw,[np.int0(cv2.boxPoints(num_rect))],0,[255,0,0],1)
		num_center=np.array([num_rect[0][0],num_rect[0][1]],dtype='int8')
		#cv2.circle(target_draw,num_center,2,[0,255,0], -1)
		
		
		#去除图像中数字外部分
		polyMask_img=np.zeros((h,w),dtype=np.uint8)
		poly_mask=cv2.fillConvexPoly(polyMask_img,np.int0(cv2.boxPoints(num_rect)),[255,255,255])
		ret,poly_mask=cv2.threshold(poly_mask,254,255,cv2.THRESH_BINARY)
		
		#cv2.imshow('poly',poly_mask)
		#print(poly_mask.shape,target_img.shape)
		target_img=cv2.cvtColor(target_img,cv2.COLOR_BGR2GRAY)
		#print(poly_mask)
		target_img=cv2.add(target_img, np.zeros(np.shape(target_img), dtype=np.uint8), mask=poly_mask)
		#cv2.imshow('target'+str(target_index),target_img)
		
		#计算目标偏转角
		#print(num_center,target_center)
		angle=rect_list[target_index][-1]	
		if (num_center[0]-target_center[0])<=0 and (num_center[1]-target_center[1])<=0:
			angle+=90
		elif (num_center[0]-target_center[0])>=0 and (num_center[1]-target_center[1])<=0:
			angle+=180
		elif (num_center[0]-target_center[0])>=0 and (num_center[1]-target_center[1])>=0:
			angle+=270
		#print(angle)
		
		#旋转标靶，对准数字
		row,col=target_img.shape
		target_img=cv2.copyMakeBorder(target_img,int(col*0.5),int(col*0.5),int(col*0.5),int(col*0.5),cv2.BORDER_CONSTANT,value=[0,0,0])
		row,col=target_img.shape
		rotate_M = cv2.getRotationMatrix2D((row*0.5, col*0.5), angle, 1)
		modified_target = cv2.warpAffine(target_img, rotate_M, (col,row))
		mask_black=cv2.inRange(modified_target,0,1)
		kernel=np.ones((3,3),np.uint8)
		#mask_black=cv2.dilate(modified_target,kernel)
		#cv2.imshow('black',mask_black)
		modified_target[mask_black!=0]=255
		modified_target=cv2.resize(modified_target,(col*4,row*4))
		modified_target=preProcess.custom_blur_demo(modified_target)
		
		#cv2.imshow('blur',modified_target)
	
		modified_target=cv2.adaptiveThreshold(modified_target,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,13,5)
		#modified_target=cv2.morphologyEx(modified_target,cv2.MORPH_OPEN,kernel,True)
		#modified_target=cv2.morphologyEx(modified_target,cv2.MORPH_CLOSE,kernel,True,iterations=1)
		#cv2.imshow('modified_target'+str(target_index),modified_target)
		big_kernel=np.ones((int(col*0.05),int(col*0.05)),np.uint8)
		number_box=cv2.morphologyEx(modified_target,cv2.MORPH_CLOSE,big_kernel,iterations=2)
		#cv2.imshow('num_box',number_box)
		numBox_contour,hier=cv2.findContours(number_box,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		if len(numBox_contour)==1:
			x,y,w,h=cv2.boundingRect(numBox_contour[0])
		else:
			continue
		num_img=modified_target[y+int(h*0.18):y+h-int(h*0.18),x+int(w*0.18):x+w-int(w*0.18)]
		#cv2.imshow('num_img'+str(target_index),num_img)
		
#		

		#截取数字
		num_img=cv2.morphologyEx(num_img,cv2.MORPH_CLOSE,kernel)
		num_contour,hier=cv2.findContours(num_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
		row,col=num_img.shape
		j=0
		if len(num_contour)<2:
			continue
		
		num_shot=[]
		num_position=[]
		for i in range(len(num_contour)):
			x,y,w,h=cv2.boundingRect(num_contour[i])
			
			if w*h<int((row*col)*0.1):
				j+=1
				continue
			#num_img=cv2.cvtColor(num_img,cv2.COLOR_GRAY2BGR)
			if i-j>1:
				break
			#print(w*h)
			#print(num_img.shape)
			#numBox_img=cv2.rectangle(num_img,(x,y),(x+w,y+h),(255,255,0),1)
			#cv2.imshow('number'+str(i-j)+'of'+str(target_index),numBox_img)
			num_shot.append(preProcess.transform_num(cv2.medianBlur(num_img[y:y+h,x:x+w],5)))
			num_position.append(x)
		if len(num_position)!=2:
			continue
		if num_position[0]>num_position[1]:
			temp=num_shot[1]
			num_shot[1]=num_shot[0]
			num_shot[0]=temp
		Total_numlist.append(num_shot)
		#print(Total_numlist)
		#print(len(Total_numlist))
		if len(Total_numlist)<target_index+1:
			continue
		
		#cv2.imshow('num1 of '+str(target_index),Total_numlist[target_index][0])
		#cv2.imshow('num2 of '+str(target_index),Total_numlist[target_index][1])
		hstack_img=my_hstack(Total_numlist[target_index][0],Total_numlist[target_index][1])
		cv2.imshow('for target '+str(target_index),hstack_img)

		#使用CNN进行数字识别
		cnn_predict=CNN()
		cnn_predict.load_state_dict(torch.load('./CNN/model_dict.pth'))
		cnn_predict.eval()
		try:
			num_shot_img = Image.fromarray(num_shot[0])
			transf = transforms.ToTensor()
			num_shot_img=transf(num_shot_img)
			num_shot_img = num_shot_img.view(1, 1, 28, 28)
			test_output = cnn_predict(num_shot_img)
			_, prediction = torch.max(test_output, 1)
			number1=prediction.numpy().tolist()
			print(number1[0],end='')
			num_shot_img = Image.fromarray(num_shot[1])
			transf = transforms.ToTensor()
			num_shot_img=transf(num_shot_img)
			num_shot_img = num_shot_img.view(1, 1, 28, 28)
			test_output = cnn_predict(num_shot_img)
			_, prediction = torch.max(test_output, 1)
			number2=prediction.numpy().tolist()
			print(number2[0])
		except:
			print('what?')
			continue





		

