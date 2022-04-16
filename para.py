import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
def ben_edge_detection(img):
    for j in range(img.shape[1]): #由上到下掃描
        for i in range(img.shape[0]):
            if img[i][j] - img[i][j-1] != 0:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img

def Gaussian_blur(img):
    blur=cv2.GaussianBlur(img, (5,5) , 0)
    return blur
def binary(img):
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return th 

def dilation1(img):
    ct = 0
    for j in range(img.shape[1]-2): #由左到右掃描
        for i in range(img.shape[0]-2): 
            # for x in range(2):
            #     for y in range(2):
            #         ct+=int(img[i+x][j+y])
            if (int(img[i][j]) + int(img[i][j+1]) + int(img[i][j+2]) + 
                int(img[i+1][j]) + int(img[i+2][j]) + int(img[i+1][j+1]) + 
                int(img[i+1][j+2]) + int(img[i+2][j+1]) + int(img[i+2][j+2])) == 0:
            # if ct == 0 :
                img[i][j] = 0 
            else:
                img[i][j] = 255
    return img

def erosion(img):
    ct = 0
    for j in range(img.shape[1]-2): #由左到右掃描
        for i in range(img.shape[0]-2): 
            # for x in range(2):
            #     for y in range(2):
            #         ct+=int(img[i+x][j+y])
            if (int(img[i][j]) > 0 and int(img[i][j+1]) >0 and int(img[i][j+2]) >0 and 
                int(img[i+1][j]) >0 and int(img[i+2][j]) >0 and int(img[i+1][j+1]) >0 and 
                int(img[i+1][j+2]) >0 and int(img[i+2][j+1]) >0 and int(img[i+2][j+2])>0) :
            # if ct == 0 :
                img[i][j] = 255 
            else:
                img[i][j] = 0
    return img

# def dilation2(img):
#     for j in range(img.shape[1]-2): #由左到右掃描
#         for i in range(img.shape[0]-2): 
#             if (img[i-1][j] + img[i-1][j] + img[i-1][j+1] + 
#                 img[i][j-1] + img[i][j] + img[i][j+1] + 
#                 img[i+1][j+1] + img[i+1][j] + img[i+1][j+1]) == 0:

#                 img[i][j] = 0 
#             else:
#                 img[i][j] = 255
#     return img

def compute_pixels(img):
    count = 0
    for j in range(img.shape[1]): #由上到下掃描
        for i in range(img.shape[0]):
            if img[i][j] ==255:
                count+=1
    return count
def canny(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #reduce noise using gaussian filter 
    blur=cv2.GaussianBlur(image, (5,5) , 0) #apply 5*5 kernal
    #Canny edge detection cv2.Canny(image , low_threshold , high_threshold), threshold: 
    canny = cv2.Canny(blur , 50 , 150)
    return canny
kernel = np.ones((3,3), np.uint8)
img1 = cv2.imread("bad IC mark2.bmp",-1)#cv2.imread("good IC mark.bmp",-1)

img4 = img1
img2 = canny(img1)
img3 = img2
white_pixel = compute_pixels(img2)
#iter 2
dilation_by_ben1 = dilation1(img3)
dilation_by_ben1 = dilation1(dilation_by_ben1)
er = erosion(dilation_by_ben1)
#dilation_by_ben2 = dilation2(img2)
dilation = cv2.dilate(img2, kernel, iterations = 1)
ret, th1 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
th_canny_img = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
print(img2.shape)
print("number of white pixels",white_pixel)
#-----------------------------------------------------

bl = Gaussian_blur(img4)
ret, th1 = cv2.threshold(bl, 150, 255, cv2.THRESH_BINARY)
# result = binary(bl)
th1 = th1[120:240,240:460]
#result = ben_edge_detection(th1)

#print(img2)
# plt.imshow(img2)
# plt.show() 


# cv2.imshow("res",th1)
# cv2.imshow("t",img2)
# cv2.imshow("a",th_canny_img)
# cv2.imshow("sx",255-dilation)
# cv2.imshow("ben",255-dilation_by_ben1)
# cv2.imshow("eee",er)
cv2.imshow("blur",th1)
cv2.waitKey(0)
cv2.destroyAllWindows() 