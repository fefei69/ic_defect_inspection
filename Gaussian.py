import cv2
import os
import numpy as np 

def intensify(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 0:
                image[i][j] = 255
    return image

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x=np.array([[-1 ,-1 ,-1],
                [ 0 , 0  ,0],
                [ 1  ,1 , 1]])

    y = np.array([[-1 , 0 , 1],
                  [-1  ,0 , 1],
                  [-1 , 0  ,1]])
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def ben_threshold(img):
    for j in range(img.shape[1]): #由上到下掃描
        for i in range(img.shape[0]):
            if img[i][j] > 150:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img 

def laplacian(image,kernel_num):
    if kernel_num == "laplacian":
        kernel_laplacian1 =  np.array([[1,1,1],
                                       [1,-8,1],
                                       [1,1,1]])

        kernel_laplacian2 =  np.array([[-1,-1,-1],
                                       [-1,8,-1],
                                       [-1,-1,-1]])
    elif kernel_num == 5 :
        kernel_laplacian =  np.array([[-1,-1,-1],
                                    [-1,8,-1],
                                    [-1,-1,-1]])
    elif kernel_num == 2 :
        kernel_laplacian = np.array([[0,-1,0],
                                    [-1,4,-1],
                                    [0,-1,0]])
    # kernel_laplacian =  np.array([[2,0,2],
    #                             [0,-8,0],
    #                             [2,0,2]])
    elif kernel_num == 3 :
        kernel_laplacian =  np.array([[1,4,1],
                                    [4,-20,4],
                                    [1,4,1]])
    #LoG laplacian of Gaussian
    elif kernel_num == 4 :
        kernel_laplacian = np.array([[0 ,0 ,-1 ,0 ,0],
                                     [0 ,-1 ,-2 ,-1 ,0],
                                     [ -1  ,-2  ,16  ,-2  ,-1],
                                     [ 0  ,-1  ,-2  ,-1  ,0],
                                     [ 0  ,0  ,-1  ,0  ,0]])
    #kernel_laplacian = np.array([0,0,0,3,0,-3,0,0])

    if kernel_num == "laplacian":
        edge1 = convolution(image,kernel_laplacian1,normalize=False)
        edge2 = convolution(image,kernel_laplacian2,normalize=False)
        edge  = edge1/2 + edge2/2
    else:
        edge = convolution(image,kernel_laplacian,normalize=False)
    return edge

def convolution(image,kernel,normalize):
    if normalize == False:
        kernel_sum = 1
    else:
        kernel_sum = kernel.sum()

    # fetch the dimensions for iteration over the pixels and weights
    i_width, i_height = image.shape[0], image.shape[1]
    k_width, k_height = kernel.shape[0], kernel.shape[1]
    print(i_width,i_height,k_width,k_height)
    # prepare the output array
    filtered = np.zeros_like(image)

    # Iterate over each (x, y) pixel in the image ...
    for y in range(i_width):
        for x in range(i_height):
            weighted_pixel_sum = 0
            for ky in range(-(k_height // 2), k_height - 1):
                for kx in range(-(k_width // 2), k_width - 1):
                    pixel = 0
                    pixel_y = y - ky 
                    pixel_x = x - kx 

                    # boundary check: all values outside the image are treated as zero.
                    # This is a definition and implementation dependent, it's not a property of the convolution itself.
                    if (pixel_y >= 0) and (pixel_y < i_width) and (pixel_x >= 0) and (pixel_x < i_height)  :
                        #print(pixel_y,pixel_x)
                        pixel = image[pixel_y, pixel_x]

                    # get the weight at the current kernel position
                    # (also un-shift the kernel coordinates into the valid range for the array.)
                    weight = kernel[ky + (k_height // 2-1), kx + (k_width // 2-1)]

                    # weigh the pixel value and sum
                    weighted_pixel_sum += pixel * weight
            #print(pixel_x)
            # finally, the pixel at location (x,y) is the sum of the weighed neighborhood
            filtered[y, x] = weighted_pixel_sum / kernel_sum
    return filtered

def canny(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #reduce noise using gaussian filter 
    blur=cv2.GaussianBlur(image, (5,5) , 0) #apply 5*5 kernal
    #Canny edge detection cv2.Canny(image , low_threshold , high_threshold), threshold: 
    canny = cv2.Canny(blur , 50 , 150)
    return canny

def erode(img):
    for j in range(img.shape[1]-2): #由左到右掃描
        for i in range(img.shape[0]-2): 
            if (int(img[i][j]) > 0 and int(img[i][j+1]) >0 and int(img[i][j+2]) >0 and 
                int(img[i+1][j]) >0 and int(img[i+2][j]) >0 and int(img[i+1][j+1]) >0 and 
                int(img[i+1][j+2]) >0 and int(img[i+2][j+1]) >0 and int(img[i+2][j+2])>0) :

                img[i][j] = 255 
            else:
                img[i][j] = 0
    return img

def dilate(img):
    for j in range(img.shape[1]-2): #由左到右掃描
        for i in range(img.shape[0]-2): 
            if (int(img[i][j]) + int(img[i][j+1]) + int(img[i][j+2]) + 
                int(img[i+1][j]) + int(img[i+2][j]) + int(img[i+1][j+1]) + 
                int(img[i+1][j+2]) + int(img[i+2][j+1]) + int(img[i+2][j+2])) == 0:

                img[i][j] = 0 
            else:
                img[i][j] = 255
    return img

def compute_pixels(img):
    count = 0
    for j in range(img.shape[1]): #由上到下掃描
        for i in range(img.shape[0]):
            if img[i][j] ==255:
                count+=1
    return count


def make_3_channels (img2_org):
    img2_org = ben_threshold(img2_org)
    #img2_org = canny(img2_org)
    img2 = np.zeros( ( np.array(img2_org).shape[0], np.array(img2_org).shape[1], 3 ) )
    img2_org = np.array(img2_org)
    img2[:,:,0] = img2_org
    img2[:,:,1] = img2_org
    img2[:,:,2] = img2_org
    return img2
'''
read image
'''
icpath = "C:/vscode/machinevision_project/ic_defect_inspection"
file_name1 = "good IC mark.bmp"
file_name2 = "via6.bmp"#bad IC mark3.bmp" #"via10.bmp"  #"bad IC mark1.bmp"  via10
img1 = cv2.imread(os.path.join(icpath,file_name1),-1)
img2_org = cv2.imread(os.path.join(icpath,file_name2),-1)
#cv2.imshow("img1",img2_org)
# cv2.imshow("can",img1_via)
# cv2.imshow("r",img2_via)
#cv2.waitKey(0)
'''
make 3 channels to show where the defect is with colors 
'''
img2_org_3ch = make_3_channels(img2_org)
'''
crop the original image to get a ideal image
'''
img1_crp = img1[120:250,240:460]
img2_crp = img2_org[120:250,240:460]
#gaussian kernel
gau = gaussian_kernel(5)
#gaussian blur
img1_blur = convolution(img1_crp,gau,normalize=True) 
img2_blur = convolution(img2_crp,gau,normalize=True)
img1_blur_th = ben_threshold(img1_blur)
img2_blur_th = ben_threshold(img2_blur)

img1 = laplacian(img1_blur_th,kernel_num="laplacian")
img2 = laplacian(img2_blur_th,kernel_num="laplacian")
cv2.imshow("img1",img2)
# cv2.imshow("can",img1_via)
# cv2.imshow("r",img2_via)
cv2.waitKey(0)
# img1 = img1_blur_th
# img2 = img2_blur_th
print(img1.shape,img2.shape)

def find_via(img2):
    for i in range(img2.shape[0]): #由上到下掃描
        for j in range(img2.shape[1]):
            if img2[i][j] - img2[i][j-1] != 0:
                print(i,j)
                pos = (i,j)
                img2_via = img2[i:i+70,j:j+185]      
                return img2_via,pos

img1_via,pos1 = find_via(img1)
img2_via,pos2 = find_via(img2)           
print("img2 shape",img2_via.shape,"img1 shape",img1_via.shape)
#cv2.imshow("img1",img1)
# cv2.imshow("can",img1_via)
# cv2.imshow("r",img2_via)
# cv2.waitKey(0)



'''
preprocessing
'''

'''
also works . just a little bit nosie ....
'''
#kernel = np.ones((3,3), np.uint8)
#黑白反轉 reverse black and white 
# th_canny_img = 255-cv2.dilate(img1_via, kernel, iterations = 1)
# th_canny_img2 = 255-cv2.dilate(img2_via, kernel, iterations = 1)

th_canny_img = 255-dilate(img1_via)#255-img1_via#255-dilate(img1_via)
th_canny_img2 = 255-dilate(img2_via)#255-img2_via#255-dilate(img2_via)
# cv2.imshow("can",th_canny_img)
# cv2.imshow("r",th_canny_img2)
# cv2.waitKey(0)


ar = np.array(th_canny_img)
th_canny_img2 = np.array(th_canny_img2)
minus = ar - th_canny_img2
print("shape of minus",minus.shape)
#kernel = np.ones((3,3), np.uint8)
#erosion = erode(minus)#cv2.erode(minus, kernel, iterations = 1)
kernel = np.ones((3,3), np.uint8)
erosion = erode(minus)
#erosion = cv2.erode(minus, kernel, iterations = 1)
# cv2.imshow("err",erosion)
# cv2.waitKey(0)
dilation = dilate(erosion)
# cv2.imshow("minus",minus)
# cv2.imshow("dilation",dilation)
# cv2.waitKey(0)
#dilation = cv2.dilate(erosion, kernel, iterations = 1)
z = 0
for x in range(dilation.shape[1]):
    for y in range(dilation.shape[0]):
        #print(mi[y][x])
        if dilation[y][x] == 255: #via10 : 104 bad ic mark1:94  bad IC mark2:161
            i , j = pos2
            cv2.circle(img2_org_3ch,(240+j+x,120+i+y+5),1,(0,0,255),-1)
            #print(y+i,x+j)
            z+=1
if z > 60:
    cv2.putText(img2_org_3ch,"bad IC mark!!",(220,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1)
    print("bad ic mark")
else:
    cv2.putText(img2_org_3ch,"good IC mark!!",(220,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
    print("good IC mark!!")

white_pixels_img1 = compute_pixels(img1_via)
white_pixels_img2 = compute_pixels(img2_via)
print("white pixels of img1",white_pixels_img1)
print("white pixels of img2",white_pixels_img2)
print(z)
if white_pixels_img2 > white_pixels_img1 :
    print("method2 - also shows bad ic mark")


#cv2.imshow("err",erosion)    
#cv2.imshow('minus',minus)
#cv2.imshow('dilation - defect only',dilation)
#cv2.imwrite("goodimage.jpg",img2_org_3ch)
cv2.imshow("defect image",img2_org_3ch)
cv2.waitKey(0)
cv2.destroyAllWindows()
