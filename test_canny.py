import numpy as np
from numpy.core.fromnumeric import shape
from scipy import ndimage
import cv2
import os
def intensify(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 0:
                image[i][j] = 255
    return image
def canny(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #reduce noise using gaussian filter 
    blur=cv2.GaussianBlur(image, (5,5) , 0) #apply 5*5 kernal
    #Canny edge detection cv2.Canny(image , low_threshold , high_threshold), threshold: 
    canny = cv2.Canny(blur , 50 , 150)
    return canny

def laplacian(image,kernel_num):
    if kernel_num == "laplacian":
        kernel_laplacian1 =  np.array([[1,1,1],
                                       [1,-8,1],
                                       [1,1,1]])

        kernel_laplacian2 =  np.array([[-1,-1,-1],
                                       [-1,8,-1],
                                       [-1,-1,-1]])
    elif kernel_num == 6:
        kernel_laplacian =  np.array([[1,1,1],
                                       [1,-8,1],
                                       [1,1,1]])
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
        print("yessas")
    return edge

def convolution(image,kernel,normalize):
    if normalize == False:
        kernel_sum = 1
    elif normalize == True:
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

def ben_threshold(img):
    for j in range(img.shape[1]): #由上到下掃描
        for i in range(img.shape[0]):
            if img[i][j] > 150:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img 
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    # x = np.array([[-2 ,-2 ,-2 ,-2 ,-2],
    #               [-1 ,-1 ,-1 ,-1 ,-1],
    #               [ 0  ,0  ,0  ,0  ,0],
    #               [ 1  ,1  ,1  ,1  ,1],
    #               [ 2  ,2  ,2  ,2  ,2]])

    # y = np.array([[-2 ,-1  ,0  ,1  ,2],
    #               [-2 ,-1  ,0  ,1  ,2],
    #               [-2 ,-1  ,0  ,1  ,2],
    #               [-2 ,-1  ,0  ,1  ,2],
    #               [-2 ,-1  ,0  ,1  ,2]])
    x=np.array([[-1 ,-1 ,-1],
                [ 0 , 0  ,0],
                [ 1  ,1 , 1]])

    y = np.array([[-1 , 0 , 1],
                  [-1  ,0 , 1],
                  [-1 , 0  ,1]])
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

# def apply_filter(img,filter):
#     print(filter.shape)
#     print(filter)
#     #newimg = np.zeros((img.shape[0],img.shape[1]))
#     for i in range(img.shape[0]): #480
#         for j in range(img.shape[1]): #640
#             for h in range(filter.shape[0]): #5
#                 for w in range(filter.shape[1]): #5
#                     newimg[i][j] += img[i][j]*filter[h][w]
#     nw = newimg
#     print(newimg)                
#     return nw



gau = gaussian_kernel(5)
print("gau :\n",gau)
print(gau)
icpath = "C:/vscode/machinevision_project"
file_name1 = "good IC mark.bmp"#"bad IC mark2.bmp"#"good IC mark.bmp"
img1 = cv2.imread(os.path.join(icpath,file_name1),-1)
img3 = img1
#th = ben_threshold(img1)
#ret, th1 = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY)
ga1 = convolution(img3,gau,normalize=True)
th = ben_threshold(ga1)
#ga1 = cv2.GaussianBlur(img1, (5,5) , 0)
#ga1 = img1
ga1 = ga1[120:240,240:460]
ga1 = ben_threshold(ga1)
#img11 = convolution(img3,gau) #blur by ben

#th2 = ben_threshold(ga1)
lp2 = laplacian(ga1,kernel_num = "laplacian")
lp2_2 = laplacian(ga1,kernel_num = 6)
lp2 = intensify(lp2)
lp2_2  = intensify(lp2_2)
#lp2_2 = laplacian(ga1,5)
#add_lp2 = cv2.addWeighted(lp2,0.5,lp2_2,0.5,0)
#lp2 = lp2/2 + lp2_2/2
# lp2 = np.hypot(lp2,lp2_2)
# lp2 = cv2.convertScaleAbs(lp2)
# for i in range(lp2.shape[0]):
#     for j in range(lp2.shape[1]):
#         if lp2[i][j] > 0:
#             lp2[i][j] = 255
#lp2 = lp2 + lp2_2
lp3 = cv2.Laplacian(ga1, cv2.CV_16S, ksize=3)
lp3 = cv2.convertScaleAbs(lp3)
# lp = cv2.Laplacian(th2, cv2.CV_16S, ksize=3)
# lp = cv2.convertScaleAbs(lp)

# Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
# Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
can = canny(ga1)
# Ix = convolution(th2,Kx)
# Iy = convolution(th2,Ky)
#G = cv2.addWeighted(Ix,0.5,Iy,0.5,0)
# G = np.hypot(Ix, Iy)
# G = G / G.max() * 255
# G = cv2.convertScaleAbs(G)
#print(img11.shape)
#print("img11",img11)
print("lp:",lp2)


# cv2.imshow("sb0",res)
#cv2.imshow("th2",th2)
cv2.imshow("canny",can)
cv2.imshow("lp2",lp2)
cv2.imshow("lp3",lp2_2)
cv2.imshow("gaussain",ga1)
cv2.imshow("th",th)
cv2.waitKey(0)
cv2.destroyAllWindows() 