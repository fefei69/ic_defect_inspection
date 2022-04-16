import cv2
import numpy as np

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
    x, y = np.mgrid[-size:size+1, -size:size+1]
    x=np.array([[-1 ,-1 ,-1],
                [ 0 , 0  ,0],
                [ 1  ,1 , 1]])

    y = np.array([[-1 , 0 , 1],
                  [-1  ,0 , 1],
                  [-1 , 0  ,1]])
    print(x,'\n',y)
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

image = cv2.imread("bad IC mark1.bmp",-1).astype(float) / 255.0
kernel = gaussian_kernel(3)
def convolution(image,kernel):
    kernel_sum = kernel.sum()

    # fetch the dimensions for iteration over the pixels and weights
    i_width, i_height = image.shape[0], image.shape[1]
    k_width, k_height = kernel.shape[0], kernel.shape[1]
    print(i_width,i_height,k_width,k_height)
    # prepare the output array
    filtered = np.zeros_like(image)

    # Iterate over each (x, y) pixel in the image ...
    for y in range(i_width):
        for x in range(640):
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

image = cv2.imread("bad IC mark1.bmp",-1).astype(float) / 255.0
kernel = gaussian_kernel(3)
filtered = convolution(image,kernel)
kernel_laplacian = np.array([[0.01,1,0.01],
                            [1,-4,1],
                            [0.01,1,0.01]])
edge = convolution(filtered,kernel_laplacian)

cv2.imshow('DIY convolution', filtered)
cv2.imshow("ed",edge)

# wait and quit
cv2.waitKey(0)
cv2.destroyAllWindows()