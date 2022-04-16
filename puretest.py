import numpy as np
import cv2
import os
icpath = "C:/vscode/machinevision_project"
file_name1 = "bad IC mark1.bmp"#"good IC mark.bmp"
img = cv2.imread(os.path.join(icpath,file_name1),-1)
newimg = np.zeros((img.shape[0],img.shape[1]))
for i in range(3):
    print(i)
print(newimg.shape[0])