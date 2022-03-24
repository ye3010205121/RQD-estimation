import cv2
import matplotlib.image as mpimg
from pylab import *

# Read the original image
img_name = "DSC06695.JPG"

img = mpimg.imread(img_name)


imshow(img)

x =ginput(4)
data = []
for i in x:
    data.append(i)


# obtain the height and width
img_height,img_width = img.shape[:2]

# define the points
points1 = np.float32([data])
points2 = np.float32([[0,0], [img_width,0], [0,img_height], [img_width,img_height]])

# calculate the transfermation matrix
M = cv2.getPerspectiveTransform(points1, points2)

# realize the perspective
processed = cv2.warpPerspective(img,M,(img_width,img_height))

#color change
processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

save_path = ".\\" + img_name
cv2.imwrite(save_path, processed)
