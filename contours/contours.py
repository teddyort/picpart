import cv2
import sys

sys.path.append('../util/')
from paths import getDropboxPath

im = cv2.imread(getDropboxPath()+'data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# draw the contours
cv2.drawContours(im, contours, -1, (0,255,0),3)

cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()