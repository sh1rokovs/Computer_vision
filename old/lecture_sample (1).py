import cv2
import numpy as np

#1 : reading and pixel access
img = cv2.imread('lena.jpg', 0)

w,h = img.shape
print(w,h)
print(img[100,100])
img[100,100] = 255

cv2.line(img, (0,0), (100,100), (255,255,255), 1)
cv2.putText(img, 'Lena', (100,100), 1, 1, (255,255,255))
cv2.imwrite("lena2.jpg", img)

cv2.namedWindow('sample')
cv2.imshow("sample", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#2 : simple operations
img = cv2.imread('lena.jpg')

cv2.imshow("input", img)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img2[200:300,:,1] = img2[200:300,:,1] * 1.15
#img2[:,:,1] = img2[:,:,1] * 1.15
img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
cv2.imshow("Saturation", img2)

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img2 = cv2.equalizeHist(img2)
ret, img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((5,5), np.uint8) 
img2 = cv2.erode(img2, kernel)

cv2.imshow("contrast binarization erode", img2)
cv2.waitKey(0)

#3 : contour processing
contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cv2.drawContours(img, contours, max_index, (0, 0, 255), 5)

cv2.imshow("Contour", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#4 : filters 
#cv2.Canny(img,100,200)
#cv2.goodFeaturesToTrack()
#cv2.medianBlur(img,5)
#cv2.blur(img,(5,5))
#cv2.GaussianBlur(img,(5,5))

#5 : fft
img = cv2.imread('lena.jpg',0)
f1 = np.fft.fft2(img)
f2 = np.fft.fftshift(f1)

mask = np.zeros(f1.shape, np.uint8)
mask = np.array(mask,  dtype = np.uint8)

k = 0.05
cv2.circle(mask, ((int)(w/2), (int)(h/2)), (int)(w*k), (255,255,255), -1)
cv2.imshow('mask', mask)

f2[mask!=0]=0

spectrum = np.log(np.abs(0.1+f2))

dst = np.array(spectrum,  dtype = np.float32)
img_dst = cv2.normalize(dst,None,0,1,cv2.NORM_MINMAX)
cv2.imshow('spectrum', img_dst)

conv_img = np.fft.ifft2(np.fft.ifftshift(f2)).real

#conv_img = np.fft.ifft2(f1)
img_dst = np.array(conv_img, dtype = np.uint8)
                        
cv2.imshow('result', img_dst)
cv2.waitKey(0)

cv2.destroyAllWindows()


#6 : face detection
img = cv2.imread('lena.jpg',0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
faces = face_cascade.detectMultiScale(img, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
# Display the output
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#video processing, optical flow

cap = cv2.VideoCapture(0) #"video.avi"
ret, prev_img = cap.read()
prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
while True:
  ret, img = cap.read()
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imshow("video", img)
  #OF
  points = cv2.goodFeaturesToTrack( img, 1000, 0.01, 10)
  points2, st, err = cv2.calcOpticalFlowPyrLK(  img, prev_img, points, None, winSize=(20, 20) )
  
  points = np.int0(points)
  points2 = np.int0(points2)
  
  prev_img = img
  for i, p in enumerate(points):
     x1,y1 = p.ravel()
     x2,y2 = points2[i].ravel()
     cv2.circle(img, (x1,y1), 2, 255)
     cv2.line(img, (x1,y1), (x2,y2), 255, 1)
  
  cv2.imshow("video result", img)
  
  if cv2.waitKey(10) == 32:
      break
  



