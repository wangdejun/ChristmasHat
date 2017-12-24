import numpy as np
import dlib
import cv2

hat_img = cv2.imread("hat_img.png")
img = cv2.imread("Torres.jpg")

r,g,b = cv2.split(hat_img)
rgb_hat = cv2.merge((r,g,b))
cv2.imwrite("hat_alpha.jpg", r)

predictor_path = "shape_predictor_5_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path) 
detector = dlib.get_frontal_face_detector() 
dets = detector(img, 1) 


if len(dets)>0: 
    for d in dets:
        x,y,w,h = d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top() 
        # x,y,w,h = faceRect 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)
        shape = predictor(img, d)
        for point in shape.parts(): 
            print "length of face points"
            print len(shape.parts())
            cv2.circle(img, (point.x, point.y), 3, color=(255, 0, 0)) 


# pick the lefteye point and the righteye point
point1 = shape.part(0)

print point1
point2 = shape.part(2)

# calculate the middle point
eyes_center = ((point1.x+point2.x)/2, (point1.y+point2.y)/2)
cv2.circle(img,eyes_center,3,color=(0,255,0)) 

# adjust the hat size to the face size
factor = 1.5 
resized_hat_h = int(round(rgb_hat.shape[0]*w/rgb_hat.shape[1]*factor))
resized_hat_w = int(round(rgb_hat.shape[1]*w/rgb_hat.shape[1]*factor))

if resized_hat_h > y:
    resized_hat_h = y-1 
    print "resized_hat_h"

# adjust the hat size to the face size
resized_hat = cv2.resize(rgb_hat, (resized_hat_w, resized_hat_h))

mask = cv2.resize(r,(resized_hat_w,resized_hat_h))
mask_inv = cv2.bitwise_not(mask)

dh = 0 
dw = 0 

bg_roi = img[y+dh-resized_hat_h:y+dh, x+dw:x+dw+resized_hat_w] 
bg_roi = img[y+dh-resized_hat_h:y+dh, (eyes_center[0]-resized_hat_w//3):(eyes_center[0] + resized_hat_w//3*2)] 

# extract the zone from Origin image ROI 
bg_roi = bg_roi.astype(float)
mask_inv = cv2.merge((mask_inv, mask_inv,mask_inv))
alpha = mask_inv.astype(float)/255 

print "K"
alpha = cv2.resize(alpha, (bg_roi.shape[1], bg_roi.shape[0])) 
print("alpha size: ", alpha.shape)
print("bg_roi size: ", bg_roi.shape)
bg = cv2.multiply(alpha, bg_roi)
bg = bg.astype('uint8')

hat = cv2.bitwise_and(resized_hat,resized_hat,mask = mask)
hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))

# 2 ROI zone plus
add_hat = cv2.add(bg,hat) 
cv2.imshow("add_hat",add_hat) 
# 
img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)] = add_hat
cv2.imshow("image", img)
cv2.waitKey()
