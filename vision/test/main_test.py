import cv2
import numpy as np
import os
import shutil
from itertools import combinations

cap=cv2.VideoCapture("../media/5.mp4")

min_area=1500
potential_rect=[]
cnt=0

close_kernel=np.ones((5,5),np.uint8)
for tmp_dir in ("../wrong_pic_pure", "../wrong_pic"):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
render = cv2.VideoWriter("../output2.mp4", fourcc, 60, (1280, 720))

while True:
    potential_rect = []
    rect=[]

    ret,frame=cap.read()
    if not ret:
        break

    blurred = cv2.GaussianBlur(frame,(5,5),0)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    binary=cv2.threshold(gray,120,255,cv2.THRESH_BINARY)[1]
    closed=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,close_kernel)

    contous,_=cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    for contour in contous:
        if(cv2.contourArea(contour)<min_area):
            continue
        peri=cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,0.02*peri,True)
        if(len(approx)!=4):
            continue

        x,y,w,h=cv2.boundingRect(contour)
        if w<h:
            continue

        if w==0 or h==0:
            continue

        rate = h/w

        potential_rect.append(contour)

    rect_features = []
    for contour in potential_rect:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w / 2
        center_y = y + h / 2
        rect_features.append((contour, x, w, h, center_x, center_y))

    matched_indexes = set()
    for (i, (_, x1, w1, h1, cx1, cy1)), (j, (_, x2, w2, h2, cx2, cy2)) in combinations(enumerate(rect_features), 2):
        x_limit = max(w1, w2) / 10
        y_limit = max(h1, h2) / 10
        if abs(cx1 - cx2) < x_limit and abs(cy1 - cy2) < y_limit and abs(x1 - x2) <= max(w1, w2)/5:
            matched_indexes.update({i, j})

    rect = [rect_features[i][0] for i in sorted(matched_indexes)]


    if(len(rect)!=2):
        cnt+=1
        cv2.drawContours(frame, potential_rect, -1, (0, 0, 255), 3)
        wrong=np.vstack((frame,cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)))
        cv2.imwrite("..\wrong_pic\wrong{}.jpg".format(cnt),wrong)
        cv2.imwrite("..\wrong_pic_pure\wrong_frame{}.jpg".format(cnt), frame)
        print("第{}张识别错误".format(cnt))
    cv2.drawContours(frame,rect,-1,(0,0,255),3)

    render.write(frame)
    cv2.imshow("frame",frame)
    if(cv2.waitKey(20)&0xFF==ord('q')):
        break


cv2.destroyAllWindows()
