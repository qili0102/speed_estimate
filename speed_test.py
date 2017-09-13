import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import datetime
import time

def perspective_transform(img, name):
#     img = cv2.imread(image_path)
    rows,cols,ch = img.shape

    pts1 = np.float32([[1174,387],[1915,481],[0,570],[1165,786]])
    # pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    heightA = np.sqrt(((1174-0) ** 2) + ((387-570) ** 2))
    heightB = np.sqrt(((1915 - 1165) ** 2) + ((481 - 786) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    widthA = np.sqrt(((1915 - 1174) ** 2) + ((481 - 387) ** 2))
    widthB = np.sqrt(((1165 - 0) ** 2) + ((786 - 570) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    pts2 = np.float32([[0,0],[maxWidth,0],[0,maxHeight],[maxWidth,maxHeight]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(maxWidth,maxHeight))

    plt.subplot(121),plt.imshow(img,interpolation='nearest'),plt.title('Input')
    plt.subplot(122),plt.imshow(dst,interpolation='nearest'),plt.title('Output')
#     plt.savefig(name+'.jpg', dpi = 1000)
    return dst

def get_distance(point1, point2):
    x = point2[0] - point1[0]
    y = point2[1] - point1[1]
    dis = np.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))
    return (dis, x, y)

#IOU matching
#return two center points of the bbox
def iou_matching(box, box_list):
    iou_p = -1
    boxes_d = None
    for box_p in box_list:
        if (abs(box[0]-box_p[0]) > box[2]) or (abs(box[1]-box_p[1]) > box[3]):
            continue
        xA = max(box[0], box_p[0])
        yA = max(box[1], box_p[1])
        xB = min(box[0] + box[2], box_p[0] + box_p[2])
        yB = min(box[1] + box[3], box_p[1] + box_p[3])

        # print(xB, xA, yB, yA)

        interArea = (xB - xA + 1) * (yB - yA + 1)

        boxAArea = (box[2] + 1) * (box[3] + 1)
        boxBArea = (box_p[2]+ 1) * (box_p[3] + 1)

        if (boxAArea + boxBArea - interArea) != 0:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        else:
            iou = 1

        # print(box, box_p, iou)

        if iou <= 1 and iou >= 0.5 and iou > iou_p:
            iou_p = iou
            boxes_d = box_p

    print(box, boxes_d)
    return boxes_d

#get Bounding box center point
#input: box=[x, y, w. h]
def getCenterPoint(box):
    return [box[0]+box[2]/2, box[1]+box[3]/2]

# initialize the first frame in the video stream
firstFrame = None

cap = cv2.VideoCapture('1080p_WALSH_ST_000.mp4')
bb_prev = []
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    bb_curr = []
    fps    = cap.get(5)
    print("fps",fps)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = perspective_transform(frame, None)
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if firstFrame is None:
        firstFrame = gray
        continue
    
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue
            
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bb_curr.append([x,y,w,h])
        # text = "Occupied"

        if len(bb_prev) > 0:
            bb_iou = iou_matching([x,y,w,h], bb_prev)
            if bb_iou != None:
                x_iou_c = int(getCenterPoint(bb_iou)[0])
                y_iou_c = int(getCenterPoint(bb_iou)[1])
                x_c = int(getCenterPoint([x,y,w,h])[0])
                y_c = int(getCenterPoint([x,y,w,h])[1])
                cv2.line(frame, (x_iou_c,y_iou_c),(x_c,y_c),(255,0,0),2)
                (distance, dir_x, dir_y) = get_distance([x_iou_c,y_iou_c], [x_c,y_c])
                # speed = distance*fps
                cv2.putText(frame, "Speed: %s/frame  x: %s  y: %s"%(distance, dir_x, dir_y), (x_c,y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    bb_prev = bb_curr
        
    # draw the text and timestamp on the frame
    # cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.putText( frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)

    firstFrame = gray

    key = cv2.waitKey(1) & 0xFF
    
#     cv2.imshow('frame',frame)
#     cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()