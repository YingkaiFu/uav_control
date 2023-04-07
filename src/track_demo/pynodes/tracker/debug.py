import cv2
window_name = 'output'

cap = cv2.VideoCapture("fan-video.mp4")
Flag = 0
while(cap.isOpened()): 
    ret, frame = cap.read()
    cv2.imshow(window_name,frame)
    if Flag==0:
        box = cv2.selectROI(window_name,
                            frame,
                            fromCenter=False,
                            showCrosshair=True)
        Flag = 1
    print(box)
    cv2.waitKey(1)