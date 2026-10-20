import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\USER\Desktop\ASL_Detection\yolo_runs\train_yolo-17\weights\best.pt")
# 0 = default webcam
cap = cv2.VideoCapture(0)  

while True:
    #grab one frame
    ok,frame=cap.read() 
    
    # no frame = camera off or video ended
    if not ok:
        break;
    
    results = model.predict(frame, conf=0.5, verbose=False)
    
    for x1,y1,x2,y2 in results[0].boxes.xyxy.int().tolist():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    cv2.imshow("Hand Dectection",frame)
    if cv2.waitKey(1) & 0XFF==ord("q"):
        break;
    
cap.release()
cv2.destroyAllWindows()