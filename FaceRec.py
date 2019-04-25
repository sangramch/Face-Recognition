import cv2

face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_face(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    coords=face_detector.detectMultiScale(gray,1.3,5)
    if len(coords)!=0:
        for (x,y,w,h) in coords:
            return gray[y:y+w,x:x+h],coords
    else:
        return None,None


recogniser=cv2.face.LBPHFaceRecognizer_create()
recogniser.read("facerecog.xml")

textlabels=["Unknown","Sangramjit","Ryan"]

def realtimeRecognition():
    cap=cv2.VideoCapture(0)

    while(True):
        _,frame=cap.read()
    
        frame=cv2.flip(frame, 1)
    
        face,square=get_face(frame)
        if face is not None:
            pred=recogniser.predict(face)
            if pred[1]>60:
                text=textlabels[0]
            else:
                text=textlabels[pred[0]]
            for (x,y,w,h) in square:
                frame=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def recogniseFromImage(image):
    face,square=get_face(image)
    if face is not None:
        pred=recogniser.predict(face)
        if pred[1]>60:
            text=textlabels[0]
        else:
            text=textlabels[pred[0]]
        for (x,y,w,h) in square:
            image=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        
        cv2.imshow("frame",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    
if __name__=="__main__":
    
    print("Do you want to: \n1: Perform Realtime Recognition\n2: Perform Recognition from Static Image")
    choice=int(input("Enter::: "))
    
    if choice==1:
        realtimeRecognition()
    elif choice==2:
        path=input("Enter full image path: ")
        image=cv2.imread(path)
        recogniseFromImage(image)
    else:
        print("Invalid Choice")
