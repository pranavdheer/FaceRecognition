import cv2
import dlib
import math
import numpy as np

detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recogonizer=cv2.face.createLBPHFaceRecognizer(threshold=95)

def detect(gray): 
 face_store=[]
 (x,y,w,h)=(0,0,0,0)
 faces = face_cascade.detectMultiScale(gray,1.3,6)
 for (x,y,w,h) in faces:   
  save.append((x,y,w,h))
  if y-10>=0 and x-10>=0:
   f=gray[y-10:(y+h+10),x-10:(x+w+10)]  
   f=cv2.resize(f,(100,100)) 
   face_store.append(f) 
  else:
   f=gray[y:(y+h+10),x:(x+w+10)]  
   f=cv2.resize(f,(100,100)) 
   face_store.append(f) 
 return face_store


def align(face_store):
 output=[]   
 flag=0  
 for ix in range(0,len(face_store)): 
  flag=0     
  detections = detector(face_store[ix],2)
  for k,d in enumerate(detections):
   shape = predictor(face_store[ix], d)
   p1=[(shape.part(45).x,shape.part(45).y),(shape.part(36).x,shape.part(36).y)]
   p2=[((int(0.7*100),33)),(int(0.3*100),33)]
   s60 = math.sin(60*math.pi/180);
   c60 = math.cos(60*math.pi/180);  
   inPts = np.copy(p1).tolist();
   outPts = np.copy(p2).tolist();
   xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
   yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
   inPts.append([np.int(xin), np.int(yin)]);
   xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
   yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
   outPts.append([np.int(xout), np.int(yout)]);
   tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);
   img2 = cv2.warpAffine(face_store[ix], tform, (100,100)); 

   detections = detector(img2,3)
   for k,d in enumerate(detections): 
     flag=1
     face=[[abs(d.left()),abs(d.right())],[abs(d.top()),abs(d.bottom())]]
     shape = predictor(img2, d)
     l_eye=np.asarray([(shape.part(36).x,shape.part(36).y),(shape.part(37).x,shape.part(37).y),(shape.part(38).x,shape.part(38).y),(shape.part(39).x,shape.part(39).y),(shape.part(40).x,shape.part(40).y),(shape.part(41).x,shape.part(41).y)])
     r_eye=np.asarray([(shape.part(42).x,shape.part(42).y),(shape.part(43).x,shape.part(43).y),(shape.part(44).x,shape.part(44).y),(shape.part(45).x,shape.part(45).y),(shape.part(46).x,shape.part(46).y),(shape.part(47).x,shape.part(47).y)])
     eye_left=np.mean(l_eye,axis=0)
     eye_right=np.mean(r_eye,axis=0) 
     face[0][0]=int((eye_left[0]+face[0][0])/2.0)
     face[0][1]=int((eye_right[0]+face[0][1])/2.0)
     face[1][1]=int((shape.part(10).y+shape.part(57).y)/2.0)
     img2_cropped=img2[face[1][0]:face[1][1],face[0][0]:face[0][1]]
     img2_cropped=cv2.resize(img2_cropped,(100,100))
     output.append(img2_cropped)   
   if flag==0:
    del(save[ix])
 if len(output)==0:   
  return ('n',1)  
 return ('y',output)


def train(): 
  test_list=[]
  test_label=[]
  for ix in range (1,24):
   face_store=[]
   image=cv2.imread('./training/'+str(ix)+'.jpg') 
   image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   image=cv2.resize(image,(100,100)) 
   face_store.append(image)
   output=align(face_store)
   if output[0]=='y':   
      out=output[1][0]    
      test_list.append(out)
      test_label.append(1)  
  img_array=np.asarray(test_list)
  test_label_array=np.asarray(test_label)
  recogonizer.train(img_array, test_label_array) 
train()    

face_store=[]
cap = cv2.VideoCapture('obama.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate=24
vid = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width,frame_height))
iy=1
frame_count=0
save=[]
while(cap.isOpened()):  
  ret, frame = cap.read()
  if ret == True:
   frame_count=frame_count+1
   (dimensions_x,dimensions_y,z)=frame.shape 
   images=frame     
   copy_image=images
   images=cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)

   if frame_count>=frame_rate or frame_count==1:
    save=[]
    face_store=detect(images)
    output=align(face_store)
    frame_count=1
   if output[0]=='y':
     for ix in range (0,len(output[1])):
      x,y,w,h=save[ix][0],save[ix][1],save[ix][2],save[ix][3]
        
      out=output[1][ix]
      lab= recogonizer.predict(out)  
      if lab[0]==1:
        cv2.rectangle(copy_image,(x,y),(x+w,y+h), (0, 255, 0), 2)
        cv2.putText(copy_image,'Obama',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
      else:
        cv2.rectangle(copy_image,(x,y),(x+w,y+h), (0, 0, 255), 2)
   vid.write(copy_image) 
  else:
    break
cap.release()
cv2.destroyAllWindows()
vid.release()