#live stream code file for face recognition and dynamic face tracking
#load all the needed library
import cv2
import pickle
import face_recognition
from PIL import *
import PIL.ImageFont
import PIL.ImageDraw
import numpy as np
import time
from datetime import datetime
import pymysql as db

cnx = db.connect(user='root', password='root',
                              host='127.0.0.1',
                              database='test',
                              charset='utf8')

cmd = cnx.cursor()


attpersons=dict()

###########Color for the text############
color = (0, 0, 255)


#will load the face trained model file
with open('encoded_people.pickle', 'rb') as filename:
    people = pickle.load(filename)
    #print(people)
print("Data loaded successfully")

#turn on the camera automically using videocapture supported by opencv
cap = cv2.VideoCapture(0)
#while true refers to one single frame captured at a time

#tree=dict()

while True:
    try:
        ret, frame = cap.read()
        start_time = time.time()
        #time.sleep(0.2)
        scaleFactor = 4
        small_frame = cv2.resize(frame, (0, 0), fx=1/scaleFactor, fy=1/scaleFactor)
        rgb_small_frame = small_frame[:,:,::-1]
        # Find all the faces and face encodings in the current frame of video(one frame every qsqec)
        #model used is hog model
        img_loc = face_recognition.face_locations(rgb_small_frame,model="hog")
        img_enc = face_recognition.face_encodings(rgb_small_frame, known_face_locations=img_loc)

        face_img = PIL.Image.fromarray(frame)
        draw = PIL.ImageDraw.Draw(face_img)
        
        for i in range(0, len(img_enc)):
            print(len(img_enc))
            
            print("*****************i -> {0}".format(i))

            for k, v in people.items():
                result = face_recognition.compare_faces(v, img_enc[i], tolerance=0.5)
                print("*****************name {0} {1} {2}".format(k , i , result ))
                
                if len(result) == result.count(True) or  result.count(True) >= 5:
                    
                #if len(result) == result.count(True):
                    
                    top, right, bottom, left = np.multiply(img_loc[i], scaleFactor)
                    #here is where the linked list comes into play
                    #the first identified will be moved to the list called f this is dynamic assignment
                    #every time the frame captures the picture
                    draw.rectangle([left , top , right , bottom] , outline="red", width=2)
                    draw.text((left, top) , k ,color)
                    attpersons[k] = str(datetime.now())
                    

        #will display the resulting frame
        # Display the resulting frame
        open_cv_image = np.array(face_img)
        # Convert RGB to BGR colored to black and white and vise-versa
        open_cv_image = open_cv_image[:, :, :].copy()
        cv2.imshow('frame',open_cv_image)
        ############################################
        print("--- %s seconds ---" % (time.time() - start_time))
        #print("f",f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            for e in attpersons.keys():
                
                q="insert into attendance ( name , logdatetime) values ('{0}','{1}')".format( e , attpersons[e])
                cmd.execute(q)
                cnx.commit()
            
            cnx.close()
            break
    except:
        print("Face is not faced to camera properly")
        pass
#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()