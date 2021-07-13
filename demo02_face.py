import os
import face_recognition
import cv2

# 总任务 把脸部特征和数据库中面部特征去匹配
# 读取数据库 并获得图片信息和特征向量
face_databases_dir= 'face_databases'
user_names=[]
user_face_encodings=[]
boss_name=['ru','zhuang']

files = os.listdir(face_databases_dir)
for image_shot_name in files:
    #截取图片名称
    user_name,_ =os.path.splitext(image_shot_name)
    user_names.append(user_name)


    # 读取面部特征向量
    # face_databases/tao.jpg
    image_file_name = os.path.join(face_databases_dir,image_shot_name)
    image_file=face_recognition.load_image_file(image_file_name)
    face_encoding=face_recognition.face_encodings(image_file)[0]
    user_face_encodings.append(face_encoding)

# 打开摄像头
video_capture = cv2.VideoCapture(0)

# 不断的在画面中寻找人脸
while True:
    ret,frame= video_capture.read()
    #找到多人的面部位置
    face_locations=face_recognition.face_locations(frame)
    #找到多人的面部特征信息
    face_encodings=face_recognition.face_encodings(frame,face_locations)
    # 接下来就是去寻找相匹配的人 找不到的话就是Unkown
    names=[]
    # 遍历face_encodings 和数据库中面部特征做匹配
    for face_encoding in face_encodings:
        #compare_faces(['面部特征1'，‘面部特征2’...],  未知面部特征) 如果未知面部特征是其中之一 那么就会返回true 否则就是false
        # 比如 [true,false,false]
        matchs=face_recognition.compare_faces(user_face_encodings,face_encoding)
        name="Unkown"
        for index,is_match in enumerate(matchs):
            if is_match:
                name= user_names[index]
                break
        names.append(name)

    # 标识人的姓名
    for (top,right,bottom,left), name in zip(face_locations,names):

        color=(0,255,0)
        if  name in boss_name:
            color=(0,0,255)
        cv2.rectangle(frame,(left,top),(right,bottom),color,2)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left,top-10),font,0.5,color,1)

    cv2.imshow("Video",frame)
    if cv2.waitKey(1) & 0XFF== ord('q'):
        break

# 释放资源
video_capture.release()
cv2.destroyAllWindows()

