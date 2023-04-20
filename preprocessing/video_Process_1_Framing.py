import cv2
import os


# Function: video process
def video_process(targe_path,save_path):

    cap = cv2. VideoCapture(targe_path)
    # total FPS
    num = 0
    while True:
        # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
        success, data = cap.read()
        if not success:
            break
        # cv2.imwrite(targe_path+str(num)+".jpg", data)
        num = num + 1
    cap.release()

    # get the number stored
    fac1 = num // 7 # first pic
    fac2 = fac1 * 2
    fac3 = fac1 * 3
    fac4 = fac1 * 4
    fac5 = fac1 * 5
    fac6 = fac1 * 6

    cap2 = cv2.VideoCapture(targe_path)
    num = 0
    while True:
        # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
        success, data = cap2.read()
        if not success:
            break
        if num == fac1:
            cv2.imwrite(save_path+"1.jpg", data)
        elif num == fac2:
            cv2.imwrite(save_path+"2.jpg", data)
        elif num == fac3:
            cv2.imwrite(save_path+"3.jpg", data)
        elif num == fac4:
            cv2.imwrite(save_path+"4.jpg", data)
        elif num == fac5:
            cv2.imwrite(save_path+"5.jpg", data)
        elif num == fac6:
            cv2.imwrite(save_path+"6.jpg", data)
        num = num + 1
    cap.release()


# Actor_list
actor_list = ["DC","JE","JK","KL"]

for actor in actor_list:
    # Source Image Data Path
    source_data_path = "F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/AudioVisualClip/"
    # Targe folder Path
    target_folder_path = "F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_1_Framing/"
    source_data_path = source_data_path + actor +"/"
    target_folder_path = target_folder_path + actor +"/"
    # Iterations
    for filepath, dirnames, filenames in os.walk(source_data_path):
        for filename in filenames:
            # get video path
            video_path = source_data_path+filename
            # create target folder
            foldername = filename.replace(".avi","/")
            targe_path = target_folder_path+foldername
            os.makedirs(targe_path)
            video_process(video_path,targe_path)
            # print(filename,"completed")
    print(actor,"completed")