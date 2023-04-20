# import modules required
import numpy as np
import cv2
import os

# Function: show detections
def show_detections(image, detections):
    h, w, c = image.shape
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            # Centre point of recognition
            core_x = round((endX - startX) / 2 + startX)
            core_y = round((endY - startY) / 2 + startY)
            # rect size get
            start_X = core_x - 95
            end_X = core_x + 95
            start_Y = core_y - 95
            end_Y = core_y + 95

            # cv2.rectangle(image, (start_X, start_Y), (end_X, end_Y),
            #               (0, 255, 0), 1)
            # cv2.putText(image, text, (start_X-70, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    crop_list = [start_X, end_X, start_Y, end_Y]
    return image, crop_list

# Function: detect images
def detect_img(net, image):
    # 其中的固定参数，我们在上面已经解释过了，固定就是如此
    blob = cv2.dnn.blobFromImage(image, 1, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()
    return show_detections(image, detections)

# Function: Batch Process
def Recognition_Processing(net,video_path,targe_path):

    # import image and get corp_list
    image_source = cv2.imread(video_path)
    # processing the images
    image = image_source.copy()
    showimg,crop_list = detect_img(net, image)
    crop = image_source[crop_list[2]:crop_list[3],crop_list[0]:crop_list[1]]
    dst = cv2.resize(crop, (64, 64),interpolation=cv2.INTER_AREA)
    # cv2.imwrite("dnn_recognition.jpg", showimg)
    cv2.imwrite(targe_path, dst)


    # cv2.imshow("img", showimg)
    # cv2.waitKey(0)

if __name__ == "__main__":
    # Import Opencv dnn trained models
    net = cv2.dnn.readNetFromCaffe(
        "F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/opencv_dnn/deploy.prototxt",
        "F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/opencv_dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel")

    # Actor_list
    actor_list = ["DC", "JE", "JK", "KL"]

    for actor in actor_list:
        # Source Image Data Path
        source_data_path = "F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_1_Framing/"
        source_data_path = source_data_path + actor + "/"

        for filepath, dirnames, filenames in os.walk(source_data_path):
            for filename in filenames:
                # get video path
                video_path = filepath + "/" +filename
                # create target folder
                targe_path = video_path.replace("Process_1_Framing","Process_2_Central_Cropping")
                # crteate folder_path
                folder_create_path = filepath.replace("Process_1_Framing","Process_2_Central_Cropping")
                if os.path.exists(folder_create_path) == False:
                    os.makedirs(folder_create_path)
                # Call Function to process
                Recognition_Processing(net,video_path,targe_path)
                print(video_path,'completed')

        print(actor, '-------------------  completed  -------------------')