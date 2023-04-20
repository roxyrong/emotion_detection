# Import Modules
import pandas as pd
import numpy as np
import cv2
import os


class Matrix_Conversion(object):

    # CNN Dataset with Crop
    def CNN_Dataset_Crop(self):
        # create a dataframe
        df_inputs = pd.DataFrame(columns=["image","label","Actor"])
        # Constrct X_input_data & Y_label_data
        row_data_list = []
        # Actor_list
        actor_list = ["DC", "JE", "JK", "KL"]
        for actor in actor_list:

            # Source Image Data Path
            source_data_path = "F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_2_Central_Cropping/"
            source_data_path = source_data_path + actor + "/"

            for filepath, dirnames, filenames in os.walk(source_data_path):
                for filename in filenames:
                    img_path = filepath + "/" +filename
                    img = cv2.imread(img_path)
                    # get actor
                    value_actor = actor
                    value_label = img_path.split("/")[-2]
                    # get label
                    value_label = value_label[0:1] if not value_label.startswith("s") else value_label[0:2]
                     # append to dataframe
                    df_inputs.loc[len(df_inputs)+1] = [img,value_label,value_actor]
                    # print processing status
                    print("Completed: ",filepath)

        # processing label
        label_dict = {"a":0,"n":1,"su":2,"sa":3,"d":4,"h":5,"f":6}
        df_inputs["label_value"] = df_inputs["label"].replace(label_dict)
        print(df_inputs["Actor"])
        # save
        np.save("F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_3_Matrix_Conversion/cnn_dataset_crop.npy",df_inputs)
        print("NPY File Save Completed")

    # CNN Dataset Without Crop
    def CNN_Dataset_Without_Crop(self):
        # create a dataframe
        df_inputs = pd.DataFrame(columns=["image","label","Actor"])
        # Constrct X_input_data & Y_label_data
        row_data_list = []
        # Actor_list
        actor_list = ["DC", "JE", "JK", "KL"]
        for actor in actor_list:
            # Source Image Data Path
            source_data_path = "F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_1_Framing/"
            source_data_path = source_data_path + actor + "/"

            for filepath, dirnames, filenames in os.walk(source_data_path):
                for filename in filenames:
                    img_path = filepath + "/" +filename
                    img = cv2.imread(img_path)
                    # resizing
                    dst = cv2.resize(img, (64, 64),interpolation=cv2.INTER_AREA)
                    # cv2.imwrite("dnn_recognition.jpg", showimg)
                    # get actor
                    value_actor = actor
                    value_label = img_path.split("/")[-2]
                    # get label
                    value_label = value_label[0:1] if not value_label.startswith("s") else value_label[0:2]
                    # append to dataframe
                    df_inputs.loc[len(df_inputs)+1] = [dst,value_label,value_actor]
                    # print processing status
                    print("Completed: ",filepath)

        # processing label
        label_dict = {"a":0,"n":1,"su":2,"sa":3,"d":4,"h":5,"f":6}
        df_inputs["label_value"] = df_inputs["label"].replace(label_dict)
        # save
        np.save("F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_3_Matrix_Conversion/cnn_dataset_without_crop.npy",df_inputs)
        print("NPY File Save Completed")

    # CNNLSTM Dataset with Crop
    def CNNLSTM_Crop(self):
        # Construct a df_inputs
        df_inputs = pd.DataFrame(columns=["image","label","Actor"])
        # Constrct X_input_data & Y_label_data
        row_data_list = []
        # Actor_list
        actor_list = ["DC", "JE", "JK", "KL"]
        for actor in actor_list:
            # Source Image Data Path
            source_data_path = "F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_2_Central_Cropping/"
            source_data_path = source_data_path + actor + "/"

            for filepath, dirnames, filenames in os.walk(source_data_path):

                if not filepath.endswith("/"):
                    # get x input
                    img1 = cv2.imread(filepath + "/1.jpg")
                    img2 = cv2.imread(filepath + "/2.jpg")
                    img3 = cv2.imread(filepath + "/3.jpg")
                    img4 = cv2.imread(filepath + "/4.jpg")
                    img5 = cv2.imread(filepath + "/5.jpg")
                    img6 = cv2.imread(filepath + "/6.jpg")

                    value_x = np.array([img1,img2,img3,img4,img5,img6],dtype=object)
                    # get actor
                    value_actor = actor
                    value_label = filepath.split("/")[-1]
                    # get label
                    value_label = value_label[0:1] if not value_label.startswith("s") else value_label[0:2]
                    # append to dataframe
                    df_inputs.loc[len(df_inputs)+1] = [value_x,value_label,value_actor]
                    # print processing status
                    print("Completed: ",filepath)

        # processing label
        label_dict = {"a":0,"n":1,"su":2,"sa":3,"d":4,"h":5,"f":6}
        df_inputs["label_value"] = df_inputs["label"].replace(label_dict)
        # Save NPY File
        np.save("F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_3_Matrix_Conversion/cnnlstm_dataset_crop.npy",df_inputs)
        print("NPY File Save Completed")

    # CNNLSTM Dataset Without Crop
    def CNNLSTM_Without_Crop(self):
        # Construct a df_inputs
        df_inputs = pd.DataFrame(columns=["image","label","Actor"])
        # Constrct X_input_data & Y_label_data
        row_data_list = []
        # Actor_list
        actor_list = ["DC", "JE", "JK", "KL"]
        for actor in actor_list:
            # Source Image Data Path
            source_data_path = "F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_1_Framing/"
            source_data_path = source_data_path + actor + "/"

            for filepath, dirnames, filenames in os.walk(source_data_path):

                if not filepath.endswith("/"):
                    # get x input
                    img1 = cv2.imread(filepath + "/1.jpg")
                    img2 = cv2.imread(filepath + "/2.jpg")
                    img3 = cv2.imread(filepath + "/3.jpg")
                    img4 = cv2.imread(filepath + "/4.jpg")
                    img5 = cv2.imread(filepath + "/5.jpg")
                    img6 = cv2.imread(filepath + "/6.jpg")
                    # resize
                    dst1 = cv2.resize(img1, (64, 64),interpolation=cv2.INTER_AREA)
                    dst2 = cv2.resize(img2, (64, 64),interpolation=cv2.INTER_AREA)
                    dst3 = cv2.resize(img3, (64, 64),interpolation=cv2.INTER_AREA)
                    dst4 = cv2.resize(img4, (64, 64),interpolation=cv2.INTER_AREA)
                    dst5 = cv2.resize(img5, (64, 64),interpolation=cv2.INTER_AREA)
                    dst6 = cv2.resize(img6, (64, 64),interpolation=cv2.INTER_AREA)
                    # contruct array
                    value_x = np.array([dst1,dst2,dst3,dst4,dst5,dst6],dtype=object)
                    # get actor
                    value_actor = actor
                    value_label = filepath.split("/")[-1]
                    # get label
                    value_label = value_label[0:1] if not value_label.startswith("s") else value_label[0:2]
                    # append to dataframe
                    df_inputs.loc[len(df_inputs)+1] = [value_x,value_label,value_actor]
                    # print processing status
                    print("Completed: ",filepath)

        # processing label
        label_dict = {"a":0,"n":1,"su":2,"sa":3,"d":4,"h":5,"f":6}
        df_inputs["label_value"] = df_inputs["label"].replace(label_dict)
        # Save NPY File
        np.save("F:/Coding Projects/Data_Spell_Workspace/project_w207_video_process/Dataset/Process_3_Matrix_Conversion/cnnlstm_dataset_without_crop.npy",df_inputs)
        print("NPY File Save Completed")


if __name__ == "__main__":
    Matrix_Computation = Matrix_Conversion()
    # === Computing Matrix
    Matrix_Computation.CNN_Dataset_Crop()
    Matrix_Computation.CNN_Dataset_Without_Crop()
    Matrix_Computation.CNNLSTM_Crop()
    Matrix_Computation.CNNLSTM_Without_Crop()