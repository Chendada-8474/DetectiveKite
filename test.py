import cv2
from PIL import Image
from detect import is_grayscale
import torch
import yolov5
import pandas as pd
import os
from datetime import datetime



model_color = yolov5.load("./model/exp_color/best.pt")
model_infrared = yolov5.load("./model/exp_infrared/best.pt")

model_color.to("cuda")
model_infrared.to("cuda")

# video_path = "D:/coding/dataset/perch-mount/unorg/大荒野/04210014.JPG"
video_path = "D:/coding/dataset/perch-mount/NPUST/test/鹽埔20220329-0424/2022鹽埔砂石場04120368.MP4"
image_path = "D:/coding/dataset/perch-mount/NPUST/test/鹽埔20220329-0424/2022鹽埔砂石場04120386.JPG"


def vid_detect(video_path: str, model, interval = 1):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    c_dt = datetime.fromtimestamp(os.path.getctime(video_path)).strftime('%Y-%m-%d %H:%M:%S')
    parent_dir = os.path.basename(os.path.abspath(os.path.join(video_path, os.pardir)))
    file_name = os.path.basename(video_path)
    index_frm = 0
    data = pd.DataFrame({
        "confidence":[],
        "class":[],
        "name":[],
        "num_inds":[],
        })

    while cap.isOpened():
        suc, frame = cap.read()

        if index_frm % int(interval*fps) == 0 and suc:
            result = model(frame, size = 640)
            result = result.pandas().xyxy[0][["confidence", "class", "name"]]
            result["num_inds"] = 1
            if len(result) > 0:
                result = result.groupby(["class", "name"]).agg({'confidence' : 'mean', 'num_inds' : 'sum'}).reset_index()
                data = pd.concat([data, result])

        elif index_frm % (interval*fps) != 0 and suc:
            pass
        else:
            break
        index_frm+=1

    if len(data) > 0:
        data = data.groupby(["class", "name"]).agg({'confidence' : 'mean', 'num_inds' : 'max'}).reset_index()
        data["file_name"] = file_name
        data["datetime"] = c_dt
        data["num_inds"] = data["num_inds"].astype(int)
        data["media"] = "video"
        data = data[["file_name", "class", "name", "num_inds", "confidence", "datetime", "media"]]
    else:
        data = pd.DataFrame({
            "file_name": [file_name],
            "class": [None],
            "name": [None],
            "num_inds": [0],
            "confidence": [None],
            "datetime": [c_dt],
            "media": ["video"],
        })
    return data

def img_detect(image_path: str, model):

    c_dt = datetime.fromtimestamp(os.path.getctime(image_path)).strftime('%Y-%m-%d %H:%M:%S')
    parent_dir = os.path.basename(os.path.abspath(os.path.join(image_path, os.pardir)))
    file_name = os.path.basename(image_path)

    result = model(image_path, size = 640)
    result = result.pandas().xyxy[0][["confidence", "class", "name"]]
    if len(result) > 0:
        result["num_inds"] = 1
        result = result.groupby(["class", "name"]).agg({'confidence' : 'mean', 'num_inds' : 'sum'}).reset_index()
        result["file_name"] = file_name
        result["datetime"] = c_dt
        result["num_inds"] = result["num_inds"].astype(int)
        result["media"] = "image"
        result = result[["file_name", "class", "name", "num_inds", "confidence", "datetime", "media"]]
    else:
        result = pd.DataFrame({
            "file_name": [file_name],
            "class": [None],
            "name": [None],
            "num_inds": [0],
            "confidence": [None],
            "datetime": [c_dt],
            "media": ["image"],
        })
    return result

def save_csv(dataframe, dir_name: str, ori_dir_name: str):
    dirs = os.listdir("./")
    if "runs" not in dirs:
        os.mkdir("./runs")
    else:
        dirs = os.listdir("./runs/")
        if "data" not in dirs:
            os.mkdir("./runs/data")
        else:
            dirs = os.listdir("./runs/data/")

        index = 0
        while True:
            if dir_name + str(index) not in dirs:
                os.mkdir("./runs/data/" + dir_name + str(index))
                break
            else:
                index+=1

        dataframe.to_csv("./runs/data/" + dir_name + str(index)+ "/" + ori_dir_name + ".csv", index = False)


image_path = "./sample/015.jpg"
print(img_detect(image_path, model_color))