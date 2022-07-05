import argparse
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
import yolov5
import torch
from torchvision import transforms
from ffmpeg import probe
from dateutil.parser import parse

PROJECT = "DetectiveKite"

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-so', '--source', type=str, help='path of image, vedeo or a folder', required=True)
    parser.add_argument('-cc', '--classes-color', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('-ci', '--classes-infrared', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('-ctc', '--conf-thres-color', type=restricted_float, default=0.25, help='confidence threshold of model color')
    parser.add_argument('-cti', '--conf-thres-infrared', type=restricted_float, default=0.25, help='confidence threshold model intrared')
    parser.add_argument('-vi', '--video-interval', type=float, default=1, help='video detection interval (s)')
    parser.add_argument('-cm', '--color-mode', type=str, choices=['all', 'color', 'infrared'], default='all', help='Color by Day, Monochrome Infrared by Night')
    parser.add_argument('-na', '--name', default='exp', help='save to project/name')
    parser.add_argument('-sn', '--sp-name', type=str, nargs='+',default=["en"], choices=["en", "sci", "ch", "jp"], help='result species name, default in Englisg common name')

    args = parser.parse_args()
    return args


def is_grayscale(img_path: str, device = "cpu"):

    if img_path.split(".")[-1].lower() in img_formats:
        open_image = Image.open(img_path)
    elif img_path.split(".")[-1].lower() in vid_formats:
        ved = cv2.VideoCapture(img_path)
        suc, frame = ved.read()
        open_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        open_image = Image.fromarray(open_image)
    else:
        print("not support .%s files" % img_path.split(".")[-1])
        return

    r,g,b = open_image.split()

    convert_tensor = transforms.ToTensor()
    r = convert_tensor(r).to(device)
    g = convert_tensor(g).to(device)
    arr = (r-g).to(device)
    percnt0 = 1 - (torch.count_nonzero(arr))/(arr.size(dim=1)*arr.size(dim=2))

    if percnt0 > 0.8:
        return True
    return False

def vid_detect(video_path: str, model, interval = 1):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    c_dt = parse(probe(video_path)["streams"][0]["tags"]["creation_time"]).strftime('%Y-%m-%d %H:%M:%S')
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
        frm_interval = int(interval*fps) if interval*fps > 2 else 1
        if index_frm % frm_interval == 0 and suc:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    img_exif = Image.open(image_path)._getexif()
    c_dt = datetime.strptime(img_exif[36867], '%Y:%m:%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S') if img_exif else None
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

def save_csv(dataframe, dir_name: str, ori_dir_name: str, sp_lang: list):
    directory = Path("./runs/data/")
    directory.mkdir(parents=True, exist_ok=True)
    dirs = os.listdir(directory)

    index = 0
    while True:
        if dir_name + str(index) not in dirs:
            os.mkdir("./runs/data/" + dir_name + str(index))
            break
        else:
            index+=1

    color_sp = pd.read_csv("./model/exp_color/classes_color.csv")
    infrared_sp = pd.read_csv("./model/exp_infrared/color_infrared.csv")
    color_sp["model"] = "color"
    infrared_sp["model"] = "infrared"
    sp_info = pd.concat([color_sp, infrared_sp])

    abb_ref = {
        "sci": "scientific_name",
        "ch": "chinese_name",
        "jp": "japanese_name",
    }

    for lang in sp_lang:
        if lang == "en":
            continue
        dataframe = pd.merge(dataframe, sp_info[["class", "model", abb_ref[lang]]], on=["class", "model"], how="left")

    dataframe.to_csv("./runs/data/" + dir_name + str(index)+ "/" + ori_dir_name + ".csv", index = False)
    return "./runs/data/" + dir_name + str(index)

def detect(opt):
    torch_version = "torch " + torch.__version__
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name()
        device_number = torch.cuda.current_device()
        print("%s %s CUDA:%s (%s)" % (PROJECT, torch_version, device_number, device_name))
    else:
        device = 'cpu'
        device_name = None
        device_number = None
        print("%s %s CPU" % (PROJECT, torch_version))


    print("Loading models...")
    model_color = yolov5.load("./model/exp_color/best.pt") if opt.color_mode in ["all", "color"] else None
    model_infrared = yolov5.load("./model/exp_infrared/best.pt") if opt.color_mode in ["all", "infrared"] else None

    if model_color and device == "cuda": model_color.cuda()
    if model_infrared and device == "cuda": model_infrared.cuda()

    if model_color: model_color.conf = opt.conf_thres_color
    if model_infrared: model_infrared.conf = opt.conf_thres_infrared

    if opt.classes_color and model_color: model_color.classes = opt.classes_color
    if opt.classes_infrared and model_infrared: model_infrared.classes = opt.classes_infrared

    # read filee in folder
    if os.path.isdir(opt.source):
        print("Path detecting: %s" % opt.source)

        files = os.listdir(opt.source)

        result = pd.DataFrame({
            "file_name": [],
            "class": [],
            "name": [],
            "num_inds": [],
            "confidence": [],
            "datetime": [],
            "media": [],
            "model": [],
        })

        num_color_imgs = 0
        num_color_vids = 0
        num_inf_imgs = 0
        num_inf_vids = 0
        num_imgs = 0
        num_vids = 0
        img_runtime = 0
        vid_runtime = 0

        for file in tqdm(files):
            file_format = file.split(".")[-1].lower()
            file_path = os.path.join(opt.source, file)

            if file_format in img_formats:
                time_start = datetime.now()
                if opt.color_mode == "all":
                    if is_grayscale(file_path, device=device):
                        prdct = img_detect(file_path, model_infrared)
                        prdct["model"] = "infrared"
                        num_inf_imgs+=1
                    else:
                        prdct = img_detect(file_path, model_color)
                        prdct["model"] = "color"
                        num_color_imgs+=1

                elif opt.color_mode == "color":
                    prdct = img_detect(file_path, model_color)
                    prdct["model"] = "color"
                    num_imgs+=1
                elif opt.color_mode == "infrared":
                    prdct = img_detect(file_path, model_infrared)
                    prdct["model"] = "infrared"
                    num_imgs+=1
                time_end = datetime.now()
                img_runtime = img_runtime + (time_end - time_start).total_seconds()*1000

            elif file_format in vid_formats:
                time_start = datetime.now()
                if opt.color_mode == "all":
                    if is_grayscale(file_path, device=device):
                        prdct = vid_detect(file_path, model_infrared, interval=opt.video_interval)
                        prdct["model"] = "infrared"
                        num_inf_vids+=1
                    else:
                        prdct = vid_detect(file_path, model_color, interval=opt.video_interval)
                        prdct["model"] = "color"
                        num_color_vids+=1
                elif opt.color_mode == "color":
                    prdct = vid_detect(file_path, model_color)
                    prdct["model"] = "color"
                    num_vids+=1
                elif opt.color_mode == "infrared":
                    prdct = vid_detect(file_path, model_infrared)
                    prdct["model"] = "infrared"
                    num_vids+=1
                time_end = datetime.now()
                vid_runtime = vid_runtime + (time_end - time_start).total_seconds()*1000
            else:
                continue

            result = pd.concat([result, prdct])

        if opt.color_mode == "all":
            per_image = int(img_runtime/(num_color_imgs + num_inf_imgs)) if num_color_imgs + num_inf_imgs > 0 else " "
            per_video = int(vid_runtime/(num_inf_vids + num_color_vids)) if num_inf_vids + num_color_vids > 0 else " "

            print("Files summary: %s color images, %s color video, %s infrared images, %s infrared video, detected" % (num_color_imgs, num_color_vids, num_inf_imgs, num_inf_vids))
            print("Speed: %sms per image, %sms per video" % (per_image, per_video))
        else:
            per_image = int(img_runtime/num_imgs) if num_imgs > 0 else " "
            per_video = int(vid_runtime/num_vids) if num_vids > 0 else " "
            print("Files summary: %s images, %s video, have been detected" % (num_imgs, num_vids))
            print("Speed: %sms per image, %sms per video" % (int(img_runtime/num_imgs), int(vid_runtime/num_vids)))

        save_dir = save_csv(result, opt.name, os.path.basename(opt.source[:-1]), opt.sp_name)
        print("Results saved to %s" % save_dir)

        return

    else:
        print("path: %s" % opt.source)
        file_format = opt.source.split(".")[-1].lower()
        file_path = opt.source
        if file_format in img_formats:
            time_start = datetime.now()
            if opt.color_mode == "all":
                if is_grayscale(file_path, device=device):
                    prdct = img_detect(file_path, model_infrared)
                else:
                    prdct = img_detect(file_path, model_color)

            elif opt.color_mode == "color":
                prdct = img_detect(file_path, model_color)
            elif opt.color_mode == "infrared":
                prdct = img_detect(file_path, model_infrared)
            time_end = datetime.now()
            img_runtime = (time_end-time_start).total_seconds()*1000
            print("Speed: %sms" % int(img_runtime))

        elif file_format in vid_formats:
            time_start = datetime.now()
            if opt.color_mode == "all":
                if is_grayscale(file_path, device=device):
                    prdct = vid_detect(file_path, model_infrared, interval=opt.video_interval)
                else:
                    prdct = vid_detect(file_path, model_color, interval=opt.video_interval)
            elif opt.color_mode == "color":
                prdct = vid_detect(file_path, model_color)
            elif opt.color_mode == "infrared":
                prdct = vid_detect(file_path, model_infrared)
            time_end = datetime.now()
            img_runtime = (time_end-time_start).total_seconds()*1000
            print("Speed: %sms" % int(img_runtime))
        else:
            print("Unsupported media type, Please check the --source")
            return
        save_dir = save_csv(prdct, opt.name, os.path.basename(file_path).split(".")[0], opt.sp_name)
        print("Results saved to %s" % save_dir)
        return

def main():
    opt = parse_opt()
    detect(opt)


if __name__ == "__main__":
    main()

