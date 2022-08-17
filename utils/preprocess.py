from multiprocessing import Pool
from dateutil.parser import parse
from ffmpeg import probe
from PIL import Image
from datetime import datetime
import pandas as pd
import numpy as np
import yolov5
import torch
import cv2
import pathlib
import os

PROJECT_PATH = pathlib.Path(__file__).parent.parent.resolve()

class MediaJudgement:
    def __init__(self):
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']
        self.vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
        self.infrad_image = ([],[])
        self.infrad_video = ([],[])
        self.color_image = ([],[])
        self.color_video = ([],[])
        self.available_cpus = os.cpu_count()

    def _judge_media(self, mid_path: str):
        try:
            if mid_path.split(".")[-1].lower() in self.img_formats:
                open_image = Image.open(mid_path)
                dt = datetime.strptime(open_image._getexif()[36867], '%Y:%m:%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S') if open_image._getexif() else None
                open_image = np.array(open_image)
                is_image = True
            elif mid_path.split(".")[-1].lower() in self.vid_formats:
                dt = parse(probe(mid_path)["streams"][0]["tags"]["creation_time"]).strftime('%Y-%m-%d %H:%M:%S')
                ved = cv2.VideoCapture(mid_path)
                suc, open_image = ved.read()
                is_image = False
                # open_image = Image.fromarray(open_image)
            else:
                print("Unsupported file type detected: %s " % os.path.basename(mid_path))
                return ()
        except:
            print("Skip one media: %s\nThis file could be corrupted" % mid_path)
            return ()

        detect_wid, detect_hei = 64, 48
        x_begin = int(round(len(open_image)/2 - detect_wid/2, 0))
        y_begin = int(round(len(open_image[0])/2 - detect_hei/2, 0))

        # convert_tensor = transforms.ToTensor()
        detect_ary = open_image[y_begin:y_begin+detect_hei, x_begin:x_begin+detect_wid]
        ch0, ch1, ch2 = detect_ary[:, :, 0], detect_ary[:, :, 1], detect_ary[:, :, 2]
        not_zero = np.count_nonzero(ch0 - ch1)/(detect_wid*detect_hei)
        is_gray = True if not_zero < 0.05 else False
        return is_gray, is_image, dt

    def classify(self, dir_path: str):
        print("Scaning files in direction...")
        begin = datetime.now()
        files = os.listdir(dir_path)
        mids = [os.path.join(dir_path, i) for i in files]
        pool = Pool(self.available_cpus)
        outputs = pool.map(self._judge_media, mids)

        for i, o in enumerate(outputs):
            if len(o) == 0:
                continue
            elif o[0] and o[1]:
                self.infrad_image[0].append(files[i])
                self.infrad_image[1].append(o[2])
            elif o[0] and not o[1]:
                self.infrad_video[0].append(files[i])
                self.infrad_video[1].append(o[2])
            elif not o[0] and o[1]:
                self.color_image[0].append(files[i])
                self.color_image[1].append(o[2])
            elif not o[0] and not o[1]:
                self.infrad_video[0].append(files[i])
                self.infrad_video[1].append(o[2])

        end = datetime.now()
        print("Time consumption: ", end - begin)

        print("%s color images, %s infrared images, %s color videos, %s infrared videos detected" % (len(self.color_image[0]), len(self.infrad_image[0]), len(self.color_video[0]), len(self.infrad_video[0])))


class PredictInit():
    def __init__(self):
        self.PROFECT = "DetectiveKite"

        color_path = os.path.join(PROJECT_PATH, "model/exp_color/classes_color.csv")
        infrad_path = os.path.join(PROJECT_PATH, "model/exp_infrared/classes_infrared.csv")
        self.species_color = pd.read_csv(color_path)
        self.species_inifrad = pd.read_csv(infrad_path)

        self.torch_version = "torch " + torch.__version__
        self.device = "cpu"
        self.device_name = None
        self.device_number = None

        if torch.cuda.is_available():
            self.device = "cuda"
            self.device_name = torch.cuda.get_device_name()
            self.device_number = torch.cuda.current_device()

        print("%s %s CUDA:%s (%s)" % (self.PROFECT, self.torch_version, self.device_number, self.device_name))
        print("Loading models...")
        self.model_color = yolov5.load("./model/exp_color/best.pt")
        self.model_infrad = yolov5.load("./model/exp_infrared/best.pt")

        self.result = pd.DataFrame({
            "file_name": [],
            "class": [],
            "name": [],
            "num_inds": [],
            "confidence": [],
            "datetime": [],
            "media": [],
            "model": [],
            "xmin": [],
            "ymin": [],
            "xmax": [],
            "ymax": [],
        })

    def set_model(self, classes_color = None, classes_infrad = None, conf_color = 0.25, conf_infrad = 0.25):
        if self.model_color and self.device == "cuda": self.model_color.cuda()
        if self.model_infrad and self.device == "cuda": self.model_infrad.cuda()

        self.model_color.conf = conf_color
        self.model_infrad.conf = conf_infrad

        print("Color model species detecting:")
        if classes_color:
            self.model_color.classes = classes_color
            species = list(self.species_color[self.species_color["class"].isin(classes_color)]["name"])
            print(*species, sep = ", ")
        else:
            print("Detecting all color model species.")

        print("Infrared model species detecting:")
        if classes_infrad:
            self.model_infrad.classes = classes_infrad
            species = list(self.classes_infrad[self.classes_infrad["class"].isin(classes_infrad)]["name"])
            print(*species, sep = ", ")
        else:
            print("Detecting all infrared model species.")

