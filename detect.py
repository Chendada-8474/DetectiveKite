import pandas as pd
import argparse
import os
from pathlib import Path
from utils.predictor import *
from utils.preprocess import *
from torchvision import transforms

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def restri_batch_size(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a integer" % (x,))

    if x < 1:
        raise argparse.ArgumentTypeError("%r have to be bigger than 0"%(x,))
    return x

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-so', '--source', type=str, help='path of image, vedeo or a folder', required=True)
    parser.add_argument('-bs', '--batch-size', type=restri_batch_size, help='detecting batch size, set the batch size as big as you can', required=True)
    parser.add_argument('-cc', '--classes-color', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('-ci', '--classes-infrared', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('-ctc', '--conf-thres-color', type=restricted_float, default=0.25, help='confidence threshold of model color')
    parser.add_argument('-cti', '--conf-thres-infrared', type=restricted_float, default=0.25, help='confidence threshold model intrared')
    parser.add_argument('-vi', '--video-interval', type=float, default=1, help='video detection interval (s)')
    parser.add_argument('-na', '--name', default='exp', help='save to project/name')
    parser.add_argument('-sn', '--sp-name', type=str, nargs='+',default=["en"], choices=["en", "sci", "ch", "jp"], help='result species name, default in Englisg common name')

    args = parser.parse_args()
    return args


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
    infrared_sp = pd.read_csv("./model/exp_infrared/classes_infrared.csv")
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
    dir_path = opt.source
    batch_size = opt.batch_size
    interval = opt.video_interval

    medias = MediaJudgement()
    medias.classify(dir_path)

    color_images = medias.color_image
    infrad_images = medias.infrad_image
    color_videos = medias.color_video
    infrad_videos = medias.infrad_video

    transform = transforms.Compose([transforms.Resize((480, 640))])

    color_img_dataset = ImageDataset(dir_path, color_images, transform=transform)
    infrad_img_dataset = ImageDataset(dir_path, infrad_images, transform=transform)

    model_init = PredictInit()
    model_init.set_model(classes_color = opt.classes_color, classes_infrad = opt.classes_infrared, conf_color = opt.conf_thres_color, conf_infrad = opt.conf_thres_infrared)
    predictor = Predictor(model_init.model_color, model_init.model_infrad)

    color_image_results = predictor.detect_imgs(color_img_dataset, model_type="color", batch_size=batch_size)
    infrad_image_results = predictor.detect_imgs(infrad_img_dataset, model_type="infrared", batch_size=batch_size)
    color_video_results = predictor.detect_vids(dir_path, color_videos, model_type="color", interval=interval)
    intrad_video_results = predictor.detect_vids(dir_path, infrad_videos, model_type="infrared", interval=interval)

    results = pd.concat([color_image_results, infrad_image_results, color_video_results, intrad_video_results], ignore_index = True)
    results["num_inds"] = results["num_inds"].astype("int")

    save_dir = save_csv(results, opt.name, os.path.basename(dir_path[:-1]), opt.sp_name)
    print("Results saved to %s" % save_dir)

def main():
    opt = parse_opt()
    detect(opt)

if __name__ == "__main__":
    main()

