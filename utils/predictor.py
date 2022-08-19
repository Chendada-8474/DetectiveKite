from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from tqdm import tqdm
from tqdm.contrib import tzip
import pandas as pd
import os
import cv2


class ImageDataset(Dataset):
    def __init__(self, img_dir: str, imgs: list, transform=None):
        self.imgs = imgs[0]
        self.dts = imgs[1]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = read_image(img_path)
        shape = image.shape
        if self.transform is not None:
            image = self.transform(image)

        return image, self.imgs[idx], self.dts[idx], shape


class Predictor:
    def __init__(self, model_color, model_infrad):
        self.model_color = model_color
        self.model_infrad = model_infrad
        pass

    def _empty_result(self):
        result_data = pd.DataFrame(
            {
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
            }
        )

        return result_data

    def _empty_row_data(self):
        empty_data = pd.DataFrame(
            {
                "confidence": [],
                "class": [],
                "name": [],
                "xmin": [],
                "ymin": [],
                "xmax": [],
                "ymax": [],
                "num_inds": [],
            }
        )
        return empty_data

    def _accident_shot(self, file_name, c_dt, media: str, model: str):
        accident_data = pd.DataFrame(
            {
                "file_name": [file_name],
                "class": [None],
                "name": [None],
                "num_inds": [0],
                "confidence": [None],
                "datetime": [c_dt],
                "media": [media],
                "model": [model],
                "xmin": [None],
                "ymin": [None],
                "xmax": [None],
                "ymax": [None],
            }
        )

        return accident_data

    def detect_vids(self, dir_path: str, files: list, model_type=None, interval=1):

        if len(files[0]) == 0:
            return

        result_data = self._empty_result()
        if model_type == "infrared":
            model = self.model_infrad
        elif model_type == "color":
            model = self.model_color
        else:
            print("model type should be infrared or color")
            return

        print("detecting %s %s videos..." % (len(files[0]), model_type))

        for f, dt in tzip(files[0], files[1]):
            video_path = os.path.join(dir_path, f)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            index_frm = 0
            data = self._empty_row_data()
            while cap.isOpened():
                suc, frame = cap.read()
                frm_interval = int(interval * fps) if interval * fps > 2 else 1
                if index_frm % frm_interval == 0 and suc:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = model(frame, size=640)
                    result = result.pandas().xyxy[0]
                    result["num_inds"] = 1
                    result = (
                        result.groupby(["class", "name"])
                        .agg(
                            {
                                "confidence": "mean",
                                "num_inds": "sum",
                                "xmin": "min",
                                "ymin": "min",
                                "xmax": "max",
                                "ymax": "max",
                            }
                        )
                        .reset_index()
                    )
                    data = pd.concat([data, result])
                elif index_frm % (interval * fps) != 0 and suc:
                    pass
                else:
                    break
                index_frm += 1

            if len(data) > 0:
                data["file_name"] = f
                data["datetime"] = dt
                data["model"] = model_type
                data["media"] = "video"
                data = (
                    data.groupby(["file_name", "class", "name", "datetime"])
                    .agg(
                        {
                            "confidence": "mean",
                            "num_inds": "max",
                            "xmin": "min",
                            "ymin": "min",
                            "xmax": "max",
                            "ymax": "max",
                            "media": "first",
                            "model": "first",
                        }
                    )
                    .reset_index()
                )
            else:
                data = self._accident_shot(f, dt, "video", model_type)

            result_data = pd.concat([result_data, data])

        return result_data

    def _to_numpy(self, tensor):
        return tensor.numpy()

    def detect_imgs(self, dataset, model_type=None, batch_size=1):

        if model_type == "infrared":
            model = self.model_infrad
        elif model_type == "color":
            model = self.model_color
        else:
            print("model type should be infrared or color")
            return

        img_loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=False)

        result_data = self._empty_result()

        print(
            "detecting %s (batch size %s) %s images..."
            % (len(dataset), batch_size, model_type)
        )
        for data in tqdm(img_loader):
            img, file_name, date_time, shape = data
            results = model([i.numpy() for i in img], size=640)
            for d, fn, dt, w, h in zip(
                results.pandas().xyxy, file_name, date_time, shape[1], shape[2]
            ):
                if len(d) > 0:
                    d["num_inds"] = 1
                    d["file_name"] = fn
                    d["datetime"] = dt
                    d["model"] = model_type
                    d["media"] = "image"
                    d = (
                        d.groupby(["file_name", "class", "name"])
                        .agg(
                            {
                                "confidence": "mean",
                                "num_inds": "sum",
                                "xmin": "min",
                                "ymin": "min",
                                "xmax": "max",
                                "ymax": "max",
                                "datetime": "first",
                                "model": "first",
                                "media": "first",
                            }
                        )
                        .reset_index()
                    )
                    d["xmin"] = d["xmin"] * int(w / 480)
                    d["ymin"] = d["ymin"] * int(h / 640)
                    d["xmax"] = d["xmax"] * int(w / 480)
                    d["ymax"] = d["ymax"] * int(h / 640)
                else:
                    d = self._accident_shot(fn, dt, "image", model_type)

                result_data = pd.concat([result_data, d])

        return result_data
