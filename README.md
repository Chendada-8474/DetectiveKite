# DetectiveKite

[DetectiveKite](https://github.com/Chendada-8474/DetectiveKite) is a free perch mount camera trap images auto-detection system which is developed base on yolov5s, a light and fast object detection architecture. I believe DetectiveKite can help biologists in Taiwan to accelerate the heavy species labeling works.

## Datasets

### Transfer Learning
- [NABirds Dataset](https://dl.allaboutbirds.org/nabirds)

### Augmentation
- Private data from [Endemic Species Research Institute](https://www.tesri.gov.tw)
- [Landscape Pictures, Kaggle](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)
    - Owner: [Arnaud ROUGETET](https://www.kaggle.com/arnaud58)
- [Landscape color and grayscale images, Kaggle](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization)
    - Owner: [Black Mamba](https://www.kaggle.com/theblackmamba31)

### Train and Validation
- Private data from [Institute of Wildlife Conservation](http://wildmic.npust.edu.tw/bin/home.php), National Pingtune University of Science and Technology
- Private data from [Department of Biological Resources](https://www.ncyu.edu.tw/biors/), National Chiayi University
- [iNaturalist](https://www.inaturalist.org)

## Install

### git clone
```
git clone https://github.com/Chendada-8474/DetectiveKite.git
cd DetectiveKite
```

### Packages
```
pip install yolov5
```

### GPU Accelerated Computing
For GPU acceleration, please install the compatible torch package with your device. see [INSTALL PYTORCH](https://pytorch.org). Futhermore, installing [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) correctly is also necessary.

## Documentation

### Detection

#### detect.py
`detect.py` runs source of images, video, or a folder. use `--source` or `-so` to set path of target media. The results will be saved to `runs/data/`.

```
python detect.py -so img.jpg   # image
                      vid.mp4   # video
                      path\     # directory
```

#### Argument Parameter

Trail camera normally fill infrared light at night or in bad light condition when animals are detected and take a black and white photo. Otherwise, it takes color photo in Natural light. Two models are used to detect color and infrared image saperatly. So, please note that some parameter have to be set saperatly.

| argument | abbreviation | parameter | default | require |
| -------- | -------- | -------- | -------- | ------- |
| `--source`              | `-so`  | path                   |  | True |
| `--classes-color`       | `-cc`  | int series             | all classes | False |
| `--classes-infrared`    | `-ci`  | int series             | all classes | False |
| `--conf-thres-color`    | `-ctc` | float, range 0~1       | 0.25 | False |
| `--conf-thres-infrared` | `-cti` | float, range 0~1       | 0.25 | False |
| `--vedio-interval`      | `-vi`  | int (second)           | 1 | False |
| `--color-mode`          | `-cm`  | all, color or infrared | all | False |
| `--name`                | `-n`   | results folder name    | exp |  False |

##### Classes


When deteting, you are able to use
`--classes-color` and `--classes-infrared` to spefify classes. Information of classes is in `./model/exp_color/classes_color.csv` and `./model/exp_infrared/classes_infrared.csv`.
```
python detect.py -so path\   # all classes
```

color
```
python detect.py -so path\ -cc 0      # only specify class o
                               0 15   # specify class 0 and 15
```

infrared
```
python detect.py -so path\ -ci 0      # only specify class o
                               0 3    # specify class 0 and 3
```

or both
```
python detect.py -so path\ -cc 0 15 -ci 0 3
```

##### Confidence Threshold
Setting the confidence threshold when detecting using `--conf-thres-color` and `conf-thres-infrared`. The results which confidence is below than confidence threshold will not be saved.

```
python detect.py -so path\ -ctc 0.6 -cti 0.75
```

##### Video Interval
Detecting EACH frame of video just for knowing what kind of birds stanted on perch mount is not a efficient way. DetectiveKite detect video in interval to speed up the processing of detection. Set `--video-interval` to adjust the suitable detection interval base on your video length.

```
python detect.py -so path\ -vi 2   # detect video frame only in 2s interval
```

##### Color Mode
DetectiveKite judge whether a file gray-scale or not before detection. If you files are all color or gray-scale, set the `--color-mode`. DetectiveKite will skip the judgement step. It may speed up the processing.

```
python detect.py -so path\ -cm color   # use color model to detect all files
```

##### Saved Folder Name
After detection, results will be saved to `./runs/data/exp/`. exp is the default folder name. You can change the name via `--name`.

```
python detect.py -so path\ -n project
```

Then, results will be saved to `./runs/data/project/`

### Draw Bounding Box and Label Name
If you want some thing cool like these:

![](https://github.com/Chendada-8474/DetectiveKite/blob/main/runs/detect/exp/016.jpg?raw=true)

![](https://github.com/Chendada-8474/DetectiveKite/blob/main/runs/detect/exp2/005.JPG?raw=true)

Please use `yolov5 detect` to do it. see [yolov5 package](https://pypi.org/project/yolov5/)

example:
```
yolov5 detect --source ./sample/016.jpg --weights ./model/exp_color/best.pt    # color media
```
```
yolov5 detect --source ./sample/005.JPG --weights ./model/exp_infrared/best.pt    # infrared media
```


## Results
Results are saved as .csv format. Columns: file_name, class, name, num_inds, confidence, media, model.

Information of columns:

| column   | data type | description |
| -------- | -------- | -------- |
| file_name       | string    | file name of image or video  |
| class           | integer   | code of class |
| name            | string    | name of detected species |
| num_inds        | integer   | number of individuals detected |
| confidence      | float     | see [this](https://chih-sheng-huang821.medium.com/深度學習-物件偵測-you-only-look-once-yolo-4fb9cf49453c) |
| datetime        | datetime  | file create datetime |
| media           | string    | image or video |
| model           | string    | detected by color model or infrared model |

*NOTE: If detected media is video, the confidence is the average of every detected frame*

## Donation
Althought, this is not a big project, I maintain this project without any financial or funds support. If you think this is great and want to encourage me, donation would be so helpful!

In addition, if you have tons of perch mount images and videos are waited to be labeled, we could cooperate! [contact me!](https://chendada-8474.github.io/)

### PayPal
<!-- PayPal Logo -->
<tr><td align="center"></td></tr><tr><td align="center"><a href="https://paypal.me/tachihchen" title="tachihchen" onclick="javascript:window.open('https://www.paypal.com/tw/webapps/mpp/paypal-popup?locale.x=zh_TW','WIPaypal','toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=1060, height=700'); return false;"><img src="https://www.paypalobjects.com/webstatic/en_US/i/buttons/pp-acceptance-medium.png" alt="使用 PayPal 立即購" /></a></td></tr>
<!-- PayPal Logo -->
<br>
https://paypal.me/tachihchen