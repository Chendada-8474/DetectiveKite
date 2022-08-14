from cmath import isnan
import csv
from PyQt5.QtWidgets import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import sys
import os

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']

with open("./bird_names.yaml", "r") as birds:
    bird_names = yaml.load(birds, Loader=yaml.CLoader)["bird_name"]


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("review.ui", self)

        # csv and dir path
        self.pred = pd.DataFrame({})
        self.dir_path = ""
        self.csv_path = ""

        # Tool box
        self.open_dir = self.findChild(QAction, "actionOpen_Dir")
        self.open_csv = self.findChild(QAction, "actionOpen_csv")
        self.save = self.findChild(QAction, "actionSave")
        self.reload = self.findChild(QAction, "actionReload")
        self.open_dir.triggered.connect(self.opendir)
        self.open_csv.triggered.connect(self.opencsv)
        self.save.triggered.connect(self.savecsv)
        self.reload.triggered.connect(self.reloadmedia)

        self.meds = None

        # status
        self.progress = self.findChild(QLabel, "progressLabel")
        self.med_path_label = self.findChild(QLabel, "pathLabel")
        self.status_label = self.findChild(QLabel, "statusLabel")
        self.progress.setStyleSheet("font-size: 16px;")
        self.med_path_label.setStyleSheet("font-size: 16px;")
        self.status_label.setStyleSheet("font-size: 16px; color: blue")

        # next and previous
        self.media_index = 0

        self.next_media = self.findChild(QPushButton, "nextButton")
        self.per_media = self.findChild(QPushButton, "preButton")
        self.next_media.clicked.connect(self.nextmedia)
        self.per_media.clicked.connect(self.previmedia)

        # video and image
        self.stack = self.findChild(QStackedWidget, "stackedWidget")
        self.image_label = self.findChild(QLabel, "imageLabel")
        self.video = self.findChild(QVideoWidget, "videoWidget")
        self.play = self.findChild(QPushButton, "playButton")
        self.pause = self.findChild(QPushButton, "pauseButton")
        self.slider = self.findChild(QSlider, "horizontalSlider")
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.setPosition)
        self.play.clicked.connect(self.playvideo)
        self.pause.clicked.connect(self.pausevideo)

        self.ori_x = 0
        self.ori_y = 0
        self.ori_w = 0
        self.ori_h = 0

        self.bbox = [[0,0,60,60,"",""]]

        self.bx_canwidget = self.findChild(QWidget, "centralwidget")
        self.bx_frame = self.findChild(QFrame, "frame")
        self.bx_stack = self.findChild(QStackedWidget, "stackedWidget")
        self.bx_page1 = self.findChild(QWidget, "page")

        self.paint_label = QWidget(self)
        self.paint_label.setGeometry(0, 0, 0, 0)
        self.paint_label.installEventFilter(self)

        # edit prediction
        self.editLayout = self.findChild(QGridLayout, "gridLayout")
        self.addlineButton = self.findChild(QPushButton, "addlineButton")
        self.comform_button = self.findChild(QPushButton, "confirmButton")
        self.completer = QCompleter(bird_names)
        self.addlineButton.clicked.connect(self.add_edit_row)
        self.comform_button.clicked.connect(self.confirm_ans)

        self.med_preds = pd.DataFrame({})
        self.row_index = 1

        # save csv
        self.saved = True
        self.save_st = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_st.activated.connect(self.savecsv)

        self.show()

    def broadcast_status(self, content: str, color = "red"):
        self.status_label.setStyleSheet("font-size: 16px; color: %s" % color)
        self.status_label.setText(content)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Paint:
            painter = QPainter()
            painter.begin(obj)
            if obj == self.paint_label:
                painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))  # Some specific painting
                painter.setFont(QFont('Arial', 20, QFont.Bold))
            for b in self.bbox:
                painter.drawRect(b[0], b[1], b[2], b[3])
                ty = b[1] - 10 if b[1] > 50 else b[1] + 50
                painter.drawText(QPoint(b[0], ty), str(b[6]) + ". " + b[4] + " " + str(b[5]))
            painter.end()
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, e):
        if e.key() == 68:
            self.nextmedia()
        elif e.key() == 65:
            self.previmedia()
        elif e.key() == 16777268:
            self.reloadmedia()
        elif e.key() == 67:
            self.confirm_ans()
        elif e.key() == 76:
            self.add_edit_row("_")
        elif e.key() in list(range(49, 58)):
            item_index = (e.key()-48)*4 + 1
            item = self.editLayout.itemAt(item_index)
            if item:
                item = item.widget()
                item.setFocus(True)


    def delete_edit_row(self):
        index = self.editLayout.indexOf(self.sender())
        for i in range(0,4):
            item = self.editLayout.itemAt(index-i).widget()
            item.deleteLater()

    def add_edit_row(self, _, pred_index = "", sp_name = None, num_inds = 0):
        index_label = QLabel(str(pred_index))
        line_edit = QLineEdit(self)
        spin_box = QSpinBox(self)
        delete_button = QPushButton("Delete", self)
        self.editLayout.addWidget(index_label, self.row_index, 0)
        self.editLayout.addWidget(line_edit, self.row_index, 1)
        self.editLayout.addWidget(spin_box, self.row_index, 2)
        self.editLayout.addWidget(delete_button, self.row_index, 3)
        line_edit.setText(sp_name)
        line_edit.setCompleter(self.completer)
        spin_box.setValue(num_inds)
        delete_button.clicked.connect(self.delete_edit_row)
        self.row_index += 1

    def closeEvent(self, event):
        if self.saved:
            event.accept()
        else:
            reply = QMessageBox.information(self, 'System alert', "File has not saved.\nOK: stay in program\nClose: End program",
                QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            if reply == QMessageBox.Close:
                event.accept()
            else:
                event.ignore()

    def confirm_ans(self):

        if self.med_preds.empty:
            return

        indexes = self.med_preds.index.to_list()

        row_indexes = []
        sp_names = []
        nums = []

        for i in range(1, self.row_index):

            index = self.editLayout.itemAtPosition(i, 0)
            sp_name = self.editLayout.itemAtPosition(i, 1)
            num = self.editLayout.itemAtPosition(i, 2)

            if index and sp_name and num:

                index = index.widget().text()
                sp_name = sp_name.widget().text()
                num = num.widget().text()

                if sp_name not in bird_names:
                    self.broadcast_status("Wrong common name: index %s" % index)
                    return
                elif int(num) == 0:
                    self.broadcast_status("Number of indviduals can't be 0: index %s" % index)
                    return

                row_indexes.append(index)
                sp_names.append(sp_name)
                nums.append(int(num))

        false_nega = pd.isnull(self.med_preds["name"].values[0]) and len(row_indexes) > 0
        false_posi = not self.med_preds.empty and len(row_indexes) == 0

        if false_nega:
            for i, s, n in zip(row_indexes, sp_names, nums):
                    file_name = self.meds[self.media_index]
                    date_time = self.med_preds["datetime"].values[0]
                    media_type = self.med_preds["media"].values[0]
                    model_ci = self.med_preds["model"].values[0]

                    new_pred = pd.DataFrame(
                        np.array([[file_name, None, None, n, None, date_time, media_type, model_ci, None, None, None, None, s, True]]),
                        columns = self.med_preds.columns.tolist()
                    )

                    biggest_index = self.pred.index.to_list()[-1] + 1
                    new_pred.index = [biggest_index]
                    self.pred = pd.concat([self.pred, new_pred])

        elif false_posi:
            for i, mi in enumerate(indexes):
                if i == 0:
                    self.pred.at[int(mi), "reviewed_name"] = None
                    self.pred.at[int(mi), "num_inds"] = 0
                    self.pred.at[int(mi), "reviewed"] = True
                    indexes.remove(int(mi))

        else:
            for i, s, n in zip(row_indexes, sp_names, nums):
                if i == "":
                    file_name = self.meds[self.media_index]
                    date_time = self.med_preds["datetime"].values[0]
                    media_type = self.med_preds["media"].values[0]
                    model_ci = self.med_preds["model"].values[0]

                    new_pred = pd.DataFrame(
                        np.array([[file_name, None, None, n, None, date_time, media_type, model_ci, None, None, None, None, s, True]]),
                        columns = self.med_preds.columns.tolist()
                    )

                    biggest_index = self.pred.index.to_list()[-1] + 1
                    new_pred.index = [biggest_index]
                    self.pred = pd.concat([self.pred, new_pred])

                else:
                    self.pred.at[int(i), "reviewed_name"] = s
                    self.pred.at[int(i), "num_inds"] = int(n)
                    self.pred.at[int(i), "reviewed"] = True
                    indexes.remove(int(i))

        self.pred = self.pred.drop(indexes, axis=0)

        self.saved = False
        self.broadcast_status("Confirmed! ", "green")
        self.nextmedia()

    def show_media(self, dir_path, media_index):
        if not self.pred.empty and self.dir_path != "":
            fat = self.filetype(self.meds[media_index])
            med_path = os.path.join(dir_path, self.meds[media_index])

            self.progress.setText(str(media_index + 1) + " / " + str(len(self.meds)))
            self.med_path_label.setText(med_path)


            self.med_preds = self.pred[self.pred["file_name"] == self.meds[media_index]]

            reviewed = self.med_preds["reviewed"].values
            if len(reviewed) > 0:
                if reviewed[0]:
                    self.broadcast_status("Confirmed! ", "green")
                else:
                    self.broadcast_status("Un confirmed", "blue")
            else:
                self.broadcast_status("Media information not found", "red")

            # edit species part
            rows = self.editLayout.rowCount()
            cols = self.editLayout.columnCount()
            num_items = rows*cols

            if num_items > cols:
                for i in range(cols, num_items):
                    item = self.editLayout.itemAt(i)
                    if item:
                        item.widget().deleteLater()

            if len(self.med_preds) > 0:
                self.row_index = 1
                for i, row in self.med_preds.iterrows():
                    if not pd.isna(row["reviewed_name"]):
                        self.add_edit_row("_", i, sp_name = row["reviewed_name"], num_inds = row["num_inds"])

            # geometry for bounding box
            cenx, ceny = self.bx_canwidget.x(), self.bx_canwidget.y()
            fx, fy = self.bx_frame.x(), self.bx_frame.y()
            sx, sy = self.bx_stack.x(), self.bx_stack.y()
            p1x, p1y = self.bx_page1.x(), self.bx_page1.y()


            # show image or video
            if fat in img_formats:
                self.stack.setCurrentIndex(0)
                self.pixmap = QPixmap(med_path).scaled(self.image_label.size(), Qt.KeepAspectRatio)
                self.image_label.setPixmap(self.pixmap)

                # caculate for bounding box
                lx, ly, lw, lh = self.image_label.x(), self.image_label.y(), self.image_label.width(), self.image_label.height()
                label_ratio = lh/lw

                img=plt.imread(med_path)
                ih, iw, _ = img.shape
                img_ratio = ih/iw

                parent_x = cenx + fx + sx + p1x + lx
                parent_y = ceny + fy + sy + p1y + ly

                # image ratio judgement
                if label_ratio > img_ratio:
                    self.ori_x = parent_x
                    self.ori_y = parent_y + lh/2 - round((lw*img_ratio)/2, 0)
                    pre_img_w = lw
                    pre_img_h = lw*img_ratio

                elif label_ratio < img_ratio:
                    self.ori_x = parent_x + lw/2 - round((lh/img_ratio)/2, 0)
                    self.ori_y = parent_y
                    pre_img_w = lh/img_ratio
                    pre_img_h = lh

                else:
                    self.ori_x = parent_x
                    self.ori_y = parent_y
                    pre_img_w = lw
                    pre_img_h = lh

                boxes = []

                for i, row in self.med_preds.iterrows():

                # draw bounding box
                    if pd.isnull(row["name"]) or pd.isnull(row["reviewed_name"]):
                        boxes.append([0, 0, 0, 0, "", "", ""])
                    else:
                        box_x = int(round((pre_img_w * (row["xmin"]/iw)), 0))
                        box_y = int(round((pre_img_h * (row["ymin"]/ih)), 0))
                        box_w = int(round((row["xmax"] - row["xmin"])/iw * pre_img_w, 0))
                        box_h = int(round((row["ymax"] - row["ymin"])/ih * pre_img_h, 0))
                        boxes.append([box_x, box_y, box_w, box_h, row["reviewed_name"], round(row["confidence"], 2), i])

                self.bbox = boxes
                self.paint_label.setGeometry(self.ori_x, self.ori_y, pre_img_w, pre_img_h)

            elif fat in vid_formats:
                self.paint_label.setGeometry(0, 0, 0, 0)
                self.bbox = [[0, 0, 0, 0, "", "", ""]]

                self.stack.setCurrentIndex(1)
                self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
                self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(med_path)))
                self.mediaPlayer.setVideoOutput(self.video)
                self.mediaPlayer.positionChanged.connect(self.positionChanged)
                self.mediaPlayer.durationChanged.connect(self.durationChanged)
                self.mediaPlayer.play()


    def filetype(self, filename:str):
        ft = filename.split(".")[-1].lower()
        return ft


    # Tool box
    def opendir(self):
        self.dir_path = QFileDialog.getExistingDirectory(self, "select a folder")

        if self.dir_path == "":
            return

        self.meds = [i for i in os.listdir(self.dir_path) if i.split(".")[-1].lower() in img_formats or i.split(".")[-1].lower() in vid_formats]

        self.broadcast_status("You selected: %s" % self.dir_path, "blue")

        if not self.pred.empty:
            for i, m in enumerate(self.meds):
                revieweds = self.pred[self.pred["file_name"] == m]["reviewed"].values
                if len(revieweds) > 0:
                    if not revieweds[0]:
                        self.media_index = i
                        break
        else:
            self.media_index = 0

        self.show_media(self.dir_path, self.media_index)

    def opencsv(self):
        csv_path = QFileDialog.getOpenFileName(self, "select the prediction csv by DetectKite", "", ".csv (*.csv)")
        self.csv_path = csv_path[0]

        if self.csv_path == "":
            return

        self.pred = pd.read_csv(self.csv_path)
        columns = self.pred.columns.to_list()

        if "reviewed_name" not in columns:

            self.pred["reviewed_name"] = self.pred["name"]
            self.pred["reviewed"] = False
            self.media_index = 0

        # find first un reviewed media
        elif self.meds:
            for i, m in enumerate(self.meds):
                bools = self.pred[self.pred["file_name"] == m]["reviewed"]
                if len(bools) > 0 and not bools.values[0]:
                    self.media_index = i
                    break

        else:
            self.media_index = 0


        self.broadcast_status("You selected: %s" % self.csv_path, "blue")
        self.show_media(self.dir_path, self.media_index)

    def savecsv(self):
        if not self.saved:
            self.pred.to_csv(self.csv_path, index=False)
            self.saved = True
            self.broadcast_status("Result has been saved to %s" % self.csv_path, "green")

    def reloadmedia(self):
        self.show_media(self.dir_path, self.media_index)

    # next and pre media
    def nextmedia(self):
        if self.meds and len(self.meds) - 1 > self.media_index:
            if self.filetype(self.meds[self.media_index]) in vid_formats:
                self.mediaPlayer.stop()

            self.media_index += 1
            self.show_media(self.dir_path, self.media_index)

    def previmedia(self):
        if self.meds and self.media_index > 0:
            if self.filetype(self.meds[self.media_index]) in vid_formats:
                self.mediaPlayer.stop()
            self.media_index -= 1
            self.show_media(self.dir_path, self.media_index)

    def playvideo(self):
        if self.meds and self.filetype(self.meds[self.media_index]) in vid_formats:
            self.mediaPlayer.play()

    def pausevideo(self):
        if self.meds and self.filetype(self.meds[self.media_index]) in vid_formats:
            self.mediaPlayer.pause()
    def positionChanged(self, position):
        if self.meds and self.filetype(self.meds[self.media_index]) in vid_formats:
            self.slider.setValue(position)

    def durationChanged(self, duration):
        if self.meds and self.filetype(self.meds[self.media_index]) in vid_formats:
            self.slider.setRange(0, duration)

    def setPosition(self, position):
        if self.meds and self.filetype(self.meds[self.media_index]) in vid_formats:
            self.mediaPlayer.setPosition(position)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = MainWindow()
    app.exec_()