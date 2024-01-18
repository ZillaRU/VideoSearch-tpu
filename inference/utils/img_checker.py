import numpy as np
import cv2
import os


class ImgChecker():
    def __init__(self, threshold=0.96, dir_path=None):
        self.dir_path = dir_path
        self.threshold = threshold
        self.bad_count = 0

    def load_img(self, jpg_path):
        img = cv2.imread(jpg_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def check_folder(self):
        bad_file = []
        files = os.listdir(self.dir_path)
        for file in files:
            if file.endswith('jpg'):
                img = self.load_img(os.path.join(self.dir_path, file))
                img_pixel = img.reshape(-1).shape
                white_pixel = np.sum(img > 220)
                black_pixel = np.sum(img < 35)
                if white_pixel / img_pixel > self.threshold or black_pixel / img_pixel > self.threshold:
                    print('bad')
                    print(file)
                    self.bad_count += 1
                    bad_file.append(file)

        print(self.bad_count)
        return bad_file

    def check_img(self, img):
        # print(img)
        img_pixel = img.reshape(-1).shape
        white_pixel = np.sum(img > 220)
        black_pixel = np.sum(img < 35)
        if white_pixel / img_pixel > self.threshold or black_pixel / img_pixel > self.threshold:
            # print('False')
            return False
        else:
            # print('True')
            return True

    def img_statis(self, img_path):
        img = self.load_img(img_path)
        img_pixel = img.reshape(-1).shape
        white_pixel = np.sum(img > 220)
        black_pixel = np.sum(img < 35)
        print("while: " + str(white_pixel))
        print("black: " + str(black_pixel))
        white_percentage = white_pixel / img_pixel
        black_percentage = black_pixel / img_pixel
        print('white_percentage: ' + str(white_percentage))
        print('black_percentage: ' + str(black_percentage))


# a = JpgChecker('./scene_snapshot', 0.96)
# a.check()
# a = JpgChecker()
# a.jpg_statis('/home/zifeng/scp_mid/scene_snapshot/9c9587d1-c109-42cc-b7fb-b4411bb8473f.jpg')