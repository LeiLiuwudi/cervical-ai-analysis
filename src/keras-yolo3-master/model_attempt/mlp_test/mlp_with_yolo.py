#%%
import sys
sys.path.append('../../')
#%%
from file import findAllFile
import colorsys
import math
import os

from sklearn.externals import joblib

from timeit import default_timer as timer
from mse import mse

import numpy as np
from cv2 import cv2
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from five_classification import train

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model


class YOLO(object):
    _defaults = {
        "model_path": '/ext/tianpeng/Lab205/cervical-ai-analysis/src/keras-yolo3-master/logs/trained_weights_final.h5',
        "anchors_path": '/ext/tianpeng/Lab205/cervical-ai-analysis/src/keras-yolo3-master/model_data/yolo_anchors.txt',
        "classes_path": '/ext/tianpeng/Lab205/cervical-ai-analysis/src/keras-yolo3-master/model_data/my_class.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        print(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image1):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image1, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image1.width - (image1.width % 32),
                              image1.height - (image1.height % 32))
            boxed_image = letterbox_image(image1, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image1.size[1], image1.size[0]],
                # K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='/ext/tianpeng/Lab205/cervical-ai-analysis/src/keras-yolo3-master/font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image1.size[1] + 0.5).astype('int32'))
        thickness = (image1.size[0] + image1.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image1)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image1.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image1.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print(end - start)
        return image1, out_boxes[0]

    def close_session(self):
        self.sess.close()

#%%
yolo = YOLO()
# base = 'E:\\graduationProject\\VGG16_TF-master\\data\\dataset\\all'
base = '/ext/tianpeng/Lab205/cervical-ai-analysis/src/keras-yolo3-master/dataset/all'
allFile = findAllFile(base)
X = []
count = [0, 0, 0, 0, 0]
for each in allFile:
    complete_path = base + '/' + each
    image = Image.open(complete_path)
    uncroped_image = cv2.imread(complete_path)
    try:
        r_image, box = yolo.detect_image(image)
    except:
        print("continue")
        continue
    
    count[int(each[0])] = count[int(each[0])] + 1
    top = box[0]
    left = box[1]
    bottom = box[2]
    right = box[3]

    top = top - 5
    left = left - 5
    bottom = bottom + 5
    right = right + 5

    # 左上角点的坐标
    top = int(max(0, np.floor(top + 0.5).astype('int32')))

    left = int(max(0, np.floor(left + 0.5).astype('int32')))
    # 右下角点的坐标
    bottom = int(min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32')))
    right = int(min(np.shape(image)[1], np.floor(right + 0.5).astype('int32')))
    croped_region = uncroped_image[top:bottom, left:right]  # 先高后宽
    grey_image = cv2.cvtColor(croped_region, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([grey_image], [0], None, [256], [0, 256])
    seq = []
    for e in hist:
        seq.append(e[0])
    # print(mse(seq, 2, 0.15, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
    X.append(mse(seq, 2, 0.15, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))

X = np.array(X)
row = X.shape[0]
col = X.shape[1]
for i in range(row):
    for j in range(col):
        if math.isinf(X[i][j]) or math.isnan(X[i][j]):
            X[i][j] = 0.333

Y = []
for i in range(count[0]):
    Y.append(0)
for i in range(count[1]):
    Y.append(1)
for i in range(count[2]):
    Y.append(2)
for i in range(count[3]):
    Y.append(3)
for i in range(count[4]):
    Y.append(4)

#%%
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


# %%
print(X)
# y = label_binarize(y, classes=[0, 1, 2, 3, 4])
print(Y)
# 设置种类
# n_classes = y.shape[1]
# print(n_classes)
# 训练模型并预测
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

# 随机化数据，并划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
model = svm.LinearSVC()
clt = model.fit(X_train, y_train)
joblib.dump(model, '/ext/tianpeng/Lab205/cervical-ai-analysis/src/keras-yolo3-master/model_attempt/mlp_test/model_trained/svm.pkl')
print("train accuracy:", model.score(X_test, y_test))

# %%
def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

print(to_categorical([1,2,3,4,0], 5))
y_train_onehot = to_categorical(y_train, num_classes = 5)
y_test_onehot = to_categorical(y_test, num_classes = 5)

#%%
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
mlp = Sequential()
mlp.add(Dense(128, input_dim=16, activation='relu'))
# mlp.add(Dropout(0.5))
mlp.add(Dense(128, activation='relu'))
# mlp.add(Dropout(0.5))
mlp.add(Dense(5, activation='sigmoid'))

mlp.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
mlp.fit(X_train, y_train_onehot,
          epochs=50,
          batch_size=32)
score = mlp.evaluate(X_test, y_test_onehot, batch_size=32)
print("test_acc: " + str(score[1]))


# %%
mlp.save('/ext/tianpeng/Lab205/cervical-ai-analysis/src/keras-yolo3-master/model_attempt/mlp_test/model_trained/mlp.h5')
# %%
