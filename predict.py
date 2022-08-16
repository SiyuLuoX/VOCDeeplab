import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from nets.deeplab import Deeplabv3 
from utils.dataloader import colormap

input_shape = (320, 480, 3) # 改
num_classes = 21  # 改
model = Deeplabv3(classes=num_classes,input_shape=input_shape)
model.load_weights("weight/last.h5")


# RGB color for each class
VOC_COLORMAP = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

def predict(img):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    feature = tf.cast(img, tf.float32)
    feature = tf.divide(feature, 255.)
    # feature = tf.divide(tf.subtract(feature, rgb_mean), rgb_std)
    x = tf.expand_dims(feature, axis=0)
    return model.predict(x)

def label2image(pred):
    colormap = np.array(VOC_COLORMAP, dtype='float32')
    x = colormap[tf.argmax(pred, axis=-1)]
    # print(pred[0, 160, 200])
    return x


root="dataset/MyDataset"
fname="2007_000175"
image = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert("RGB")
label = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert("RGB")
image = np.array(image)
label = np.array(label)

label = tf.image.resize_with_crop_or_pad(label,input_shape[0], input_shape[1])
image = tf.image.resize_with_crop_or_pad(image,input_shape[0], input_shape[1])

pred = predict(image)
pred = label2image(pred)


plt.subplot(1, 3, 1)
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.imshow(label)
plt.subplot(1, 3, 3)
plt.imshow(pred)
plt.show()