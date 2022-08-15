import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from nets.deeplab import Deeplabv3 


input_shape = (2048, 4000, 3)
num_classes = 6
model = Deeplabv3(classes=num_classes,input_shape=input_shape)
model.load_weights("weight/last1.h5")

# RGB color for each class
VOC_COLORMAP = [[0,0,0],[0,0,128],[0,128,0],[128,0,0], [128,0,128], [128,0,0]]

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
fname="DJI_0001"
image = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert("RGB")
label = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert("RGB")

label = tf.image.resize_with_crop_or_pad(label,2048, 4000)
image = tf.image.resize_with_crop_or_pad(image,2048, 4000)

pred = predict(image)
pred = label2image(pred)


plt.subplot(1, 3, 1)
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.imshow(label)
plt.subplot(1, 3, 3)
plt.imshow(pred)
plt.show()