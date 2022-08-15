import os
# from keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence
import numpy as np
import tensorflow as tf 
from PIL import Image


classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']


# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]


def read_images(root="voc_root", is_train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label


cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引


def image2label(img):
    data = np.array(img, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64') # 根据索引得到 label 矩阵


def voc_rand_crop(feature, label, height, width):
    """
    Random crop feature (PIL image) and label (PIL image).
    先将channel合并,剪裁之后再分开
    """
    combined = tf.concat([feature, label], axis=2)
    #print(combined.shape)
    
    last_label_dim = tf.shape(label)[-1]
    last_feature_dim = tf.shape(feature)[-1]
    
    combined_crop = tf.image.random_crop(combined,
                                         size=tf.concat([(height, width), [last_label_dim + last_feature_dim]], 
                                         axis=0))
    return combined_crop[:, :, :last_feature_dim], combined_crop[:, :, last_feature_dim:]


class VOCSegDataset(Sequence):
    def __init__(self, train, crop_size, voc_root, batch_size):
        """
        crop_size: (h, w)
        """
        self.crop_size = crop_size  # (h, w)
        self.batch_size = batch_size
        images, labels = read_images(root=voc_root, is_train=train)
        self.images = self.filter(images)  # images list
        self.labels = self.filter(labels)  # labels list
        print('Read ' + str(len(self.images)) + ' valid examples')

    def filter(self, imgs):  # 过滤掉尺寸小于crop_size的图片
        return [img for img in imgs if (
                Image.open(img).size[1] >= self.crop_size[0] and
                Image.open(img).size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        image = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        label = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        jpgs, pngs = [None] * self.batch_size, [None] * self.batch_size

        for i,(img,png) in enumerate(zip(image,label)):
            jpgs[i] = Image.open(img)  # PIL images (w,h,c)
            pngs[i] = Image.open(png).convert('RGB')  # PIL images 
            jpgs[i], pngs[i] = voc_rand_crop(jpgs[i], pngs[i],self.crop_size[0],self.crop_size[1])
            jpgs[i] = tf.divide(jpgs[i],255)  # 归一化处理
            pngs[i] = image2label(pngs[i])
        return tf.cast(jpgs,dtype=tf.float32), tf.cast(pngs,dtype=tf.uint8)  # float32 tensor, uint8 tensor

    def __len__(self):
        return len(self.images)
