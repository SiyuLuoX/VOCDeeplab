import os
from PIL import Image
import numpy as np
from tensorflow.keras.callbacks import (EarlyStopping,ModelCheckpoint,
                                        ReduceLROnPlateau,TensorBoard)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from nets.deeplab import Deeplabv3
from utils.dataloader import image2label


def read_images(root="voc_root", train=True):
    '''返回图片路径'''
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label


def voc_rand_crop(feature, label, height, width):
    """
    Random crop feature (PIL image)(w,h) and label (PIL image).
    先将channel合并,剪裁之后再分开
    """
    combined = tf.concat([feature, label], axis=2)
    last_label_dim = tf.shape(label)[-1]
    last_feature_dim = tf.shape(feature)[-1]
    combined_crop = tf.image.random_crop(combined,
                        size=tf.concat([(height, width), [last_label_dim + last_feature_dim]],axis=0))
    return combined_crop[:, :, :last_feature_dim], combined_crop[:, :, last_feature_dim:]

def filter(images, crop_size):
    '''
    PIL (w,h)   crop_size = (320, 480)
    PIL (480,320)   crop_size = (320, 480)
    '''
    return [im for im in images if (Image.open(im).size[1] >= crop_size[0] and 
            Image.open(im).size[0] >= crop_size[1])]

def generate_arrays_from_file(images, labels, batch_size):
    n = len(images)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            image = Image.open(images[i]).convert("RGB")
            image = np.array(image)
            label = Image.open(labels[i]).convert("RGB")
            label = np.array(label)
            image , label = voc_rand_crop(image, label,crop_size[0],crop_size[1])
            label = image2label(label) 
            one_hot_label = np.eye(num_classes)[np.array(label, np.int32)]
            X_train.append(image/255)
            Y_train.append(one_hot_label)

            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

if __name__ == "__main__":
    input_shape = (320, 480, 3) # 改
    num_classes = 21  # 改
    weigth_dir = "weight/"
    model = Deeplabv3(classes=num_classes,input_shape=input_shape)
    
    voc_root = './dataset/MyDataset' # maybe改

    # 训练参数的设置
    checkpoint = ModelCheckpoint(weigth_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                            monitor='val_loss', save_weights_only=True, save_best_only=False, period=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 超参数设置
    lr = 1e-3
    batch_size = 4
    crop_size = (320, 480) #(h,w)

    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
            optimizer = Adam(learning_rate=lr),
            metrics = ['accuracy'])


    # 读取图片地址
    train_data,train_label = read_images(voc_root,True)
    val_data,val_label = read_images(voc_root,False)

    # 剔除小于裁剪窗口的数据
    train_data ,train_label = filter(train_data,crop_size),filter(train_label,crop_size)
    val_data ,val_label = filter(val_data,crop_size),filter(val_label,crop_size)

    num_val = int(len(train_data))
    num_train = int(len(val_data))
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))


    model.fit(generate_arrays_from_file(train_data, train_label, batch_size),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=generate_arrays_from_file(val_data, val_label, batch_size),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[checkpoint, reduce_lr,early_stopping])
    model.save_weights(weigth_dir+'last.h5')