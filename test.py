from nets.deeplab import Deeplabv3
from utils.dataloader import VOCSegDataset
import tensorflow as tf 


train = True
crop_size = (320, 480)
voc_root = './dataset/VOCdevkit/VOC2012'
batch_size = 32
voc_train = VOCSegDataset(train,crop_size,voc_root,batch_size)

# for x, y in iter(voc_train):
#     print(x.dtype, x.shape)
#     print(y.dtype, y.shape)
#     break


input_shape = (320, 480, 3)
num_classes = 21
    
net = Deeplabv3(input_shape,num_classes)


for x, y in iter(voc_train):
    print(x.shape)
    pred = net.predict(x)
    pred = tf.convert_to_tensor(pred)
    print(x.shape, y.shape, y[0, 170, 240])
    print(net(x).shape)
    break