from nets.deeplab import Deeplabv3

if __name__ == "__main__":
    
    input_shape = (320, 480, 3) # 改
    num_classes = 21  # 改

    model = Deeplabv3(input_shape,num_classes)
    model.summary()