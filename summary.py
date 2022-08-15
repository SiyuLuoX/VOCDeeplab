from nets.deeplab import Deeplabv3


if __name__ == "__main__":
    input_shape     = (512, 512,3)
    num_classes     = 21

    model = Deeplabv3(input_shape,num_classes)
    model.summary()