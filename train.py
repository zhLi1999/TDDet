
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    # model = YOLO('/home/lizhihao/code/ultralytics-A100/ultralytics-A100/ultralytics/cfg/models/v8/yolov8s-mobilenetv4.yaml')
    model = YOLO('D:/code/ultralytics-A100/ultralytics/cfg/models/v8/yolov8-mobilenetv4.yaml')
    model.load('yolov8s.pt') # loading pretrain weights
    # model.train(data='/home/lizhihao/code/ultralytics-A100/ultralytics-A100/dataset/chayev11/data.yaml',
    model.train(data='D:/code/ultralytics-A100/dataset/chayev11/data.yaml',
                device='0',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=10,
                workers=8,
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='/home/lizhihao/code/ultralytics-A100/ultralytics-A100/runs/train',
                name='exp',
                )