import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/code/ultralytics-A100/runs/train-yolov5/train2/exp2/weights/best.pt') # select your model.pt path
    model.predict(source='D:/code/ultralytics-A100/dataset/chayev11/val/images',
                  imgsz=640,
                  project='D:/code/ultralytics-A100/runs/detect-yolov5',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )