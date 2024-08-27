import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/lizhihao/code/ultralytics-A100/ultralytics-A100/runs/train/exp35/weights/best.pt')
    model.val(data='/home/lizhihao/code/ultralytics-main/dataset/chayev11/data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )