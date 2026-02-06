import warnings
warnings.filterwarnings('ignore')
import os
from ultralytics import RTDETR

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = 'your-model-path/best.pt'
    model = RTDETR(model_path)
    result = model.val(data='your-dataset.yaml',
                      split='test',
                      imgsz=640,
                      batch=4,
                      project='your-project-path',
                      name='project-name',
                      )
