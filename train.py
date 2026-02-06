import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('your-yaml-path')
    model.train(data='your-dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                patience=0,
                project='your-project-path',
                name='project-name',
                )
