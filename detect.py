import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('your-model-path')
    model.predict(source='your-image-path',
                  conf=0.25,
                  project='your-project-path',
                  name='project-name',
                  save=True,
                  # visualize=True 
                  # line_width=2, 
                  # show_conf=False, 
                  # show_labels=False, 
                  )
