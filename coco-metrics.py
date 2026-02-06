import argparse
import json
import os
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets

"""
  your need to prepare two JSON files:
   1. 'anno_json': annotation json path
   2. 'pred_json': prediction json path
"""

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default=r'your\anno_json.json', help='annotation json path')
    parser.add_argument('--pred_json', type=str, default=r'your\pred_json.json', help='prediction json path') 
    
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    
    print(f"Loading annotations: {anno_json}")
    anno = COCO(anno_json)
    
    print(f"Loading predictions: {pred_json}")
    with open(pred_json, 'r') as f:
        preds = json.load(f)
    
    valid_img_ids = set(anno.getImgIds())

    filtered_preds = []
    skipped_count = 0
    
    first_id = list(valid_img_ids)[0] if valid_img_ids else None
    is_int_id = isinstance(first_id, int)
    
    print("Filtering predictions...")
    for p in preds:
        img_id = p['image_id']
        
        if is_int_id and isinstance(img_id, str):
            try:
                img_id = int(img_id)
                p['image_id'] = img_id 
            except ValueError:
                pass
        
        if img_id in valid_img_ids:
            filtered_preds.append(p)
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Filtered out {skipped_count} predictions that had image IDs not present in the annotation file.")
    
    if not filtered_preds:
        print("\n❌ Error: No remaining predictions after filtering!")
        print("This means that the image IDs in your prediction file do not match any image IDs in the annotation file.")
        print("Please check the above debug information to confirm that you are using the correct annotation file (anno_json).")
        sys.exit(1)

    pred = anno.loadRes(filtered_preds)
    
    print("Running COCO evaluation...")
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    
    print("Running TIDE evaluation...")
    temp_pred_path = 'temp_filtered_predictions.json'
    with open(temp_pred_path, 'w') as f:
        json.dump(filtered_preds, f)
        
    try:
        tide = TIDE()
        tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(temp_pred_path), mode=TIDE.BOX)
        tide.summarize()
        tide.plot(out_dir='result')
        print("TIDE evaluation finished. Results saved to 'result' folder.")
    except Exception as e:
        print(f"An error occurred during TIDE evaluation: {e}")
    finally:
        if os.path.exists(temp_pred_path):
            os.remove(temp_pred_path)
