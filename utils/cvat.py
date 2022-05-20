import sys
import os
sys.path.append(os.getcwd())

import torch
import cv2
import numpy as np
from PIL import Image
from models.models import Darknet
from utils.datasets import LoadImages, letterbox
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from detect import load_classes

def load_model(weights: str):
    # define device
    device = select_device('cpu') # TODO: change to gpu 
    
    # model = Darknet(cfg, 1280).cuda()
    model = Darknet(f'./cfg/{weights}.cfg', 1280)
    model.load_state_dict(torch.load(f'{weights}.pt', map_location=device)['model'])
    model.to(device).eval()
    
    return model

def preprocessing(img0, img_size=640, auto_size=64):
    img0_arr = np.array(img0)
    img = letterbox(img0_arr, new_shape=img_size, auto_size=auto_size)[0]
    
    # Convert to RGB 
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    # Convert to Tensor
    img = torch.from_numpy(img).to('cpu')
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    return img, img0_arr

def inference(context, img, img0, names_path, conf_thres=0.4, iou_thres=0.5):
    # preds dict
    RESULTS = { 0: {'confidence': None, 'class': None, 'xmin': None, 'ymin': None, 'xmax': None,'ymax': None} }
    # class names
    names = load_classes(names_path)
    
    # inference
    pred = context.user_data.model(img)[0]
    # nms
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # Process detections
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            for i, (*xyxy, conf, cls) in enumerate(det):
                xyxy = np.squeeze((torch.tensor(xyxy).view(1, 4)).tolist())
                RESULTS[i] = {}
                RESULTS[i]['confidence'] = round(conf.item(), 2)
                RESULTS[i]['class'] = names[int(cls)]
                RESULTS[i]['xmin'] = xyxy[0]
                RESULTS[i]['ymin'] = xyxy[1]
                RESULTS[i]['xmax'] = xyxy[2]
                RESULTS[i]['ymax'] = xyxy[3]
                                
    return RESULTS

def _inference(model, img, img0, names_path, conf_thres=0.4, iou_thres=0.5):
    """
    for testing only
    """
    # preds dict
    RESULTS = { 0: {'confidence': [], 'class': [], 'xmin': [], 'ymin': [], 'xmax': [],'ymax': []} }
    # class names
    names = load_classes(names_path)

    # inference
    pred = model(img)[0]
    # nms
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # Process detections
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            for i, (*xyxy, conf, cls) in enumerate(det):
                xyxy = np.squeeze((torch.tensor(xyxy).view(1, 4)).tolist())
                RESULTS[i] = {}
                RESULTS[i]['confidence'] = round(conf.item(), 2)
                RESULTS[i]['class'] = names[int(cls)]
                RESULTS[i]['xmin'] = xyxy[0]
                RESULTS[i]['ymin'] = xyxy[1]
                RESULTS[i]['xmax'] = xyxy[2]
                RESULTS[i]['ymax'] = xyxy[3]
                                
    return RESULTS

if __name__ == '__main__':
    
    img_size = 1280
    auto_size = 64
    names_path = './data/coco.names'
    
    img0 = Image.open('./inference/images/horses.jpg')
    
    # preprocessing image
    img, img0 = preprocessing(img0)

    # inference
    model = load_model('yolor_p6')
    pred = _inference(model, img, img0, names_path)

    encoded_results = []
    for key, value in pred.items():
        encoded_results.append({
            'confidence': value['confidence'],
            'label': value['class'],
            'points': [
                value['xmin'],
                value['ymin'],
                value['xmax'],
                value['ymax']
            ],
            'type': 'rectangle'
        })
    print(encoded_results)