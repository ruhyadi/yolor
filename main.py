import json
import base64
from PIL import Image
import io
from utils.cvat import inference, load_model, preprocessing

def init_context(context):
    context.logger.info("Init context...  0%")
    
    context.logger.info("Load model")
    model = load_model('yolor_p6')
    context.logger.info("Success downloaded model... ")

    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run YOLO-R model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    
    # preprocessing image
    img, img0 = preprocessing(img0=image, img_size=1280, auto_size=64)
    
    # inference
    names_path = './data/coco.names'
    results = inference(context, img, img0, names_path, threshold)

    encoded_results = []
    for key, value in results.items():
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

    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)
