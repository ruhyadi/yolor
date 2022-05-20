"""
Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = torch.hub.load('ultralytics/yolov5:master', 'custom', 'path/to/yolov5s.onnx')  # file from branch
"""

import os
from pathlib import Path
import subprocess
import torch
from models.models import Darknet

from utils.torch_utils import select_device
from utils.cvat import preprocessing

def attempt_download(file, repo='ruhyadi/yolov5n'):  # from utils.downloads import *; attempt_download()
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                print(f'Found {url} locally at {file}')  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
            assets = [x['name'] for x in response['assets']]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            tag = response['tag_name']  # i.e. 'v1.0'
        except Exception:  # fallback plan
            assets = [
                'yolov5_nodeflux.pt','yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov5n6.pt', 'yolov5s6.pt',
                'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except Exception:
                tag = 'cvat'  # current release

        if name in assets:
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')

    return str(file)

def _create(name, tag=None, imgsize=640,  device='cpu'):
    dev = select_device(device)

    cwd = Path(os.getcwd())
    if tag is not None:
        tag = subprocess.check_output(
            'git tag', 
            shell=True, 
            stderr=subprocess.STDOUT).decode().split()[-1]

    url=f'https://github.com/ruhyadi/yolor/releases/download/{tag}/{name}.pt',

    model = Darknet(f'./cfg/{name}.cfg', img_size=imgsize)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(url, map_location=device)['model'])
    
    return model.to(device).eval()

def yolor_nodeflux(imgsize=640, device='cpu'):
    # nodeflux yolor model
    return _create('yolor_nodeflux', imgsize=imgsize, device=device)

def yolor_p6(imgsize=640, device='cpu'):
    # nodeflux yolor model
    return _create('yolor_p6', imgsize=imgsize, device=device)

if __name__ == '__main__':

    # model = torch.hub.load('ruhyadi/yolov5n:cvat-v1.2', 'yolov5_nodeflux')
    model = torch.hub.load('ruhyadi/yolor:v1.3-cvat', 'yolor_p6')

    print(model)

    # imgs = ['data/images/bus.jpg']

    # results = model(imgs, size=640)
    # json = results.pandas().xyxy[0].to_dict('records')
    # print(json)