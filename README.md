# YOLO-R for Nuclio/CVAT

## Code Structure
```bash
├── serverless
│   ├── your-model-folder
│   │   ├── yolor
│   │   │   └── nuclio
│   │   │       ├── **yolor-file    # clone from github
│   │   │       ├── function.yaml   # define name, docker image, etc. for model
│   │   │       ├── main.py         # define hook for inference model
│   │   │       ├── utils           # utils folder from yolor github repository
│   │   │       │   ├── cvat.py     # utils to run main.py
│   │   │       └── yolor_p6.pt     # model weight (optional, but recommended)
```
## Dockerize
```bash
docker build -t username/yolor:cvat-1.0
```
## Development
Crete conda environment
```bash
cd yolor/nuclio
conda create -n yolor python=3.8 numpy
```
Install `requirements.txt`
```bash
cd yolor/nuclio
conda activate yolor
pip install -r requirements.txt
```
