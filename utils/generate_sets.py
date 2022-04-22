import os
from glob import glob
import argparse

def generate(datapath:str, img_format: str, split:str, output: str):

    dump_txt = open(os.path.join('.', output), 'w')

    files = sorted(glob(os.path.join(datapath, f'*.{img_format}')))
    files = [os.path.join('.', file.split(split)[-1]) for file in files]

    for i, file in enumerate(files):
        dump_txt.write(file + '\n')
        
def main():
    parser = argparse.ArgumentParser(description="Generate image sets for training")
    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--img_format', type=str, default='jpg')
    parser.add_argument('--split', type=str, default='/')
    parser.add_argument('--output', type=str, default='sets.txt')
    cfg = parser.parse_args()
    
    # generate
    generate(cfg.datapath, cfg.img_format, cfg.split, cfg.output)