import os
import cv2
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob
from wolta.visual_tools import get_extensions, dataset_size_same, dataset_ratio_same, dir_split
from concurrent.futures import ThreadPoolExecutor

p_paths = [
    "/Users/nayanadorairaj/Desktop/Machine Learning/project/New Plant Diseases Dataset(Augmented)/valid",
    "/Users/nayanadorairaj/Desktop/Machine Learning/project/New Plant Diseases Dataset(Augmented)/train"
]

d_paths = [d for p in p_paths for d in glob(f'{p}/*')]
print("Classes found:", len(d_paths))

i_paths = [i for d in d_paths for i in glob(f'{d}/*')]
print("Total images:", len(i_paths))

get_extensions(i_paths)
dataset_size_same(i_paths)
dataset_ratio_same(i_paths)

temp_img = cv2.imread(i_paths[0])
h, w = temp_img.shape[:2]
print(f' Image resolution — width: {w}, height: {h}, ratio: {w/h:.2f}')

raw_out = r'~path/working/raw'
if os.path.exists(raw_out):
    shutil.rmtree(raw_out)
os.makedirs(raw_out, exist_ok=True)

def preprocess_image(i_path, out_path, size=(128, 128)):
    try:
        img = cv2.imread(i_path)
        if img is None:
            return False
        resized = cv2.resize(img, size)
        cv2.imwrite(out_path, resized)
        return True
    except Exception as e:
        print(f"Error processing {i_path}: {e}")
        return False

tasks = []
for d_path in d_paths:
    label = Path(d_path).name
    dest_dir = os.path.join(raw_out, label)
    os.makedirs(dest_dir, exist_ok=True)
    for i_path in glob(f'{d_path}/*'):
        out_path = os.path.join(dest_dir, Path(i_path).name)
        tasks.append((i_path, out_path))

with ThreadPoolExecutor(max_workers=8) as executor: 
    results = list(tqdm(
        executor.map(lambda args: preprocess_image(*args), tasks),
        total=len(tasks),
        desc="Preprocessing",
        unit="img"
    ))

split_out = r'~path/working/data'
if not all(os.path.exists(os.path.join(split_out, sub)) and os.listdir(os.path.join(split_out, sub))
           for sub in ['train', 'val', 'test']):
    dir_split(raw_out, split_out, test_size=0.2, val_size=0.2)
    print("Dataset split completed.")
else:
    print("Using existing split.")
