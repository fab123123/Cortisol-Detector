# RUN AND PIP KAGGLEHUB TO DOWNLOAD DATASET TO .cache

import kagglehub

path = kagglehub.dataset_download("msambare/fer2013")

print("Path to dataset files:", path)