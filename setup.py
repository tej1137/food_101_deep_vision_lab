"""
Food-101 Deep Learning Project - Setup Script
Run this at the start of every RunPod session
"""

import subprocess
import sys
import os


# GIT CREDENTIALS
print("=" * 60)
print("STEP 1: Configuring Git")
print("=" * 60)

os.system("git config --global user.email 'thjssunil@gmail.com'")
os.system("git config --global user.name 'tej1137'")
os.system("git config --global credential.helper 'store --file /workspace/.git-credentials'")

print("Git configured!")

#INSTALLING DEPENDENCIES

print("=" * 60)
print("STEP 2: Installing dependencies")
print("=" * 60)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/workspace/food_101_deep_vision_lab/requirements.txt", "-q"
])

print("All dependencies installed!")


#CREATING FOLDER STRUCTURE

print()
print("=" * 60)
print("STEP 3: Checking folder structure")
print("=" * 60)

BASE_DIR    = "/workspace/food101-dl-project"
DATASET_DIR = f"{BASE_DIR}/dataset"
REPO_DIR    = "/workspace/food_101_deep_vision_lab"

folders = [
    f"{BASE_DIR}/dataset",
    f"{BASE_DIR}/checkpoints",
    f"{BASE_DIR}/outputs",
    f"{REPO_DIR}/notebooks",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(folder)

#VERIFY DATASET EXISTS

print("=" * 60)
print("STEP 4: Verifying our dataset")
print("=" * 60)

dataset_path = f"{DATASET_DIR}/food101/2.0.0"
dataset_info = f"{dataset_path}/dataset_info.json"

if os.path.exists(dataset_info):
    files       = os.listdir(dataset_path)
    train_files = [f for f in files if "train" in f and ".tfrecord" in f]
    val_files   = [f for f in files if "validation" in f and ".tfrecord" in f]
    result      = subprocess.run(["du", "-sh", dataset_path], capture_output=True, text=True)

    print(f"  Dataset found at {dataset_path}")
    print(f"  Train shards : {len(train_files)}")
    print(f"  Val shards   : {len(val_files)}")
    print(f"  Total size   : {result.stdout.split()[0]}")
else:
    print("  Dataset NOT found!")

#CHECKING GPU

print("=" * 60)
print("STEP 5: Checking GPU")
print("=" * 60)

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
