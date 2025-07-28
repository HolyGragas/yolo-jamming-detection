import os
import yaml
import shutil
import zipfile
import subprocess

# --- SETTINGS ---
CUSTOM_DATA_DIR = "custom_data"
TRAIN_PCT = 0.9
CLASSES_TXT = os.path.join(CUSTOM_DATA_DIR, "classes.txt")
DATA_YAML = "data.yaml"
MODEL_NAME = "yolov8n.pt"  # change if needed
EPOCHS = 60
IMG_SIZE = 640

# --- 1. Split train/val ---
if not os.path.exists("train_val_split.py"):
    print("Downloading train_val_split.py...")
    subprocess.run(["wget", "-O", "train_val_split.py",
                    "https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/main/utils/train_val_split.py"])

print("Splitting train and validation sets...")
subprocess.run(["python", "train_val_split.py", f"--datapath={CUSTOM_DATA_DIR}", f"--train_pct={TRAIN_PCT}"])

# --- 2. Create data.yaml ---
def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    data = {
        'path': CUSTOM_DATA_DIR,
        'train': 'train/images',
        'val': 'validation/images',
        'nc': len(classes),
        'names': classes
    }
    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"✅ data.yaml created at {path_to_data_yaml}")

create_data_yaml(CLASSES_TXT, DATA_YAML)

# --- 3. Train ---
print("Starting training...")
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")  # or your custom .pt file

# Train
model.train(data="data.yaml", epochs=60, imgsz=640)
subprocess.run([
    "yolo", "detect", "train",
    f"data={DATA_YAML}",
    f"model={MODEL_NAME}",
    f"epochs={EPOCHS}",
    f"imgsz={IMG_SIZE}"
])

# --- 4. Predict (optional) ---
print("Running prediction on validation set...")
subprocess.run([
    "yolo", "detect", "predict",
    "model=runs/detect/train/weights/best.pt",
    f"source={CUSTOM_DATA_DIR}/validation/images",
    "save=True"
])

# --- 5. Zip model folder ---
print("Zipping model folder...")
os.makedirs("my_model", exist_ok=True)
shutil.copy("runs/detect/train/weights/best.pt", "my_model/my_model.pt")
shutil.copytree("runs/detect/train", "my_model/train", dirs_exist_ok=True)
shutil.make_archive("my_model", 'zip', "my_model")
print("✅ Training complete. Model saved as my_model.zip.")