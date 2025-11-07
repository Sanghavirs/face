import cv2
import os
import numpy as np

DATASET = "dataset"
MODEL = "model_lbph.yml"
LABELS_FILE = "labels.txt"

def train():
    faces = []
    labels = []
    label_dict = {}
    current_id = 0

    for folder in os.listdir(DATASET):
        if "_" not in folder:
            continue
        person_id, person_name = folder.split("_", 1)
        person_id = int(person_id)
        folder_path = os.path.join(DATASET, folder)

        label_dict[person_id] = person_name

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(person_id)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL)

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        for _id, name in label_dict.items():
            f.write(f"{_id},{name}\n")

    print("✅ Training complete. Model and labels saved.")

if __name__ == "__main__":
    train()
