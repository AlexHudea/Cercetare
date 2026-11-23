import os
import glob
import json
import time
import csv
import psutil
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Mapare unificată DAWN & Cityscapes -> ID-uri COCO
LABEL_TO_COCO = {
    'person': 0, 'rider': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7
}

def get_memory_usage():
    """Returnează memoria RAM (MB) și VRAM GPU (MB) utilizate curent."""
    ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    return ram_mb, vram_mb

def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = YOLO('yolov8s.pt')
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    faster_rcnn_model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    faster_rcnn_model.eval()
    return yolo_model, faster_rcnn_model, device

def parse_dawn_xml(xml_path):
    """Citește etichetele PASCAL VOC (.xml) din DAWN."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        if name in LABEL_TO_COCO:
            bndbox = obj.find('bndbox')
            boxes.append([
                float(bndbox.find('xmin').text),
                float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text),
                float(bndbox.find('ymax').text)
            ])
            labels.append(LABEL_TO_COCO[name])
    if not boxes:
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

def parse_cityscapes_json(json_path):
    """Citește poligoanele Cityscapes (.json) și le convertește în bounding boxes."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    boxes, labels = [], []
    for obj in data.get('objects', []):
        label = obj.get('label', '').lower()
        if label in LABEL_TO_COCO:
            poly = np.array(obj['polygon'])
            if len(poly) > 0:
                xmin, ymin = np.min(poly, axis=0)
                xmax, ymax = np.max(poly, axis=0)
                if xmax > xmin and ymax > ymin:
                    boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                    labels.append(LABEL_TO_COCO[label])
    if not boxes:
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

def run_yolo(model, image):
    start_time = time.time()
    results = model(image, verbose=False)
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    boxes = results[0].boxes.xyxy.cpu()
    scores = results[0].boxes.conf.cpu()
    labels = results[0].boxes.cls.cpu().to(torch.int64)
    return dict(boxes=boxes, scores=scores, labels=labels), len(boxes), inference_time, fps

def run_faster_rcnn(model, image, device):
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    start_time = time.time()
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    mask = predictions['scores'] > 0.5
    boxes = predictions['boxes'][mask].cpu()
    scores = predictions['scores'][mask].cpu()
    labels = predictions['labels'][mask].cpu() - 1  # Aliniere COCO (eliminare clasa 0 background)
    return dict(boxes=boxes, scores=scores, labels=labels), len(boxes), inference_time, fps

def evaluate_dataset(dataset_name, dataset_type, img_dir, ann_dir, yolo_model, rcnn_model, device, limit=None):
    print(f"\n==================================================")
    print(f"RULARE BENCHMARK: {dataset_name}")
    print(f"==================================================")

    if dataset_type == "DAWN":
        images = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    else:
        images = glob.glob(os.path.join(img_dir, "**", "*_leftImg8bit.png"), recursive=True)

    if limit:
        images = images[:limit]

    if not images:
        print(f"[EROARE] Nu s-au găsit imagini în: {img_dir}")
        return None

    map_yolo = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    map_rcnn = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    y_times, y_fps_list, y_objs = [], [], []
    r_times, r_fps_list, r_objs = [], [], []
    count = 0

    for img_path in images:
        if dataset_type == "DAWN":
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            ann_path = os.path.join(ann_dir, base_name + ".xml")
            if not os.path.exists(ann_path):
                continue
            gt_boxes, gt_labels = parse_dawn_xml(ann_path)
        else:
            # Înlocuiește calea și sufixul pentru a găsi JSON-ul corespunzător din gtFine
            rel_path = os.path.relpath(img_path, img_dir)
            json_rel = rel_path.replace("_leftImg8bit.png", "_gtFine_polygons.json")
            ann_path = os.path.join(ann_dir, json_rel)
            if not os.path.exists(ann_path):
                continue
            gt_boxes, gt_labels = parse_cityscapes_json(ann_path)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = [dict(boxes=gt_boxes, labels=gt_labels)]

        # YOLO
        y_preds, y_obj, y_t, y_fps = run_yolo(yolo_model, img_rgb)
        map_yolo.update([y_preds], target)
        y_times.append(y_t)
        y_fps_list.append(y_fps)
        y_objs.append(y_obj)

        # Faster R-CNN
        r_preds, r_obj, r_t, r_fps = run_faster_rcnn(rcnn_model, img_rgb, device)
        map_rcnn.update([r_preds], target)
        r_times.append(r_t)
        r_fps_list.append(r_fps)
        r_objs.append(r_obj)

        count += 1
        if count % 25 == 0:
            print(f"Procesate {count}/{len(images)} imagini...")

    if count == 0:
        print("[EROARE] Nicio pereche imagine-etichetă validă.")
        return None

    y_map_res = map_yolo.compute()['map_50'].item()
    r_map_res = map_rcnn.compute()['map_50'].item()
    ram_mb, vram_mb = get_memory_usage()

    summary = {
        "Dataset": dataset_name,
        "Imagini": count,
        "YOLO_mAP50": round(y_map_res, 4),
        "YOLO_Timp_s": round(np.mean(y_times), 4),
        "YOLO_FPS": round(np.mean(y_fps_list), 2),
        "RCNN_mAP50": round(r_map_res, 4),
        "RCNN_Timp_s": round(np.mean(r_times), 4),
        "RCNN_FPS": round(np.mean(r_fps_list), 2),
        "RAM_MB": round(ram_mb, 1),
        "VRAM_MB": round(vram_mb, 1)
    }

    print(f"\n--- REZULTATE {dataset_name} ({count} imagini) ---")
    print(f"YOLO         -> mAP@50: {summary['YOLO_mAP50']} | Timp: {summary['YOLO_Timp_s']}s | FPS: {summary['YOLO_FPS']}")
    print(f"Faster R-CNN -> mAP@50: {summary['RCNN_mAP50']} | Timp: {summary['RCNN_Timp_s']}s | FPS: {summary['RCNN_FPS']}")
    print(f"Hardware     -> RAM: {summary['RAM_MB']} MB | VRAM GPU: {summary['VRAM_MB']} MB")
    return summary

if __name__ == "__main__":
    BASE_DAWN = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN"
    CITY_IMG = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/Cityscrapes/leftImg8bit_trainvaltest/leftImg8bit/val"
    CITY_ANN = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/Cityscrapes/gtFine_trainvaltest/gtFine/val"

    DATASETS = {
        "1": ("DAWN - Fog", "DAWN", f"{BASE_DAWN}/Fog/Fog", f"{BASE_DAWN}/Fog/Fog/Fog_PASCAL_VOC"),
        "2": ("DAWN - Rain", "DAWN", f"{BASE_DAWN}/Rain/Rain", f"{BASE_DAWN}/Rain/Rain/Rain_PASCAL_VOC"),
        "3": ("DAWN - Sand", "DAWN", f"{BASE_DAWN}/Sand/Sand", f"{BASE_DAWN}/Sand/Sand/Sand_PASCAL_VOC"),
        "4": ("DAWN - Snow", "DAWN", f"{BASE_DAWN}/Snow/Snow", f"{BASE_DAWN}/Snow/Snow/Snow_PASCAL_VOC"),
        "5": ("Cityscapes - Clear", "CITYSCAPES", CITY_IMG, CITY_ANN)
    }

    print('''Alege experimentul de rulat:
        [1] DAWN - Fog (Ceață)
        [2] DAWN - Rain (Ploaie)
        [3] DAWN - Sand (Furtună de nisip)
        [4] DAWN - Snow (Zăpadă)
        [5] Cityscapes - Vreme Bună (Baseline)
        [6] RULEAZĂ TOATE EXPERIMENTELE ȘI SALVEAZĂ CSV''')

    opt = input("Opțiune (1-6): ").strip()
    # None pentru toate
    LIMIT_IMAGINI = 50

    yolo_model, rcnn_model, device = load_models()
    print(f"Dispozitiv detectat: {device}")

    all_results = []
    keys_to_run = ["1", "2", "3", "4", "5"] if opt == "6" else [opt]

    for k in keys_to_run:
        if k in DATASETS:
            name, dtype, i_dir, a_dir = DATASETS[k]
            res = evaluate_dataset(name, dtype, i_dir, a_dir, yolo_model, rcnn_model, device, limit=LIMIT_IMAGINI)
            if res:
                all_results.append(res)

    if all_results:
        csv_file = "rezultate_experiment.csv"
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nToate rezultatele au fost salvate în '{csv_file}'.")