import os
import glob
import torch
import cv2
import time
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# aceasta functie traduce din XML-ul din Dawn in limbaj cunoscut de YOLO si R-CNN
# in acest limbaj o persona este clasa 0, bicicleta clasa 1, etc.
DAWN_TO_COCO = {
    'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3,
    'bus': 5, 'truck': 7
}


# folosim gpu sau cpu(daca nu exista gpu)
# descarca yolo si cnn si le evaluaza
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = YOLO('yolov8s.pt')
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    faster_rcnn_model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    faster_rcnn_model.eval()
    return yolo_model, faster_rcnn_model, device


# aflam raspunsurile corecte din XML-ulrile din DAWN pentru a face statistica
# "GROUND TRUTH"
def parse_dawn_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        if name in DAWN_TO_COCO:
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(DAWN_TO_COCO[name])
    if len(boxes) == 0:
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


# Rulam modelul YOLO
# porneste un cornometru, modelul prea o poza, calculeaza timpul si FPS-ul
# Timpul/Viteza = masoara cat dureaza sa analizeze o singura imagine,
#                 se exprima in milisecunde, cu cat mai mica este valoare cu atat mai rapid este modelul
# FPS = masoara cate imagini pot fi analizate intr-o singura secunda.
#       Se calculeaza impatind 1 secunda la timpul de inferenta( ex : 1/ 0.02(viteaza) = 50FPS). Cu cat valoare este mai mare cu atat modelul este mai rapid
# extrage rezultatele YOLO
def run_yolo(model, image):
    start_time = time.time()
    results = model(image, verbose=False)
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    boxes = results[0].boxes.xyxy.cpu()
    scores = results[0].boxes.conf.cpu()
    labels = results[0].boxes.cls.cpu().to(torch.int64)
    return dict(boxes=boxes, scores=scores, labels=labels), len(boxes), inference_time, fps


# Rulam modeul faster R-CNN
# pornim cronometrul, ...(la fel ca la yolo)
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
    labels = predictions['labels'][mask].cpu() - 1
    return dict(boxes=boxes, scores=scores, labels=labels), len(boxes), inference_time, fps


# Alegem setul de date,
if __name__ == "__main__":
    print('''Alege un set de date pentru a rula YOLO vs Faster R-CNN:
        [1] Fog 
        [2] Rain
        [3] Sand
        [4] Snow''')

    file = input("Alege un fișier (1/2/3/4): ")

    # SETĂRI CĂI - AICI S-AR PUTEA SĂ FIE PROBLEMA
    # Dacă imaginile sunt de fapt în folderul DAWN/Fog/Fog, modifică calea de mai jos!
    if file == "1":
        folder_imagini = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Fog/Fog"
        folder_xml = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Fog/Fog/Fog_PASCAL_VOC"
    elif file == "2":
        folder_imagini = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Rain/Rain"
        folder_xml = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Rain/Rain/Rain_PASCAL_VOC"
    elif file == "3":
        folder_imagini = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Sand/Sand"
        folder_xml = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Sand/Sand/Sand_PASCAL_VOC"
    elif file == "4":
        folder_imagini = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Snow/Snow"
        folder_xml = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Snow/Snow/Snow_PASCAL_VOC"
    else:
        print("Opțiune invalidă! Setăm implicit pe Fog.")
        folder_imagini = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Fog/Fog"
        folder_xml = "C:/Users/hudea/UBB/Anu3/Sem1/Proiect_cerce/DAWN/Fog/Fog/Fog_PASCAL_VOC"

    print(f"\n[DEBUG] Caut imagini în folderul: {folder_imagini}")
    print(f"[DEBUG] Caut XML-uri în folderul: {folder_xml}")

    # Căutăm mai multe tipuri de extensii, pentru siguranță
    imagini_test = glob.glob(os.path.join(folder_imagini, "*.jpg"))
    imagini_test += glob.glob(os.path.join(folder_imagini, "*.jpeg"))
    imagini_test += glob.glob(os.path.join(folder_imagini, "*.png"))
    imagini_test += glob.glob(os.path.join(folder_imagini, "*.JPG"))

    print(f"[DEBUG] Am găsit {len(imagini_test)} imagini în total.")

    if len(imagini_test) == 0:
        print("\nEROARE: Nu am găsit nicio imagine! Verifică dacă calea este corectă și dacă fișierele sunt acolo.")
    else:
        print("\nÎncărcare modele (YOLO și Faster R-CNN)...")
        yolo_model, rcnn_model, device = load_models()

        map_metric_yolo = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
        map_metric_rcnn = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

        imagini_subset = imagini_test[:10]

        total_yolo_obj, total_yolo_time, total_yolo_fps = 0, 0, 0
        total_rcnn_obj, total_rcnn_time, total_rcnn_fps = 0, 0, 0
        count = 0

        # pentru fiecare poza:
        # gaseste fisierul XML cu rezultatele
        # incarcam poza
        # trimitem poza in YOLO si R-CNN si salvam rezultatele la ambele
        # trimitem rezultatele cate calculatorul de mAP
        for image_path in imagini_subset:
            nume_fisier = os.path.basename(image_path)
            # Indiferent ce extensie are imaginea, căutăm fișierul .xml aferent
            nume_fara_extensie = os.path.splitext(nume_fisier)[0]
            xml_path = os.path.join(folder_xml, nume_fara_extensie + '.xml')

            if not os.path.exists(xml_path):
                print(f"Sărit {nume_fisier} - Nu am găsit XML-ul la: {xml_path}")
                continue

            gt_boxes, gt_labels = parse_dawn_xml(xml_path)
            target = [dict(boxes=gt_boxes, labels=gt_labels)]

            print(f"\n--- Procesare imagine: {nume_fisier} (Obiecte reale: {len(gt_boxes)}) ---")
            img = cv2.imread(image_path)
            if img is None:
                print(f"Eroare la citirea imaginii: {image_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            yolo_preds, y_obj, y_time, y_fps = run_yolo(yolo_model, img_rgb)
            map_metric_yolo.update([yolo_preds], target)
            print(f"[YOLO]         Detecții: {y_obj} | Timp: {y_time:.4f}s | FPS: {y_fps:.1f}")
            total_yolo_obj += y_obj
            total_yolo_time += y_time
            total_yolo_fps += y_fps

            rcnn_preds, r_obj, r_time, r_fps = run_faster_rcnn(rcnn_model, img_rgb, device)
            map_metric_rcnn.update([rcnn_preds], target)
            print(f"[Faster R-CNN] Detecții: {r_obj} | Timp: {r_time:.4f}s | FPS: {r_fps:.1f}")
            total_rcnn_obj += r_obj
            total_rcnn_time += r_time
            total_rcnn_fps += r_fps

            count += 1

        if count > 0:
            print("\nCalculare mAP total... (vă rugăm așteptați)")
            yolo_map = map_metric_yolo.compute()
            rcnn_map = map_metric_rcnn.compute()

            print("\n==================================================")
            print(f"REZULTATE MEDII (pe {count} imagini validate):")
            print("==================================================")
            print(
                f"YOLO         -> Timp mediu: {total_yolo_time / count:.4f}s | FPS mediu: {total_yolo_fps / count:.1f} | mAP: {yolo_map['map']:.4f}")
            print(
                f"Faster R-CNN -> Timp mediu: {total_rcnn_time / count:.4f}s | FPS mediu: {total_rcnn_fps / count:.1f} | mAP: {rcnn_map['map']:.4f}")
        else:
            print("\nNu s-au putut procesa imagini (probabil nu s-au potrivit pozele cu XML-urile).")

    # Pauză la final pentru a nu se închide automat fereastra
    input("\nApasă Enter pentru a închide programul...")