from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import cv2
import numpy as np
from PIL import Image

# DETR Modelini ve İşleyiciyi Yükle
processor_detr = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model_detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def object_detection_and_visualize(image_path, threshold=0.9):
    """
    DETR modelini kullanarak nesneleri tespit eder, OpenCV ile kutular çizer 
    ve tespit edilen nesne etiketlerini döndürür.
    """
    # 1. Görüntüyü Yükle
    im = Image.open(image_path).convert("RGB")
    
    # 2. Ön İşleme
    inputs = processor_detr(images=im, return_tensors="pt")
    
    # 3. Model Tahmini
    outputs = model_detr(**inputs)

    # 4. Tahminleri İşle
    target_sizes = torch.tensor([im.size[::-1]])
    results = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    detected_objects = []
    
    # 5. OpenCV ile Görselleştirme
    img_cv = cv2.imread(image_path)
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Yüksek güvenilirlikteki nesneleri seç
        if score > threshold:
            box = [round(i) for i in box.tolist()]
            xmin, ymin, xmax, ymax = box
            
            # Nesne etiketini al
            label_text = model_detr.config.id2label[label.item()]
            detected_objects.append(label_text)
            
            # OpenCV ile kutu çiz
            cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2) # Kırmızı kutu
            
            # OpenCV ile etiket yaz
            cv2.putText(
                img_cv, 
                f"{label_text}: {score.item():.2f}", 
                (xmin, ymin - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                2
            )

    # Görüntüyü OpenCV penceresinde göster
    cv2.imshow("OpenCV Nesne Tespiti", img_cv)
    cv2.waitKey(1) # Pencerelerin görünmesi için kısa bir bekleme
    
    return list(set(detected_objects)) # Tekrarlanan etiketleri kaldır

# --- KULLANIM VE ENTEGRASYON ---
# Önceki altyazı çıktınız: 'a red barn sits on a grassy hill'
# --- KULLANIM VE ENTEGRASYON (OpenCV PENCERE DÜZELTMESİ) ---

# image_file tanımı
image_file = "doga_resmi.jpg" 

detected_list = object_detection_and_visualize(image_file, threshold=0.9)

print("\n--- OpenCV Nesne Tespit Ciktilari ---")
print(f"Tespit Edilen Ana Nesneler (OpenCV): {detected_list}")

# Pencerenin açılması için programı bekleten ana komut:
cv2.waitKey(0) # Bir tuşa basılana kadar pencereyi açık tutar
cv2.destroyAllWindows() # Sonra pencereleri kapatır