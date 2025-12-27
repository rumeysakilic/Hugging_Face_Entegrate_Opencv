import os
import cv2
import numpy as np
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration, # Captioning için
    DetrImageProcessor, DetrForObjectDetection, # Nesne tespiti için
    AutoTokenizer, AutoModelForCausalLM # Metin üretimi için (GPT-2)
)
import torch

# 1. Görüntü Dosyası Adınız
IMAGE_FILE = "doga_resmi.jpg" 

# 2. Nesne Tespiti Güven Eşiği (0.9 çok katıydı, 0.7 daha fazla nesne bulabilir)
THRESHOLD = 0.7

# ----------------------------------------------------------------------
# 1. ADIM: GÖRÜNTÜ AÇIKLAMA (CAPTIONING)
# ----------------------------------------------------------------------

try:
    # Modelleri tek bir yerde yüklemek daha verimli
    print("--- Modeller Yükleniyor... (İlk çalıştırmada zaman alabilir) ---")
    
    # Captioning Modeli
    processor_cap = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_cap = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Nesne Tespiti Modeli
    processor_detr = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model_detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    # Metin Üretimi Modeli
    tokenizer_gen = AutoTokenizer.from_pretrained("gpt2")
    model_gen = AutoModelForCausalLM.from_pretrained("gpt2")
    print("--- Modeller Başarıyla Yüklendi. ---")

except Exception as e:
    print(f"\n HATA: Modeller yüklenirken bir sorun oluştu: {e}")
    exit() # Hata oluşursa programı sonlandır

def generate_caption(image_path):
    """BLIP modelini kullanarak temel altyazı üretir."""
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor_cap(raw_image, return_tensors="pt")
        out = model_cap.generate(**inputs, max_length=50)
        caption = processor_cap.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f" HATA: Caption üretilemedi. {e}")
        return "a simple image" # Hata durumunda varsayılan çıktı

# ----------------------------------------------------------------------
# 2. ADIM: NESNE TESPİTİ (OBJECT DETECTION)
# ----------------------------------------------------------------------

def object_detection_and_visualize(image_path, threshold):
    """DETR ile nesneleri tespit eder, OpenCV ile çizer ve etiketleri döndürür."""
    
    detected_objects = []
    
    try:
        im = Image.open(image_path).convert("RGB")
        inputs = processor_detr(images=im, return_tensors="pt")
        outputs = model_detr(**inputs)
        
        target_sizes = torch.tensor([im.size[::-1]])
        results = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        
        img_cv = cv2.imread(image_path)
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > threshold:
                box = [round(i) for i in box.tolist()]
                xmin, ymin, xmax, ymax = box
                
                label_text = model_detr.config.id2label[label.item()]
                detected_objects.append(label_text)
                
                # OpenCV ile kutu çizimi
                cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(
                    img_cv, 
                    f"{label_text}: {score.item():.2f}", 
                    (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    2
                )

        # OpenCV Penceresini Göster
        cv2.imshow("OpenCV Nesne Tespiti", img_cv)
        cv2.waitKey(1) 
        
        return list(set(detected_objects))
        
    except Exception as e:
        print(f" HATA: Nesne tespiti yapılamadı. {e}")
        return []

# ----------------------------------------------------------------------
# 3. ADIM: HİKAYE ÜRETİMİ (TEXT GENERATION)
# ----------------------------------------------------------------------

def generate_story(caption, detected_objects):
    """Caption ve tespit edilen nesneleri kullanarak yaratıcı bir hikaye üretir."""
    
    objects_str = ", ".join(detected_objects) 
    
    # Prompt Engineering: Tüm girdileri birleştir
    prompt = f"""
    Based on the image description: '{caption}' and the detected objects: '{objects_str}'.
    Write a short, mysterious story (3-5 sentences), continuing from the description.
    [STORY START]: 
    """
    
    print("\n--- GPT-2 Icin Girdi Prompt'u ---")
    print(prompt.strip())
    print("-----------------------------------")
    
    try:
        inputs = tokenizer_gen(prompt, return_tensors='pt')
        
        output_sequences = model_gen.generate(
            inputs['input_ids'],
            max_length=150, 
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7, 
            top_k=50
        )
        
        generated_story = tokenizer_gen.decode(output_sequences[0], skip_special_tokens=True)
        # Sadece hikaye kısmını al
        story_only = generated_story.replace(prompt, "", 1).strip()
        return story_only
    
    except Exception as e:
        print(f" HATA: Hikaye üretilemedi. {e}")
        return "Hikaye üretimi sirasinda bir sorun yasandi."
 
if __name__ == "__main__":
    
    if not os.path.exists(IMAGE_FILE):
        print(f"\n HATA: '{IMAGE_FILE}' dosyasi bulunamadi. Lutfen dosya yolunu kontrol edin.")
        exit()
        
    print(f"\n--- BILGI: '{IMAGE_FILE}' dosyasi bulundu. Uygulama akisi baslatiliyor... ---")

    # ADIM 1: Görüntü Açıklama (Captioning)
    print("\n--- ADIM 1: Görüntü Aciklamasi Üretiliyor... ---")
    caption_output = generate_caption(IMAGE_FILE)
    print(f" Caption Ciktisi: '{caption_output}'")

    # ADIM 2: Nesne Tespiti (OpenCV Görselleştirme ile)
    print("\n--- ADIM 2: Nesne Tespiti Yapiliyor... ---")
    objects_output = object_detection_and_visualize(IMAGE_FILE, threshold=THRESHOLD)
    print(f" Tespit Edilen Nesneler: {objects_output}")
    
    # ADIM 3: Hikaye Üretimi
    print("\n--- ADIM 3: Yaratici Hikaye Üretiliyor... ---")
    final_story = generate_story(caption_output, objects_output)

    # Nihai Sonuç
    print("\n\n-------------------------------------------------")
    print(" NIHAI YARATICI HIKAYE ")
    print("-------------------------------------------------")
    print(final_story)
    print("-------------------------------------------------")
    
    # OpenCV pencerelerini bir tuşa basılana kadar açık tut
    print("\n[BILGI] OpenCV penceresini kapatmak için herhangi bir tusa basin...")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()