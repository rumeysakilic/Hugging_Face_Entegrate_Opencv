# Hugging_Face_Entegrate_Opencv
🤖 Görsel Temelli Hikaye Oluşturucu ve Nesne Analiz Aracı
Bu proje, Yapay Zeka ile Uygulama Geliştirme ve Doğal Dil İşleme (NLP) tekniklerini birleştirerek, yüklenen bir görselden otomatik olarak altyazı üreten, nesneleri tespit eden ve bu verilerden yola çıkarak yaratıcı bir hikaye yazan bir yapay zeka uygulamasıdır.

🚀 Proje Mimarisi
Uygulama, üç farklı yapay zeka modelinin bir boru hattı (pipeline) şeklinde birleştirilmesiyle çalışır:

Görüntü Açıklama (Image Captioning): Salesforce/blip-image-captioning-base modeli ile görselin genel bir özeti çıkarılır.

Nesne Tespiti (Object Detection): facebook/detr-resnet-50 modeli kullanılarak görseldeki spesifik nesneler bulunur ve OpenCV kütüphanesi ile görsel üzerine işaretlenir.

Metin Üretimi (Text Generation): Elde edilen altyazı ve nesne etiketleri birleştirilerek GPT-2 modeli üzerinden yaratıcı bir hikaye kurgulanır.

🛠️ Kullanılan Teknolojiler ve Kütüphaneler
Python 3.10

OpenCV: Görsel işleme ve nesne tespiti görselleştirmesi.

Hugging Face Transformers: Önceden eğitilmiş (pre-trained) modellerin kullanımı.

PyTorch: Modellerin arka plan hesaplamaları.

Pillow (PIL): Görüntü dosyalarının yönetimi.

📋 Kurulum
Projenin çalışması için gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

pip install opencv-python transformers torch Pillow

<img width="1000" height="635" alt="Image" src="https://github.com/user-attachments/assets/1e585a0b-3d34-41ce-87ae-b4fe6ca2c951" />

⚙️ Uygulama Akışı ve Örnek Çıktı
Uygulama çalıştırıldığında aşağıdaki adımları izler:

1. Görsel Analizi
Yüklenen görsel (örneğin: doga_resmi.jpg) analiz edilir.

Üretilen Altyazı: "a red barn sits on a grassy hill"

<img width="519" height="168" alt="Image" src="https://github.com/user-attachments/assets/9c908ee1-9abe-4ec0-9138-a25279676acf" />

2. Nesne Tespiti (OpenCV Entegrasyonu)
Model, görseldeki nesneleri tespit eder ve OpenCV penceresinde kırmızı çerçeveler içinde gösterir.

Tespit Edilenler: ['barn', 'fire hydrant']

3. Hikaye Üretimi
Tüm veriler GPT-2 modeline "Prompt Engineering" teknikleri ile beslenir ve final metni oluşturulur.

<img width="1065" height="159" alt="Image" src="https://github.com/user-attachments/assets/5084e402-3b9a-4b8f-9c65-5519aae54f91" />

Örnek Çıktı: "For a while, I would get this strange feeling. This barn was a house on a hill top. It was on a hill. It was in a shady corner. It was a house. The barn was on a hill. It was in a shady corner. It was a house. The barn was on a hill. It was in a shady corner. It was a house. The barn was on a hill"

"Bir süredir bu tuhaf hissi yaşıyordum. Bu ahır, bir tepe üzerindeki bir evdi. Bir tepedeydi. Gölgeli bir köşedeydi. O bir evdi. Ahır bir tepedeydi. Gölgeli bir köşedeydi. O bir evdi. Ahır bir tepedeydi. Gölgeli bir köşedeydi. O bir evdi. Ahır bir tepedeydi..."

<img width="894" height="173" alt="Image" src="https://github.com/user-attachments/assets/cc8aa889-9c69-40ad-a9b5-74172d6b8d4c" />
