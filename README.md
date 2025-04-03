# Türkçe Duygu Analizi Web Uygulaması

Bu proje, Türkçe metinler için duygu analizi (pozitif/negatif) yapan basit bir web uygulamasıdır. **Flask** web çatısı ve **TensorFlow/Keras** ile eğitilmiş bir **GRU** modeli kullanılarak geliştirilmiştir. Kullanıcı arayüzü üzerinden girilen metnin duygu skorunu tahmin eder.

## 🚀 Kurulum

Projeyi yerel makinenizde kurmak için aşağıdaki adımları izleyin:

1.  **Gereksinimler:**
    * Python 3.8 veya üzeri
    * Git

2.  **Adımlar:**
    * Depoyu klonlayın ve dizine gidin:
      ```bash
      git clone [https://github.com/](https://github.com/)<kullanici-adin>/<depo-adin>.git
      cd <depo-adin>
      ```
    * (Önerilir) Sanal ortam oluşturup aktifleştirin:
      ```bash
      python -m venv venv
      # Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
      ```
    * Gerekli Python kütüphanelerini yükleyin:
      ```bash
      pip install tensorflow pandas numpy Flask pytz
      ```
    * **Veri Seti ve Yollar:** `hepsiburada.csv` dosyasını `train_sentiment.py` içindeki `csv_path` ile belirtilen konuma yerleştirin (veya betikteki yolu güncelleyin). `output_dir` yolunun (`train_sentiment.py` ve `app.py` içinde) geçerli olduğundan emin olun (veya betiklerdeki yolu güncelleyin).
      *(Not: Kodda belirtilen varsayılan yollar (`C:\Users\ismsa\...` gibi) sizin sisteminize uymayabilir, kontrol edilmelidir.)*

## 🧠 Modelin Eğitilmesi (Gerekirse)

Uygulama, çalışmak için önceden eğitilmiş bir modele ve ilgili dosyalara ihtiyaç duyar (`output/` klasöründeki `.keras`, `.pickle`, `.json` dosyaları).

* **Eğer projenizde (örneğin `output/` klasöründe) bu dosyalar mevcut değilse** veya modeli farklı verilerle/parametrelerle **yeniden eğitmek istiyorsanız**, aşağıdaki komutu çalıştırın:
    ```bash
    python train_sentiment.py
    ```
    Bu komut, gerekli dosyaları yapılandırılan `output/` klasörüne üretecektir. Bu işlem veri setinin büyüklüğüne göre zaman alabilir.

* **Eğer gerekli model dosyaları zaten `output/` klasöründe mevcutsa** (örneğin depoyu klonladığınızda geldiyse veya daha önce eğittiyseniz), bu adımı atlayıp doğrudan bir sonraki adıma geçebilirsiniz.
    *(Not: Büyük model dosyalarının doğrudan Git deposunda tutulması genellikle iyi bir pratik olarak önerilmez.)*

## ▶️ Uygulamanın Çalıştırılması

Gerekli model dosyaları `output/` klasöründe hazır olduğunda, Flask web sunucusunu başlatın:

```bash
python app.py
