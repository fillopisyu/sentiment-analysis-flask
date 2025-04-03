# app.py dosyası (Tamamı)

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import json
import os

# Mevcut zaman ve konumu ekleyelim (opsiyonel, sadece bilgi için)
from datetime import datetime
import pytz # Zaman dilimi için (pip install pytz gerekebilir)

# Zaman dilimini ayarla (Türkiye için)
turkey_tz = pytz.timezone('Europe/Istanbul')
current_time = datetime.now(turkey_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
current_location = "Konya, Konya, Türkiye" # Bilgi olarak eklendi

print(f"Uygulama başlatılıyor...")
print(f"Mevcut Konum: {current_location}")
print(f"Mevcut Zaman: {current_time}")
print(f"TensorFlow Sürümü: {tf.__version__}")


# --- Yapılandırma ---
# DİKKAT: Bu yolun, train_sentiment.py'nin çıktıları kaydettiği yerle aynı olduğundan emin olun!
# Örnek yol: r'C:\Users\KULLANICI_ADINIZ\...' veya projenizin olduğu yer.
output_dir = r'C:\Users\ismsa\OneDrive\Masaüstü\Eğitimler\NLP\SemantikAnalysis\output' # BU YOLU KONTROL EDİN!
model_path = os.path.join(output_dir, 'sentiment_model.keras')
tokenizer_path = os.path.join(output_dir, 'tokenizer.pickle')
config_path = os.path.join(output_dir, 'config.json')
# --- ---

# Global değişkenler
model = None
tokenizer = None
max_tokens = 100 # Varsayılan, config'den yüklenecek

# --- Gerekli Dosyaları Yükle ---
print("Yapılandırma dosyası yükleniyor...")
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    max_tokens = config['max_tokens']
    print(f"Yapılandırma yüklendi: max_tokens={max_tokens}")
except FileNotFoundError:
    print(f"HATA: Yapılandırma dosyası bulunamadı: {config_path}")
    print("Lütfen önce train_sentiment.py betiğini HATASIZ çalıştırdığınızdan emin olun.")
    exit()
except Exception as e:
    print(f"HATA: Yapılandırma dosyası yüklenemedi: {e}")
    exit()

print("Tokenizer yükleniyor...")
try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer başarıyla yüklendi.")
except FileNotFoundError:
    print(f"HATA: Tokenizer dosyası bulunamadı: {tokenizer_path}")
    print("Lütfen önce train_sentiment.py betiğini HATASIZ çalıştırdığınızdan emin olun.")
    exit()
except Exception as e:
    print(f"HATA: Tokenizer yüklenemedi: {e}")
    exit()

print("Keras modeli yükleniyor...")
try:
    # TensorFlow'un bilgi mesajlarını bastırmak için log seviyesini ayarla (opsiyonel)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Sadece hataları göster
    tf.get_logger().setLevel('ERROR')       # TensorFlow logger seviyesi

    model = load_model(model_path)
    print("Model başarıyla yüklendi.")
    # Modelin özetini görmek isterseniz açabilirsiniz:
    # model.summary()
except FileNotFoundError:
     print(f"HATA: Model dosyası bulunamadı: {model_path}")
     print("Lütfen önce train_sentiment.py betiğini HATASIZ çalıştırdığınızdan emin olun.")
     exit()
except Exception as e:
    print(f"HATA: Model yüklenemedi: {e}")
    exit()
# --- Yükleme Sonu ---


# Flask uygulamasını başlat
app = Flask(__name__)
print("Flask uygulaması yapılandırıldı.")

# Ana sayfa route'u
@app.route("/", methods=["GET", "POST"])
def index():
    # Her istek için değişkenleri sıfırla
    sentiment_label = None      # "Olumlu" veya "Olumsuz" metni
    sentiment_status = None     # "positive" veya "negative" durumu (CSS/JS için)
    prediction_value = None     # Modelin ham skoru
    error_message = None        # Olası hata mesajları
    user_input = ""             # Formdaki önceki girişi tutmak için

    if request.method == "POST":
        user_input = request.form.get("review", "").strip() # Formdan yorumu al, boşlukları temizle

        if not user_input:
            error_message = "Lütfen analiz edilecek bir yorum girin."
        # Model veya tokenizer yüklenememişse (yukarıda kontrol ediliyor ama yine de ekleyelim)
        elif model is None or tokenizer is None:
             error_message = "Model veya Tokenizer yüklenemediği için tahmin yapılamıyor. Lütfen sunucu loglarını kontrol edin."
        else:
            try:
                # 1. Yorumu tokenize et
                tokens = tokenizer.texts_to_sequences([user_input])

                # 2. Padding uygula (config'den gelen max_tokens ile)
                tokens_pad = pad_sequences(tokens, maxlen=max_tokens)

                # 3. Model ile tahmin yap
                # Not: model.predict tek bir örnek için bile 2D array döndürebilir
                prediction = model.predict(tokens_pad, verbose=0)[0][0] # verbose=0 logları azaltır
                prediction_value = f"{prediction:.4f}" # Skoru formatla

                # 4. Sonucu belirle (Etiket ve Durum olarak)
                if prediction > 0.5:
                    sentiment_label = "Olumlu"
                    sentiment_status = "positive"
                else:
                    sentiment_label = "Olumsuz"
                    sentiment_status = "negative"

                # Sunucu loguna yazdırma (opsiyonel)
                print(f"Analiz: Girdi='{user_input[:50]}...' -> Skor={prediction_value} -> Durum={sentiment_status}")

            except Exception as e:
                print(f"Tahmin sırasında hata oluştu: {e}") # Sunucu tarafında hatayı logla
                # Kullanıcıya daha genel bir hata göster
                error_message = "Tahmin sırasında beklenmedik bir sorun oluştu. Lütfen tekrar deneyin veya sistem yöneticisine başvurun."

    # HTML şablonunu render et ve ilgili değişkenleri gönder
    return render_template("index.html",
                           sentiment_label=sentiment_label,     # "Olumlu" veya "Olumsuz" metni
                           sentiment_status=sentiment_status,   # "positive" veya "negative" durumu
                           prediction_value=prediction_value,   # Ham skor
                           error_message=error_message,         # Hata mesajı
                           user_review=user_input)              # Kullanıcının girdiği metni formda tut

# Uygulamayı çalıştır (sadece betik doğrudan çalıştırıldığında)
if __name__ == "__main__":
    # Dosyaların varlığını tekrar kontrol etmeye gerek yok, yukarıda yapıldı ve exit() ile çıkıldı.
    print(f"Flask sunucusu başlatılıyor... http://127.0.0.1:5000 adresinden erişilebilir.")
    print("Sunucuyu durdurmak için CTRL+C tuşlarına basın.")
    # debug=True geliştirme sırasında hataları tarayıcıda gösterir ve otomatik yeniden başlatma sağlar.
    # Production (gerçek kullanım) ortamında debug=False olarak ayarlanmalıdır.
    app.run(host='0.0.0.0', port=5000, debug=True)
    # host='0.0.0.0' eklemek, aynı ağdaki diğer cihazlardan erişime izin verebilir (güvenlik duvarı ayarlarınıza bağlıdır).
    # Sadece kendi makinenizden erişmek için host='127.0.0.1' veya host parametresini hiç vermeyebilirsiniz.