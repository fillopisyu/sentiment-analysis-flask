# train_sentiment.py dosyası

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
import pickle
import json
import os

print("TensorFlow Sürümü:", tf.__version__)

# --- Yapılandırma ---
# DİKKAT: Bu yolu kendi CSV dosyanızın konumuyla değiştirin!
csv_path = r'C:\Users\ismsa\OneDrive\Masaüstü\Eğitimler\NLP\SemantikAnalysis\hepsiburada.csv'
# DİKKAT: Model, tokenizer ve config dosyalarının kaydedileceği yer
output_dir = r'C:\Users\ismsa\OneDrive\Masaüstü\Eğitimler\NLP\SemantikAnalysis\output' # Çıktı için ayrı bir klasör önerilir
num_words = 10000  # Kullanılacak maksimum kelime sayısı
embedding_size = 50 # Kelime vektör boyutu
epochs = 5         # Eğitim döngüsü sayısı
batch_size = 256   # Her eğitim adımındaki örnek sayısı
# --- ---

# Çıktı klasörünü oluştur (varsa dokunmaz)
os.makedirs(output_dir, exist_ok=True)
print(f"Çıktı dosyaları '{output_dir}' klasörüne kaydedilecek.")

# 1. Veriyi Yükle
print(f"Veri yükleniyor: {csv_path}")
try:
    data = pd.read_csv(csv_path)
    print("Veri başarıyla yüklendi.")
    print(data.head())
except FileNotFoundError:
    print(f"HATA: CSV dosyası bulunamadı: {csv_path}")
    print("Lütfen `train_sentiment.py` içindeki `csv_path` değişkenini kontrol edin.")
    exit() # Dosya yoksa devam etme

# 2. Veriyi Hazırla
print("Veri ön işleniyor...")
# Eksik verileri temizle (varsa)
data.dropna(subset=['Review', 'Rating'], inplace=True)
# Rating'i tamsayı yap (varsayılan 1 = olumlu, 0 = olumsuz olmalı, kontrol et!)
# Eğer Rating sütunu farklı değerler içeriyorsa (örn: 1-5 yıldız), önce 0/1'e dönüştürmen gerekir.
# Örnek: data['Rating'] = data['Rating'].apply(lambda x: 1 if x > 3 else 0) # 4 ve 5 yıldız olumlu varsayımı
# Bu örnekte Hepsiburada verisinde 1 ve 0 olduğu varsayılıyor.
data['Rating'] = data['Rating'].astype(int)

target = data['Rating'].values.tolist()
reviews = data['Review'].values.tolist()
print(f"İşlenecek {len(reviews)} yorum bulundu.")

# 3. Tokenizer Oluştur, Eğit ve Kaydet
print("Tokenizer oluşturuluyor ve eğitiliyor...")
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(reviews) # Tüm verilere göre kelime haznesini oluştur

# Tokenizer'ı kaydet
tokenizer_path = os.path.join(output_dir, 'tokenizer.pickle')
try:
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer başarıyla kaydedildi: {tokenizer_path}")
except Exception as e:
    print(f"HATA: Tokenizer kaydedilemedi: {e}")
    exit()

# 4. Metinleri Sekanslara Çevir
print("Metinler sekanslara çevriliyor...")
all_tokens = tokenizer.texts_to_sequences(reviews)

# 5. Padding İçin max_tokens Hesapla ve Config Kaydet
num_tokens = [len(tokens) for tokens in all_tokens]
num_tokens = np.array(num_tokens)

if len(num_tokens) > 0:
    mean_tokens = np.mean(num_tokens)
    std_tokens = np.std(num_tokens)
    max_tokens = int(mean_tokens + 2 * std_tokens) # Ortalama + 2 standart sapma
    print(f"Ortalama token sayısı: {mean_tokens:.2f}")
    print(f"Hesaplanan max_tokens (padding için): {max_tokens}")

    coverage = np.sum(num_tokens < max_tokens) / len(num_tokens)
    print(f"Bu max_tokens ile yorumların kapsanma oranı: {coverage:.2%}")
else:
    print("Uyarı: Token bulunamadı, max_tokens varsayılan olarak 100 ayarlandı.")
    max_tokens = 100 # Varsayılan bir değer

# Config dosyasını kaydet
config_path = os.path.join(output_dir, 'config.json')
config = {'max_tokens': max_tokens, 'num_words': num_words, 'embedding_size': embedding_size}
try:
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"Yapılandırma kaydedildi: {config_path}")
except Exception as e:
    print(f"HATA: Yapılandırma kaydedilemedi: {e}")
    exit()

# 6. Padding Uygula
print(f"Sekanslara padding uygulanıyor (max uzunluk = {max_tokens})...")
all_pad = pad_sequences(all_tokens, maxlen=max_tokens)

# 7. Veriyi Eğitim ve Test Olarak Ayır (Padding yapılmış veri üzerinden)
print("Veri eğitim ve test setlerine ayrılıyor...")
cutoff = int(len(all_pad) * 0.80)
x_train_pad, x_test_pad = all_pad[:cutoff], all_pad[cutoff:]
y_train, y_test = target[:cutoff], target[cutoff:] # Target'ı da ayır

# Numpy array'e çevir (TensorFlow için)
x_train_pad = np.array(x_train_pad)
x_test_pad = np.array(x_test_pad)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(f"Eğitim seti boyutu: {len(x_train_pad)}")
print(f"Test seti boyutu: {len(x_test_pad)}")

# 8. Modeli Oluştur
print("Keras modeli oluşturuluyor...")
model = Sequential()
# Embedding katmanı (input_length önemlidir!)
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens, # Padding uzunluğu ile aynı olmalı
                    name='embedding_layer'))
# GRU katmanları
model.add(GRU(units=32, return_sequences=True)) # Biraz daha fazla ünite
model.add(GRU(units=16))
# Çıkış katmanı (Sigmoid ikili sınıflandırma için)
model.add(Dense(1, activation='sigmoid'))

# Optimizasyon ve Derleme
optimizer = Adam(learning_rate=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary() # Model özetini yazdır

# 9. Modeli Eğit
print("Model eğitiliyor...")
try:
    model.fit(x_train_pad, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_test_pad, y_test)) # Test verisiyle doğrula
    print("Model eğitimi tamamlandı.")
except Exception as e:
    print(f"HATA: Model eğitimi sırasında hata oluştu: {e}")
    exit()

# 10. Modeli Kaydet
model_path = os.path.join(output_dir, 'sentiment_model.keras')
try:
    model.save(model_path)
    print(f"Model başarıyla kaydedildi: {model_path}")
except Exception as e:
    print(f"HATA: Model kaydedilemedi: {e}")
    exit()

# 11. (Opsiyonel) Modeli Test Et
print("Model test seti üzerinde değerlendiriliyor...")
loss, accuracy = model.evaluate(x_test_pad, y_test)
print(f"Test Kaybı (Loss): {loss:.4f}")
print(f"Test Doğruluğu (Accuracy): {accuracy:.4f}")

print("\nEğitim betiği başarıyla tamamlandı!")