❤️ Kalp Yetmezliği Risk ve Sağkalım Tahmin Sistemi
📌 Proje Hakkında

Bu proje, kalp yetmezliği hastalarında mortalite riskini tahmin etmek amacıyla geliştirilmiş bir makine öğrenmesi tabanlı klinik karar destek uygulamasıdır.

Model, Heart Failure Clinical Records Dataset (299 hasta) kullanılarak eğitilmiş ve 5 Katlı Çapraz Doğrulama (5-Fold Cross Validation) ile değerlendirilmiştir.

Uygulama, Streamlit kullanılarak geliştirilmiş olup kullanıcı dostu bir arayüz üzerinden hasta bilgileri girilerek anlık risk tahmini yapılabilmektedir.

🚀 Özellikler
Kalp yetmezliği mortalite risk tahmini
Streamlit tabanlı web arayüzü
Random Forest modeli
SHAP ile model açıklanabilirliği
Klinik risk yorumları
Türkçe kullanıcı arayüzü


📊 Kullanılan Teknolojiler
Python
Streamlit
Pandas
NumPy
Scikit-learn
SHAP
Matplotlib
Joblib

📈 Makine Öğrenmesi İş Akışı
Veri ön işleme
Özellik seçimi
Model eğitimi
5 Katlı Çapraz Doğrulama
Model değerlendirme
SHAP analizi
Streamlit ile dağıtım

📂 Veri Seti

Heart Failure Clinical Records Dataset

Toplam hasta sayısı: 299

Hedef değişken:

Mortalite (DEATH_EVENT)

📌 Proje Yapısı
Heart-Failure-Risk/
│
├── app.py
├── rf_model.pkl
├── requirements.txt
├── README.md
├── images/
│   └── app.png
└── notebooks/

⚠️ Uyarı : Bu uygulama yalnızca eğitim ve araştırma amacıyla geliştirilmiştir. Klinik karar verme sürecinin yerine geçmez.

👩‍💻 Geliştirici

Menekşe Ayhan

Klinik Araştırma Koordinatörü (5+ yıl)
İstatistiksel Danışman (10+ yıl)
Veri Bilimi ve Makine Öğrenmesi

