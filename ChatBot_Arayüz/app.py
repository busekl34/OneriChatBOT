import re
import string
import pandas as pd
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer

from flask import Flask, render_template, request

nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))
stemmer = TurkishStemmer()

app = Flask(__name__)

# Veri yükleme ve model hazırlama (uygulama başladığında sadece 1 kere çalışacak)
df = pd.read_csv('user_chat_dataset.csv')

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

df['cleaned_user'] = df['user'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_user'])

le = LabelEncoder()
y = le.fit_transform(df['kategori'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)


@app.route('/', methods=['GET', 'POST'])
def home():
    cevap = ""
    tahmin_kategori = ""
    tahmin_guven = 0.0
    if request.method == 'POST':
        soru = request.form.get('soru')
        soru_cleaned = clean_text(soru)
        soru_vec = vectorizer.transform([soru_cleaned])

        tahmin_olasiliklari = model.predict_proba(soru_vec)[0]
        en_yuksek_ihtimal = max(tahmin_olasiliklari)
        tahmin_index = tahmin_olasiliklari.argmax()

        if en_yuksek_ihtimal < 0.6:
            cevap = "Bu konuda emin değilim ama başka bir soru sorabilirsin."
        else:
            tahmin_kategori = le.inverse_transform([tahmin_index])[0]
            cevaplar = df[df['kategori'] == tahmin_kategori]['chat'].tolist()
            if cevaplar:
                cevap = random.choice(cevaplar)
            else:
                cevap = "Üzgünüm, bu konuda elimde bir cevap yok."
            tahmin_guven = en_yuksek_ihtimal * 100

    return render_template('index.html', cevap=cevap, kategori=tahmin_kategori, guven=round(tahmin_guven, 2))

if __name__ == '__main__':
    app.run(debug=True)
