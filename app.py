from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

# Download stopwords once (first run may take a bit)
nltk.download('stopwords')

app = Flask(__name__)

# ---------- Load model and vectorizer ----------
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('sentiment_model.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ---------- Text cleaning function ----------
def clean_review(text: str) -> str:
    text = re.sub('[^a-zA-Z]', ' ', text)      # keep only letters
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# ---------- Simple aspect extraction (keywords) ----------
def extract_aspects(original_text: str):
    text = original_text.lower()
    aspects = []

    service_keywords = ['service', 'waiter', 'staff', 'server', 'slow', 'fast', 'rude', 'polite']
    food_keywords = ['food', 'taste', 'tasty', 'flavor', 'dish', 'meal', 'burger', 'pizza', 'dessert']
    hygiene_keywords = ['clean', 'dirty', 'hygiene', 'smell', 'messy', 'neat']
    ambience_keywords = ['ambience', 'atmosphere', 'music', 'noise', 'lighting']

    if any(k in text for k in service_keywords):
        aspects.append('Service')
    if any(k in text for k in food_keywords):
        aspects.append('Food Quality')
    if any(k in text for k in hygiene_keywords):
        aspects.append('Hygiene')
    if any(k in text for k in ambience_keywords):
        aspects.append('Ambience')

    if not aspects:
        aspects.append('General Experience')

    return aspects

# ---------- Route ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_label = None
    confidence = None
    aspects = []
    review_text = ""

    if request.method == 'POST':
        review_text = request.form.get('review', '').strip()

        if review_text:
            cleaned = clean_review(review_text)
            vec = tfidf.transform([cleaned]).toarray()

            # prediction (0 = negative, 1 = positive)
            pred = model.predict(vec)[0]

            # confidence score (if model supports predict_proba)
            try:
                proba = model.predict_proba(vec)[0]
                # probability of the predicted class
                confidence = float(np.max(proba) * 100)
            except Exception:
                confidence = None

            if pred == 1:
                prediction_label = "Positive"
                sentiment_class = "positive"
            else:
                prediction_label = "Negative"
                sentiment_class = "negative"

            aspects = extract_aspects(review_text)

            return render_template(
                'index.html',
                review=review_text,
                prediction_label=prediction_label,
                sentiment_class=sentiment_class,
                confidence=confidence,
                aspects=aspects
            )

    # GET request or empty form
    return render_template('index.html',
                           review=review_text,
                           prediction_label=prediction_label,
                           sentiment_class=None,
                           confidence=confidence,
                           aspects=aspects)


if __name__ == '__main__':
    app.run(debug=True)
