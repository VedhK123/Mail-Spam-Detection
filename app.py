from flask import Flask, request, jsonify
from flask_cors import CORS
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib

app = Flask(__name__)
CORS(app)  # This allows your Chrome Extension to talk to this API

# Load the saved model and vectorizer
clf = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(raw_text):
    """Replicates the exact preprocessing steps from training"""
    text = raw_text.replace('\r\n', ' ').lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get('text', '')

    if not email_text:
        return jsonify({'error': 'No text provided'}), 400

    # 1. Preprocess the incoming text
    clean_text = preprocess_text(email_text)

    # 2. Vectorize the text using the trained vocabulary
    vectorized_text = vectorizer.transform([clean_text]).toarray()

    # 3. Predict (0 = Ham, 1 = Spam)
    prediction = int(clf.predict(vectorized_text)[0])
    
    # Optional: Get probability score
    probabilities = clf.predict_proba(vectorized_text)[0]
    confidence = float(probabilities[prediction])

    return jsonify({
        'prediction': 'spam' if prediction == 1 else 'ham',
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)