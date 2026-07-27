from flask import Flask, render_template, request
import pickle

from utils.preprocess import clean_text

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None
    confidence = None

    if request.method == 'POST':

        review = request.form['review']

        # Clean review
        cleaned_review = clean_text(review)

        # Convert review into vector
        vector_input = vectorizer.transform([cleaned_review])

        # Predict sentiment
        prediction = model.predict(vector_input)[0]

        # Predict probability
        confidence = model.predict_proba(vector_input).max() * 100

        confidence = round(confidence, 2)

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence
    )

if __name__ == '__main__':
    app.run(debug=True)