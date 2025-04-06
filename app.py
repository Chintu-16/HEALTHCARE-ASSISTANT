from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python3 -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Sample disease prediction dictionary (Replace with ML model in future)
DISEASE_SYMPTOMS = {
    "flu": ["fever", "cough", "body ache", "fatigue"],
    "cold": ["sneezing", "runny nose", "sore throat"],
    "covid-19": ["fever", "cough", "shortness of breath", "loss of taste"],
    "diabetes": ["excessive thirst", "frequent urination", "blurred vision"]
}

def predict_disease(user_input):
    """ Simple rule-based prediction (Replace with ML model for better accuracy) """
    user_symptoms = set(user_input.lower().split())

    best_match = None
    max_match_count = 0

    for disease, symptoms in DISEASE_SYMPTOMS.items():
        match_count = len(user_symptoms.intersection(symptoms))
        if match_count > max_match_count:
            max_match_count = match_count
            best_match = disease

    return best_match if best_match else "No matching disease found"

@app.route("/")
def home():
    return jsonify({"message": "AI Healthcare Chatbot Backend Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """API Endpoint to predict disease from symptoms"""
    data = request.get_json()
    symptoms = data.get("symptoms", "")

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # NLP processing
    doc = nlp(symptoms)
    extracted_symptoms = " ".join([token.lemma_ for token in doc if token.is_alpha])

    # Predict disease
    predicted_disease = predict_disease(extracted_symptoms)

    return jsonify({"disease": predicted_disease})

if __name__ == "__main__":
    app.run(debug=True)
