import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
CORS(app)

class ToxicityModel:
    def __init__(self, model_path, tokenizer_path, max_len=200):
        """
        Inicjalizacja modelu i tokenizatora.
        :param model_path: Ścieżka do zapisanego modelu Keras.
        :param tokenizer_path: Ścieżka do zapisanego tokenizatora (pickle).
        :param max_len: Maksymalna długość sekwencji.
        """
        self.model = load_model(model_path)
        self.max_len = max_len

        # Wczytaj tokenizator
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def preprocess_input(self, comment):
        """
        Tokenizuje i normalizuje tekst wejściowy.
        :param comment: Tekst komentarza.
        :return: Padded sequence jako wejście dla modelu.
        """
        # Tokenizuj tekst
        sequences = self.tokenizer.texts_to_sequences([comment])
        # Paduj sekwencje do odpowiedniej długości
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        return padded_sequences

    def predict_toxicity(self, comment):
        """
        Przeprowadza predykcję toksyczności dla komentarza.
        :param comment: Tekst komentarza.
        :return: Wynik predykcji (np. prawdopodobieństwo toksyczności).
        """
        # input_data = self.preprocess_input(comment)
        # prediction = self.model.predict(input_data)
        # return prediction[0].tolist()  # Zwróć wynik jako listę (JSON-friendly)
        input_data = self.preprocess_input(comment)
        prediction = self.model.predict(input_data)
        # Zastosowanie progowania do predykcji binarnej
        binary_prediction = (prediction >= 0.5).astype(int)  # Zamiana na 0/1
        return binary_prediction[0].tolist()

# Wczytanie modelu i tokenizatora
model = ToxicityModel('model/best_model_lstm.keras', 'model/tokenizer.pkl')

@app.route('/api/check_toxicity', methods=['POST'])
def check_toxicity():
    try:
        # Pobieranie komentarza z żądania
        data = request.get_json()
        comment = data.get('comment')
        if not comment:
            return jsonify({'error': 'Comment is required'}), 400

        # Analiza toksyczności
        toxicity_score = model.predict_toxicity(comment)
        return jsonify({'comment': comment, 'toxicity_score': toxicity_score}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
