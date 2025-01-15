import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ToxicityModel:
    def __init__(self, model_path, tokenizer_path, max_len=200):
        self.model = load_model(model_path)
        self.max_len = max_len

        # Wczytaj tokenizator
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def preprocess_input(self, comment):
        # Tokenizuj tekst i dopasuj długość sekwencji
        sequences = self.tokenizer.texts_to_sequences([comment])
        return pad_sequences(sequences, maxlen=self.max_len)

    def predict_toxicity(self, comment):
        # Przeprowadź predykcję
        input_data = self.preprocess_input(comment)
        prediction = self.model.predict(input_data)
        # Zastosuj progowanie
        binary_prediction = (prediction >= 0.5).astype(int)
        return binary_prediction[0].tolist()
