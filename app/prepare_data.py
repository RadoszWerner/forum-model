import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class DataPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        """
        Oczyszcza tekst z HTML, znaków specjalnych itp.
        """
        clean_column = re.sub('<.*?>', ' ', str(text))
        clean_column = re.sub('[^a-zA-Z0-9.]+', ' ', clean_column)
        tokenized_column = word_tokenize(clean_column)
        return tokenized_column

    def lemmatize_text(self, tokens):
        """
        Lematyzacja tokenów.
        """
        lemmatizer = WordNetLemmatizer()
        lemmatized_list = [lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized_list

    def remove_stopwords(self, tokens):
        """
        Usuwa stop słowa z listy tokenów.
        """
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in tokens if word not in stop_words]
        return filtered_words

    def preprocess_text(self, text):
        """
        Przetwarza pojedynczy string i zwraca przetworzony tekst.
        """
        cleaned = self.clean_text(text.lower())
        lemmatized = self.lemmatize_text(cleaned)
        processed = self.remove_stopwords(lemmatized)
        return ' '.join(processed)
