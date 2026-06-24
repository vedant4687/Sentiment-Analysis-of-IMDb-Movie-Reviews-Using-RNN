import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def preprocess(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = stopwords.words("english")

    filtered = []

    for word in tokens:
        if word not in stop_words:
            filtered.append(word)

    ps = PorterStemmer()

    stemmed = []

    for word in filtered:
        stemmed.append(ps.stem(word))

    return " ".join(stemmed)