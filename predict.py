import torch
import pickle

from model import RNN
from preprocess import preprocess


# Load TF-IDF
with open("models/tfidf.pkl", "rb") as f:
    tf = pickle.load(f)


# Load model
model = RNN()

model.load_state_dict(
    torch.load(
        "models/rnn_model.pth",
        map_location="cpu"
    )
)

model.eval()


def predict_sentiment(review):

    review = preprocess(review)

    vector = tf.transform([review])

    x = torch.tensor(
        vector.toarray(),
        dtype=torch.float32
    ).unsqueeze(1)

    with torch.no_grad():

        output = model(x)

        probability = torch.sigmoid(
            output.squeeze()
        ).item()

    if probability > 0.46:
        return "Positive", probability

    else:
        return "Negative", 1 - probability