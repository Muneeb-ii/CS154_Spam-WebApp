import string

import pickle
from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

CWD = Path.cwd()
APP = CWD.joinpath("My_App")
model_file = APP.joinpath("model.pkl")
vectorizer_file = APP.joinpath("vectorizer.pkl")

with open(model_file, "rb") as file:
    model = pickle.load(file)

with open(vectorizer_file, "rb") as file:
    vectorizer = pickle.load(file)

templates = Jinja2Templates(directory=APP.joinpath("templates"))


def preprocess_text(text: str) -> str:
    lowercase_text = text.lower()
    lowercase_text = lowercase_text.translate(str.maketrans("", "", string.punctuation))
    split_text = lowercase_text.split()

    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in split_text if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return " ".join(lemmatized_words)


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_sentiment(request: Request, text: str = Form(...)):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)[0]
    label = "Ham" if prediction == 1 else "Spam"
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "text": text, "prediction": label},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
