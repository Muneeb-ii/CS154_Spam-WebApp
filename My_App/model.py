# Python built-in libraries
import pickle
import string
from pathlib import Path

# Core Math and Visualization Libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# sci-kit (sklearn) is a Machine Learning Library
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# Change this based on the number of labels you have
def return_label(label):
    if label == "ham":
        return 1
    else:
        return 0


def preprocess_text(text: str) -> str:
    lowercase_text = text.lower()
    lowercase_text = lowercase_text.translate(str.maketrans("", "", string.punctuation))
    split_text = lowercase_text.split()

    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in split_text if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return " ".join(lemmatized_words)


def create_bag_of_words_model(preprocessed_X_train):
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(preprocessed_X_train)
    return vectorized_data, vectorizer


def create_tfidf_model(preprocessed_X_train):
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(preprocessed_X_train)
    return vectorized_data, vectorizer


def train_model(x_train, y_train):
    model = MultinomialNB()
    model.fit(x_train, y_train)
    return model


if __name__ == "__main__":
    # CWD stands for current working directory
    CWD = Path.cwd()
    APP = CWD.joinpath("My_App")
    DATA_DIR = APP.joinpath("data")
    dataset_filepath = DATA_DIR.joinpath("SMSSpamCollection.txt")

    data = pd.read_csv(dataset_filepath, encoding="utf-8", sep="\t")
    y = data["label"].apply(return_label)

    X_train, X_test, y_train, y_test = train_test_split(
        data["Message"],
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessed_X_train = X_train.apply(preprocess_text)
    preprocessed_X_test = X_test.apply(preprocess_text)

    x_train, vectorizer = create_bag_of_words_model(preprocessed_X_train)
    x_test = vectorizer.transform(preprocessed_X_test)

    x_train, vectorizer = create_bag_of_words_model(preprocessed_X_train)
    x_test = vectorizer.transform(preprocessed_X_test)

    model = train_model(x_train, y_train)

    MODEL_FILEPATH = APP.joinpath("model.pkl")
    VECTORIZER_FILEPATH = APP.joinpath("vectorizer.pkl")

    with open(MODEL_FILEPATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(VECTORIZER_FILEPATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
