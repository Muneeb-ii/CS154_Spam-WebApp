# SMS Spam Classifier Web App

This project packages a trained Naive Bayes classifier as a simple web application using FastAPI. The model classifies SMS messages as **spam** or **not spam** based on the `SMSSpamCollection` dataset. Built as part of **CS154**, an introductory Python course with NLP components.

---

## 📌 Features

- Trained Naive Bayes model with `TfidfVectorizer`
- Web interface built using **FastAPI**
- Accepts user input for real-time classification
- Local web server hosted using **Uvicorn**
- Exports and loads `.pkl` model and vectorizer files for production use

---

## 🧠 Reflections

This was a fairly straightforward project. Converting the notebook to Python scripts and setting up the FastAPI server went smoothly with no major hurdles.

---

## 🛠️ How to Use

### ✅ Prerequisites

Before running the web app:

1. Install required dependencies:
   ```bash
   pip install fastapi uvicorn python-multipart scikit-learn pandas
   ```

2. Run the model script to generate serialized files:
   ```bash
   python model.py
   ```
   This creates:
   - `model.pkl` — the trained classifier
   - `vectorizer.pkl` — the fitted TfidfVectorizer

3. Launch the web app:
   ```bash
   python app/main.py
   ```

4. Open your browser and visit:
   ```
   http://0.0.0.0:8000
   ```

---

## 🧪 Dataset

- **SMSSpamCollection**: A labeled dataset for binary classification of SMS messages into spam or ham.

---

## 🧰 Files Overview

- `model.py`: Converts the Jupyter Notebook into a trainable pipeline and saves the model
- `app/main.py`: FastAPI web interface for real-time classification
- `model.pkl`, `vectorizer.pkl`: Serialized model and vectorizer
- `index.html`: HTML interface served via FastAPI

---

## 📚 References

- [W3Schools CSS Reference](https://www.w3schools.com/css/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [How to Deploy a Machine Learning Model to the Web](https://blog.bolajiayodeji.com/how-to-deploy-a-machine-learning-model-to-the-web)

---

## 🪪 License

This project is licensed under the MIT License.
