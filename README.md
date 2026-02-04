
---

## 📊 Dataset Information

The dataset used for training is **large in size**, so it is hosted externally on Google Drive.

### 🔗 Dataset Download Link
👉 **[Click here to download the dataset](https://drive.google.com/drive/folders/1nOETzcds3C2dRVOpTE4qn2AUD3W0hIjL?usp=drive_link)**

> The dataset contains labeled news articles for Fake and Real news classification.

---

## 🧠 Machine Learning Approach

1. **Data Cleaning & Preprocessing**
   - Lowercasing text
   - Removing punctuation and stopwords

2. **Feature Extraction**
   - TF-IDF Vectorization (Term Frequency–Inverse Document Frequency)

3. **Model Training**
   - Linear Machine Learning Classifier
   - Train-test split for evaluation

4. **Model Evaluation**
   - Accuracy metric used to measure performance

5. **Model Saving**
   - Trained model and vectorizer saved as `.pkl` files using `pickle`

---

## ▶ How to Run the Project Locally

### 1️⃣ Download the Repository
Download ZIP from GitHub and extract it.

---

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
