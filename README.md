# 📧 Mail Spam Django — Malagasy Email Spam Detection

**Mail Spam Django** is a web application built with Django and Bootstrap that explores email messaging and AI-assisted spam detection for the Malagasy language.

The project integrates a machine learning model to classify email messages as either legitimate (*ham*) or spam, helping users identify potentially unwanted messages. It combines web development, machine learning, and natural language processing in a language-specific application.

## 🎯 Project Objectives

- Develop a web-based messaging or mailing application.
- Integrate machine learning into the email classification workflow.
- Explore automated spam detection for Malagasy-language messages.
- Provide a simple and accessible interface using Bootstrap.
- Investigate the application of natural language processing to a less-resourced language.

## ✨ Features

- 📬 **Email Messaging:** A web interface for working with email messages.
- 🤖 **AI-Assisted Spam Detection:** Use a machine learning model to classify messages as spam or legitimate.
- 🇲🇬 **Malagasy Language Focus:** Explore spam classification for emails written in Malagasy.
- 🖥️ **Web Interface:** Django templates styled with Bootstrap.
- 🔎 **Message Classification:** Support the identification and filtering of potentially unwanted messages.

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| Python | Main programming language |
| Django | Backend web framework |
| Bootstrap | User interface styling |
| Machine Learning | Email spam classification |
| Natural Language Processing (NLP) | Text analysis for spam detection |

The specific machine learning algorithm and its training configuration can be documented once confirmed from the source code.

## 🧠 Machine Learning Pipeline

The application uses a text classification pipeline based on **TF-IDF (Term Frequency–Inverse Document Frequency)** and **Logistic Regression** to detect spam in Malagasy-language emails.

### How It Works

1. **Text Input:** The application receives an email message.
2. **Text Vectorization:** TF-IDF converts the text into numerical features based on word importance within the message and across the dataset.
3. **Classification:** A Logistic Regression model uses these features to predict the message category.
4. **Prediction Output:** The application identifies the message as either spam or legitimate (*ham*).

### Technologies

| Component | Purpose |
|---|---|
| TF-IDF | Converts email text into numerical feature vectors |
| Logistic Regression | Classifies messages as spam or legitimate |
| Scikit-learn | Machine learning tools for vectorization and classification, if used in the implementation |

### Why TF-IDF and Logistic Regression?

This combination provides a practical baseline for text classification. TF-IDF represents the importance of terms in a document, while Logistic Regression learns to distinguish between spam and legitimate messages based on the resulting feature vectors.

The project applies this approach to Malagasy-language email classification, a use case that can be challenging when labeled datasets and language-processing resources are limited.

### Evaluation

Model performance should be assessed using metrics such as:

- **Precision:** How many messages predicted as spam are actually spam?
- **Recall:** How many actual spam messages are detected?
- **F1-score:** A balance between precision and recall.
- **Confusion matrix:** A breakdown of correct and incorrect classifications.

Actual evaluation results should be added once they have been measured on a suitable test dataset.

## 🌍 Why Malagasy Spam Detection?

Many NLP resources and spam-classification datasets focus on widely represented languages. This project explores a more language-specific use case by applying machine learning to Malagasy email messages.

Potential challenges include spelling variations, informal language, code-switching between Malagasy and other languages, and limited availability of labeled training data.

This makes the project an opportunity to explore machine learning beyond conventional English-language spam detection.

## 🚀 Getting Started

### Prerequisites

- Python
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/JoharyJonathan/Mail-spam-django.git
cd Mail-spam-django
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install Django and the project's other dependencies according to its configuration.

### 4. Configure the application

Check the Django settings and any required environment variables or model files before starting the application.

### 5. Run database migrations

```bash
python manage.py migrate
```

### 6. Start the development server

```bash
python manage.py runserver
```

Open the local development URL displayed by Django, typically `http://127.0.0.1:8000/`.

> **Note:** These are standard Django development steps. Additional configuration may be required depending on how the machine learning model and email functionality are integrated.

## 🔬 Potential Future Improvements

- Evaluate precision, recall, and F1-score on a labeled Malagasy email dataset.
- Analyze false positives and false negatives.
- Improve robustness to spelling variations and mixed-language messages.
- Expand and document the training dataset.
- Add automated tests for the classification pipeline.
- Improve the user experience for reviewing and filtering suspicious messages.

## 👨‍💻 Author

**Johary Jonathan**

- GitHub: [@JoharyJonathan](https://github.com/JoharyJonathan)
- Repository: [Mail-spam-django](https://github.com/JoharyJonathan/Mail-spam-django)

---

*An exploration of Django, machine learning, and spam detection for Malagasy-language email.*