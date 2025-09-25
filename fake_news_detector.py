#!/usr/bin/env python3
"""
Fake News Detector (REAL / FAKE)

Использует:
- TfidfVectorizer (sklearn)
- PassiveAggressiveClassifier (sklearn)

Датасет должен содержать колонки: title, text, label
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os


# Загружаем датасет
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]  # на всякий случай
    if not {'title', 'text', 'label'}.issubset(df.columns):
        raise ValueError("CSV должен содержать столбцы: title, text, label")
    # Объединяем заголовок и текст в одно поле
    df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
    df['label'] = df['label'].str.upper().str.strip()
    return df[['content', 'label']]


# Обучение модели
def train_model(df: pd.DataFrame):
    X = df['content']
    y = df['label']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Преобразуем текст → TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Классификатор
    clf = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # Предсказание
    y_pred = clf.predict(X_test_tfidf)

    # Метрики
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (acc={acc:.2%})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    path = os.path.join(script_dir, "traindata.csv")
    df = load_dataset(path)
    train_model(df)
