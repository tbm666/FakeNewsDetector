#!/usr/bin/env python3
"""
Fake News Detector - Модель для классификации новостей (REAL/FAKE)
Поиск датасета в корневой директории скрипта
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_curve, auc)
import re
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        
    def find_dataset(self):
        """Поиск датасета в корневой директории скрипта"""
        print("🔍 Поиск датасета в корневой директории...")
        
        # Получаем абсолютный путь к директории скрипта
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"📁 Текущая директория: {script_dir}")
        
        # Ищем CSV файлы
        csv_files = glob.glob(os.path.join(script_dir, "*.csv"))
        
        if not csv_files:
            # Покажем все файлы в директории для отладки
            all_files = glob.glob(os.path.join(script_dir, "*"))
            print("📁 Содержимое директории:")
            for file in all_files:
                file_name = os.path.basename(file)
                print(f"   - {file_name}")
            raise FileNotFoundError("Не найдено CSV файлов в директории!")
        
        print("📋 Найдены CSV файлы:")
        for file in csv_files:
            file_name = os.path.basename(file)
            file_size = os.path.getsize(file) / 1024  # размер в KB
            print(f"   - {file_name} ({file_size:.1f} KB)")
        
        # Выбираем самый подходящий файл
        preferred_names = ['traindata.csv', 'train.csv', 'data.csv', 'dataset.csv']
        for preferred in preferred_names:
            for file in csv_files:
                if os.path.basename(file).lower() == preferred:
                    print(f"✅ Выбран файл: {preferred}")
                    return file
        
        # Если не нашли предпочтительные, берем первый CSV
        selected_file = csv_files[0]
        print(f"✅ Автовыбор файла: {os.path.basename(selected_file)}")
        return selected_file
    
    def load_and_prepare_data(self, file_path=None):
        """Загрузка и подготовка данных"""
        if file_path is None:
            file_path = self.find_dataset()
            
        print(f"📊 Загрузка данных из {os.path.basename(file_path)}...")
        try:
            # Пробуем разные кодировки
            encodings = ['utf-8', 'latin-1', 'windows-1251', 'cp1251']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ Успешная загрузка с кодировкой: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                # Последняя попытка с обработкой ошибок
                df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
                print("⚠️ Загрузка с игнорированием ошибок кодировки")
                
            print(f"✅ Данные загружены: {len(df)} записей, {len(df.columns)} колонок")
            print(f"📋 Колонки: {list(df.columns)}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return None
        
        # Стандартизация названий колонок
        df.columns = [col.lower().strip() for col in df.columns]
        print(f"📋 Стандартизированные колонки: {list(df.columns)}")
        
        # Покажем немного данных для отладки
        print("\n📊 Первые 3 строки данных:")
        print(df.head(3))
        
        # Определение колонок с текстом и метками
        text_columns = []
        label_column = None
        
        # Поиск колонки с метками
        label_keywords = ['label', 'class', 'target', 'is_fake', 'type', 'category']
        for col in df.columns:
            if any(keyword in col for keyword in label_keywords):
                label_column = col
                break
        
        if not label_column:
            # Попробуем найти колонку с двумя уникальными значениями
            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2:
                    label_column = col
                    print(f"🔍 Найдена колонка с метками (2 значения): {col}")
                    print(f"   Значения: {unique_vals}")
                    break
        
        # Поиск текстовых колонок
        text_keywords = ['text', 'content', 'article', 'title', 'news', 'message', 'headline']
        for col in df.columns:
            if col != label_column and df[col].dtype == 'object':
                if any(keyword in col for keyword in text_keywords):
                    text_columns.append(col)
        
        if not text_columns:
            # Возьмем первую текстовую колонку
            for col in df.columns:
                if col != label_column and df[col].dtype == 'object':
                    text_columns.append(col)
                    break
        
        if not label_column or not text_columns:
            print("❌ Не удалось определить колонки автоматически")
            print("📊 Статистика по колонкам:")
            for col in df.columns:
                print(f"   - {col}: {df[col].dtype}, уникальных: {df[col].nunique()}")
                if df[col].dtype == 'object':
                    sample = df[col].iloc[0] if len(df) > 0 else "N/A"
                    print(f"     пример: {str(sample)[:50]}...")
            raise ValueError("Необходимо указать колонки вручную")
        
        print(f"🏷️ Колонка с метками: {label_column}")
        print(f"📝 Колонки с текстом: {text_columns}")
        
        # Объединение текстовых колонок
        df['content'] = ''
        for col in text_columns:
            df['content'] += ' ' + df[col].astype(str)
        
        # Очистка и стандартизация меток
        df['label'] = df[label_column].astype(str).str.upper().str.strip()
        
        # Приведение к стандартным значениям
        label_mapping = {
            'FAKE': 'FAKE', 'FALSE': 'FAKE', '0': 'FAKE', 'F': 'FAKE', 'FAKE NEWS': 'FAKE',
            'REAL': 'REAL', 'TRUE': 'REAL', '1': 'REAL', 'R': 'REAL', 'REAL NEWS': 'REAL'
        }
        
        df['label'] = df['label'].map(label_mapping)
        
        # Если остались неизвестные метки, спросим пользователя
        unknown_labels = df[~df['label'].isin(['FAKE', 'REAL'])]['label'].unique()
        if len(unknown_labels) > 0:
            print(f"⚠️ Обнаружены неизвестные метки: {unknown_labels}")
            print("🔄 Приведение к FAKE/REAL...")
            # Автоматически определяем по первому символу
            for label in unknown_labels:
                if label and label[0] in ['F', '0']:
                    df.loc[df['label'] == label, 'label'] = 'FAKE'
                elif label and label[0] in ['R', '1', 'T']:
                    df.loc[df['label'] == label, 'label'] = 'REAL'
        
        print(f"📊 Распределение меток после обработки:")
        print(df['label'].value_counts())
        
        # Удаляем строки с пустыми метками
        initial_count = len(df)
        df = df[df['label'].isin(['FAKE', 'REAL'])]
        final_count = len(df)
        
        if final_count < initial_count:
            print(f"🗑️ Удалено {initial_count - final_count} строк с некорректными метками")
        
        return df[['content', 'label']]
    
    def preprocess_text(self, text):
        """Базовая очистка текста"""
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)  # Удаляем пунктуацию
            text = re.sub(r'\s+', ' ', text).strip()  # Убираем лишние пробелы
            return text
        return ""
    
    def create_visualizations(self, df):
        """Создание визуализаций данных"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📈 Анализ данных новостей', fontsize=16, fontweight='bold')
        
        # 1. Распределение классов
        label_counts = df['label'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4']
        axes[0,0].pie(label_counts.values, labels=label_counts.index, 
                     autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0,0].set_title('Распределение классов новостей')
        
        # 2. Длина текста по классам
        df['text_length'] = df['content'].str.len()
        sns.boxplot(data=df, x='label', y='text_length', ax=axes[0,1], palette=colors)
        axes[0,1].set_title('Длина текста по классам')
        axes[0,1].set_ylabel('Длина текста')
        axes[0,1].set_xlabel('Класс')
        
        # 3. Количество слов по классам
        df['word_count'] = df['content'].str.split().str.len()
        sns.histplot(data=df, x='word_count', hue='label', 
                    ax=axes[1,0], palette=colors, alpha=0.7)
        axes[1,0].set_title('Распределение количества слов')
        axes[1,0].set_xlabel('Количество слов')
        
        # 4. Примеры текстов
        axes[1,1].text(0.1, 0.9, "Примеры данных:", fontsize=12, fontweight='bold')
        
        fake_sample = df[df['label']=='FAKE'].iloc[0]['content'] if len(df[df['label']=='FAKE']) > 0 else "Нет данных"
        real_sample = df[df['label']=='REAL'].iloc[0]['content'] if len(df[df['label']=='REAL']) > 0 else "Нет данных"
        
        axes[1,1].text(0.1, 0.7, f"FAKE: {str(fake_sample)[:100]}...", 
                      fontsize=9, color='red')
        axes[1,1].text(0.1, 0.5, f"REAL: {str(real_sample)[:100]}...", 
                      fontsize=9, color='green')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def train_model(self, df):
        """Обучение модели"""
        print("🎯 Обучение модели...")
        
        # Предобработка текста
        df['content_clean'] = df['content'].apply(self.preprocess_text)
        X = df['content_clean']
        y = df['label']
        
        # Проверяем, что есть оба класса
        if len(y.unique()) < 2:
            raise ValueError("Необходимы оба класса (FAKE и REAL) для обучения")
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📚 Данные для обучения: {len(X_train)} записей")
        print(f"📚 Данные для теста: {len(X_test)} записей")
        
        # TF-IDF векторизация
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.7,
            min_df=2,
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        
        # Обучение векторизатора
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"🔡 Создано признаков TF-IDF: {X_train_tfidf.shape[1]}")
        
        # Обучение PassiveAggressiveClassifier
        self.model = PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=42,
            C=0.5,
            early_stopping=True
        )
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X_train_tfidf, y_train, cv=5)
        print(f"✅ Кросс-валидация (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return X_test_tfidf, y_test, X_train_tfidf, y_train
    
    def evaluate_model(self, X_test, y_test, X_train, y_train):
        """Оценка модели и визуализация результатов"""
        print("📊 Оценка модели...")
        
        # Предсказания
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.decision_function(X_test)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 ТОЧНОСТЬ МОДЕЛИ: {accuracy:.2%}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Визуализация результатов
        self._plot_results(y_test, y_pred, y_pred_proba, accuracy)
        
        # Анализ важности признаков
        self._plot_feature_importance()
        
        return accuracy
    
    def _plot_results(self, y_test, y_pred, y_pred_proba, accuracy):
        """Визуализация результатов классификации"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Результаты классификации (Accuracy: {accuracy:.2%})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Матрица ошибок
        cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'],
                   ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Предсказанный класс')
        axes[0,0].set_ylabel('Реальный класс')
        
        # 2. ROC кривая (если есть вероятности)
        try:
            fpr, tpr, _ = roc_curve((y_test == 'REAL').astype(int), y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                          label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0,1].set_xlim([0.0, 1.0])
            axes[0,1].set_ylim([0.0, 1.05])
            axes[0,1].set_xlabel('False Positive Rate')
            axes[0,1].set_ylabel('True Positive Rate')
            axes[0,1].set_title('ROC Curve')
            axes[0,1].legend(loc="lower right")
            axes[0,1].grid(True)
        except Exception as e:
            axes[0,1].text(0.5, 0.5, "ROC curve not available", 
                          ha='center', va='center', fontsize=12)
            axes[0,1].set_title('ROC Curve')
        
        # 3. Распределение confidence scores
        if len(np.unique(y_pred_proba)) > 1:
            axes[1,0].hist(y_pred_proba[y_test == 'REAL'], bins=30, alpha=0.7, 
                          label='REAL', color='green')
            axes[1,0].hist(y_pred_proba[y_test == 'FAKE'], bins=30, alpha=0.7, 
                          label='FAKE', color='red')
            axes[1,0].set_xlabel('Confidence Score')
            axes[1,0].set_ylabel('Частота')
            axes[1,0].set_title('Распределение Confidence Scores')
            axes[1,0].legend()
            axes[1,0].grid(True)
        else:
            axes[1,0].text(0.5, 0.5, "Confidence scores not available", 
                          ha='center', va='center', fontsize=12)
            axes[1,0].set_title('Confidence Scores')
        
        # 4. Сравнение метрик
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['FAKE', 'REAL']
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in classes]
            axes[1,1].bar(x + i*width, values, width, label=metric)
        
        axes[1,1].set_xlabel('Классы')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_title('Метрики по классам')
        axes[1,1].set_xticks(x + width)
        axes[1,1].set_xticklabels(classes)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_importance(self, top_n=20):
        """Визуализация самых важных признаков"""
        if hasattr(self.model, 'coef_'):
            feature_importance = np.abs(self.model.coef_[0])
            top_indices = np.argsort(feature_importance)[-top_n:]
            top_features = self.feature_names[top_indices]
            top_scores = feature_importance[top_indices]
            
            plt.figure(figsize=(10, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            
            bars = plt.barh(range(len(top_features)), top_scores, color=colors)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Важность признака')
            plt.title(f'Топ-{top_n} самых важных слов для классификации')
            
            # Добавляем значения на бары
            for bar, score in zip(bars, top_scores):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.show()

def main():
    """Основная функция"""
    detector = FakeNewsDetector()
    
    try:
        # Загрузка данных
        df = detector.load_and_prepare_data()
        
        if df is not None and len(df) > 0:
            # Визуализация данных
            df = detector.create_visualizations(df)
            
            # Обучение модели
            X_test, y_test, X_train, y_train = detector.train_model(df)
            
            # Оценка модели
            accuracy = detector.evaluate_model(X_test, y_test, X_train, y_train)
            
            # Проверка достижения целевой точности
            if accuracy >= 0.90:
                print(f"🎉 ЦЕЛЬ ДОСТИГНУТА! Точность: {accuracy:.2%} > 90%")
            else:
                print(f"⚠️ Точность {accuracy:.2%} ниже целевой 90%")
        else:
            print("❌ Не удалось загрузить данные или данные пустые")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()