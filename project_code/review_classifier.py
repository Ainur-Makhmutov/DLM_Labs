import re
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import pymorphy2
from catboost import CatBoostClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

class TextPreprocessor:
    """Очистка, нормализация и лемматизация русскоязычных текстов"""

    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

        self.stop_words = set(
            ['это', 'как', 'так', 'и', 'в', 'на', 'с', 'по', 'к', 'у', 'же', 'не', 'то', 'все', 'было', 'но', 'за',
             'из', 'о', 'от', 'для', 'без', 'до', 'при', 'через', 'тоже', 'еще', 'только', 'очень', 'быть', 'весь',
             'или', 'иметь', 'более', 'чем', 'можно', 'нужно', 'такой', 'там', 'тут', 'где', 'когда', 'тогда', 'теперь',
             'уже', 'потом', 'сразу', 'опять', 'вдруг', 'совсем', 'около', 'вокруг', 'будто', 'словно', 'прямо', 'даже',
             'неужели', 'разве', 'какой', 'который'])

    def clean_text(self, text):
        """Удаление шумов: эмодзи, спецсимволы, лишние пробелы, приведение к нижнему регистру"""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s\u0400-\u052f,.\-!?]', ' ', text)
        text = re.sub(r'[^\w\s\u0400-\u052f!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def lemmatize(self, text):
        """Лемматизация слов с помощью pymorphy2"""
        words = text.split()
        lemmas = []

        for word in words:
            if word in self.stop_words:
                continue

            p = self.morph.parse(word)[0]
            lemma = p.normal_form
            lemmas.append(lemma)

        return ' '.join(lemmas)

    def preprocess(self, text):
        """Полный цикл предобработки"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        cleaned = self.clean_text(text)
        lemmatized = self.lemmatize(cleaned)

        return lemmatized


class AspectExtractor:
    """Извлечение бинарных признаков наличия упоминаний по 4 аспектам на основе ключевых слов"""

    def __init__(self):
        self.aspect_keywords = {
            'graphics': ['графика', 'текстуры', 'визуал', 'картинка', 'красиво', 'графон',
                         'пейзаж', 'анимация', 'модели'],
            'performance': ['фпс', 'оптимизация', 'лаги', 'вылетает', 'баги', 'тормозит',
                            '30 фпс', 'производительность', 'зависание', 'фризы'],
            'story': ['сюжет', 'история', 'диалоги', 'концовка', 'персонажи', 'квесты',
                      'трогательная', 'повествование', 'нарратив'],
            'gameplay': ['боевка', 'геймплей', 'механика', 'заказы', 'гвинт', 'прокачка',
                         'однотипно', 'управление', 'босс']
        }

        self.aspect_regex = {}

        for aspect, keywords in self.aspect_keywords.items():
            pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.aspect_regex[aspect] = re.compile(pattern, re.IGNORECASE | re.UNICODE)

    def extract(self, text):
        """Возвращает словарь с бинарными флагами для каждого аспекта"""
        result = {}

        for aspect, regex in self.aspect_regex.items():
            result[aspect] = 1 if regex.search(text) else 0

        return result

class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts

        if hasattr(labels, 'values'):
            self.labels = labels.values
        else:
            self.labels = np.array(labels)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ReviewClassifier:
    def __init__(self, csv_path, sep=','):
        self.csv_path = csv_path
        self.sep = sep
        self.df = None
        self.text_preprocessor = TextPreprocessor()
        self.aspect_extractor = AspectExtractor()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

        self.aspect_models = {}
        self.sentiment_model = None
        self.sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}

        self.rule_based_predictions = None

    def load_and_prepare_data(self):
        print("Загрузка данных...")
        self.df = pd.read_csv(self.csv_path, sep=self.sep, encoding='utf-8')
        print(f"Исходное количество записей: {len(self.df)}")
        if 'language' in self.df.columns:
            self.df = self.df[self.df['language'] == 'russian']
        # print(f"После фильтрации по языку: {len(self.df)}")
        self.df = self.df.dropna(subset=['review'])
        self.df = self.df.reset_index(drop=True)
        print("Предобработка текстов (лемматизация, очистка)...")
        self.df['clean_review'] = self.df['review'].apply(self.text_preprocessor.preprocess)
        self.df = self.df[self.df['clean_review'].str.len() > 0]
        self.df = self.df.reset_index(drop=True)
        print(f"После предобработки: {len(self.df)}")

        print("Извлечение аспектов регулярными выражениями...")
        aspect_features = self.df['review'].apply(self.aspect_extractor.extract)
        self.df['aspect_graphics'] = aspect_features.apply(lambda x: x['graphics'])
        self.df['aspect_performance'] = aspect_features.apply(lambda x: x['performance'])
        self.df['aspect_story'] = aspect_features.apply(lambda x: x['story'])
        self.df['aspect_gameplay'] = aspect_features.apply(lambda x: x['gameplay'])

        self._prepare_sentiment_labels()

    def _prepare_sentiment_labels(self):
        pos_markers = ['хорош', 'отличн', 'класс', 'бомба', 'рекомендую', 'шикарн', 'нравится', 'шедевр']
        neg_markers = ['скучно', 'худш', 'бесит', 'разочарован', 'не рекомендую', 'ужасно', 'баг', 'вылета']

        def get_sentiment(row):
            text = row['review'].lower()
            rec = row.get('recommended', None)
            pos_score = sum(1 for m in pos_markers if m in text)
            neg_score = sum(1 for m in neg_markers if m in text)
            if pos_score > neg_score and pos_score > 0:
                return 'positive'
            elif neg_score > pos_score and neg_score > 0:
                return 'negative'
            else:
                if rec == True:
                    return 'positive'
                elif rec == False:
                    return 'negative'
                else:
                    return 'negative'

        self.df['sentiment'] = self.df.apply(get_sentiment, axis=1)
        self.df['sentiment_label'] = self.df['sentiment'].map(self.sentiment_map)

    def rule_based_classify(self):
        print("\n=== Rule-based модель (регулярные выражения) ===")
        aspect_cols = ['aspect_graphics', 'aspect_performance', 'aspect_story', 'aspect_gameplay']
        return self.df[aspect_cols], self.df['sentiment_label']

    def train_logreg(self, test_size=0.2):
        self._train_ml_model('logreg', test_size)

    def train_catboost(self, test_size=0.2):
        self._train_ml_model('catboost', test_size)

    def train_rubert(self, test_size=0.2, max_length=128, batch_size=16, epochs=3):
        print("\n=== Обучение RuBERT ===")

        texts = self.df['review'].astype(str).values
        aspect_cols = ['aspect_graphics', 'aspect_performance', 'aspect_story', 'aspect_gameplay']


        X_train_text, X_test_text, y_train_aspects, y_test_aspects = train_test_split(
            texts, self.df[aspect_cols], test_size=test_size, random_state=42, stratify=self.df['sentiment_label']
        )
        _, _, y_train_sent, y_test_sent = train_test_split(
            texts, self.df['sentiment_label'], test_size=test_size, random_state=42, stratify=self.df['sentiment_label']
        )

        model_name = "./rubert_model"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        train_dataset = ReviewsDataset(X_train_text, y_train_sent.values, tokenizer, max_length)
        test_dataset = ReviewsDataset(X_test_text, y_test_sent.values, tokenizer, max_length)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        training_args = TrainingArguments(
            output_dir='./rubert_results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",  # <-- было evaluation_strategy
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self._compute_metrics
        )

        trainer.train()
        self.rubert_model = model
        self.rubert_tokenizer = tokenizer

        preds = trainer.predict(test_dataset)
        y_pred = np.argmax(preds.predictions, axis=1)
        print("\nRuBERT - тональность:")
        print(classification_report(y_test_sent, y_pred, labels=[0, 2],
                                    target_names=['negative', 'positive'], zero_division=0))


        self._train_rubert_for_aspects(X_train_text, X_test_text, y_train_aspects, y_test_aspects, tokenizer,
                                       max_length, batch_size, epochs)

    def _train_rubert_for_aspects(self, X_train, X_test, y_train, y_test, tokenizer, max_length, batch_size, epochs):
        """Обучает отдельную модель RuBERT для каждого аспекта (бинарная классификация)"""
        aspect_names = ['graphics', 'performance', 'story', 'gameplay']
        self.rubert_aspect_models = {}

        for aspect in aspect_names:
            print(f"\nОбучение RuBERT для аспекта {aspect}...")
            col_name = f'aspect_{aspect}'
            y_train_aspect = y_train[col_name].values
            y_test_aspect = y_test[col_name].values

            train_ds = ReviewsDataset(X_train, y_train_aspect, tokenizer, max_length)
            test_ds = ReviewsDataset(X_test, y_test_aspect, tokenizer, max_length)

            model = AutoModelForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased', num_labels=2)
            training_args = TrainingArguments(
                output_dir=f'./rubert_{aspect}',
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy="epoch",
                save_strategy="no",
                logging_dir='./logs',
                logging_steps=10,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                tokenizer=tokenizer,
                compute_metrics=self._compute_metrics_binary
            )
            trainer.train()
            self.rubert_aspect_models[aspect] = trainer

            # Оценка
            preds = trainer.predict(test_ds)
            y_pred = np.argmax(preds.predictions, axis=1)
            print(f"Результаты для {aspect}:")
            print(classification_report(y_test_aspect, y_pred, zero_division=0))

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}

    def _compute_metrics_binary(self, eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {'accuracy': accuracy_score(labels, preds)}

    def _train_ml_model(self, model_type, test_size=0.2):
        print(f"\n=== Обучение {model_type.upper()} ===")
        X_tfidf = self.tfidf_vectorizer.fit_transform(self.df['clean_review'])
        aspect_columns = ['aspect_graphics', 'aspect_performance', 'aspect_story', 'aspect_gameplay']
        y_aspects = self.df[aspect_columns]
        y_sentiment = self.df['sentiment_label']

        X_train, X_test, y_train_aspects, y_test_aspects = train_test_split(
            X_tfidf, y_aspects, test_size=test_size, random_state=42
        )
        _, _, y_train_sent, y_test_sent = train_test_split(
            X_tfidf, y_sentiment, test_size=test_size, random_state=42, stratify=y_sentiment
        )

        if model_type == 'logreg':
            base_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        elif model_type == 'catboost':
            base_clf = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6,
                                          verbose=False, random_seed=42, class_weights=[1, 1, 1])  # позже подстроим
        else:
            raise ValueError("Неизвестный тип модели")

        self.aspect_models = {}
        for col in aspect_columns:
            aspect_name = col.replace('aspect_', '')
            if model_type == 'catboost':
                clf = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6, verbose=False, random_seed=42)
            else:
                clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            clf.fit(X_train, y_train_aspects[col])
            self.aspect_models[aspect_name] = clf


        if model_type == 'catboost':
            self.sentiment_model = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6,
                                                      verbose=False, random_seed=42)
        else:
            self.sentiment_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        self.sentiment_model.fit(X_train, y_train_sent)


        print(f"\nОценка {model_type.upper()} (аспекты):")
        for aspect_name, clf in self.aspect_models.items():
            y_pred = clf.predict(X_test)
            y_true = y_test_aspects[f'aspect_{aspect_name}']
            print(f"{aspect_name}: F1 = {f1_score(y_true, y_pred, zero_division=0):.4f}")

        y_pred_sent = self.sentiment_model.predict(X_test)
        print("\nТональность:")
        print(classification_report(y_test_sent, y_pred_sent, labels=[0, 2],
                                    target_names=['negative','positive'], zero_division=0))

    def save_enhanced_dataset(self, output_path, model_type='logreg', include_predictions=False):
        """Сохраняет обогащённый датасет в CSV, оставляя только строки, где есть хотя бы один аспект."""

        base_columns = [
            'review', 'clean_review',
            'aspect_graphics', 'aspect_performance', 'aspect_story', 'aspect_gameplay',
            'sentiment'
        ]

        # Фильтруем строки: оставляем только те, где хотя бы один аспект = 1
        aspect_cols = ['aspect_graphics', 'aspect_performance', 'aspect_story', 'aspect_gameplay']
        mask = (self.df[aspect_cols].sum(axis=1) > 0)
        df_filtered = self.df[mask].copy()

        if include_predictions:
            if model_type == 'rubert':
                if not hasattr(self, 'rubert_aspect_models'):
                    raise ValueError("Сначала обучите RuBERT методом train_rubert()")
                texts = df_filtered['review'].astype(str).values
                for aspect, trainer in self.rubert_aspect_models.items():
                    dataset = ReviewsDataset(texts, [0] * len(texts), self.rubert_tokenizer, 128)
                    preds = trainer.predict(dataset)
                    df_filtered[f'pred_{aspect}'] = np.argmax(preds.predictions, axis=1)
                sent_dataset = ReviewsDataset(texts, [0] * len(texts), self.rubert_tokenizer, 128)
                sent_preds = self.rubert_model.predict(sent_dataset)
                sent_labels = np.argmax(sent_preds.predictions, axis=1)
                inv_map = {v: k for k, v in self.sentiment_map.items()}
                df_filtered['pred_sentiment'] = [inv_map[l] for l in sent_labels]
            else:
                X_all = self.tfidf_vectorizer.transform(df_filtered['clean_review'])
                for aspect, clf in self.aspect_models.items():
                    df_filtered[f'pred_{aspect}'] = clf.predict(X_all)
                sent_pred = self.sentiment_model.predict(X_all)
                inv_map = {v: k for k, v in self.sentiment_map.items()}
                df_filtered['pred_sentiment'] = [inv_map[p] for p in sent_pred]

            pred_columns = [f'pred_{aspect}' for aspect in ['graphics', 'performance', 'story', 'gameplay']] + [
                'pred_sentiment']
            columns_to_save = base_columns + pred_columns
        else:
            columns_to_save = base_columns

        df_filtered[columns_to_save].to_csv(output_path, index=False, encoding='utf-8')
        print(f"Обогащённый датасет сохранён в {output_path}")



if __name__ == "__main__":
    DATA_PATH = "Data/Raw/steam_reviews_russian_balanced.csv"

    classifier = ReviewClassifier(DATA_PATH, sep=',')
    classifier.load_and_prepare_data()

    # 1. Rule-based
    rule_preds, rule_sent = classifier.rule_based_classify()

    # 2. Логистическая регрессия
    classifier.train_logreg(test_size=0.2)
    classifier.save_enhanced_dataset("Data/Processed/steam_reviews_logreg.csv", model_type='logreg')

    # 3. CatBoost
    classifier.train_catboost(test_size=0.2)
    classifier.save_enhanced_dataset("Data/Processed/steam_reviews_catboost.csv", model_type='catboost')

