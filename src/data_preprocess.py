"""
資料前處理模組 (Data Preprocessing Module)

負責：
  - 文字清理（小寫、去標點、去特殊符號）
  - 詞彙標準化
  - 斷詞 (Tokenization)
  - 停用詞移除
  - TF-IDF 向量化（Word n-grams + Character n-grams）
  - 前處理後資料儲存

Usage:
    from src.data_preprocess import TextPreprocessor, TfidfFeatureBuilder

    preprocessor = TextPreprocessor()
    clean_text = preprocessor.transform("i need you to book me a flight")

    builder = TfidfFeatureBuilder()
    X_train, X_eval = builder.fit_transform(train_texts, eval_texts)
"""

import json
import re
import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


# ============================================================
# 停用詞列表
# ============================================================
ENGLISH_STOP_WORDS = frozenset([
    # 人稱代名詞
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
    'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself',
    'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves',
    # 疑問詞 / 指示詞
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those',
    # Be 動詞
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    # Have / Do 動詞
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    # 冠詞 / 連接詞
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while',
    # 介係詞
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under',
    # 副詞 / 其他
    'again', 'further', 'then', 'once',
])


# ============================================================
# TextPreprocessor：文字前處理器
# ============================================================
class TextPreprocessor:
    """
    文字前處理器

    處理流程：
        1. 小寫轉換 (Lowercasing)
        2. 移除標點符號與特殊字元 (Punctuation Removal)
        3. 空白正規化 (Whitespace Normalization)
        4. 斷詞 — 空白分割 (Whitespace Tokenization)
        5. 停用詞移除 (Stopword Removal)
        6. 重組為字串 (Rejoin)

    斷詞說明：
        本資料集為英文短句，英文天然以空白分隔詞彙，
        因此使用 str.split() 空白斷詞即可。
        若處理中文，需改用 jieba / ckiptagger 等工具。
    """

    def __init__(self, stop_words: Optional[frozenset] = None, remove_stopwords: bool = True):
        self.stop_words = stop_words or ENGLISH_STOP_WORDS
        self.remove_stopwords = remove_stopwords

    def transform(self, text: str, verbose: bool = False) -> str:
        """對單筆文字進行前處理"""
        if verbose:
            print(f"  [原始] {text}")

        # Step 1: 小寫轉換
        text = text.lower()

        # Step 2: 移除標點符號（正則：非英數字且非空白 → 替換為空白）
        text = re.sub(r'[^\w\s]', ' ', text)

        # Step 3: 壓縮多餘空白
        text = re.sub(r'\s+', ' ', text).strip()

        # Step 4: 斷詞（英文：空白分割）
        tokens = text.split()
        if verbose:
            print(f"  [斷詞] {tokens}")

        # Step 5: 停用詞移除
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
            if verbose:
                print(f"  [去停用詞] {tokens}")

        # Step 6: 重組
        result = ' '.join(tokens)
        if verbose:
            print(f"  [結果] {result}")
        return result

    def transform_batch(self, texts: List[str]) -> List[str]:
        """批次前處理"""
        return [self.transform(t) for t in texts]


# ============================================================
# TfidfFeatureBuilder：TF-IDF 特徵建構器
# ============================================================
class TfidfFeatureBuilder:
    """
    TF-IDF 特徵建構器

    雙層 TF-IDF 策略：
        1. Word-level TF-IDF (unigram + bigram)
           → 捕捉詞彙與片語語義
        2. Character-level TF-IDF (3-gram ~ 5-gram)
           → 捕捉拼寫模式、字尾變化、子詞資訊

    兩者水平拼接為最終特徵矩陣。
    """

    def __init__(
        self,
        word_max_features: int = 50000,
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_min_df: int = 2,
        char_max_features: int = 50000,
        char_ngram_range: Tuple[int, int] = (3, 5),
    ):
        self.tfidf_word = TfidfVectorizer(
            max_features=word_max_features,
            ngram_range=word_ngram_range,
            min_df=word_min_df,
            sublinear_tf=True,
        )
        self.tfidf_char = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=char_ngram_range,
            max_features=char_max_features,
            sublinear_tf=True,
        )
        self._is_fitted = False

    def fit_transform(self, train_texts: List[str], eval_texts: Optional[List[str]] = None):
        """
        訓練 TF-IDF 並轉換資料

        Returns:
            如果有 eval_texts → (X_train, X_eval)
            否則               → X_train
        """
        X_train_w = self.tfidf_word.fit_transform(train_texts)
        X_train_c = self.tfidf_char.fit_transform(train_texts)
        X_train = hstack([X_train_w, X_train_c])
        self._is_fitted = True

        if eval_texts is not None:
            X_eval_w = self.tfidf_word.transform(eval_texts)
            X_eval_c = self.tfidf_char.transform(eval_texts)
            X_eval = hstack([X_eval_w, X_eval_c])
            return X_train, X_eval
        return X_train

    def transform(self, texts: List[str]) -> csr_matrix:
        """對新資料做 TF-IDF 轉換（需先 fit）"""
        assert self._is_fitted, "請先呼叫 fit_transform()"
        X_w = self.tfidf_word.transform(texts)
        X_c = self.tfidf_char.transform(texts)
        return hstack([X_w, X_c])

    def get_word_feature_names(self) -> np.ndarray:
        return self.tfidf_word.get_feature_names_out()

    def get_feature_dims(self) -> Dict[str, int]:
        return {
            'word': len(self.tfidf_word.get_feature_names_out()),
            'char': len(self.tfidf_char.get_feature_names_out()),
            'total': (len(self.tfidf_word.get_feature_names_out())
                      + len(self.tfidf_char.get_feature_names_out())),
        }

    def save(self, path: str):
        """儲存 TF-IDF 向量化器"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'word': self.tfidf_word, 'char': self.tfidf_char}, f)

    @classmethod
    def load(cls, path: str) -> 'TfidfFeatureBuilder':
        """載入已訓練的 TF-IDF 向量化器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        builder = cls()
        builder.tfidf_word = data['word']
        builder.tfidf_char = data['char']
        builder._is_fitted = True
        return builder


# ============================================================
# 資料載入輔助函式
# ============================================================
def load_dataset(path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    載入 JSON 資料集

    Returns:
        (texts, labels, ids)
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [d['text'] for d in data]
    labels = [d['intent'] for d in data]
    ids = [d['id'] for d in data]
    return texts, labels, ids


def save_processed_data(texts: List[str], labels: List[str], ids: List[str], path: str):
    """將前處理後的資料存為 JSON"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [{'id': i, 'text': t, 'intent': l} for i, t, l in zip(ids, texts, labels)]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 主程式：完整前處理 Pipeline
# ============================================================
if __name__ == '__main__':
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    from src.logger import setup_logger
    logger = setup_logger('data_preprocess')

    TRAIN_PATH = os.path.join(project_root, 'data', 'raw', 'train.json')
    EVAL_PATH = os.path.join(project_root, 'data', 'raw', 'eval.json')

    # 如果 raw/ 裡的檔名是中文
    if not os.path.exists(TRAIN_PATH):
        raw_dir = os.path.join(project_root, 'data', 'raw')
        for f in os.listdir(raw_dir):
            if 'train' in f:
                TRAIN_PATH = os.path.join(raw_dir, f)
            elif 'eval' in f:
                EVAL_PATH = os.path.join(raw_dir, f)

    logger.info("=" * 60)
    logger.info("  資料前處理 Pipeline")
    logger.info("=" * 60)

    # 1. 載入
    train_texts, train_labels, train_ids = load_dataset(TRAIN_PATH)
    eval_texts, eval_labels, eval_ids = load_dataset(EVAL_PATH)
    logger.info(f"訓練集: {len(train_texts)} 筆 | 評估集: {len(eval_texts)} 筆")
    logger.info(f"訓練集意圖數: {len(set(train_labels))}")
    logger.info(f"訓練集檔案: {TRAIN_PATH}")
    logger.info(f"評估集檔案: {EVAL_PATH}")

    # 2. 前處理
    preprocessor = TextPreprocessor()

    logger.info("--- 斷詞範例 ---")
    for i in range(3):
        original = train_texts[i]
        cleaned = preprocessor.transform(original)
        logger.info(f"  範例 {i} | 意圖: {train_labels[i]}")
        logger.info(f"    原始: {original}")
        logger.info(f"    結果: {cleaned}")

    train_clean = preprocessor.transform_batch(train_texts)
    eval_clean = preprocessor.transform_batch(eval_texts)

    avg_before = np.mean([len(t.split()) for t in train_texts])
    avg_after = np.mean([len(t.split()) for t in train_clean])
    logger.info(f"前處理前平均詞數: {avg_before:.1f} → 前處理後: {avg_after:.1f}")

    # 3. 儲存前處理結果
    train_out = os.path.join(project_root, 'data', 'processed', 'train_clean.json')
    eval_out = os.path.join(project_root, 'data', 'processed', 'eval_clean.json')
    save_processed_data(train_clean, train_labels, train_ids, train_out)
    save_processed_data(eval_clean, eval_labels, eval_ids, eval_out)
    logger.info(f"前處理資料已儲存至 data/processed/")
    logger.info(f"  train: {train_out}")
    logger.info(f"  eval:  {eval_out}")

    # 4. TF-IDF 向量化
    builder = TfidfFeatureBuilder()
    X_train, X_eval = builder.fit_transform(train_clean, eval_clean)
    dims = builder.get_feature_dims()
    logger.info(f"TF-IDF 特徵維度:")
    logger.info(f"  Word n-gram: {dims['word']}")
    logger.info(f"  Char n-gram: {dims['char']}")
    logger.info(f"  合計:        {dims['total']}")
    logger.info(f"訓練矩陣: {X_train.shape} | 評估矩陣: {X_eval.shape}")

    # 5. 儲存 TF-IDF
    tfidf_path = os.path.join(project_root, 'models', 'tfidf_vectorizer.pkl')
    builder.save(tfidf_path)
    logger.info(f"TF-IDF 向量化器已儲存至 {tfidf_path}")

    logger.info("前處理完成！")
