"""
模型評估模組 (Evaluator Module)

負責：
  - 計算分類指標（Accuracy, Precision, Recall, F1）
  - 混淆矩陣分析
  - 錯誤分析（找出最常混淆的意圖對）
  - 按領域 (Domain) 分組評估
  - 產出評估報告

Usage:
    from src.evaluator import IntentEvaluator

    evaluator = IntentEvaluator(y_true, y_pred, texts)
    report = evaluator.full_report()
    evaluator.save_report("models/eval_report.json")
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import INTENT_DOMAINS, INTENT_TO_DOMAIN


# ============================================================
# IntentEvaluator
# ============================================================
class IntentEvaluator:
    """
    意圖分類模型評估器

    提供：
        - overall_accuracy: 整體準確率
        - per_class_report: 每類 Precision / Recall / F1
        - domain_accuracy: 按領域分組的準確率
        - confusion_pairs: 最常見的混淆意圖對
        - error_examples: 錯誤案例清單
    """

    def __init__(
        self,
        y_true: List[str],
        y_pred: List[str],
        texts: Optional[List[str]] = None,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.texts = texts
        self.n = len(y_true)

    # ─── 整體準確率 ───
    def overall_accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    # ─── 每類詳細報告 ───
    def per_class_report(self, as_dict: bool = False):
        if as_dict:
            return classification_report(self.y_true, self.y_pred, output_dict=True, zero_division=0)
        return classification_report(self.y_true, self.y_pred, zero_division=0)

    # ─── 按領域準確率 ───
    def domain_accuracy(self) -> Dict[str, Dict]:
        results = {}
        for domain, intents in INTENT_DOMAINS.items():
            intent_set = set(intents)
            mask = [self.y_true[i] in intent_set for i in range(self.n)]
            if not any(mask):
                continue
            true_d = [self.y_true[i] for i in range(self.n) if mask[i]]
            pred_d = [self.y_pred[i] for i in range(self.n) if mask[i]]
            results[domain] = {
                'accuracy': accuracy_score(true_d, pred_d),
                'n_samples': len(true_d),
                'n_intents': len([it for it in intents if it in set(true_d)]),
            }
        # 排序
        results = dict(sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True))
        return results

    # ─── 混淆意圖對 ───
    def confusion_pairs(self, top_k: int = 10) -> List[Dict]:
        pairs = defaultdict(int)
        for t, p in zip(self.y_true, self.y_pred):
            if t != p:
                pairs[(t, p)] += 1

        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {'true': t, 'predicted': p, 'count': c}
            for (t, p), c in sorted_pairs
        ]

    # ─── 錯誤案例 ───
    def error_examples(self, max_per_pair: int = 3) -> List[Dict]:
        if self.texts is None:
            return []

        errors_by_pair = defaultdict(list)
        for i in range(self.n):
            if self.y_true[i] != self.y_pred[i]:
                pair = (self.y_true[i], self.y_pred[i])
                if len(errors_by_pair[pair]) < max_per_pair:
                    errors_by_pair[pair].append({
                        'text': self.texts[i],
                        'true_intent': self.y_true[i],
                        'pred_intent': self.y_pred[i],
                    })

        all_errors = []
        for pair_errors in errors_by_pair.values():
            all_errors.extend(pair_errors)
        return all_errors

    # ─── 完整報告 ───
    def full_report(self) -> Dict:
        return {
            'overall_accuracy': self.overall_accuracy(),
            'total_samples': self.n,
            'total_errors': sum(1 for t, p in zip(self.y_true, self.y_pred) if t != p),
            'domain_accuracy': self.domain_accuracy(),
            'top_confusion_pairs': self.confusion_pairs(),
            'per_class_report': self.per_class_report(as_dict=True),
        }

    # ─── 儲存報告 ───
    def save_report(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        report = self.full_report()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # ─── 印出摘要 ───
    def print_summary(self, logger=None):
        """印出評估摘要，支援 logger 或 print"""
        def log(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        acc = self.overall_accuracy()
        errors = sum(1 for t, p in zip(self.y_true, self.y_pred) if t != p)

        log("=" * 60)
        log(f"  整體準確率: {acc:.4f} ({acc*100:.2f}%)")
        log(f"  總樣本數: {self.n} | 錯誤數: {errors}")
        log("=" * 60)

        log("  按領域準確率:")
        for domain, info in self.domain_accuracy().items():
            bar = "█" * int(info['accuracy'] * 20)
            log(f"    {domain:15s}  {info['accuracy']:.2%}  {bar}  ({info['n_samples']} samples)")

        log(f"  Top-10 混淆意圖對:")
        for pair in self.confusion_pairs(10):
            log(f"    {pair['true']:30s} → {pair['predicted']:30s} ({pair['count']})")

        if self.texts:
            log(f"  錯誤案例（前 5 筆）:")
            for ex in self.error_examples()[:5]:
                log(f"    \"{ex['text']}\"")
                log(f"      真實: {ex['true_intent']} | 預測: {ex['pred_intent']}")


# ============================================================
# 主程式
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_preprocess import TextPreprocessor, TfidfFeatureBuilder, load_dataset
    from src.model_trainer import TraditionalMLTrainer
    from src.logger import setup_logger

    logger = setup_logger('evaluator')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, 'data', 'raw')
    train_file = [f for f in os.listdir(raw_dir) if 'train' in f][0]
    eval_file = [f for f in os.listdir(raw_dir) if 'eval' in f][0]

    # Pipeline
    logger.info("載入資料...")
    train_texts, train_labels, _ = load_dataset(os.path.join(raw_dir, train_file))
    eval_texts, eval_labels, _ = load_dataset(os.path.join(raw_dir, eval_file))

    logger.info("前處理...")
    pp = TextPreprocessor()
    train_clean = pp.transform_batch(train_texts)
    eval_clean = pp.transform_batch(eval_texts)

    logger.info("TF-IDF 建構...")
    fb = TfidfFeatureBuilder()
    X_train, X_eval = fb.fit_transform(train_clean, eval_clean)

    logger.info("模型訓練 (LinearSVC)...")
    trainer = TraditionalMLTrainer('linear_svc', C=1.0)
    trainer.train(X_train, train_labels)
    preds = trainer.predict(X_eval)

    # 評估
    logger.info("開始評估...")
    evaluator = IntentEvaluator(eval_labels, preds.tolist(), eval_texts)
    evaluator.print_summary(logger=logger)

    report_path = os.path.join(project_root, 'models', 'eval_report.json')
    evaluator.save_report(report_path)
    logger.info(f"報告已儲存至 {report_path}")
