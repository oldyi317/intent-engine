"""
模型訓練模組 (Model Trainer Module)

負責：
  - 傳統 ML 模型建置（Logistic Regression, LinearSVC）
  - BERT fine-tuning 訓練
  - 交叉驗證
  - 超參數調整 (GridSearchCV)
  - 模型儲存 / 載入

Usage:
    from src.model_trainer import TraditionalMLTrainer, BertTrainer

    # 傳統 ML
    trainer = TraditionalMLTrainer()
    trainer.train(X_train, y_train)
    trainer.save("models/svm_best.pkl")

    # BERT
    bert_trainer = BertTrainer(num_labels=150)
    bert_trainer.train(train_data, eval_data, epochs=5)
"""

import os
import pickle
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder


# ============================================================
# TraditionalMLTrainer：傳統 ML 訓練器
# ============================================================
class TraditionalMLTrainer:
    """
    傳統機器學習模型訓練器

    支援模型：
        - logistic_regression: Logistic Regression (multinomial)
        - linear_svc: Linear Support Vector Classification

    功能：
        - 訓練 / 預測
        - 交叉驗證
        - 超參數搜索 (GridSearchCV)
        - 模型儲存 / 載入
    """

    MODELS = {
        'logistic_regression': lambda **kw: LogisticRegression(
            max_iter=1000, solver='lbfgs', n_jobs=-1, **kw
        ),
        'linear_svc': lambda **kw: LinearSVC(max_iter=5000, **kw),
    }

    def __init__(self, model_type: str = 'linear_svc', **model_kwargs):
        assert model_type in self.MODELS, f"支援的模型: {list(self.MODELS.keys())}"
        self.model_type = model_type
        self.model = self.MODELS[model_type](**model_kwargs)
        self.label_encoder = LabelEncoder()
        self._is_trained = False

    def train(self, X_train, y_train) -> Dict[str, float]:
        """
        訓練模型

        Returns:
            {"train_accuracy": float, "train_time_sec": float}
        """
        start = time.time()
        self.label_encoder.fit(y_train)
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start
        self._is_trained = True

        train_acc = (self.model.predict(X_train) == np.array(y_train)).mean()
        return {
            'train_accuracy': float(train_acc),
            'train_time_sec': round(elapsed, 2),
        }

    def predict(self, X) -> np.ndarray:
        """預測意圖標籤"""
        assert self._is_trained, "模型尚未訓練"
        return self.model.predict(X)

    def cross_validate(self, X, y, cv: int = 5) -> Dict[str, Any]:
        """
        交叉驗證

        Returns:
            {"cv_scores": list, "mean": float, "std": float}
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        return {
            'cv_scores': scores.tolist(),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
        }

    def grid_search(
        self, X, y, param_grid: Dict[str, list], cv: int = 5
    ) -> Dict[str, Any]:
        """
        超參數搜索

        Args:
            param_grid: 例如 {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}

        Returns:
            {"best_params": dict, "best_score": float, "all_results": list}
        """
        grid = GridSearchCV(
            self.model, param_grid, cv=cv,
            scoring='accuracy', n_jobs=-1, verbose=1, refit=True
        )
        grid.fit(X, y)

        # 用最佳模型替換
        self.model = grid.best_estimator_
        self._is_trained = True

        return {
            'best_params': grid.best_params_,
            'best_score': float(grid.best_score_),
            'all_results': [
                {
                    'params': p,
                    'mean_score': float(m),
                    'std_score': float(s),
                }
                for p, m, s in zip(
                    grid.cv_results_['params'],
                    grid.cv_results_['mean_test_score'],
                    grid.cv_results_['std_test_score'],
                )
            ],
        }

    def get_feature_importance(self, feature_names: np.ndarray, intent: str, top_k: int = 10):
        """
        取得指定意圖的 Top-K 重要特徵詞

        僅適用於 LinearSVC / LogisticRegression（具有 coef_ 屬性的模型）
        """
        assert self._is_trained, "模型尚未訓練"
        assert hasattr(self.model, 'coef_'), "此模型不支援特徵重要性分析"

        classes = self.model.classes_
        if intent not in classes:
            raise ValueError(f"意圖 '{intent}' 不在模型的類別中")

        idx = list(classes).index(intent)
        coef = self.model.coef_[idx]
        weights = coef.toarray().flatten() if hasattr(coef, 'toarray') else np.array(coef).flatten()

        n_feats = min(len(feature_names), len(weights))
        weights = weights[:n_feats]

        top_indices = weights.argsort()[-top_k:][::-1]
        return [(feature_names[i], float(weights[i])) for i in top_indices]

    def save(self, path: str):
        """儲存模型與 LabelEncoder"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'label_encoder': self.label_encoder,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'TraditionalMLTrainer':
        """載入已儲存的模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        trainer = cls(model_type=data['model_type'])
        trainer.model = data['model']
        trainer.label_encoder = data['label_encoder']
        trainer._is_trained = True
        return trainer


# ============================================================
# BertTrainer：BERT Fine-tuning 訓練器
# ============================================================
class BertTrainer:
    """
    BERT Fine-tuning 訓練器

    需要 GPU 環境，請在有 CUDA 的機器或 Google Colab 上執行。

    Usage:
        trainer = BertTrainer(num_labels=150, model_name='bert-base-uncased')
        history = trainer.train(train_data, eval_data, epochs=5, batch_size=32)
        trainer.save("models/bert_intent/")
    """

    def __init__(
        self,
        num_labels: int = 150,
        model_name: str = 'bert-base-uncased',
        max_length: int = 64,
        learning_rate: float = 2e-5,
    ):
        self.num_labels = num_labels
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.label_encoder = LabelEncoder()

        # 延遲 import，避免沒裝 torch 的環境報錯
        self._model = None
        self._tokenizer = None

    def _lazy_import(self):
        """延遲載入 torch 和 transformers"""
        try:
            import torch
            from transformers import BertTokenizer, BertForSequenceClassification
            self._torch = torch
            self._BertTokenizer = BertTokenizer
            self._BertForSequenceClassification = BertForSequenceClassification
        except ImportError:
            raise ImportError(
                "BERT 訓練需要 torch 和 transformers，請執行:\n"
                "  pip install torch transformers"
            )

    def train(
        self,
        train_data: List[Dict],
        eval_data: List[Dict],
        epochs: int = 5,
        batch_size: int = 32,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        logger=None,
    ) -> Dict[str, list]:
        """
        執行 BERT fine-tuning

        Args:
            train_data: [{"text": str, "intent": str}, ...]
            eval_data: 同上
            epochs: 訓練輪數
            batch_size: 批次大小
            logger: 可選的 logger 物件

        Returns:
            {"train_loss": [...], "train_acc": [...], "eval_loss": [...], "eval_acc": [...]}
        """
        def log(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        self._lazy_import()
        torch = self._torch
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log(f"Device: {device}")
        if device.type == 'cuda':
            log(f"GPU: {torch.cuda.get_device_name(0)}")
            log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Label encoding
        all_intents = sorted(set(d['intent'] for d in train_data))
        self.label_encoder.fit(all_intents)
        log(f"標籤類別數: {len(all_intents)}")

        # Tokenizer & Model
        tokenizer = self._BertTokenizer.from_pretrained(self.model_name)
        model = self._BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        ).to(device)

        self._tokenizer = tokenizer
        self._model = model

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"模型參數量: {total_params:,} (可訓練: {trainable_params:,})")

        # Dataset class
        class IntentDS(Dataset):
            def __init__(ds_self, data, le):
                ds_self.texts = [d['text'].lower().strip() for d in data]
                ds_self.labels = le.transform([d['intent'] for d in data])
                ds_self.tokenizer = tokenizer
                ds_self.max_len = self.max_length

            def __len__(ds_self):
                return len(ds_self.texts)

            def __getitem__(ds_self, idx):
                enc = ds_self.tokenizer(
                    ds_self.texts[idx],
                    max_length=ds_self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                return {
                    'input_ids': enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze(),
                    'label': torch.tensor(ds_self.labels[idx], dtype=torch.long),
                }

        train_loader = DataLoader(
            IntentDS(train_data, self.label_encoder),
            batch_size=batch_size, shuffle=True,
        )
        eval_loader = DataLoader(
            IntentDS(eval_data, self.label_encoder),
            batch_size=batch_size, shuffle=False,
        )

        log(f"訓練集 batches: {len(train_loader)} | 評估集 batches: {len(eval_loader)}")

        # Optimizer & Scheduler
        total_steps = len(train_loader) * epochs
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        log(f"Optimizer: AdamW (lr={self.learning_rate}, weight_decay={weight_decay})")
        log(f"Scheduler: linear warmup ({int(total_steps * warmup_ratio)} steps) + decay")
        log(f"Total steps: {total_steps}")

        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'eval_loss': [], 'eval_acc': []}
        best_acc = 0.0

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            model.train()
            total_loss, correct, total = 0, 0, 0
            for batch_idx, batch in enumerate(train_loader):
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += out.loss.item()
                correct += (out.logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                # 每 100 batch 輸出一次進度
                if (batch_idx + 1) % 100 == 0:
                    log(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                        f"Running Loss: {total_loss/(batch_idx+1):.4f} | "
                        f"Running Acc: {correct/total:.4f}")

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Eval
            model.eval()
            total_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for batch in eval_loader:
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    out = model(input_ids=ids, attention_mask=mask, labels=labels)
                    total_loss += out.loss.item()
                    correct += (out.logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)

            eval_loss = total_loss / len(eval_loader)
            eval_acc = correct / total
            history['eval_loss'].append(eval_loss)
            history['eval_acc'].append(eval_acc)

            epoch_time = time.time() - epoch_start
            log(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Eval Loss: {eval_loss:.4f} Acc: {eval_acc:.4f} | "
                f"Time: {epoch_time:.1f}s")

            if eval_acc > best_acc:
                best_acc = eval_acc
                log(f"  → New best! ({best_acc:.4f})")

        log(f"\nBest Eval Accuracy: {best_acc:.4f}")
        return history

    def predict(self, texts: List[str]) -> List[str]:
        """使用訓練好的 BERT 模型預測意圖"""
        self._lazy_import()
        torch = self._torch
        device = next(self._model.parameters()).device

        self._model.eval()
        preds = []
        with torch.no_grad():
            for text in texts:
                enc = self._tokenizer(
                    text.lower().strip(),
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                ids = enc['input_ids'].to(device)
                mask = enc['attention_mask'].to(device)
                out = self._model(input_ids=ids, attention_mask=mask)
                pred_idx = out.logits.argmax(dim=1).item()
                preds.append(self.label_encoder.inverse_transform([pred_idx])[0])
        return preds

    def save(self, directory: str):
        """儲存 BERT 模型、tokenizer、label_encoder"""
        os.makedirs(directory, exist_ok=True)
        if self._model:
            self._model.save_pretrained(directory)
        if self._tokenizer:
            self._tokenizer.save_pretrained(directory)
        with open(os.path.join(directory, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

    @classmethod
    def load(cls, directory: str) -> 'BertTrainer':
        """載入已儲存的 BERT 模型，自動偵測 GPU"""
        import torch
        trainer = cls()
        trainer._lazy_import()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer._tokenizer = trainer._BertTokenizer.from_pretrained(directory)
        trainer._model = trainer._BertForSequenceClassification.from_pretrained(directory).to(device)
        with open(os.path.join(directory, 'label_encoder.pkl'), 'rb') as f:
            trainer.label_encoder = pickle.load(f)
        return trainer


# ============================================================
# 主程式
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_preprocess import TextPreprocessor, TfidfFeatureBuilder, load_dataset
    from src.logger import setup_logger

    logger = setup_logger('model_trainer')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, 'data', 'raw')
    train_file = [f for f in os.listdir(raw_dir) if 'train' in f][0]
    eval_file = [f for f in os.listdir(raw_dir) if 'eval' in f][0]

    # 載入 & 前處理
    train_texts, train_labels, _ = load_dataset(os.path.join(raw_dir, train_file))
    eval_texts, eval_labels, _ = load_dataset(os.path.join(raw_dir, eval_file))

    pp = TextPreprocessor()
    train_clean = pp.transform_batch(train_texts)
    eval_clean = pp.transform_batch(eval_texts)

    fb = TfidfFeatureBuilder()
    X_train, X_eval = fb.fit_transform(train_clean, eval_clean)

    logger.info("=" * 60)
    logger.info(" 模型訓練")
    logger.info("=" * 60)
    logger.info(f"訓練集: {X_train.shape} | 評估集: {X_eval.shape}")
    logger.info(f"意圖數: {len(set(train_labels))}")

    # --- Logistic Regression ---
    logger.info("--- Logistic Regression ---")
    lr_trainer = TraditionalMLTrainer('logistic_regression', C=5.0)
    result = lr_trainer.train(X_train, train_labels)
    logger.info(f"  Train Acc: {result['train_accuracy']:.4f} ({result['train_time_sec']}s)")

    from sklearn.metrics import accuracy_score
    lr_preds = lr_trainer.predict(X_eval)
    lr_acc = accuracy_score(eval_labels, lr_preds)
    logger.info(f"  Eval  Acc: {lr_acc:.4f}")

    # --- LinearSVC ---
    logger.info("--- LinearSVC ---")
    svm_trainer = TraditionalMLTrainer('linear_svc', C=1.0)
    result = svm_trainer.train(X_train, train_labels)
    logger.info(f"  Train Acc: {result['train_accuracy']:.4f} ({result['train_time_sec']}s)")

    svm_preds = svm_trainer.predict(X_eval)
    svm_acc = accuracy_score(eval_labels, svm_preds)
    logger.info(f"  Eval  Acc: {svm_acc:.4f}")

    # --- GridSearch ---
    logger.info("--- GridSearch for SVM ---")
    gs_trainer = TraditionalMLTrainer('linear_svc')
    gs_result = gs_trainer.grid_search(X_train, train_labels, {'C': [0.1, 0.5, 1.0, 2.0, 5.0]})
    logger.info(f"  Best C: {gs_result['best_params']}")
    logger.info(f"  Best CV Score: {gs_result['best_score']:.4f}")
    for r in gs_result['all_results']:
        logger.info(f"    C={r['params']['C']}: {r['mean_score']:.4f} (±{r['std_score']:.4f})")

    gs_preds = gs_trainer.predict(X_eval)
    gs_acc = accuracy_score(eval_labels, gs_preds)
    logger.info(f"  Eval Acc: {gs_acc:.4f}")

    # --- Cross Validation ---
    logger.info("--- Cross Validation (best model) ---")
    cv_result = gs_trainer.cross_validate(X_train, train_labels, cv=5)
    logger.info(f"  Scores: {[f'{s:.4f}' for s in cv_result['cv_scores']]}")
    logger.info(f"  Mean: {cv_result['mean']:.4f} (±{cv_result['std']:.4f})")

    # --- 模型比較摘要 ---
    logger.info("=" * 60)
    logger.info(" 模型比較摘要")
    logger.info("=" * 60)
    logger.info(f"  Logistic Regression: {lr_acc:.4f}")
    logger.info(f"  LinearSVC (C=1.0):   {svm_acc:.4f}")
    logger.info(f"  GridSearch Best:     {gs_acc:.4f} (C={gs_result['best_params']['C']})")
    logger.info(f"  5-Fold CV Mean:      {cv_result['mean']:.4f} (±{cv_result['std']:.4f})")

    # --- 儲存 ---
    model_path = os.path.join(project_root, 'models', 'svm_best.pkl')
    gs_trainer.save(model_path)
    logger.info(f"模型已儲存至 {model_path}")
