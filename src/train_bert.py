"""
BERT Fine-tuning 訓練腳本

在有 GPU 的本機環境執行：
    conda activate pytorch
    cd fubon_intent_project
    python -m src.train_bert

預計訓練時間：約 10-15 分鐘（RTX 3060 以上）
產出：models/bert_intent/ 目錄（含模型權重、tokenizer、label_encoder）
"""

import os
import sys
import json
import time

# 確保專案根目錄在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import BertTrainer
from src.logger import setup_logger


def main():
    logger = setup_logger('train_bert')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, 'data', 'raw')

    # ── 1. 載入資料 ──
    logger.info("=" * 60)
    logger.info(" BERT Fine-tuning for Intent Classification")
    logger.info("=" * 60)

    train_file = [f for f in os.listdir(raw_dir) if 'train' in f][0]
    eval_file = [f for f in os.listdir(raw_dir) if 'eval' in f][0]

    with open(os.path.join(raw_dir, train_file), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(os.path.join(raw_dir, eval_file), 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    logger.info(f"訓練集: {len(train_data)} 筆")
    logger.info(f"評估集: {len(eval_data)} 筆")
    logger.info(f"意圖數: {len(set(d['intent'] for d in train_data))}")
    logger.info(f"訓練集檔案: {train_file}")
    logger.info(f"評估集檔案: {eval_file}")

    # ── 2. 初始化 BertTrainer ──
    trainer = BertTrainer(
        num_labels=150,
        model_name='bert-base-uncased',
        max_length=64,
        learning_rate=2e-5,
    )
    logger.info(f"模型: {trainer.model_name}")
    logger.info(f"Max length: {trainer.max_length}")
    logger.info(f"Learning rate: {trainer.learning_rate}")

    # ── 3. 訓練 ──
    logger.info("開始訓練...")
    start = time.time()

    history = trainer.train(
        train_data=train_data,
        eval_data=eval_data,
        epochs=5,
        batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logger=logger,
    )

    elapsed = time.time() - start
    logger.info(f"總訓練時間: {elapsed / 60:.1f} 分鐘")

    # ── 4. 儲存模型 ──
    save_dir = os.path.join(project_root, 'models', 'bert_intent')
    trainer.save(save_dir)
    logger.info(f"模型已儲存至 {save_dir}/")

    # ── 5. 儲存訓練歷史 ──
    history_path = os.path.join(project_root, 'models', 'bert_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"訓練歷史已儲存至 {history_path}")

    # ── 6. 訓練曲線摘要 ──
    logger.info("=" * 60)
    logger.info(" 訓練曲線摘要")
    logger.info("=" * 60)
    for i in range(len(history['train_loss'])):
        logger.info(
            f"  Epoch {i+1}: "
            f"Train Loss={history['train_loss'][i]:.4f} Acc={history['train_acc'][i]:.4f} | "
            f"Eval Loss={history['eval_loss'][i]:.4f} Acc={history['eval_acc'][i]:.4f}"
        )

    # ── 7. 快速測試 ──
    logger.info("=" * 60)
    logger.info(" 快速測試")
    logger.info("=" * 60)

    test_texts = [
        "what is my bank balance",
        "book a flight to tokyo",
        "set an alarm for 7am",
        "how do i improve my credit score",
        "find me a good italian restaurant nearby",
    ]

    preds = trainer.predict(test_texts)
    for text, pred in zip(test_texts, preds):
        logger.info(f"  [{text}] → {pred}")

    # ── 8. 完整評估 ──
    logger.info("=" * 60)
    logger.info(" 完整評估")
    logger.info("=" * 60)

    all_texts = [d['text'] for d in eval_data]
    all_labels = [d['intent'] for d in eval_data]
    all_preds = trainer.predict(all_texts)

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    logger.info(f"  Eval Accuracy: {correct}/{total} = {correct/total:.4f}")

    # 儲存 BERT 預測結果供 evaluator 使用
    bert_eval = {
        'accuracy': correct / total,
        'history': history,
        'predictions': all_preds,
        'labels': all_labels,
    }
    bert_eval_path = os.path.join(project_root, 'models', 'bert_eval_result.json')
    with open(bert_eval_path, 'w') as f:
        json.dump(bert_eval, f, indent=2)
    logger.info(f"  評估結果已儲存至 {bert_eval_path}")

    logger.info("BERT fine-tuning 完成！")


if __name__ == '__main__':
    main()
