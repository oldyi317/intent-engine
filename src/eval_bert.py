"""
BERT 模型驗證腳本

功能：
  1. 載入已訓練好的 BERT 模型（models/bert_intent/）
  2. 對 eval set 做完整預測
  3. 用 IntentEvaluator 產出領域準確率、混淆對、錯誤案例分析
  4. 輸出四大評估指標（Accuracy, Precision, Recall, F1-Score）
  5. 自動產出視覺化圖表（儲存至 outputs/ 目錄）
  6. 與 SVM 結果做對比
  7. 支援自訂文字的互動測試（加 --interactive 參數）

執行方式：
    conda activate pytorch
    cd fubon_intent_project
    python -m src.eval_bert                 # 評估 + 圖表（不進互動模式）
    python -m src.eval_bert --interactive   # 評估 + 圖表 + 互動測試

前置條件：
    先跑完 python -m src.train_bert（產出 models/bert_intent/）
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import BertTrainer
from src.evaluator import IntentEvaluator
from src.logger import setup_logger


def print_four_metrics(evaluator, logger, model_name="BERT"):
    """印出四大評估指標（Accuracy, Macro Precision, Macro Recall, Macro F1）"""
    report_dict = evaluator.per_class_report(as_dict=True)
    acc = evaluator.overall_accuracy()
    macro = report_dict.get('macro avg', {})
    weighted = report_dict.get('weighted avg', {})

    logger.info("=" * 60)
    logger.info(f"  {model_name} — 四大評估指標")
    logger.info("=" * 60)
    logger.info(f"  {'指標':<20s} {'Macro Avg':>10s}   {'Weighted Avg':>12s}")
    logger.info(f"  {'-'*20} {'-'*10}   {'-'*12}")
    logger.info(f"  {'Accuracy':<20s} {acc:>10.4f}   {acc:>12.4f}")
    logger.info(f"  {'Precision':<20s} {macro.get('precision', 0):>10.4f}   {weighted.get('precision', 0):>12.4f}")
    logger.info(f"  {'Recall':<20s} {macro.get('recall', 0):>10.4f}   {weighted.get('recall', 0):>12.4f}")
    logger.info(f"  {'F1-Score':<20s} {macro.get('f1-score', 0):>10.4f}   {weighted.get('f1-score', 0):>12.4f}")
    logger.info("")

    return {
        'accuracy': acc,
        'macro_precision': macro.get('precision', 0),
        'macro_recall': macro.get('recall', 0),
        'macro_f1': macro.get('f1-score', 0),
        'weighted_precision': weighted.get('precision', 0),
        'weighted_recall': weighted.get('recall', 0),
        'weighted_f1': weighted.get('f1-score', 0),
    }


def generate_charts(evaluator, output_dir, svm_report_path=None, logger=None):
    """
    產出 BERT 評估圖表，委派給 plot_model_metrics 中的共用繪圖函式。

    產出檔案：
      - bert_4metrics.png         BERT 四大指標長條圖
      - bert_domain_accuracy.png  BERT 各領域準確率
      - bert_confusion_pairs.png  BERT Top 混淆意圖對
      - model_comparison.png      SVM vs BERT 四指標對比（若有 SVM 報告）
      - domain_comparison.png     SVM vs BERT 各領域準確率對比（若有 SVM 報告）
    """
    from src.plot_model_metrics import (
        load_report, extract_macro_metrics,
        plot_4metrics, plot_domain_accuracy, plot_confusion_pairs,
        plot_comparison_4metrics, plot_domain_comparison,
    )

    os.makedirs(output_dir, exist_ok=True)

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # 先儲存 BERT 報告到暫存，讓共用函式讀取
    bert_report = evaluator.full_report()
    report_dict = evaluator.per_class_report(as_dict=True)
    macro = report_dict.get('macro avg', {})
    bert_metrics = {
        'Accuracy':  evaluator.overall_accuracy() * 100,
        'Precision': macro.get('precision', 0) * 100,
        'Recall':    macro.get('recall', 0) * 100,
        'F1-Score':  macro.get('f1-score', 0) * 100,
    }

    # Chart 1: BERT 四大指標
    p = plot_4metrics(bert_metrics, 'BERT Fine-tune — Macro Average Metrics', 'bert_4metrics.png', output_dir)
    log(f"  圖表已儲存: {p}")

    # Chart 2: BERT 各領域準確率
    p = plot_domain_accuracy(bert_report, 'BERT', 'bert_domain_accuracy.png', output_dir)
    log(f"  圖表已儲存: {p}")

    # Chart 3: BERT Top 混淆意圖對
    p = plot_confusion_pairs(bert_report, 'BERT', 'bert_confusion_pairs.png', output_dir)
    if p:
        log(f"  圖表已儲存: {p}")

    # Chart 4 & 5: SVM vs BERT 對比
    if svm_report_path and os.path.exists(svm_report_path):
        svm_report = load_report(svm_report_path)
        if svm_report:
            svm_metrics = extract_macro_metrics(svm_report)
            p = plot_comparison_4metrics(svm_metrics, bert_metrics, output_dir)
            log(f"  圖表已儲存: {p}")
            p = plot_domain_comparison(svm_report, bert_report, output_dir)
            log(f"  圖表已儲存: {p}")

    log(f"\n  所有圖表已儲存至 {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='BERT 模型驗證')
    parser.add_argument('--interactive', action='store_true', help='啟用互動測試模式')
    parser.add_argument('--no-charts', action='store_true', help='跳過圖表產出')
    args = parser.parse_args()

    logger = setup_logger('eval_bert')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, 'models', 'bert_intent')
    raw_dir = os.path.join(project_root, 'data', 'raw')
    output_dir = os.path.join(project_root, 'outputs')

    # ── 1. 檢查模型是否存在 ──
    if not os.path.exists(model_dir):
        logger.error(f"找不到 BERT 模型目錄: {model_dir}")
        logger.error("請先執行: python -m src.train_bert")
        return

    # ── 2. 載入模型 ──
    logger.info("=" * 60)
    logger.info(" BERT 模型驗證")
    logger.info("=" * 60)

    logger.info(f"載入模型: {model_dir}")
    start = time.time()
    trainer = BertTrainer.load(model_dir)
    load_time = time.time() - start
    logger.info(f"模型載入完成 ({load_time:.1f}s)")

    # 確認 device
    import torch
    device = next(trainer._model.parameters()).device
    logger.info(f"Device: {device}")

    # ── 3. 載入評估資料 ──
    eval_file = [f for f in os.listdir(raw_dir) if 'eval' in f][0]
    with open(os.path.join(raw_dir, eval_file), 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    all_texts = [d['text'] for d in eval_data]
    all_labels = [d['intent'] for d in eval_data]
    logger.info(f"評估集: {len(eval_data)} 筆 | 意圖數: {len(set(all_labels))}")

    # ── 4. 批次預測 ──
    logger.info("開始預測...")
    start = time.time()
    all_preds = trainer.predict(all_texts)
    pred_time = time.time() - start
    logger.info(f"預測完成 ({pred_time:.1f}s, {len(all_texts)/pred_time:.0f} samples/sec)")

    # ── 5. 完整評估 ──
    logger.info("=" * 60)
    logger.info(" BERT 評估結果")
    logger.info("=" * 60)

    evaluator = IntentEvaluator(all_labels, all_preds, all_texts)
    evaluator.print_summary(logger=logger)

    # ── 6. 四大評估指標 ──
    metrics = print_four_metrics(evaluator, logger, model_name="BERT")

    # 儲存報告（含四大指標摘要）
    report_path = os.path.join(project_root, 'models', 'bert_eval_report.json')
    evaluator.save_report(report_path)

    # 額外寫入 macro_metrics 方便後續讀取
    with open(report_path, 'r') as f:
        report_data = json.load(f)
    report_data['macro_metrics'] = metrics
    with open(report_path, 'w') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    logger.info(f"評估報告已儲存至 {report_path}")

    # ── 7. 產出圖表 ──
    if not args.no_charts:
        logger.info("=" * 60)
        logger.info(" 產出評估圖表")
        logger.info("=" * 60)

        svm_report_path = os.path.join(project_root, 'models', 'eval_report.json')
        generate_charts(evaluator, output_dir, svm_report_path, logger)

    # ── 8. 與 SVM 結果比較 ──
    svm_report_path = os.path.join(project_root, 'models', 'eval_report.json')
    if os.path.exists(svm_report_path):
        logger.info("=" * 60)
        logger.info(" BERT vs SVM 比較")
        logger.info("=" * 60)

        with open(svm_report_path, 'r') as f:
            svm_report = json.load(f)

        bert_acc = evaluator.overall_accuracy()
        svm_acc = svm_report['overall_accuracy']
        diff = bert_acc - svm_acc

        logger.info(f"  SVM  Accuracy: {svm_acc:.4f}")
        logger.info(f"  BERT Accuracy: {bert_acc:.4f}")
        logger.info(f"  差異: {diff:+.4f} ({'BERT 勝' if diff > 0 else 'SVM 勝'})")

        # 各領域比較
        logger.info("")
        logger.info(f"  {'領域':<15s} {'SVM':>8s} {'BERT':>8s} {'差異':>8s}")
        logger.info(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")

        bert_domains = evaluator.domain_accuracy()
        svm_domains = svm_report.get('domain_accuracy', {})

        for domain in bert_domains:
            b_acc = bert_domains[domain]['accuracy']
            s_acc = svm_domains.get(domain, {}).get('accuracy', 0)
            d = b_acc - s_acc
            marker = '+' if d > 0 else ''
            logger.info(f"  {domain:<15s} {s_acc:>7.2%} {b_acc:>7.2%} {marker}{d:>7.2%}")

    # ── 9. 互動測試（含多意圖拆解）──
    if args.interactive:
        from app.llm_handler import MultiIntentHandler

        def bert_classify(text):
            pred = trainer.predict([text])[0]
            return pred, 1.0

        multi_handler = MultiIntentHandler(classifier_fn=bert_classify)

        logger.info("=" * 60)
        logger.info(" 互動測試模式（支援多意圖拆解）")
        logger.info("=" * 60)
        logger.info("輸入文字進行意圖預測（輸入 'q' 離開）")

        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() == 'q':
                break

            result = multi_handler.analyze(text)
            logger.info(f"  輸入: {text}")
            logger.info(f"  複合句: {result['is_compound']}")

            if result['is_compound']:
                logger.info(f"  拆解為 {len(result['sub_queries'])} 個子意圖:")
                for i, sq in enumerate(result['sub_queries'], 1):
                    logger.info(f"    {i}. [{sq['text']}] → {sq['intent']}")
            else:
                logger.info(f"  預測: {result['sub_queries'][0]['intent']}")
    else:
        logger.info("")
        logger.info("提示：加 --interactive 參數可進入互動測試模式")

    logger.info("")
    logger.info("BERT 評估完成！")


if __name__ == '__main__':
    main()
