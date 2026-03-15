"""
模型評估圖表產生器 (Standalone)

從已儲存的 JSON 評估報告產出視覺化圖表。
可在沒有 GPU / 模型的環境下重新產圖。

產出圖表：
  1. bert_4metrics.png         — BERT 四大指標長條圖
  2. svm_4metrics.png          — SVM 四大指標長條圖
  3. model_comparison.png      — SVM vs BERT 四指標對比
  4. model_accuracy_compare.png — LR vs SVM vs BERT 準確率
  5. bert_domain_accuracy.png  — BERT 各領域準確率
  6. domain_comparison.png     — SVM vs BERT 各領域準確率對比
  7. bert_confusion_pairs.png  — BERT Top 混淆意圖對
  8. bert_training_progress.png— BERT 訓練曲線

執行方式：
    cd fubon_intent_project
    python -m src.plot_model_metrics

    # 或指定輸出目錄：
    python -m src.plot_model_metrics --output outputs/charts
"""

import os
import sys
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# 字型設定（確保中文顯示正常）
plt.rcParams['font.family'] = 'Microsoft JhengHei, DejaVu Sans, Arial, sans-serif'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Color Palette ──
NAVY = '#1A2744'
GOLD = '#B8860B'
TEAL = '#0D7C66'
BLUE = '#2563EB'
RED  = '#C0392B'
GRAY = '#8899AA'


def load_report(path):
    """載入 JSON 報告"""
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_macro_metrics(report):
    """從報告提取 macro avg 四大指標"""
    # 先看有沒有預先計算好的 macro_metrics
    if 'macro_metrics' in report:
        mm = report['macro_metrics']
        return {
            'Accuracy':  mm.get('accuracy', report.get('overall_accuracy', 0)) * 100,
            'Precision': mm.get('macro_precision', 0) * 100,
            'Recall':    mm.get('macro_recall', 0) * 100,
            'F1-Score':  mm.get('macro_f1', 0) * 100,
        }

    # 從 per_class_report 計算
    pcr = report.get('per_class_report', {})
    macro = pcr.get('macro avg', {})

    if not macro:
        # 手動計算
        precs, recs, f1s = [], [], []
        for cls, m in pcr.items():
            if isinstance(m, dict) and 'precision' in m and cls not in ('macro avg', 'weighted avg', 'accuracy'):
                precs.append(m['precision'])
                recs.append(m['recall'])
                f1s.append(m['f1-score'])
        if precs:
            macro = {
                'precision': sum(precs) / len(precs),
                'recall': sum(recs) / len(recs),
                'f1-score': sum(f1s) / len(f1s),
            }

    return {
        'Accuracy':  report.get('overall_accuracy', 0) * 100,
        'Precision': macro.get('precision', 0) * 100,
        'Recall':    macro.get('recall', 0) * 100,
        'F1-Score':  macro.get('f1-score', 0) * 100,
    }


def plot_4metrics(metrics, title, filename, output_dir):
    """繪製四大指標長條圖"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names  = list(metrics.keys())
    vals   = list(metrics.values())
    colors = [NAVY, BLUE, TEAL, GOLD]

    bars = ax.bar(names, vals, color=colors, width=0.55, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=14, fontweight='bold', color=NAVY)

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(max(0, min(vals) - 8), min(100, max(vals) + 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def plot_model_accuracy_comparison(lr_acc, svm_acc, bert_acc, output_dir):
    """LR vs SVM vs BERT 準確率對比"""
    fig, ax = plt.subplots(figsize=(7, 4))
    models = ['Logistic\nRegression', 'LinearSVC\n+ TF-IDF', 'BERT\nFine-tune']
    accuracies = [lr_acc, svm_acc, bert_acc]
    colors = [GRAY, BLUE, TEAL]

    bars = ax.bar(models, accuracies, color=colors, width=0.55, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=16, fontweight='bold', color=NAVY)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(80, 102)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # improvement arrow
    ax.annotate('', xy=(2, bert_acc + 0.8), xytext=(1, svm_acc + 1),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2))
    diff = bert_acc - svm_acc
    ax.text(1.7, (svm_acc + bert_acc) / 2 + 1, f'+{diff:.1f}%',
            fontsize=12, fontweight='bold', color=RED)

    plt.tight_layout()
    path = os.path.join(output_dir, 'model_accuracy_compare.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def plot_comparison_4metrics(svm_metrics, bert_metrics, output_dir):
    """SVM vs BERT 四指標對比"""
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(bert_metrics.keys())
    x = np.arange(len(names))
    width = 0.32

    svm_vals  = [svm_metrics[n] for n in names]
    bert_vals = [bert_metrics[n] for n in names]

    bars1 = ax.bar(x - width/2, svm_vals,  width, label='SVM (LinearSVC)', color=BLUE, edgecolor='white')
    bars2 = ax.bar(x + width/2, bert_vals, width, label='BERT Fine-tune',  color=TEAL, edgecolor='white')

    for bar, val in zip(bars1, svm_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color=BLUE)
    for bar, val in zip(bars2, bert_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color=TEAL)

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('SVM vs BERT — 四大評估指標比較', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(max(0, min(svm_vals + bert_vals) - 8), min(100, max(bert_vals) + 5))
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def plot_domain_comparison(svm_report, bert_report, output_dir):
    """SVM vs BERT 各領域準確率對比"""
    svm_domains  = svm_report.get('domain_accuracy', {})
    bert_domains = bert_report.get('domain_accuracy', {})

    all_domains = sorted(set(list(svm_domains.keys()) + list(bert_domains.keys())))
    svm_accs  = [svm_domains.get(d, {}).get('accuracy', 0) * 100 for d in all_domains]
    bert_accs = [bert_domains.get(d, {}).get('accuracy', 0) * 100 for d in all_domains]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(all_domains))
    width = 0.35

    bars1 = ax.bar(x - width/2, svm_accs,  width, label='SVM', color=BLUE, edgecolor='white')
    bars2 = ax.bar(x + width/2, bert_accs, width, label='BERT', color=TEAL, edgecolor='white')

    for bar, val in zip(bars1, svm_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.0f}%', ha='center', fontsize=8, fontweight='bold', color=BLUE)
    for bar, val in zip(bars2, bert_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.0f}%', ha='center', fontsize=8, fontweight='bold', color=TEAL)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('SVM vs BERT — 各領域準確率比較', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_domains, fontsize=10, rotation=15, ha='right')
    ax.set_ylim(max(70, min(svm_accs + bert_accs) - 5), 105)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'domain_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def plot_domain_accuracy(report, model_name, filename, output_dir):
    """單一模型各領域準確率"""
    domain_acc = report.get('domain_accuracy', {})
    sorted_domains = sorted(domain_acc.items(), key=lambda x: x[1]['accuracy'])
    d_names   = [d for d, _ in sorted_domains]
    d_accs    = [v['accuracy'] * 100 for _, v in sorted_domains]
    d_samples = [v['n_samples'] for _, v in sorted_domains]

    color_list = ['#95A5A6', '#3498DB', '#C0392B', '#8E44AD', '#E67E22',
                  '#0D7C66', '#2563EB', '#1A2744', '#B8860B']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(d_names, d_accs, color=color_list[:len(d_names)], edgecolor='white', height=0.65)
    for bar, val, n in zip(bars, d_accs, d_samples):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%  (n={n})', va='center', fontsize=10, fontweight='bold', color=NAVY)

    ax.set_xlabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'{model_name} Accuracy by Domain', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(max(70, min(d_accs) - 5), 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def plot_confusion_pairs(report, model_name, filename, output_dir, top_k=10):
    """混淆意圖對"""
    confusion = report.get('top_confusion_pairs', [])[:top_k]
    if not confusion:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    pair_labels = [f"{c['true']} → {c['predicted']}" for c in reversed(confusion)]
    pair_counts = [c['count'] for c in reversed(confusion)]
    colors_c = [RED if c >= 5 else GOLD if c >= 3 else GRAY for c in pair_counts]

    bars = ax.barh(pair_labels, pair_counts, color=colors_c, edgecolor='white', height=0.6)
    for bar, val in zip(bars, pair_counts):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=11, fontweight='bold', color=NAVY)

    ax.set_xlabel('Misclassification Count', fontsize=11)
    ax.set_title(f'Top Confusion Pairs ({model_name})', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(0, max(pair_counts) * 1.4 if pair_counts else 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def plot_bert_training_progress(history_path, output_dir):
    """BERT 訓練曲線（Loss + Accuracy）"""
    if not os.path.exists(history_path):
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = list(range(1, len(history['eval_acc']) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(epochs, history['train_loss'], 'o-', color=BLUE, linewidth=2, label='Train', markersize=6)
    ax1.plot(epochs, history['eval_loss'],  'o-', color=TEAL, linewidth=2, label='Eval', markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Eval Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'o-', color=BLUE, linewidth=2, label='Train', markersize=6)
    ax2.plot(epochs, [a * 100 for a in history['eval_acc']],  'o-', color=TEAL, linewidth=2, label='Eval', markersize=6)
    for ep, acc in zip(epochs, history['eval_acc']):
        ax2.text(ep, acc * 100 + 1.5, f'{acc*100:.1f}%', ha='center', fontsize=9, fontweight='bold', color=NAVY)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Eval Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('BERT Fine-tuning Progress', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'bert_training_progress.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description='從已儲存的 JSON 報告產出模型評估圖表')
    parser.add_argument('--output', default=None, help='圖表輸出目錄（預設: outputs/）')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output or os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    models_dir = os.path.join(project_root, 'models')

    # ── 載入報告 ──
    svm_report  = load_report(os.path.join(models_dir, 'eval_report.json'))
    bert_report = load_report(os.path.join(models_dir, 'bert_eval_report.json'))
    bert_history_path = os.path.join(models_dir, 'bert_history.json')

    generated = []

    print("=" * 60)
    print("  模型評估圖表產生器")
    print("=" * 60)

    # ── SVM 圖表 ──
    if svm_report:
        svm_metrics = extract_macro_metrics(svm_report)
        print(f"\n  SVM 指標: Acc={svm_metrics['Accuracy']:.1f}%  "
              f"P={svm_metrics['Precision']:.1f}%  R={svm_metrics['Recall']:.1f}%  "
              f"F1={svm_metrics['F1-Score']:.1f}%")

        p = plot_4metrics(svm_metrics, 'LinearSVC — Macro Average Metrics', 'svm_4metrics.png', output_dir)
        generated.append(p)

        p = plot_domain_accuracy(svm_report, 'SVM', 'svm_domain_accuracy.png', output_dir)
        generated.append(p)

        p = plot_confusion_pairs(svm_report, 'SVM', 'svm_confusion_pairs.png', output_dir)
        if p: generated.append(p)
    else:
        print("\n  ⚠️  找不到 SVM 報告 (models/eval_report.json)")

    # ── BERT 圖表 ──
    if bert_report:
        bert_metrics = extract_macro_metrics(bert_report)
        print(f"\n  BERT 指標: Acc={bert_metrics['Accuracy']:.1f}%  "
              f"P={bert_metrics['Precision']:.1f}%  R={bert_metrics['Recall']:.1f}%  "
              f"F1={bert_metrics['F1-Score']:.1f}%")

        p = plot_4metrics(bert_metrics, 'BERT Fine-tune — Macro Average Metrics', 'bert_4metrics.png', output_dir)
        generated.append(p)

        p = plot_domain_accuracy(bert_report, 'BERT', 'bert_domain_accuracy.png', output_dir)
        generated.append(p)

        p = plot_confusion_pairs(bert_report, 'BERT', 'bert_confusion_pairs.png', output_dir)
        if p: generated.append(p)
    else:
        print("\n  ⚠️  找不到 BERT 報告 (models/bert_eval_report.json)")
        print("     請先執行: python -m src.eval_bert")

    # ── 對比圖表（需要兩份報告都有）──
    if svm_report and bert_report:
        svm_metrics  = extract_macro_metrics(svm_report)
        bert_metrics = extract_macro_metrics(bert_report)

        # 四指標對比
        p = plot_comparison_4metrics(svm_metrics, bert_metrics, output_dir)
        generated.append(p)

        # 領域對比
        p = plot_domain_comparison(svm_report, bert_report, output_dir)
        generated.append(p)

        # 三模型準確率（LR = 88.4% baseline）
        p = plot_model_accuracy_comparison(88.4, svm_metrics['Accuracy'], bert_metrics['Accuracy'], output_dir)
        generated.append(p)

    # ── BERT 訓練曲線 ──
    p = plot_bert_training_progress(bert_history_path, output_dir)
    if p:
        generated.append(p)
    else:
        print("\n  ⚠️  找不到 BERT 訓練歷史 (models/bert_history.json)")

    # ── 結果摘要 ──
    print("\n" + "=" * 60)
    print(f"  共產出 {len(generated)} 張圖表，儲存至: {output_dir}/")
    print("=" * 60)
    for p in generated:
        print(f"    ✅ {os.path.basename(p)}")

    print("\n  完成！")


if __name__ == '__main__':
    main()
