# 富邦金控 MA 甄選 — 題目二：客戶意圖分類

## 專案概述

本專案針對 **150 類客戶意圖** 進行自動分類，涵蓋完整的 NLP Pipeline：資料前處理、特徵工程、多層次模型訓練與評估、多意圖拆解，以及 RESTful API 部署與互動式 Demo Dashboard。

### 模型成績

| 模型 | Eval Accuracy | 備註 |
|------|:---:|------|
| Logistic Regression | 88.4% | Baseline |
| LinearSVC + TF-IDF | 89.1% | 最佳傳統模型，5-fold CV: 92.5% |
| **BERT fine-tune** | **96.7%** | **Best Model**，較 SVM 提升 7.6% |

## 專案結構

```
fubon_intent_project/
├── data/
│   ├── raw/                          # 原始資料集（未上傳，見 Data 章節）
│   └── processed/                    # 前處理後的資料（未上傳）
├── notebooks/
│   ├── 01_EDA_and_cleaning.ipynb     # 探索式分析與資料清理
│   └── 02_model_experiments.ipynb    # 模型實驗與比較
├── src/
│   ├── data_preprocess.py            # 文字前處理 + TF-IDF 向量化
│   ├── model_trainer.py              # 傳統 ML + BERT 訓練器
│   ├── evaluator.py                  # 模型評估與報告產出
│   ├── train_bert.py                 # BERT fine-tuning 獨立腳本
│   ├── eval_bert.py                  # BERT 模型驗證腳本
│   └── logger.py                     # 共用 logging 模組
├── models/                           # 訓練產出（.pkl 未上傳）
├── app/
│   ├── main.py                       # FastAPI 服務主程式
│   ├── llm_handler.py                # 多意圖拆解模組
│   ├── dashboard.html                # 互動式 Demo Dashboard（API 版）
│   └── dashboard_static.html         # 離線展示版 Dashboard
├── logs/                             # 執行日誌
├── requirements.txt
└── README.md
```

## Data

> ⚠️ 本專案使用的資料集為富邦金控甄選提供，受保密協議約束，**不包含在此 Repository 中**。

資料規格：
- 訓練集：15,000 筆（150 類 × 100 筆）
- 評估集：3,000 筆
- 格式：JSON（含 `text` 和 `intent` 欄位）
- 語言：英文短句，平均 8.3 字

如需復現結果，請將資料放置於 `data/raw/` 後執行前處理。

## 快速開始

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 完整 Pipeline

```bash
# Step 1: 資料前處理
python -m src.data_preprocess

# Step 2: 傳統 ML 訓練（LR + SVM）
python -m src.model_trainer

# Step 3: 模型評估
python -m src.evaluator

# Step 4: BERT Fine-tuning（需 GPU）
python -m src.train_bert

# Step 5: BERT 評估
python -m src.eval_bert

# Step 6: 啟動 API + Dashboard
uvicorn app.main:app --reload --port 8000
```

## 技術架構

### 前處理 Pipeline
小寫轉換 → 移除標點符號 → 空白正規化 → 停用詞移除 → TF-IDF 向量化

### 特徵工程
- **Word TF-IDF**：unigram + bigram（8,998 維），捕捉詞彙與片語語義
- **Char TF-IDF**：3-gram ~ 5-gram（28,456 維），捕捉拼寫模式與子詞資訊
- 合併為 37,454 維特徵矩陣，搭配 Sublinear TF 平滑高頻詞

### BERT Fine-tuning
- 模型：`bert-base-uncased`
- 訓練配置：lr=2e-5, batch_size=32, epochs=5, warmup_ratio=10%
- 訓練過程：Eval Acc 從 71.6%（Epoch 1）→ 96.7%（Epoch 5）
- 使用 PyTorch AdamW + Linear Warmup Scheduler

### 多意圖拆解
規則式複合句偵測 + 子句分割 + 逐句分類。支援連接詞偵測（and + 動詞、also、then、分號等），使用 lookahead regex 保留動詞語義。

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| POST | `/predict` | 統一意圖預測（自動偵測單一/多重意圖） |
| GET | `/health` | 健康檢查 |
| GET | `/intents` | 列出所有意圖類別 |
| GET | `/bert-history` | BERT 訓練歷史 |
| GET | `/bert-eval-report` | BERT 評估報告 |
| GET | `/` | 互動式 Demo Dashboard |

### 範例

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "check my balance and book a flight to tokyo"}'
```

```json
{
  "original_text": "check my balance and book a flight to tokyo",
  "is_compound": true,
  "intents": [
    {"intent": "balance", "text": "check my balance", "confidence": 1.4787},
    {"intent": "book_flight", "text": "book a flight to tokyo", "confidence": 1.6234}
  ]
}
```

## 領域分類

系統涵蓋 9 大領域、150 個意圖類別：

| 領域 | 意圖數 | 範例 |
|------|:---:|------|
| Finance | 36 | balance, transfer, credit_score, pay_bill |
| Assistant | 28 | alarm, reminder, calendar, weather |
| Chitchat | 27 | greeting, are_you_a_bot, tell_joke |
| Travel | 15 | book_flight, flight_status, lost_luggage |
| Food | 15 | recipe, restaurant_suggestion, nutrition |
| Entertainment | 12 | play_music, smart_home, volume |
| Vehicle | 10 | oil_change, tire_pressure, gas |
| Shopping | 5 | order_status, exchange_rate |
| Work_HR | 4 | pto_request, pto_balance |

## 環境需求

- Python 3.8+
- CUDA GPU（BERT 訓練用，推論可用 CPU）
- 主要套件：scikit-learn, transformers, torch, fastapi
