# 客戶意圖分類

## 專案概述

本專案針對 **150 類客戶意圖** 進行自動分類，涵蓋完整的 NLP Pipeline：資料前處理、特徵工程、多層次模型訓練與評估、多意圖拆解、雙層智慧路由、LLM Prompt 工程，以及 RESTful API 部署與互動式 Dashboard。

### 模型成績

| 模型 | Eval Accuracy | 備註 |
|------|:---:|------|
| Logistic Regression | 88.4% | Baseline |
| LinearSVC + TF-IDF | 89.1% | 最佳傳統模型，5-fold CV: 92.5% |
| **BERT fine-tune** | **96.7%** | **Best Model**，較 SVM 提升 7.6% |

### Live Demo

> **線上服務**：[https://intent-engine-dp45.onrender.com](https://intent-engine-dp45.onrender.com)
>
> API 文件：[https://intent-engine-dp45.onrender.com/docs](https://intent-engine-dp45.onrender.com/docs)

## 系統架構

```
使用者輸入
    │
    ▼
┌──────────────────────────────────┐
│  雙層智慧路由 (DualLayerRouter)   │
│                                  │
│  SVM 輕量層（~80% 流量）          │
│    confidence >= 0.8 → 直接回傳   │
│    confidence < 0.8  ↓           │
│  BERT 重量層（~20% 流量）         │
│    fine-tuned, 96.7% accuracy    │
│                                  │
│  LLM 層（Claude API，可選）       │
│    複合意圖拆解 / 隱藏意圖分析     │
└──────────────────────────────────┘
```

## 專案結構

```
fubon_intent_project/
├── app/
│   ├── main.py                       # FastAPI 服務主程式（12 個端點）
│   ├── dual_router.py                # 雙層智慧路由器（SVM + BERT + LLM）
│   ├── llm_classifier.py             # LLM 意圖分類（Claude API，二階段分類）
│   ├── llm_handler.py                # 多意圖拆解模組
│   ├── intent_mining.py              # 隱藏意圖挖掘（規則引擎 + SQLite）
│   ├── dashboard.html                # 互動式 Demo Dashboard（API 版）
│   └── dashboard_static.html         # 離線展示版 Dashboard
├── src/
│   ├── config.py                     # 意圖領域映射（9 領域 × 150 意圖）
│   ├── data_preprocess.py            # 文字前處理 + TF-IDF 向量化
│   ├── model_trainer.py              # 傳統 ML 訓練器（LR + SVM）
│   ├── train_bert.py                 # BERT fine-tuning 腳本
│   ├── evaluator.py                  # SVM 模型評估
│   ├── eval_bert.py                  # BERT 模型評估
│   ├── plot_model_metrics.py         # 圖表產生
│   └── logger.py                     # 共用 logging 模組
├── models/
│   ├── svm_best.pkl                  # SVM 模型（LinearSVC）
│   ├── tfidf_vectorizer.pkl          # TF-IDF 向量器
│   ├── bert_intent/                  # BERT 模型（從 HF Hub 下載，不含在 git 中）
│   ├── eval_report.json              # SVM 評估報告
│   ├── bert_eval_report.json         # BERT 評估報告
│   └── bert_history.json             # BERT 訓練歷史
├── data/                             # 訓練資料（不含在 git 中）
├── notebooks/
│   ├── 01_EDA_and_cleaning.ipynb     # 探索式分析與資料清理
│   └── 02_model_experiments.ipynb    # 模型實驗與比較
├── tests/                            # 單元測試
├── outputs/                          # 圖表產出
├── download_model.py                 # Render 部署時從 HF Hub 下載 BERT
├── render.yaml                       # Render 部署設定
├── Procfile                          # 部署啟動指令
├── requirements.txt
└── README.md
```

## Data

> ⚠️ 本專案使用的資料集受保密協議約束，**不包含在此 Repository 中**。

資料規格：
- 訓練集：15,000 筆（150 類 × 100 筆）
- 評估集：3,000 筆
- 格式：JSON（含 `text` 和 `intent` 欄位）
- 語言：英文短句，平均 8.3 字

如需復現結果，請將資料放置於 `data/raw/` 後執行前處理。

## 快速開始

### 本地開發

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2. 資料前處理
python -m src.data_preprocess

# 3. 傳統 ML 訓練（LR + SVM）
python -m src.model_trainer

# 4. 模型評估
python -m src.evaluator

# 5. BERT Fine-tuning（需 GPU）
python -m src.train_bert

# 6. BERT 評估
python -m src.eval_bert

# 7. 啟動 API + Dashboard
uvicorn app.main:app --reload --port 8000
```

### 部署至 Render

本專案已部署於 Render，BERT 模型透過 Hugging Face Hub 下載，避免存放於 Git。

```bash
# 環境變數（在 Render Dashboard 設定）
ANTHROPIC_API_KEY=your_key_here   # Claude API（可選，LLM 功能用）
PYTHON_VERSION=3.12.12
```

## 技術架構

### 前處理 Pipeline
小寫轉換 → 移除標點符號 → 空白正規化 → 停用詞移除 → TF-IDF 向量化

### 特徵工程
- **Word TF-IDF**：unigram + bigram（8,998 維），捕捉詞彙與片語語義
- **Char TF-IDF**：3-gram ~ 5-gram（28,456 維），捕捉拼寫模式與子詞資訊
- 合併為 37,454 維特徵矩陣，搭配 Sublinear TF 平滑高頻詞

### BERT Fine-tuning
- 模型：`bert-base-uncased`（託管於 [Hugging Face](https://huggingface.co/yitommy317/fubon-intent-classifier)）
- 訓練配置：lr=2e-5, batch_size=32, epochs=5, warmup_ratio=10%
- 訓練過程：Eval Acc 從 71.6%（Epoch 1）→ 96.7%（Epoch 5）
- 使用 PyTorch AdamW + Linear Warmup Scheduler

### 雙層智慧路由
SVM 輕量層處理 ~80% 高信心度請求（免費、< 5ms），BERT 重量層處理 ~20% 低信心度或複合意圖請求。LLM（Claude API）作為可選的深度分析層，提供隱藏意圖挖掘與複合句理解。

### LLM Prompt 工程
- **二階段分類**：先判斷領域（9 類）再分類意圖，縮小搜索空間
- **多意圖 Prompt**：一次辨識文本中所有意圖，含信心度與證據
- **隱藏意圖分析**：輸入客戶歷史意圖，推理潛在需求
- **成本控制**：LRU Cache + MD5 避免重複呼叫、max_tokens 限制、錯誤時自動降級回 BERT

### 多意圖拆解
規則式複合句偵測 + 子句分割 + 逐句分類。支援連接詞偵測（and + 動詞、also、then、分號等），使用 lookahead regex 保留動詞語義。

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| POST | `/predict` | 統一意圖預測（自動偵測單一/多重意圖） |
| POST | `/predict/smart` | 雙層智慧路由預測 |
| POST | `/predict/llm` | LLM 多意圖分類（Claude API） |
| POST | `/compare` | SVM vs LLM 並排比較 |
| POST | `/intent-log` | 紀錄意圖事件 |
| POST | `/llm/hidden-intents/{customer_id}` | LLM 隱藏意圖分析 |
| GET | `/routing-stats` | 雙層路由統計 |
| GET | `/llm-stats` | LLM 使用統計 |
| GET | `/intent-mining/{customer_id}` | 客戶隱藏意圖分析 |
| GET | `/intent-mining` | 批次挖掘所有客戶 |
| GET | `/health` | 健康檢查 |
| GET | `/intents` | 列出所有意圖類別 |
| GET | `/` | 互動式 Demo Dashboard |

### 範例

```bash
# 基本意圖預測
curl -X POST https://intent-engine-dp45.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "check my balance and book a flight to tokyo"}'

# 雙層智慧路由
curl -X POST https://intent-engine-dp45.onrender.com/predict/smart \
  -H "Content-Type: application/json" \
  -d '{"text": "check my balance", "customer_id": "CUST001"}'
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
- 主要套件：scikit-learn, transformers, torch, fastapi, anthropic
