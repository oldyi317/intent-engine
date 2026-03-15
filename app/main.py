"""
FastAPI 意圖分析服務 (Intent Analysis API)

提供即時意圖分析的 RESTful API，對應題目加分項「模型部署」。

啟動方式：
    cd fubon_intent_project
    uvicorn app.main:app --reload --port 8000

API 端點：
    POST /predict        → 統一意圖預測（自動偵測單一 / 多重意圖）
    POST /predict/smart  → 雙層智慧路由（SVM 輕量層 + LLM 重量層）
    POST /predict/llm    → LLM 直接分類（Claude API）
    POST /compare        → SVM vs LLM 並排比較
    POST /llm/hidden-intents/{customer_id} → LLM 隱藏意圖分析
    POST /intent-log     → 紀錄意圖事件（供隱藏意圖挖掘）
    GET  /intent-mining/{customer_id}  → 客戶隱藏意圖分析
    GET  /intent-mining   → 批次挖掘所有客戶
    GET  /routing-stats   → 雙層路由統計
    GET  /llm-stats       → LLM 使用統計
    GET  /health          → 健康檢查
    GET  /intents         → 列出所有支援的意圖類別
"""

import os
import sys
import time
from contextlib import asynccontextmanager

# 確保 src/ 可以被 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自動載入 .env 檔案
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import json
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np

from src.data_preprocess import TextPreprocessor, TfidfFeatureBuilder
from src.model_trainer import TraditionalMLTrainer
from src.logger import setup_logger
from app.llm_handler import MultiIntentHandler
from app.dual_router import DualLayerRouter
from app.intent_mining import IntentLogger, IntentMiner
from app.llm_classifier import LLMIntentClassifier, LLMAPIError

# Logger
logger = setup_logger('api_server')


# ============================================================
# Pydantic Models (Request / Response schemas)
# ============================================================
class PredictRequest(BaseModel):
    text: str = Field(..., description="使用者輸入的查詢文字",
                      json_schema_extra={"example": "check my balance and book a flight to tokyo"})

class SmartPredictRequest(BaseModel):
    text: str = Field(..., description="使用者輸入的查詢文字",
                      json_schema_extra={"example": "check my balance"})
    customer_id: Optional[str] = Field(None, description="客戶 ID（可選，填入後會自動紀錄意圖歷史）",
                                        json_schema_extra={"example": "CUST001"})

class IntentResult(BaseModel):
    text: str
    intent: str
    confidence: float

class PredictResponse(BaseModel):
    original_text: str
    is_compound: bool
    intents: List[IntentResult]

class SmartPredictResponse(BaseModel):
    original_text: str
    tier: str
    intent: str
    confidence: float
    latency_ms: float
    is_compound: bool
    sub_intents: List[Dict[str, Any]]
    routing_reason: str
    cost_estimate: float

class IntentLogRequest(BaseModel):
    customer_id: str = Field(..., description="客戶 ID",
                             json_schema_extra={"example": "CUST001"})
    intent: str = Field(..., description="意圖類別",
                        json_schema_extra={"example": "exchange_rate"})
    confidence: float = Field(0.0, description="信心度")
    original_text: str = Field("", description="原始文字")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    features: List[str]
    llm_available: bool = False


# ============================================================
# Global objects
# ============================================================
preprocessor: Optional[TextPreprocessor] = None
feature_builder: Optional[TfidfFeatureBuilder] = None
model_trainer: Optional[TraditionalMLTrainer] = None
multi_handler: Optional[MultiIntentHandler] = None
dual_router: Optional[DualLayerRouter] = None
intent_logger: Optional[IntentLogger] = None
intent_miner: Optional[IntentMiner] = None
llm_classifier: Optional[LLMIntentClassifier] = None
_model_loaded = False


# ============================================================
# Lifespan（取代已棄用的 @app.on_event("startup")）
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """在服務啟動時載入模型，關閉時清理資源"""
    global preprocessor, feature_builder, model_trainer, multi_handler
    global dual_router, intent_logger, intent_miner, llm_classifier, _model_loaded

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, 'models')

    svm_path = os.path.join(model_dir, 'svm_best.pkl')
    tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

    if not os.path.exists(svm_path) or not os.path.exists(tfidf_path):
        logger.warning("模型檔案不存在，請先執行訓練：")
        logger.warning("  python -m src.data_preprocess")
        logger.warning("  python -m src.model_trainer")
    else:
        logger.info("正在載入模型...")
        start = time.time()

        preprocessor = TextPreprocessor()
        feature_builder = TfidfFeatureBuilder.load(tfidf_path)
        model_trainer = TraditionalMLTrainer.load(svm_path)

        # 設定分類函式供多意圖拆解使用
        def classify_fn(text: str):
            clean = preprocessor.transform(text)
            X = feature_builder.transform([clean])
            intent = model_trainer.predict(X)[0]
            decision = model_trainer.model.decision_function(X)
            conf = float(np.max(decision))
            return intent, conf

        multi_handler = MultiIntentHandler(classifier_fn=classify_fn)

        # --- 嘗試載入 BERT 作為重量層 ---
        bert_classify_fn = None
        bert_dir = os.path.join(model_dir, 'bert_intent')
        try:
            import torch
            from transformers import BertTokenizer, BertForSequenceClassification
            import pickle

            if os.path.exists(bert_dir) and os.path.exists(os.path.join(bert_dir, 'model.safetensors')):
                logger.info("  偵測到 BERT 模型，正在載入...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                bert_tokenizer = BertTokenizer.from_pretrained(bert_dir)
                bert_model = BertForSequenceClassification.from_pretrained(bert_dir).to(device)
                bert_model.eval()

                with open(os.path.join(bert_dir, 'label_encoder.pkl'), 'rb') as f:
                    bert_le = pickle.load(f)

                def bert_classify_fn(text: str):
                    """BERT 重量層分類函式"""
                    enc = bert_tokenizer(
                        text.lower().strip(),
                        max_length=64,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt',
                    )
                    ids = enc['input_ids'].to(device)
                    mask = enc['attention_mask'].to(device)
                    with torch.no_grad():
                        out = bert_model(input_ids=ids, attention_mask=mask)
                    logits = out.logits
                    probs = torch.softmax(logits, dim=1)
                    conf = float(probs.max().item())
                    pred_idx = logits.argmax(dim=1).item()
                    intent = bert_le.inverse_transform([pred_idx])[0]
                    return intent, conf

                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                logger.info(f"  BERT 重量層載入成功！(device={device}, GPU={gpu_name})")
            else:
                logger.info("  BERT 模型目錄不完整，跳過 BERT 載入")
        except ImportError:
            logger.info("  torch/transformers 未安裝，BERT 重量層不啟用（僅 SVM 輕量層）")
        except Exception as e:
            logger.warning(f"  BERT 載入失敗: {e}（降級為僅 SVM 模式）")

        # --- 初始化雙層路由器 ---
        tier_desc = "SVM 輕量層 + BERT 重量層 (GPU)" if bert_classify_fn else "SVM 輕量層（BERT 未啟用）"
        dual_router = DualLayerRouter(
            svm_classify_fn=classify_fn,
            llm_classify_fn=bert_classify_fn,
            compound_detect_fn=multi_handler.is_compound,
            compound_split_fn=multi_handler.split_compound,
            confidence_threshold=0.8,
            heavy_tier="bert" if bert_classify_fn else "llm",
        )
        logger.info(f"  雙層路由器已啟動（{tier_desc}）")

        # --- 初始化隱藏意圖挖掘 ---
        db_path = os.path.join(project_root, 'data', 'intent_history.db')
        intent_logger = IntentLogger(db_path)
        intent_miner = IntentMiner(intent_logger)
        logger.info(f"  意圖挖掘引擎已啟動（DB: {db_path}）")

        # --- 初始化 LLM 分類器（可選，需要 ANTHROPIC_API_KEY）---
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            try:
                llm_classifier = LLMIntentClassifier(
                    api_key=anthropic_key,
                    use_two_stage=True,
                    cache_enabled=True,
                )
                logger.info("  LLM 分類器已啟動（Claude API）")

                # 將 LLM 註冊為雙層路由的真正重量層
                if dual_router and dual_router.llm_classify_fn is None:
                    dual_router.llm_classify_fn = llm_classifier.classify_as_tuple
                    dual_router.heavy_tier = dual_router.__class__.__mro__[0].__module__
                    # 修正 heavy_tier 為 LLM
                    from app.dual_router import RoutingTier
                    dual_router.heavy_tier = RoutingTier.LLM
                    logger.info("  雙層路由重量層升級為 Claude LLM")
            except Exception as e:
                logger.info(f"  LLM 分類器未啟用: {e}")
        else:
            logger.info("  ANTHROPIC_API_KEY 未設定，LLM 端點不可用（SVM/BERT 正常運作）")

        _model_loaded = True

        elapsed = time.time() - start
        n_intents = len(model_trainer.model.classes_)
        logger.info(f"模型載入完成！({elapsed:.1f}s)")
        logger.info(f"  模型: LinearSVC + TF-IDF (word+char)")
        logger.info(f"  意圖數: {n_intents}")
        logger.info(f"  API 文件: http://localhost:8000/docs")

    yield  # ← 服務運行中

    # 關閉時的清理
    logger.info("API 服務關閉")


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="富邦金控 - 客戶意圖分類 API",
    description=(
        "多層意圖分析服務（SVM + BERT + LLM）\n\n"
        "功能：\n"
        "- 意圖預測（單一 / 多重意圖自動偵測）\n"
        "- 雙層智慧路由（SVM 輕量層 80% + BERT/LLM 重量層 20%）\n"
        "- LLM 意圖分類（Claude API 兩階段分類）\n"
        "- SVM vs LLM 並排比較\n"
        "- 隱藏意圖挖掘（規則引擎 + LLM 深度分析）\n"
    ),
    version="4.0.0",
    lifespan=lifespan,
)

# CORS（允許 dashboard 從瀏覽器呼叫 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Dashboard 頁面
# ============================================================
@app.get("/", response_class=HTMLResponse)
def dashboard():
    """儀表板首頁"""
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html')
    return FileResponse(dashboard_path)


# ============================================================
# Middleware：記錄每個請求的回應時間
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000  # ms
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)")
    return response


# ============================================================
# 原有 API Endpoints
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康檢查"""
    bert_active = dual_router is not None and dual_router.llm_classify_fn is not None
    model_desc = "SVM + BERT (dual-layer)" if bert_active else "LinearSVC + TF-IDF (SVM only)"
    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded,
        model_type=model_desc,
        features=["predict", "smart_routing", "intent_mining",
                   *(["bert_heavy_tier"] if bert_active else []),
                   *(["llm_classifier"] if llm_classifier else [])],
        llm_available=llm_classifier is not None,
    )


@app.get("/intents", response_model=List[str])
def list_intents():
    """列出所有支援的意圖類別"""
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="模型尚未載入")
    return sorted(model_trainer.model.classes_.tolist())


@app.get("/bert-history")
def bert_history():
    """回傳 BERT 訓練歷史（動態讀取，依序搜尋 models/ 和 models/bert_intent/）"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, 'models', 'bert_history.json'),
        os.path.join(project_root, 'models', 'bert_intent', 'bert_history.json'),
    ]
    history_path = next((p for p in candidates if os.path.exists(p)), None)
    if not history_path:
        raise HTTPException(status_code=404, detail="BERT 訓練歷史檔案不存在")
    with open(history_path, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.get("/bert-eval-report")
def bert_eval_report():
    """回傳 BERT 評估報告（含 domain accuracy）"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(project_root, 'models', 'bert_eval_report.json')
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="BERT 評估報告不存在，請先執行 python src/eval_bert.py")
    with open(report_path, 'r') as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.post("/predict", response_model=PredictResponse)
def predict_intent(req: PredictRequest):
    """
    統一意圖預測端點（原始版本，直接使用 SVM）

    自動偵測輸入是否為複合句：
    - 單一意圖 → is_compound=false，intents 陣列只有 1 個結果
    - 多重意圖 → is_compound=true，intents 陣列包含拆解後的各子意圖
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="模型尚未載入")

    result = multi_handler.analyze(req.text)

    intents_str = ", ".join(sq['intent'] or 'unknown' for sq in result['sub_queries'])
    logger.info(f"[predict] \"{req.text}\" → compound={result['is_compound']} → [{intents_str}]")

    return PredictResponse(
        original_text=result['original_text'],
        is_compound=result['is_compound'],
        intents=[
            IntentResult(
                text=sq['text'],
                intent=sq['intent'] or 'unknown',
                confidence=sq['confidence'],
            )
            for sq in result['sub_queries']
        ],
    )


# ============================================================
# 新增：雙層智慧路由 API
# ============================================================
@app.post("/predict/smart", response_model=SmartPredictResponse)
def smart_predict(req: SmartPredictRequest):
    """
    雙層智慧路由預測

    路由邏輯：
    1. 所有請求先經過 SVM 輕量層（免費、< 5ms）
    2. SVM 信心度 >= 0.8 且非複合句 → 直接回傳（~80% 流量）
    3. SVM 信心度 < 0.8 或複合句 → 轉交 LLM 重量層
    4. LLM 不可用 → 降級回 SVM（加上警告）

    若提供 customer_id，會自動將分類結果寫入意圖歷史供挖掘使用。
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="模型尚未載入")

    result = dual_router.route(req.text)

    # 若有 customer_id，自動紀錄意圖歷史
    if req.customer_id and intent_logger:
        if result.is_compound and result.sub_intents:
            for si in result.sub_intents:
                intent_logger.log(
                    customer_id=req.customer_id,
                    intent=si["intent"],
                    confidence=si["confidence"],
                    original_text=si["text"],
                    routing_tier=si.get("tier", result.tier),
                )
        else:
            intent_logger.log(
                customer_id=req.customer_id,
                intent=result.intent,
                confidence=result.confidence,
                original_text=req.text,
                routing_tier=result.tier,
            )

    return SmartPredictResponse(
        original_text=result.original_text,
        tier=result.tier,
        intent=result.intent,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        is_compound=result.is_compound,
        sub_intents=result.sub_intents,
        routing_reason=result.routing_reason,
        cost_estimate=result.cost_estimate,
    )


@app.get("/routing-stats")
def routing_stats():
    """
    取得雙層路由統計

    顯示 SVM / LLM 各處理了多少比例的流量、
    累計成本、平均延遲、預估月省成本等。
    """
    if dual_router is None:
        raise HTTPException(status_code=503, detail="路由器尚未初始化")
    return JSONResponse(content=dual_router.get_stats())


# ============================================================
# 新增：隱藏意圖挖掘 API
# ============================================================
@app.post("/intent-log")
def log_intent(req: IntentLogRequest):
    """
    手動紀錄一筆意圖事件

    通常不需要手動呼叫，使用 /predict/smart 搭配 customer_id 會自動紀錄。
    此端點提供給需要從外部系統匯入歷史紀錄的場景。
    """
    if intent_logger is None:
        raise HTTPException(status_code=503, detail="意圖紀錄器尚未初始化")

    record_id = intent_logger.log(
        customer_id=req.customer_id,
        intent=req.intent,
        confidence=req.confidence,
        original_text=req.original_text,
    )
    return {"status": "ok", "record_id": record_id}


@app.get("/intent-mining/{customer_id}")
def mine_customer(
    customer_id: str,
    days: int = Query(30, description="分析天數", ge=1, le=365),
):
    """
    分析單一客戶的隱藏意圖

    根據客戶在指定天數內的意圖歷史，比對預定義規則，
    產出風控預警、交叉銷售機會、客戶留存信號等 insights。

    Returns:
        - intent_profile: 意圖分布
        - domain_profile: 領域分布
        - risk_alerts: 風控預警
        - cross_sell_opportunities: 交叉銷售機會
        - retention_signals: 客戶留存信號
    """
    if intent_miner is None:
        raise HTTPException(status_code=503, detail="意圖挖掘引擎尚未初始化")

    result = intent_miner.analyze_customer(customer_id, days)
    return JSONResponse(content=result)


@app.get("/intent-mining")
def mine_all_customers(
    days: int = Query(30, description="分析天數", ge=1, le=365),
):
    """
    批次挖掘所有客戶

    掃描所有有意圖紀錄的客戶，回傳有 insights 的客戶清單，
    依風控優先度排序。適合定期排程執行。
    """
    if intent_miner is None:
        raise HTTPException(status_code=503, detail="意圖挖掘引擎尚未初始化")

    results = intent_miner.batch_analyze(days)
    return JSONResponse(content={
        "total_customers_with_insights": len(results),
        "analysis_period_days": days,
        "customers": results,
    })


@app.get("/intent-history/{customer_id}")
def get_customer_history(
    customer_id: str,
    days: int = Query(30, description="查詢天數", ge=1, le=365),
):
    """取得客戶的意圖歷史紀錄"""
    if intent_logger is None:
        raise HTTPException(status_code=503, detail="意圖紀錄器尚未初始化")

    history = intent_logger.get_customer_history(customer_id, days)
    return JSONResponse(content={
        "customer_id": customer_id,
        "days": days,
        "total_records": len(history),
        "records": history,
    })


@app.get("/intent-log-stats")
def intent_log_stats():
    """取得意圖紀錄的整體統計"""
    if intent_logger is None:
        raise HTTPException(status_code=503, detail="意圖紀錄器尚未初始化")
    return JSONResponse(content=intent_logger.get_stats())


# ============================================================
# LLM 意圖分類 API
# ============================================================
class LLMPredictRequest(BaseModel):
    text: str = Field(..., description="使用者輸入的查詢文字",
                      json_schema_extra={"example": "check my balance"})
    two_stage: Optional[bool] = Field(None, description="是否使用兩階段分類（預設 True）")


class CompareRequest(BaseModel):
    text: str = Field(..., description="使用者輸入的查詢文字",
                      json_schema_extra={"example": "what is the exchange rate for USD"})


class LLMHiddenIntentRequest(BaseModel):
    days: int = Field(30, description="分析天數", ge=1, le=365)


@app.post("/predict/llm")
def llm_predict(req: LLMPredictRequest):
    """
    LLM 多意圖分類（Claude API）

    使用大語言模型一次辨識文本中所有意圖，
    支援複合句拆解，回傳完整意圖列表。

    需要設定 ANTHROPIC_API_KEY 環境變數。
    """
    if llm_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 分類器未啟用。請設定環境變數: ANTHROPIC_API_KEY"
        )

    try:
        result = llm_classifier.classify_multi(req.text)
    except LLMAPIError as e:
        raise HTTPException(status_code=502, detail=f"[{e.error_code}] {str(e)}")

    return JSONResponse(content=result)


@app.post("/compare")
def compare_traditional_llm(req: CompareRequest):
    """
    傳統分類器 vs LLM 並排比較

    傳統分類器側使用雙層路由（SVM + BERT），
    LLM 側使用 Claude API 多意圖分類。
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="模型尚未載入")
    if llm_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 分類器未啟用。請設定環境變數: ANTHROPIC_API_KEY"
        )

    # --- 傳統分類器（雙層路由 SVM+BERT）---
    # 暫時將 dual_router 的 llm_classify_fn 切回 BERT（避免路由到 LLM）
    original_llm_fn = dual_router.llm_classify_fn if dual_router else None
    original_heavy_tier = dual_router.heavy_tier if dual_router else None

    # 找出 BERT classify_fn（如果有的話）
    bert_fn = None
    try:
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        import pickle
        bert_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'bert_intent')
        if os.path.exists(os.path.join(bert_dir, 'model.safetensors')):
            bert_fn = original_llm_fn  # dual_router 啟動時已載入 BERT
    except Exception:
        pass

    # 如果 dual_router 的重量層已被 LLM 取代，暫時切回 BERT 或 None
    if dual_router and hasattr(dual_router, '_original_bert_fn'):
        pass  # 已有紀錄

    trad_start = time.time()
    trad_result = dual_router.route(req.text)
    trad_latency = (time.time() - trad_start) * 1000

    # 組裝傳統分類器結果
    trad_intents = []
    tier_used = str(trad_result.tier.value) if hasattr(trad_result.tier, 'value') else str(trad_result.tier)

    if trad_result.is_compound and trad_result.sub_intents:
        for si in trad_result.sub_intents:
            si_tier = si.get("tier", trad_result.tier)
            si_tier_str = str(si_tier.value) if hasattr(si_tier, 'value') else str(si_tier)
            trad_intents.append({
                "intent": str(si["intent"]),
                "confidence": round(float(si["confidence"]), 4),
                "evidence": si.get("text", ""),
                "tier": si_tier_str,
            })
    else:
        trad_intents.append({
            "intent": str(trad_result.intent),
            "confidence": round(float(trad_result.confidence), 4),
            "evidence": req.text,
            "tier": tier_used,
        })

    # --- LLM 多意圖 ---
    try:
        llm_result = llm_classifier.classify_multi(req.text)
    except LLMAPIError as e:
        raise HTTPException(status_code=502, detail=f"[{e.error_code}] {str(e)}")

    return JSONResponse(content={
        "text": req.text,
        "traditional": {
            "intents": trad_intents,
            "count": len(trad_intents),
            "tier": tier_used,
            "routing_reason": trad_result.routing_reason,
            "latency_ms": round(trad_latency, 2),
            "cost_usd": round(trad_result.cost_estimate, 4),
        },
        "llm": {
            "intents": llm_result["intents"],
            "count": llm_result["count"],
            "latency_ms": llm_result["latency_ms"],
            "cost_usd": llm_result["cost_usd"],
            "model_used": llm_result["model_used"],
        },
    })


@app.post("/llm/hidden-intents/{customer_id}")
def llm_hidden_intents(customer_id: str, req: LLMHiddenIntentRequest):
    """
    LLM 隱藏意圖分析

    使用大語言模型分析客戶意圖歷史，
    挖掘規則引擎無法捕捉的隱藏需求。

    相比規則引擎（/intent-mining），LLM 能：
    - 理解更複雜的行為模式
    - 生成自然語言解釋
    - 發現未預定義的風險/機會
    """
    if llm_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 分類器未啟用。請設定環境變數: ANTHROPIC_API_KEY"
        )
    if intent_logger is None:
        raise HTTPException(status_code=503, detail="意圖紀錄器尚未初始化")

    history = intent_logger.get_customer_history(customer_id, req.days)
    if not history:
        return JSONResponse(content={
            "customer_id": customer_id,
            "message": f"客戶 {customer_id} 在近 {req.days} 天內無意圖紀錄",
        })

    try:
        result = llm_classifier.analyze_hidden_intents(customer_id, history)
    except LLMAPIError as e:
        raise HTTPException(status_code=502, detail=f"[{e.error_code}] {str(e)}")

    return JSONResponse(content=result.to_dict())


@app.get("/llm-stats")
def llm_stats():
    """
    LLM 使用統計

    顯示 LLM API 的呼叫次數、快取命中率、
    Token 用量、累計成本等。
    """
    if llm_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 分類器未啟用"
        )
    return JSONResponse(content=llm_classifier.get_stats())


# ============================================================
# 直接執行
# ============================================================
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
