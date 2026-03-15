"""
雙層意圖路由器 (Dual-Layer Intent Router)

設計理念：
  如果把所有客戶提問都丟給 LLM 處理，Token 費用極高且延遲長，
  對於百萬級用戶的金融機構來說無法負荷。

  本模組實作「BERT/SVM 輕量層 + LLM 重量層」的雙層架構：
    - 第一層（SVM）：80% 的常規單一意圖問題，由輕量級免 Token 費用的
      SVM 模型在地端瞬間完成分類（< 5ms）
    - 第二層（LLM）：剩餘 20% 的低信心度 / 複合意圖問題，才動用 LLM

  路由決策依據：
    1. SVM decision_function 的 max score（信心度）
    2. 是否為複合句（多意圖）
    3. 可配置的信心度閾值

Usage:
    from app.dual_router import DualLayerRouter

    router = DualLayerRouter(
        svm_classify_fn=my_svm_fn,
        llm_classify_fn=my_llm_fn,  # 可選
        confidence_threshold=0.8,
    )
    result = router.route("check my balance")
"""

import time
import logging
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# 路由層級定義
# ============================================================
class RoutingTier(str, Enum):
    """路由層級"""
    SVM = "svm"          # 輕量層：SVM + TF-IDF（免費、< 5ms）
    BERT = "bert"        # 中間層：BERT fine-tune（低成本、~50ms）
    LLM = "llm"          # 重量層：LLM API（高成本、~1-3s）
    FALLBACK = "fallback" # 降級：當重量層不可用時回退


@dataclass
class RoutingResult:
    """路由結果"""
    original_text: str
    tier: str                    # 實際使用的層級
    intent: str                  # 最終意圖
    confidence: float            # 信心度
    latency_ms: float            # 推論延遲（毫秒）
    is_compound: bool = False    # 是否為複合句
    sub_intents: List[Dict] = field(default_factory=list)
    routing_reason: str = ""     # 路由原因（為什麼選這一層）
    cost_estimate: float = 0.0   # 預估成本（美元）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingStats:
    """路由統計"""
    total_requests: int = 0
    svm_handled: int = 0
    bert_handled: int = 0
    llm_handled: int = 0
    fallback_count: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    _latency_sum: float = 0.0

    @property
    def svm_ratio(self) -> float:
        return self.svm_handled / max(self.total_requests, 1)

    @property
    def llm_ratio(self) -> float:
        return self.llm_handled / max(self.total_requests, 1)

    def record(self, result: RoutingResult):
        self.total_requests += 1
        self._latency_sum += result.latency_ms
        self.avg_latency_ms = self._latency_sum / self.total_requests
        self.total_cost += result.cost_estimate

        if result.tier == RoutingTier.SVM:
            self.svm_handled += 1
        elif result.tier == RoutingTier.BERT:
            self.bert_handled += 1
        elif result.tier == RoutingTier.LLM:
            self.llm_handled += 1
        else:
            self.fallback_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "svm_handled": self.svm_handled,
            "svm_ratio": f"{self.svm_ratio:.1%}",
            "bert_handled": self.bert_handled,
            "llm_handled": self.llm_handled,
            "llm_ratio": f"{self.llm_ratio:.1%}",
            "fallback_count": self.fallback_count,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "estimated_monthly_savings_vs_full_llm": self._estimate_savings(),
        }

    def _estimate_savings(self) -> str:
        """預估相比全部走 LLM 的月省成本"""
        if self.total_requests == 0:
            return "$0"
        # 假設每次 LLM 呼叫 ~$0.003 (GPT-3.5) 或 ~$0.03 (GPT-4)
        full_llm_cost = self.total_requests * 0.003
        actual_cost = self.total_cost
        savings = full_llm_cost - actual_cost
        # 推算月度（假設當前統計期間等比放大到日均 10,000 筆）
        if self.total_requests > 0:
            ratio = 10000 / self.total_requests
            monthly_savings = savings * ratio * 30
            return f"~${monthly_savings:,.0f}/month"
        return "$0"


# ============================================================
# 成本模型
# ============================================================
class CostModel:
    """各層級的成本與延遲估算"""

    # 每次推論的預估成本（美元）
    COST_PER_CALL = {
        RoutingTier.SVM: 0.0,       # 地端推論，無 API 費用
        RoutingTier.BERT: 0.0001,   # 地端 GPU，只有電費
        RoutingTier.LLM: 0.003,     # OpenAI GPT-3.5 Turbo 平均
    }

    @classmethod
    def estimate_cost(cls, tier: RoutingTier) -> float:
        return cls.COST_PER_CALL.get(tier, 0.0)


# ============================================================
# 雙層路由器
# ============================================================
class DualLayerRouter:
    """
    雙層意圖路由器

    路由邏輯：
      1. 所有請求先經過 SVM 輕量層
      2. 若 SVM 信心度 >= threshold 且非複合句 → 直接回傳（~80% 的流量）
      3. 若 SVM 信心度 < threshold 或為複合句 → 轉交 LLM 重量層
      4. 若 LLM 不可用 → 降級回 SVM 結果（加上警告）

    Args:
        svm_classify_fn: SVM 分類函式
            輸入 text (str) → 輸出 (intent: str, confidence: float)
        llm_classify_fn: LLM 分類函式（可選）
            輸入 text (str) → 輸出 (intent: str, confidence: float)
        compound_detect_fn: 複合句偵測函式（可選）
            輸入 text (str) → 輸出 bool
        compound_split_fn: 複合句拆解函式（可選）
            輸入 text (str) → 輸出 List[str]
        confidence_threshold: SVM 信心度閾值（預設 0.8）
    """

    def __init__(
        self,
        svm_classify_fn: Callable[[str], tuple],
        llm_classify_fn: Optional[Callable[[str], tuple]] = None,
        compound_detect_fn: Optional[Callable[[str], bool]] = None,
        compound_split_fn: Optional[Callable[[str], List[str]]] = None,
        confidence_threshold: float = 0.8,
        heavy_tier: Optional[str] = None,
    ):
        self.svm_classify_fn = svm_classify_fn
        self.llm_classify_fn = llm_classify_fn
        self.compound_detect_fn = compound_detect_fn
        self.compound_split_fn = compound_split_fn
        self.confidence_threshold = confidence_threshold
        # 自動判斷重量層名稱：可指定 "bert" / "llm"，預設為 "llm"
        if heavy_tier:
            self.heavy_tier = RoutingTier(heavy_tier)
        else:
            self.heavy_tier = RoutingTier.LLM
        self.stats = RoutingStats()

    def route(self, text: str) -> RoutingResult:
        """
        對輸入文字執行雙層路由

        Returns:
            RoutingResult 包含意圖、信心度、使用層級、延遲等資訊
        """
        start = time.time()

        # --- Step 1: 複合句偵測 ---
        is_compound = False
        if self.compound_detect_fn:
            is_compound = self.compound_detect_fn(text)

        # --- Step 2: 複合句 → 直接走 LLM/拆解路線 ---
        if is_compound and self.compound_split_fn:
            return self._handle_compound(text, start)

        # --- Step 3: 單一意圖 → SVM 先行 ---
        svm_intent, svm_conf = self.svm_classify_fn(text)

        # --- Step 4: 信心度判斷 ---
        if svm_conf >= self.confidence_threshold:
            # 高信心度 → SVM 直接回傳
            latency = (time.time() - start) * 1000
            result = RoutingResult(
                original_text=text,
                tier=RoutingTier.SVM,
                intent=svm_intent,
                confidence=round(svm_conf, 4),
                latency_ms=round(latency, 2),
                is_compound=False,
                routing_reason=f"SVM confidence ({svm_conf:.3f}) >= threshold ({self.confidence_threshold})",
                cost_estimate=CostModel.estimate_cost(RoutingTier.SVM),
            )
            self.stats.record(result)
            logger.debug(f"[SVM] '{text}' → {svm_intent} (conf={svm_conf:.3f})")
            return result

        # --- Step 5: 低信心度 → 嘗試重量層 (BERT 或 LLM) ---
        if self.llm_classify_fn:
            try:
                llm_intent, llm_conf = self.llm_classify_fn(text)
                latency = (time.time() - start) * 1000
                tier_name = self.heavy_tier.value.upper()
                result = RoutingResult(
                    original_text=text,
                    tier=self.heavy_tier,
                    intent=llm_intent,
                    confidence=round(llm_conf, 4),
                    latency_ms=round(latency, 2),
                    is_compound=False,
                    routing_reason=(
                        f"SVM confidence ({svm_conf:.3f}) < threshold ({self.confidence_threshold}), "
                        f"escalated to {tier_name}"
                    ),
                    cost_estimate=CostModel.estimate_cost(self.heavy_tier),
                )
                self.stats.record(result)
                logger.info(f"[{tier_name}] '{text}' → {llm_intent} (svm_conf={svm_conf:.3f})")
                return result
            except Exception as e:
                logger.warning(f"LLM 推論失敗，降級至 SVM: {e}")

        # --- Step 6: LLM 不可用 → 降級回 SVM ---
        latency = (time.time() - start) * 1000
        result = RoutingResult(
            original_text=text,
            tier=RoutingTier.FALLBACK,
            intent=svm_intent,
            confidence=round(svm_conf, 4),
            latency_ms=round(latency, 2),
            is_compound=False,
            routing_reason=(
                f"SVM confidence ({svm_conf:.3f}) < threshold ({self.confidence_threshold}), "
                f"LLM unavailable, fallback to SVM"
            ),
            cost_estimate=CostModel.estimate_cost(RoutingTier.SVM),
        )
        self.stats.record(result)
        logger.info(f"[FALLBACK] '{text}' → {svm_intent} (conf={svm_conf:.3f}, LLM unavailable)")
        return result

    def _handle_compound(self, text: str, start: float) -> RoutingResult:
        """處理複合句：拆解後逐句分類"""
        segments = self.compound_split_fn(text)
        sub_intents = []

        for seg in segments:
            seg_intent, seg_conf = self.svm_classify_fn(seg)

            # 子句也適用信心度判斷
            if seg_conf < self.confidence_threshold and self.llm_classify_fn:
                try:
                    seg_intent, seg_conf = self.llm_classify_fn(seg)
                    tier_used = self.heavy_tier
                except Exception:
                    tier_used = RoutingTier.FALLBACK
            else:
                tier_used = RoutingTier.SVM

            sub_intents.append({
                "text": seg,
                "intent": seg_intent,
                "confidence": round(float(seg_conf), 4),
                "tier": tier_used,
            })

        # 主意圖取第一個子句的意圖
        primary = sub_intents[0] if sub_intents else {"intent": "unknown", "confidence": 0.0}
        latency = (time.time() - start) * 1000

        # 成本：有多少子句走了重量層
        heavy_count = sum(1 for s in sub_intents if s["tier"] == self.heavy_tier)
        svm_count = len(sub_intents) - heavy_count
        total_cost = (
            svm_count * CostModel.estimate_cost(RoutingTier.SVM) +
            heavy_count * CostModel.estimate_cost(self.heavy_tier)
        )

        result = RoutingResult(
            original_text=text,
            tier=self.heavy_tier if heavy_count > 0 else RoutingTier.SVM,
            intent=primary["intent"],
            confidence=primary["confidence"],
            latency_ms=round(latency, 2),
            is_compound=True,
            sub_intents=sub_intents,
            routing_reason=f"Compound query detected, split into {len(sub_intents)} sub-intents",
            cost_estimate=total_cost,
        )
        self.stats.record(result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """取得路由統計"""
        return self.stats.to_dict()

    def reset_stats(self):
        """重置統計"""
        self.stats = RoutingStats()


# ============================================================
# Demo / 單元測試
# ============================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 模擬 SVM 分類函式（用隨機信心度示範）
    import random
    random.seed(42)

    def mock_svm(text: str):
        intents = ["balance", "transfer", "book_flight", "pay_bill", "greeting"]
        intent = random.choice(intents)
        conf = random.uniform(0.3, 1.5)
        return intent, conf

    def mock_llm(text: str):
        """模擬 LLM（高準確但慢）"""
        time.sleep(0.05)  # 模擬延遲
        intents = ["balance", "transfer", "book_flight", "pay_bill", "greeting"]
        intent = random.choice(intents)
        conf = random.uniform(0.85, 0.99)
        return intent, conf

    def mock_compound_detect(text: str) -> bool:
        return " and " in text.lower() or " also " in text.lower()

    def mock_compound_split(text: str) -> list:
        import re
        return [s.strip() for s in re.split(r'\band\b|\balso\b', text, flags=re.I) if s.strip()]

    router = DualLayerRouter(
        svm_classify_fn=mock_svm,
        llm_classify_fn=mock_llm,
        compound_detect_fn=mock_compound_detect,
        compound_split_fn=mock_compound_split,
        confidence_threshold=0.8,
    )

    test_queries = [
        "check my balance",
        "what is my credit score",
        "book a flight to tokyo and also check my balance",
        "transfer money to savings",
        "help me pay my credit card bill",
        "what's the weather",
        "report lost card",
        "I need to check exchange rate",
        "remind me tomorrow and set alarm at 7am",
        "cancel my reservation",
    ]

    print("=" * 70)
    print("  Dual-Layer Router Demo")
    print("=" * 70)

    for q in test_queries:
        result = router.route(q)
        tier_emoji = {"svm": "⚡", "llm": "🧠", "fallback": "⚠️", "bert": "🔬"}
        emoji = tier_emoji.get(result.tier, "❓")
        print(f"\n  {emoji} [{result.tier.upper():8s}] \"{q}\"")
        print(f"     → {result.intent} (conf={result.confidence:.3f}, {result.latency_ms:.1f}ms, ${result.cost_estimate:.4f})")
        if result.sub_intents:
            for si in result.sub_intents:
                print(f"       ├─ \"{si['text']}\" → {si['intent']} ({si['tier']})")

    print("\n" + "=" * 70)
    print("  Routing Statistics")
    print("=" * 70)
    stats = router.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")
