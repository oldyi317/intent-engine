"""
LLM 意圖分類模組 (LLM Intent Classifier via Anthropic Claude API)

使用 Claude API 進行兩階段意圖分類：
  Stage 1: 領域分類 (9 domains) — 縮小搜索範圍
  Stage 2: 精確意圖分類 (domain-specific intents)

同時提供：
  - 隱藏意圖分析 (hidden intent mining via LLM)
  - 與 SVM/BERT 的比較介面
  - 成本追蹤與快取機制

Usage:
    from app.llm_classifier import LLMIntentClassifier

    classifier = LLMIntentClassifier(api_key="sk-ant-...")
    result = classifier.classify("check my balance")
    # {"intent": "balance", "domain": "Finance", "confidence": 0.95, ...}

    # 隱藏意圖分析
    analysis = classifier.analyze_hidden_intents(intent_history)
    # {"risk_alerts": [...], "cross_sell": [...], "reasoning": "..."}

環境變數：
    ANTHROPIC_API_KEY — Claude API 金鑰（必填）

Author: Fubon Intent Intelligence Project
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field, asdict

from src.config import INTENT_DOMAINS, INTENT_TO_DOMAIN

logger = logging.getLogger(__name__)


# ============================================================
# 設定常數
# ============================================================
DEFAULT_MODEL = "claude-sonnet-4-20250514"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# 每 1M tokens 的成本（USD）— 依 Anthropic 2025 定價
MODEL_COSTS = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}

# LRU 快取大小
CACHE_MAX_SIZE = 500


# ============================================================
# 自訂例外
# ============================================================
class LLMAPIError(Exception):
    """LLM API 呼叫失敗的例外"""
    def __init__(self, message: str, error_code: str = "UNKNOWN"):
        super().__init__(message)
        self.error_code = error_code


# ============================================================
# 資料結構
# ============================================================
@dataclass
class LLMClassifyResult:
    """LLM 分類結果"""
    intent: str
    domain: str
    confidence: float
    reasoning: str = ""
    latency_ms: float = 0.0
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMHiddenIntentResult:
    """LLM 隱藏意圖分析結果"""
    customer_id: str
    risk_alerts: List[Dict[str, Any]] = field(default_factory=list)
    cross_sell_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    retention_signals: List[Dict[str, Any]] = field(default_factory=list)
    overall_reasoning: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMUsageStats:
    """LLM 使用統計"""
    total_calls: int = 0
    cache_hits: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    _latency_sum: float = 0.0

    def record(self, result: LLMClassifyResult):
        self.total_calls += 1
        if result.cached:
            self.cache_hits += 1
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.total_cost_usd += result.cost_usd
        self._latency_sum += result.latency_ms
        self.avg_latency_ms = self._latency_sum / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{self.cache_hits / max(self.total_calls, 1):.1%}",
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


# ============================================================
# LRU 快取
# ============================================================
class LRUCache:
    """簡易 LRU 快取"""

    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def _make_key(self, text: str) -> str:
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        key = self._make_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, text: str, value: Any):
        key = self._make_key(text)
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()


# ============================================================
# Prompt 模板
# ============================================================

# Stage 1: 領域分類
DOMAIN_CLASSIFY_PROMPT = """You are a financial customer service intent classifier.

Given a customer query, classify it into ONE of these 9 domains:
{domains_list}

Respond in JSON format ONLY:
{{"domain": "<domain_name>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}

Rules:
- confidence should reflect how certain you are (0.0 = guess, 1.0 = absolutely certain)
- If the query doesn't fit any domain well, pick the closest and lower the confidence
- Keep reasoning under 20 words

Customer query: "{text}"
"""

# Stage 2: 精確意圖分類
INTENT_CLASSIFY_PROMPT = """You are a financial customer service intent classifier.

The customer query belongs to the "{domain}" domain.
Classify it into ONE of these specific intents:
{intents_list}

Respond in JSON format ONLY:
{{"intent": "<intent_label>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}

Rules:
- Use EXACTLY one of the intent labels listed above
- confidence should reflect certainty (0.0 = guess, 1.0 = certain)
- Keep reasoning under 20 words

Customer query: "{text}"
"""

# 一步到位分類（用於 Haiku 快速模式）
SINGLE_STEP_PROMPT = """You are a financial customer service intent classifier.

Classify this customer query into one intent from the list below.

Available intents by domain:
{all_intents}

Respond in JSON format ONLY:
{{"intent": "<intent_label>", "domain": "<domain>", "confidence": <0.0-1.0>}}

Customer query: "{text}"
"""

# 多意圖分類（複合句拆解）
MULTI_INTENT_PROMPT = """You are a financial customer service intent classifier.

This customer query may contain MULTIPLE intents. Identify ALL intents present.

Available intents by domain:
{all_intents}

Respond in JSON format ONLY:
{{"intents": [
  {{"intent": "<label>", "domain": "<domain>", "confidence": <0.0-1.0>, "evidence": "<which part of text>"}}
]}}

Rules:
- Use EXACTLY the intent labels listed above
- List every distinct intent found, even if there is only one
- confidence should reflect certainty (0.0 = guess, 1.0 = certain)
- evidence should be a short quote from the original text

Customer query: "{text}"
"""

# 隱藏意圖分析
HIDDEN_INTENT_PROMPT = """You are a financial customer behavior analyst.

Analyze this customer's recent intent history and identify hidden needs, risks, or opportunities.

Customer ID: {customer_id}
Intent History (most recent first):
{history_formatted}

Analyze for:
1. **Risk Alerts**: Signs of fraud, theft, account compromise, or financial distress
2. **Cross-sell Opportunities**: Unspoken needs that suggest product recommendations
3. **Retention Signals**: Signs of dissatisfaction or potential churn

Respond in JSON format ONLY:
{{
  "risk_alerts": [
    {{"signal": "<description>", "severity": "high|medium|low", "evidence": "<which intents>", "recommended_action": "<action>"}}
  ],
  "cross_sell_opportunities": [
    {{"opportunity": "<description>", "product_suggestion": "<product>", "evidence": "<which intents>", "confidence": <0.0-1.0>}}
  ],
  "retention_signals": [
    {{"signal": "<description>", "severity": "high|medium|low", "evidence": "<which intents>", "recommended_action": "<action>"}}
  ],
  "overall_reasoning": "<2-3 sentence summary of customer behavior pattern>",
  "recommended_actions": ["<action1>", "<action2>"]
}}

Rules:
- Only report signals with real evidence from the history
- Be specific about which intents form the evidence
- If no signals found for a category, return empty array []
"""


# ============================================================
# LLM 意圖分類器
# ============================================================
class LLMIntentClassifier:
    """
    使用 Anthropic Claude API 的意圖分類器

    支援兩種模式：
    1. 兩階段分類（Two-stage）：先判領域，再判意圖 — 更精準
    2. 一步分類（Single-step）：直接判意圖 — 更快、更省

    Args:
        api_key: Anthropic API Key（或從 ANTHROPIC_API_KEY 環境變數讀取）
        model: 預設模型（claude-sonnet-4-20250514）
        use_two_stage: 是否使用兩階段分類（預設 True）
        cache_enabled: 是否啟用快取（預設 True）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        use_two_stage: bool = True,
        cache_enabled: bool = True,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "需要 Anthropic API Key！\n"
                "請設定環境變數: export ANTHROPIC_API_KEY='sk-ant-...'\n"
                "或傳入參數: LLMIntentClassifier(api_key='sk-ant-...')"
            )

        import anthropic
        import httpx
        # 部分企業環境有 SSL proxy，需要跳過憑證驗證
        http_client = httpx.Client(verify=False)
        self.client = anthropic.Anthropic(api_key=self.api_key, http_client=http_client)
        self.model = model
        self.use_two_stage = use_two_stage
        self.cache = LRUCache() if cache_enabled else None
        self.stats = LLMUsageStats()

        # 預建領域和意圖列表字串
        self._domains_list = "\n".join(
            f"- {domain}: {', '.join(intents[:5])}... ({len(intents)} intents)"
            for domain, intents in INTENT_DOMAINS.items()
        )
        self._all_intents_str = "\n".join(
            f"[{domain}]: {', '.join(intents)}"
            for domain, intents in INTENT_DOMAINS.items()
        )

        logger.info(f"LLM 分類器初始化完成 (model={model}, two_stage={use_two_stage})")

    # ----------------------------------------------------------
    # 核心：呼叫 Claude API
    # ----------------------------------------------------------
    def _call_api(self, prompt: str, model: Optional[str] = None, max_tokens: int = 300) -> Tuple[str, int, int]:
        """
        呼叫 Claude API

        Returns:
            (response_text, input_tokens, output_tokens)

        Raises:
            LLMAPIError: 當 API 回傳錯誤時（餘額不足、認證失敗等）
        """
        import anthropic

        model = model or self.model
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,  # 確定性輸出
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.BadRequestError as e:
            error_msg = str(e)
            if "credit balance" in error_msg.lower():
                raise LLMAPIError(
                    "Anthropic API 餘額不足，請至 https://console.anthropic.com 加值。",
                    error_code="INSUFFICIENT_CREDITS",
                )
            raise LLMAPIError(f"API 請求錯誤: {error_msg}", error_code="BAD_REQUEST")
        except anthropic.AuthenticationError:
            raise LLMAPIError(
                "API Key 無效或已過期，請檢查 ANTHROPIC_API_KEY。",
                error_code="AUTH_FAILED",
            )
        except anthropic.RateLimitError:
            raise LLMAPIError(
                "API 速率限制，請稍後再試。",
                error_code="RATE_LIMITED",
            )
        except anthropic.APIConnectionError as e:
            raise LLMAPIError(
                f"無法連線至 Anthropic API: {e}",
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            raise LLMAPIError(f"LLM API 未預期錯誤: {e}", error_code="UNKNOWN")

        text = response.content[0].text.strip()
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return text, input_tokens, output_tokens

    def _parse_json(self, text: str) -> Dict:
        """從 API 回應中解析 JSON"""
        # 嘗試直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 嘗試提取 JSON 區塊
        import re
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"無法解析 JSON: {text[:200]}")
        return {}

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """計算 API 呼叫成本"""
        costs = MODEL_COSTS.get(model, MODEL_COSTS[DEFAULT_MODEL])
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

    # ----------------------------------------------------------
    # 兩階段分類
    # ----------------------------------------------------------
    def _classify_two_stage(self, text: str) -> LLMClassifyResult:
        """兩階段分類：領域 → 意圖"""
        total_input = 0
        total_output = 0
        start = time.time()

        # Stage 1: 領域分類
        domain_prompt = DOMAIN_CLASSIFY_PROMPT.format(
            domains_list=self._domains_list,
            text=text,
        )
        resp1, in1, out1 = self._call_api(domain_prompt)
        total_input += in1
        total_output += out1

        domain_result = self._parse_json(resp1)
        domain = domain_result.get("domain", "Finance")

        # 驗證領域
        if domain not in INTENT_DOMAINS:
            # 模糊匹配
            for d in INTENT_DOMAINS:
                if d.lower() == domain.lower():
                    domain = d
                    break
            else:
                domain = "Finance"  # fallback

        # Stage 2: 意圖分類
        intents_in_domain = INTENT_DOMAINS[domain]
        intent_prompt = INTENT_CLASSIFY_PROMPT.format(
            domain=domain,
            intents_list=", ".join(intents_in_domain),
            text=text,
        )
        resp2, in2, out2 = self._call_api(intent_prompt)
        total_input += in2
        total_output += out2

        intent_result = self._parse_json(resp2)
        intent = intent_result.get("intent", intents_in_domain[0])
        confidence = float(intent_result.get("confidence", 0.5))
        reasoning = intent_result.get("reasoning", "")

        # 驗證意圖標籤
        if intent not in intents_in_domain:
            # 模糊匹配
            intent_lower = intent.lower().replace(" ", "_")
            for i in intents_in_domain:
                if i.lower() == intent_lower:
                    intent = i
                    break
            else:
                # 在所有意圖中搜索
                if intent in INTENT_TO_DOMAIN:
                    domain = INTENT_TO_DOMAIN[intent]
                else:
                    intent = intents_in_domain[0]
                    confidence *= 0.5  # 降低信心度

        latency = (time.time() - start) * 1000
        cost = self._calculate_cost(self.model, total_input, total_output)

        return LLMClassifyResult(
            intent=intent,
            domain=domain,
            confidence=round(confidence, 4),
            reasoning=reasoning,
            latency_ms=round(latency, 2),
            model_used=self.model,
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=round(cost, 6),
        )

    # ----------------------------------------------------------
    # 一步分類
    # ----------------------------------------------------------
    def _classify_single_step(self, text: str) -> LLMClassifyResult:
        """一步分類：直接判斷意圖"""
        start = time.time()

        prompt = SINGLE_STEP_PROMPT.format(
            all_intents=self._all_intents_str,
            text=text,
        )
        resp, in_tokens, out_tokens = self._call_api(prompt)

        result = self._parse_json(resp)
        intent = result.get("intent", "greeting")
        domain = result.get("domain", INTENT_TO_DOMAIN.get(intent, "Unknown"))
        confidence = float(result.get("confidence", 0.5))

        # 驗證
        if intent not in INTENT_TO_DOMAIN:
            intent_lower = intent.lower().replace(" ", "_")
            for valid_intent in INTENT_TO_DOMAIN:
                if valid_intent == intent_lower:
                    intent = valid_intent
                    break

        if intent in INTENT_TO_DOMAIN:
            domain = INTENT_TO_DOMAIN[intent]

        latency = (time.time() - start) * 1000
        cost = self._calculate_cost(self.model, in_tokens, out_tokens)

        return LLMClassifyResult(
            intent=intent,
            domain=domain,
            confidence=round(confidence, 4),
            latency_ms=round(latency, 2),
            model_used=self.model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=round(cost, 6),
        )

    # ----------------------------------------------------------
    # 公開介面：分類
    # ----------------------------------------------------------
    def classify(self, text: str) -> LLMClassifyResult:
        """
        對輸入文字進行意圖分類

        Returns:
            LLMClassifyResult
        """
        # 檢查快取
        if self.cache:
            cached = self.cache.get(text)
            if cached:
                cached_result = LLMClassifyResult(**cached)
                cached_result.cached = True
                cached_result.latency_ms = 0.0
                self.stats.record(cached_result)
                logger.debug(f"[LLM-Cache] '{text}' → {cached_result.intent}")
                return cached_result

        # 呼叫 API
        if self.use_two_stage:
            result = self._classify_two_stage(text)
        else:
            result = self._classify_single_step(text)

        # 存入快取
        if self.cache:
            cache_data = result.to_dict()
            cache_data.pop("cached", None)
            self.cache.put(text, cache_data)

        self.stats.record(result)
        logger.info(
            f"[LLM] '{text}' → {result.intent} ({result.domain}, "
            f"conf={result.confidence}, {result.latency_ms:.0f}ms, ${result.cost_usd:.6f})"
        )
        return result

    def classify_as_tuple(self, text: str) -> Tuple[str, float]:
        """
        分類並回傳 (intent, confidence) tuple

        與 SVM/BERT 的 classify_fn 介面相容，
        可直接插入 DualLayerRouter 作為 llm_classify_fn
        """
        result = self.classify(text)
        return result.intent, result.confidence

    # ----------------------------------------------------------
    # 公開介面：多意圖分類
    # ----------------------------------------------------------
    def classify_multi(self, text: str) -> Dict[str, Any]:
        """
        對輸入文字進行多意圖分類，回傳所有偵測到的意圖

        Returns:
            {"intents": [...], "count": N, "latency_ms": ..., "cost_usd": ...}
        """
        start = time.time()

        prompt = MULTI_INTENT_PROMPT.format(
            all_intents=self._all_intents_str,
            text=text,
        )
        resp, in_tokens, out_tokens = self._call_api(prompt, model=self.model, max_tokens=800)
        parsed = self._parse_json_deep(resp)

        latency = (time.time() - start) * 1000
        cost = self._calculate_cost(self.model, in_tokens, out_tokens)

        intents = parsed.get("intents", [])

        # 驗證每個意圖標籤
        for item in intents:
            intent = item.get("intent", "")
            if intent not in INTENT_TO_DOMAIN:
                # 模糊匹配
                intent_lower = intent.lower().replace(" ", "_")
                for valid in INTENT_TO_DOMAIN:
                    if valid == intent_lower:
                        item["intent"] = valid
                        break

        return {
            "intents": intents,
            "count": len(intents),
            "latency_ms": round(latency, 2),
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": round(cost, 6),
            "model_used": self.model,
        }

    # ----------------------------------------------------------
    # 公開介面：隱藏意圖分析
    # ----------------------------------------------------------
    def analyze_hidden_intents(
        self,
        customer_id: str,
        intent_history: List[Dict[str, Any]],
    ) -> LLMHiddenIntentResult:
        """
        使用 LLM 分析客戶意圖歷史，挖掘隱藏需求

        Args:
            customer_id: 客戶 ID
            intent_history: 意圖歷史列表，每筆包含
                {"intent": str, "domain": str, "timestamp": str, "original_text": str}

        Returns:
            LLMHiddenIntentResult
        """
        if not intent_history:
            return LLMHiddenIntentResult(customer_id=customer_id)

        start = time.time()

        # 格式化歷史
        history_lines = []
        for h in intent_history[:50]:  # 限制最多 50 筆
            ts = h.get("timestamp", "")[:19]  # 截取到秒
            intent = h.get("intent", "unknown")
            domain = h.get("domain", "Unknown")
            text = h.get("original_text", "")
            history_lines.append(f"  [{ts}] {intent} ({domain}) — \"{text}\"")

        prompt = HIDDEN_INTENT_PROMPT.format(
            customer_id=customer_id,
            history_formatted="\n".join(history_lines),
        )

        resp, in_tokens, out_tokens = self._call_api(prompt, max_tokens=1024)
        parsed = self._parse_json_deep(resp)

        latency = (time.time() - start) * 1000
        cost = self._calculate_cost(self.model, in_tokens, out_tokens)

        return LLMHiddenIntentResult(
            customer_id=customer_id,
            risk_alerts=parsed.get("risk_alerts", []),
            cross_sell_opportunities=parsed.get("cross_sell_opportunities", []),
            retention_signals=parsed.get("retention_signals", []),
            overall_reasoning=parsed.get("overall_reasoning", ""),
            recommended_actions=parsed.get("recommended_actions", []),
            latency_ms=round(latency, 2),
            cost_usd=round(cost, 6),
        )

    def _parse_json_deep(self, text: str) -> Dict:
        """解析可能包含巢狀結構的 JSON"""
        # 嘗試找到最外層的 { }
        import re
        # 找到第一個 { 和最後一個 }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass
        return self._parse_json(text)

    # ----------------------------------------------------------
    # 公開介面：比較
    # ----------------------------------------------------------
    def compare_with_svm(
        self,
        text: str,
        svm_classify_fn,
    ) -> Dict[str, Any]:
        """
        同時用 SVM 和 LLM 分類，回傳比較結果

        Args:
            text: 輸入文字
            svm_classify_fn: SVM 分類函式 (text) → (intent, confidence)

        Returns:
            比較結果 dict
        """
        # SVM
        svm_start = time.time()
        svm_intent, svm_conf = svm_classify_fn(text)
        svm_latency = (time.time() - svm_start) * 1000

        # LLM
        llm_result = self.classify(text)

        agree = svm_intent == llm_result.intent

        return {
            "text": text,
            "svm": {
                "intent": svm_intent,
                "confidence": round(float(svm_conf), 4),
                "latency_ms": round(svm_latency, 2),
                "cost_usd": 0.0,
            },
            "llm": {
                "intent": llm_result.intent,
                "domain": llm_result.domain,
                "confidence": llm_result.confidence,
                "reasoning": llm_result.reasoning,
                "latency_ms": llm_result.latency_ms,
                "cost_usd": llm_result.cost_usd,
                "model": llm_result.model_used,
                "cached": llm_result.cached,
            },
            "agreement": agree,
            "speed_ratio": f"SVM is {llm_result.latency_ms / max(svm_latency, 0.01):.0f}x faster"
            if not llm_result.cached else "LLM cached",
        }

    # ----------------------------------------------------------
    # 統計與管理
    # ----------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """取得 LLM 使用統計"""
        return self.stats.to_dict()

    def clear_cache(self):
        """清除快取"""
        if self.cache:
            self.cache.clear()
            logger.info("LLM 快取已清除")

    def reset_stats(self):
        """重置統計"""
        self.stats = LLMUsageStats()


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("請設定 ANTHROPIC_API_KEY 環境變數：")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print("\n以 mock 模式示範架構...")

        # Mock demo
        print("\n" + "=" * 60)
        print("  LLM Intent Classifier — Architecture Demo (Mock)")
        print("=" * 60)

        print("\n[兩階段分類流程]")
        print("  Input:  'check my balance'")
        print("  Stage 1 → Domain: Finance (conf=0.98)")
        print("  Stage 2 → Intent: balance (conf=0.96)")
        print("  Latency: ~800ms | Cost: ~$0.000045")

        print("\n[隱藏意圖分析流程]")
        print("  Customer: CUST_A")
        print("  History:  lost_luggage → report_lost_card → freeze_account")
        print("  LLM Analysis:")
        print("    🚨 Risk: 旅途盜竊風險 (high severity)")
        print("    📋 Action: 自動降額 → 啟動詐欺防護")

        print("\n[SVM vs LLM 比較]")
        print("  Input:  'I need to check exchange rates for USD'")
        print("  SVM  → exchange_rate (conf=1.23, 2ms,  $0)")
        print("  LLM  → exchange_rate (conf=0.95, 850ms, $0.00005)")
        print("  Agreement: ✅ | SVM is 425x faster")
    else:
        # 真實 API 測試
        classifier = LLMIntentClassifier(api_key=api_key)

        test_texts = [
            "check my balance",
            "I want to book a flight to Tokyo",
            "what's the USD to TWD exchange rate",
            "report my lost credit card immediately",
            "set a reminder for tomorrow at 9am",
        ]

        print("=" * 60)
        print("  LLM Intent Classifier — Live Demo")
        print("=" * 60)

        for text in test_texts:
            result = classifier.classify(text)
            print(f"\n  Input: '{text}'")
            print(f"  → {result.intent} ({result.domain})")
            print(f"    conf={result.confidence}, {result.latency_ms:.0f}ms, ${result.cost_usd:.6f}")
            if result.reasoning:
                print(f"    reasoning: {result.reasoning}")

        # 測試快取
        print("\n--- 快取測試 ---")
        result2 = classifier.classify("check my balance")
        print(f"  'check my balance' (cached={result2.cached}, {result2.latency_ms:.0f}ms)")

        print(f"\n--- 使用統計 ---")
        stats = classifier.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
