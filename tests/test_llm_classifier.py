"""
LLM 意圖分類器測試

測試層級：
  1. 單元測試 — LLMIntentClassifier 內部邏輯（不需 API Key）
  2. API 端點測試 — /predict/llm, /compare, /llm/hidden-intents（需要 API Key）
  3. 整合比較測試 — SVM vs LLM 批次比較

執行方式：
    # 單元測試（不需 API Key）
    cd fubon_intent_project
    python -m pytest tests/test_llm_classifier.py -v -k "unit"

    # API 測試（需要啟動 server + ANTHROPIC_API_KEY）
    export ANTHROPIC_API_KEY='sk-ant-...'
    uvicorn app.main:app --port 8000 &
    python -m pytest tests/test_llm_classifier.py -v -k "api"

    # 完整測試
    python -m pytest tests/test_llm_classifier.py -v
"""

import os
import sys
import json
import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000"
HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ============================================================
# Unit Tests（不需 API Key）
# ============================================================
class TestLRUCache:
    """LRU 快取單元測試"""

    def test_cache_hit(self):
        from app.llm_classifier import LRUCache
        cache = LRUCache(max_size=10)
        cache.put("hello world", {"intent": "greeting"})
        result = cache.get("hello world")
        assert result == {"intent": "greeting"}

    def test_cache_miss(self):
        from app.llm_classifier import LRUCache
        cache = LRUCache(max_size=10)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_case_insensitive(self):
        from app.llm_classifier import LRUCache
        cache = LRUCache(max_size=10)
        cache.put("Check My Balance", {"intent": "balance"})
        result = cache.get("check my balance")
        assert result == {"intent": "balance"}

    def test_cache_eviction(self):
        from app.llm_classifier import LRUCache
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # 'a' should be evicted
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_cache_clear(self):
        from app.llm_classifier import LRUCache
        cache = LRUCache(max_size=10)
        cache.put("test", {"data": True})
        cache.clear()
        assert cache.get("test") is None


class TestPromptTemplates:
    """Prompt 模板測試"""

    def test_domain_prompt_format(self):
        from app.llm_classifier import DOMAIN_CLASSIFY_PROMPT
        prompt = DOMAIN_CLASSIFY_PROMPT.format(
            domains_list="- Finance\n- Travel",
            text="check my balance",
        )
        assert "check my balance" in prompt
        assert "Finance" in prompt

    def test_intent_prompt_format(self):
        from app.llm_classifier import INTENT_CLASSIFY_PROMPT
        prompt = INTENT_CLASSIFY_PROMPT.format(
            domain="Finance",
            intents_list="balance, transfer, pay_bill",
            text="check my balance",
        )
        assert "Finance" in prompt
        assert "balance" in prompt

    def test_hidden_intent_prompt_format(self):
        from app.llm_classifier import HIDDEN_INTENT_PROMPT
        prompt = HIDDEN_INTENT_PROMPT.format(
            customer_id="CUST001",
            history_formatted="  [2024-01-01] balance (Finance)",
        )
        assert "CUST001" in prompt
        assert "balance" in prompt


class TestCostCalculation:
    """成本計算測試"""

    def test_cost_model(self):
        from app.llm_classifier import MODEL_COSTS, DEFAULT_MODEL
        costs = MODEL_COSTS[DEFAULT_MODEL]
        # 100 input + 50 output tokens
        cost = (100 * costs["input"] + 50 * costs["output"]) / 1_000_000
        assert cost > 0
        assert cost < 0.01  # 應該遠小於 1 分美元


class TestUsageStats:
    """使用統計測試"""

    def test_record_stats(self):
        from app.llm_classifier import LLMUsageStats, LLMClassifyResult
        stats = LLMUsageStats()
        result = LLMClassifyResult(
            intent="balance", domain="Finance", confidence=0.95,
            latency_ms=800, input_tokens=100, output_tokens=50,
            cost_usd=0.001,
        )
        stats.record(result)
        assert stats.total_calls == 1
        assert stats.total_cost_usd == 0.001

    def test_cache_hit_stats(self):
        from app.llm_classifier import LLMUsageStats, LLMClassifyResult
        stats = LLMUsageStats()
        result = LLMClassifyResult(
            intent="balance", domain="Finance", confidence=0.95,
            cached=True,
        )
        stats.record(result)
        assert stats.cache_hits == 1
        d = stats.to_dict()
        assert d["cache_hit_rate"] == "100.0%"


# ============================================================
# API Tests（需要 API Key + Server）
# ============================================================
def server_is_running():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not server_is_running(),
    reason="API server not running (start with: uvicorn app.main:app --port 8000)"
)
class TestLLMEndpoints:
    """LLM API 端點測試"""

    def test_health_shows_llm(self):
        """健康檢查應顯示 llm_classifier feature（如有 API Key）"""
        r = requests.get(f"{BASE_URL}/health")
        data = r.json()
        # 不強制要求 LLM 啟用（沒有 API Key 時不會啟用）
        assert "features" in data

    @pytest.mark.skipif(not HAS_API_KEY, reason="ANTHROPIC_API_KEY not set")
    def test_llm_predict_basic(self):
        """LLM 基本分類"""
        r = requests.post(f"{BASE_URL}/predict/llm", json={
            "text": "check my balance"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "balance"
        assert data["domain"] == "Finance"
        assert data["confidence"] > 0.5
        assert data["latency_ms"] > 0
        assert data["cost_usd"] > 0

    @pytest.mark.skipif(not HAS_API_KEY, reason="ANTHROPIC_API_KEY not set")
    def test_llm_predict_two_stage(self):
        """LLM 兩階段分類"""
        r = requests.post(f"{BASE_URL}/predict/llm", json={
            "text": "book a flight to tokyo",
            "two_stage": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "book_flight"
        assert data["domain"] == "Travel"

    @pytest.mark.skipif(not HAS_API_KEY, reason="ANTHROPIC_API_KEY not set")
    def test_llm_predict_single_step(self):
        """LLM 一步分類"""
        r = requests.post(f"{BASE_URL}/predict/llm", json={
            "text": "what's the weather today",
            "two_stage": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "weather"

    @pytest.mark.skipif(not HAS_API_KEY, reason="ANTHROPIC_API_KEY not set")
    def test_llm_predict_cache(self):
        """LLM 快取機制"""
        # 第一次呼叫
        r1 = requests.post(f"{BASE_URL}/predict/llm", json={
            "text": "check my balance"
        })
        data1 = r1.json()

        # 第二次呼叫（應命中快取）
        r2 = requests.post(f"{BASE_URL}/predict/llm", json={
            "text": "check my balance"
        })
        data2 = r2.json()
        assert data2["cached"] is True
        assert data2["latency_ms"] == 0.0

    @pytest.mark.skipif(not HAS_API_KEY, reason="ANTHROPIC_API_KEY not set")
    def test_compare_endpoint(self):
        """SVM vs LLM 比較"""
        r = requests.post(f"{BASE_URL}/compare", json={
            "text": "check my balance"
        })
        assert r.status_code == 200
        data = r.json()
        assert "svm" in data
        assert "llm" in data
        assert "agreement" in data
        assert "speed_ratio" in data
        assert data["svm"]["cost_usd"] == 0.0
        assert data["llm"]["cost_usd"] > 0

    @pytest.mark.skipif(not HAS_API_KEY, reason="ANTHROPIC_API_KEY not set")
    def test_llm_stats(self):
        """LLM 使用統計"""
        r = requests.get(f"{BASE_URL}/llm-stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_calls" in data
        assert "total_cost_usd" in data
        assert "cache_hit_rate" in data

    def test_llm_predict_no_key(self):
        """無 API Key 時應回 503"""
        if HAS_API_KEY:
            pytest.skip("API Key is set, skip no-key test")
        r = requests.post(f"{BASE_URL}/predict/llm", json={
            "text": "hello"
        })
        assert r.status_code == 503


# ============================================================
# 批次比較測試（需要 API Key + Server）
# ============================================================
@pytest.mark.skipif(
    not (server_is_running() and HAS_API_KEY),
    reason="需要 API server + ANTHROPIC_API_KEY"
)
class TestBatchComparison:
    """SVM vs LLM 批次比較"""

    SAMPLE_TEXTS = [
        ("check my balance", "balance"),
        ("book a flight to tokyo", "book_flight"),
        ("what is the exchange rate", "exchange_rate"),
        ("set a reminder for tomorrow", "reminder"),
        ("report my lost credit card", "report_lost_card"),
        ("play some music", "play_music"),
        ("how many calories in a burger", "calories"),
        ("tell me a joke", "tell_joke"),
        ("what is my pto balance", "pto_balance"),
        ("track my order", "order_status"),
    ]

    def test_batch_compare(self):
        """批次比較 SVM vs LLM"""
        agreements = 0
        results = []

        for text, expected in self.SAMPLE_TEXTS:
            r = requests.post(f"{BASE_URL}/compare", json={"text": text})
            data = r.json()
            results.append({
                "text": text,
                "expected": expected,
                "svm_intent": data["svm"]["intent"],
                "llm_intent": data["llm"]["intent"],
                "agree": data["agreement"],
                "svm_correct": data["svm"]["intent"] == expected,
                "llm_correct": data["llm"]["intent"] == expected,
            })
            if data["agreement"]:
                agreements += 1

        # 印出比較表
        print("\n" + "=" * 80)
        print("  SVM vs LLM Batch Comparison")
        print("=" * 80)
        svm_correct = sum(1 for r in results if r["svm_correct"])
        llm_correct = sum(1 for r in results if r["llm_correct"])
        total = len(results)

        for r in results:
            svm_mark = "✅" if r["svm_correct"] else "❌"
            llm_mark = "✅" if r["llm_correct"] else "❌"
            agree_mark = "🤝" if r["agree"] else "⚔️"
            print(f"  {agree_mark} '{r['text']}'")
            print(f"     Expected: {r['expected']}")
            print(f"     SVM: {r['svm_intent']} {svm_mark}  |  LLM: {r['llm_intent']} {llm_mark}")

        print(f"\n  Summary: SVM {svm_correct}/{total} | LLM {llm_correct}/{total} | Agreement {agreements}/{total}")

        # LLM 準確率應該至少 70%
        assert llm_correct / total >= 0.7, f"LLM accuracy too low: {llm_correct}/{total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
