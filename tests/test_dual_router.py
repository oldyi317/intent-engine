"""
雙層路由器測試 (Dual-Layer Router Tests)

測試項目：
  1. 高信心度 → SVM 直接回傳
  2. 低信心度 → LLM 接手（或降級回 SVM）
  3. 複合句 → 拆解後逐句處理
  4. 路由統計正確性
  5. 成本計算
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dual_router import DualLayerRouter, RoutingTier, RoutingResult, CostModel


# ============================================================
# Fixtures
# ============================================================
def make_svm_fn(conf: float = 1.0, intent: str = "balance"):
    """建立一個回傳固定信心度的 mock SVM 函式"""
    def fn(text: str):
        return intent, conf
    return fn


def make_llm_fn(conf: float = 0.95, intent: str = "balance", delay: float = 0.01):
    """建立一個回傳固定信心度的 mock LLM 函式（帶延遲）"""
    def fn(text: str):
        time.sleep(delay)
        return intent, conf
    return fn


def make_failing_llm_fn():
    """建立一個永遠失敗的 mock LLM"""
    def fn(text: str):
        raise RuntimeError("LLM API error")
    return fn


def mock_compound_detect(text: str) -> bool:
    return " and " in text.lower()


def mock_compound_split(text: str) -> list:
    parts = text.lower().split(" and ")
    return [p.strip() for p in parts if p.strip() and len(p.strip().split()) >= 2]


# ============================================================
# 測試：高信心度走 SVM
# ============================================================
class TestHighConfidenceSVM:
    def test_routes_to_svm_when_confidence_high(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=1.2, intent="balance"),
            confidence_threshold=0.8,
        )
        result = router.route("check my balance")
        assert result.tier == RoutingTier.SVM
        assert result.intent == "balance"
        assert result.confidence == 1.2

    def test_svm_has_zero_cost(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=1.0),
            confidence_threshold=0.8,
        )
        result = router.route("check my balance")
        assert result.cost_estimate == 0.0

    def test_svm_latency_under_10ms(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=1.0),
            confidence_threshold=0.8,
        )
        result = router.route("check my balance")
        assert result.latency_ms < 10  # SVM 應該非常快


# ============================================================
# 測試：低信心度走 LLM
# ============================================================
class TestLowConfidenceLLM:
    def test_routes_to_llm_when_confidence_low(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=0.3, intent="greeting"),
            llm_classify_fn=make_llm_fn(conf=0.95, intent="balance"),
            confidence_threshold=0.8,
        )
        result = router.route("what is my account status")
        assert result.tier == RoutingTier.LLM
        assert result.intent == "balance"
        assert result.confidence == 0.95

    def test_llm_has_nonzero_cost(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=0.3),
            llm_classify_fn=make_llm_fn(conf=0.95),
            confidence_threshold=0.8,
        )
        result = router.route("ambiguous query")
        assert result.cost_estimate > 0

    def test_fallback_when_llm_unavailable(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=0.3, intent="greeting"),
            llm_classify_fn=None,  # LLM 未接入
            confidence_threshold=0.8,
        )
        result = router.route("ambiguous query")
        assert result.tier == RoutingTier.FALLBACK
        assert result.intent == "greeting"

    def test_fallback_when_llm_fails(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=0.3, intent="greeting"),
            llm_classify_fn=make_failing_llm_fn(),
            confidence_threshold=0.8,
        )
        result = router.route("ambiguous query")
        assert result.tier == RoutingTier.FALLBACK


# ============================================================
# 測試：複合句處理
# ============================================================
class TestCompoundQueries:
    def test_compound_detected_and_split(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=1.0, intent="balance"),
            compound_detect_fn=mock_compound_detect,
            compound_split_fn=mock_compound_split,
            confidence_threshold=0.8,
        )
        result = router.route("check my balance and book a flight")
        assert result.is_compound is True
        assert len(result.sub_intents) == 2

    def test_single_query_not_compound(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=1.0),
            compound_detect_fn=mock_compound_detect,
            compound_split_fn=mock_compound_split,
            confidence_threshold=0.8,
        )
        result = router.route("check my balance")
        assert result.is_compound is False
        assert result.sub_intents == []


# ============================================================
# 測試：路由統計
# ============================================================
class TestRoutingStats:
    def test_stats_accumulate(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=1.0),
            confidence_threshold=0.8,
        )
        for _ in range(5):
            router.route("test query")

        stats = router.get_stats()
        assert stats["total_requests"] == 5
        assert stats["svm_handled"] == 5
        assert stats["svm_ratio"] == "100.0%"

    def test_mixed_routing_stats(self):
        # 用不同信心度模擬混合路由
        call_count = [0]

        def varying_svm(text: str):
            call_count[0] += 1
            if call_count[0] <= 4:
                return "balance", 1.0  # 高信心
            else:
                return "unknown", 0.3  # 低信心

        router = DualLayerRouter(
            svm_classify_fn=varying_svm,
            confidence_threshold=0.8,
        )
        for _ in range(5):
            router.route("test")

        stats = router.get_stats()
        assert stats["svm_handled"] == 4
        assert stats["fallback_count"] == 1

    def test_reset_stats(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=1.0),
            confidence_threshold=0.8,
        )
        router.route("test")
        assert router.get_stats()["total_requests"] == 1
        router.reset_stats()
        assert router.get_stats()["total_requests"] == 0


# ============================================================
# 測試：成本模型
# ============================================================
class TestCostModel:
    def test_svm_free(self):
        assert CostModel.estimate_cost(RoutingTier.SVM) == 0.0

    def test_llm_costs_money(self):
        assert CostModel.estimate_cost(RoutingTier.LLM) > 0

    def test_bert_cheaper_than_llm(self):
        bert_cost = CostModel.estimate_cost(RoutingTier.BERT)
        llm_cost = CostModel.estimate_cost(RoutingTier.LLM)
        assert bert_cost < llm_cost


# ============================================================
# 測試：信心度閾值邊界
# ============================================================
class TestThresholdBoundary:
    def test_exactly_at_threshold_goes_svm(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=0.8),
            llm_classify_fn=make_llm_fn(conf=0.95),
            confidence_threshold=0.8,
        )
        result = router.route("test")
        assert result.tier == RoutingTier.SVM

    def test_just_below_threshold_goes_llm(self):
        router = DualLayerRouter(
            svm_classify_fn=make_svm_fn(conf=0.79),
            llm_classify_fn=make_llm_fn(conf=0.95),
            confidence_threshold=0.8,
        )
        result = router.route("test")
        assert result.tier == RoutingTier.LLM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
