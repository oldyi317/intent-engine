"""
隱藏意圖挖掘測試 (Intent Mining Tests)

測試項目：
  1. IntentLogger — 紀錄/查詢意圖歷史
  2. IntentMiner — 規則比對與 insight 產出
  3. 風控預警規則觸發
  4. 交叉銷售規則觸發
  5. 客戶留存信號
  6. 批次分析
"""

import os
import sys
import tempfile
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.intent_mining import IntentLogger, IntentMiner, INTENT_TO_DOMAIN


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def db_path():
    """建立暫存資料庫路徑"""
    path = os.path.join(tempfile.gettempdir(), "test_mining.db")
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def intent_logger(db_path):
    """建立 IntentLogger 實例"""
    return IntentLogger(db_path)


@pytest.fixture
def intent_miner(intent_logger):
    """建立 IntentMiner 實例"""
    return IntentMiner(intent_logger)


# ============================================================
# 測試：IntentLogger
# ============================================================
class TestIntentLogger:
    def test_log_and_retrieve(self, intent_logger):
        intent_logger.log("CUST001", "balance", 0.95, "check my balance")
        history = intent_logger.get_customer_history("CUST001")
        assert len(history) == 1
        assert history[0]["intent"] == "balance"
        assert history[0]["domain"] == "Finance"

    def test_multiple_logs(self, intent_logger):
        intent_logger.log("CUST001", "balance", 0.95)
        intent_logger.log("CUST001", "transfer", 0.90)
        intent_logger.log("CUST001", "exchange_rate", 0.88)
        history = intent_logger.get_customer_history("CUST001")
        assert len(history) == 3

    def test_customer_isolation(self, intent_logger):
        intent_logger.log("CUST001", "balance", 0.95)
        intent_logger.log("CUST002", "book_flight", 0.90)
        h1 = intent_logger.get_customer_history("CUST001")
        h2 = intent_logger.get_customer_history("CUST002")
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0]["intent"] == "balance"
        assert h2[0]["intent"] == "book_flight"

    def test_time_window_filtering(self, intent_logger):
        now = datetime.now()
        # 40 天前的紀錄
        intent_logger.log("CUST001", "balance", 0.95,
                          timestamp=(now - timedelta(days=40)).isoformat())
        # 5 天前的紀錄
        intent_logger.log("CUST001", "transfer", 0.90,
                          timestamp=(now - timedelta(days=5)).isoformat())

        # 30 天內只有 1 筆
        history = intent_logger.get_customer_history("CUST001", days=30)
        assert len(history) == 1
        assert history[0]["intent"] == "transfer"

    def test_domain_mapping(self, intent_logger):
        intent_logger.log("CUST001", "book_flight", 0.90)
        history = intent_logger.get_customer_history("CUST001")
        assert history[0]["domain"] == "Travel"

    def test_get_all_customers(self, intent_logger):
        intent_logger.log("CUST_A", "balance", 0.95)
        intent_logger.log("CUST_B", "transfer", 0.90)
        intent_logger.log("CUST_C", "book_flight", 0.88)
        customers = intent_logger.get_all_customers()
        assert set(customers) == {"CUST_A", "CUST_B", "CUST_C"}

    def test_stats(self, intent_logger):
        intent_logger.log("CUST001", "balance", 0.95)
        intent_logger.log("CUST001", "balance", 0.93)
        intent_logger.log("CUST002", "transfer", 0.90)
        stats = intent_logger.get_stats()
        assert stats["total_logs"] == 3
        assert stats["unique_customers"] == 2


# ============================================================
# 測試：風控預警
# ============================================================
class TestRiskAlerts:
    def test_travel_loss_risk(self, intent_logger, intent_miner):
        """旅途遺失風險：lost_luggage + report_lost_card"""
        now = datetime.now()
        intent_logger.log("CUST_R1", "lost_luggage", 0.88,
                          timestamp=(now - timedelta(days=2)).isoformat())
        intent_logger.log("CUST_R1", "report_lost_card", 0.95,
                          timestamp=(now - timedelta(days=1)).isoformat())

        result = intent_miner.analyze_customer("CUST_R1")
        assert len(result["risk_alerts"]) >= 1
        rule_ids = [a["rule_id"] for a in result["risk_alerts"]]
        assert "RISK_001" in rule_ids

    def test_account_anomaly_risk(self, intent_logger, intent_miner):
        """帳戶異常：transactions + freeze_account"""
        now = datetime.now()
        intent_logger.log("CUST_R2", "transactions", 0.92,
                          timestamp=(now - timedelta(days=1)).isoformat())
        intent_logger.log("CUST_R2", "freeze_account", 0.90,
                          timestamp=(now - timedelta(hours=2)).isoformat())

        result = intent_miner.analyze_customer("CUST_R2")
        risk_ids = [a["rule_id"] for a in result["risk_alerts"]]
        assert "RISK_002" in risk_ids

    def test_no_risk_for_normal_user(self, intent_logger, intent_miner):
        """正常客戶不應觸發風控"""
        now = datetime.now()
        intent_logger.log("CUST_NORMAL", "balance", 0.95,
                          timestamp=(now - timedelta(days=1)).isoformat())
        intent_logger.log("CUST_NORMAL", "weather", 0.88,
                          timestamp=now.isoformat())

        result = intent_miner.analyze_customer("CUST_NORMAL")
        assert len(result["risk_alerts"]) == 0


# ============================================================
# 測試：交叉銷售
# ============================================================
class TestCrossSell:
    def test_forex_opportunity(self, intent_logger, intent_miner):
        """外幣理財：exchange_rate(2+) + transfer(2+)"""
        now = datetime.now()
        intent_logger.log("CUST_S1", "exchange_rate", 0.91,
                          timestamp=(now - timedelta(days=10)).isoformat())
        intent_logger.log("CUST_S1", "exchange_rate", 0.89,
                          timestamp=(now - timedelta(days=7)).isoformat())
        intent_logger.log("CUST_S1", "transfer", 0.93,
                          timestamp=(now - timedelta(days=5)).isoformat())
        intent_logger.log("CUST_S1", "transfer", 0.90,
                          timestamp=(now - timedelta(days=3)).isoformat())

        result = intent_miner.analyze_customer("CUST_S1")
        sell_ids = [o["rule_id"] for o in result["cross_sell_opportunities"]]
        assert "SELL_001" in sell_ids

        # 確認有推薦產品
        sell_001 = [o for o in result["cross_sell_opportunities"] if o["rule_id"] == "SELL_001"][0]
        assert len(sell_001["products"]) > 0

    def test_loan_opportunity(self, intent_logger, intent_miner):
        """信貸需求：credit_score + interest_rate"""
        now = datetime.now()
        intent_logger.log("CUST_S2", "credit_score", 0.94,
                          timestamp=(now - timedelta(days=8)).isoformat())
        intent_logger.log("CUST_S2", "interest_rate", 0.91,
                          timestamp=(now - timedelta(days=6)).isoformat())

        result = intent_miner.analyze_customer("CUST_S2")
        sell_ids = [o["rule_id"] for o in result["cross_sell_opportunities"]]
        assert "SELL_002" in sell_ids

    def test_forex_requires_min_occurrences(self, intent_logger, intent_miner):
        """外幣理財需要 exchange_rate 至少 2 次"""
        now = datetime.now()
        intent_logger.log("CUST_S3", "exchange_rate", 0.91,
                          timestamp=(now - timedelta(days=5)).isoformat())
        intent_logger.log("CUST_S3", "transfer", 0.93,
                          timestamp=(now - timedelta(days=3)).isoformat())

        result = intent_miner.analyze_customer("CUST_S3")
        sell_ids = [o["rule_id"] for o in result["cross_sell_opportunities"]]
        # 只有 1 次 exchange_rate，不夠觸發 SELL_001（需要 2 次）
        assert "SELL_001" not in sell_ids


# ============================================================
# 測試：客戶留存
# ============================================================
class TestRetention:
    def test_card_dissatisfaction(self, intent_logger, intent_miner):
        """卡片不滿意：credit_limit + international_fees"""
        now = datetime.now()
        intent_logger.log("CUST_RET", "credit_limit", 0.92,
                          timestamp=(now - timedelta(days=5)).isoformat())
        intent_logger.log("CUST_RET", "international_fees", 0.88,
                          timestamp=(now - timedelta(days=3)).isoformat())

        result = intent_miner.analyze_customer("CUST_RET")
        ret_ids = [s["rule_id"] for s in result["retention_signals"]]
        assert "RET_001" in ret_ids


# ============================================================
# 測試：批次分析
# ============================================================
class TestBatchAnalysis:
    def test_batch_returns_customers_with_insights(self, intent_logger, intent_miner):
        now = datetime.now()
        # 客戶 A 有風控 insight
        intent_logger.log("CUST_A", "lost_luggage", 0.88,
                          timestamp=(now - timedelta(days=2)).isoformat())
        intent_logger.log("CUST_A", "report_lost_card", 0.95,
                          timestamp=(now - timedelta(days=1)).isoformat())
        # 客戶 B 無 insight
        intent_logger.log("CUST_B", "weather", 0.90,
                          timestamp=now.isoformat())

        results = intent_miner.batch_analyze()
        customer_ids = [r["customer_id"] for r in results]
        assert "CUST_A" in customer_ids
        assert "CUST_B" not in customer_ids  # 無 insight 不會出現

    def test_batch_sorted_by_priority(self, intent_logger, intent_miner):
        now = datetime.now()
        # 客戶 A：風控（priority=1）
        intent_logger.log("A", "lost_luggage", 0.88,
                          timestamp=(now - timedelta(days=2)).isoformat())
        intent_logger.log("A", "report_lost_card", 0.95,
                          timestamp=(now - timedelta(days=1)).isoformat())
        # 客戶 B：銷售（priority=3）
        intent_logger.log("B", "book_flight", 0.90,
                          timestamp=(now - timedelta(days=5)).isoformat())
        intent_logger.log("B", "travel_suggestion", 0.85,
                          timestamp=(now - timedelta(days=3)).isoformat())

        results = intent_miner.batch_analyze()
        if len(results) >= 2:
            # 風控應該排在銷售前面
            ids = [r["customer_id"] for r in results]
            assert ids.index("A") < ids.index("B")


# ============================================================
# 測試：空資料
# ============================================================
class TestEdgeCases:
    def test_no_history(self, intent_miner):
        result = intent_miner.analyze_customer("NONEXISTENT")
        assert result["total_interactions"] == 0
        assert result["insights"] == []

    def test_domain_mapping_coverage(self):
        """確認常用意圖都有對應的 domain"""
        common_intents = ["balance", "transfer", "book_flight", "greeting",
                          "exchange_rate", "credit_score", "report_lost_card"]
        for intent in common_intents:
            assert intent in INTENT_TO_DOMAIN, f"{intent} 缺少 domain 映射"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
