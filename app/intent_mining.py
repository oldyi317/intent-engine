"""
隱藏意圖挖掘模組 (Hidden Intent Mining)

功能：
  1. IntentLogger — 紀錄每次意圖分類結果到 SQLite
     (customer_id, timestamp, intent, domain, confidence)
  2. IntentMiner — 分析客戶意圖歷史，挖掘隱藏需求
     - 高頻意圖偵測
     - 意圖共現模式（association rules）
     - 風控預警觸發
     - 交叉銷售推薦
  3. ProductRecommender — 基於意圖模式的產品推薦引擎

設計理念：
  BERT 模型能對每句話分類出意圖，但單次分類只是起點。
  將分類結果累積成客戶意圖歷史檔案，就能挖掘出客戶
  「沒有明說但可能需要」的隱藏需求。

  例如：
  - 客戶近期頻繁詢問 exchange_rate + transfer → 外幣理財潛在客戶
  - 客戶出現 lost_luggage + report_lost_card → 風控預警
  - 客戶查詢 credit_score + interest_rate → 信貸需求

Usage:
    from app.intent_mining import IntentLogger, IntentMiner

    # 紀錄意圖
    log = IntentLogger("intent_history.db")
    log.log("CUST001", "balance", "Finance", 0.95)
    log.log("CUST001", "exchange_rate", "Shopping", 0.87)

    # 挖掘隱藏意圖
    miner = IntentMiner(log)
    insights = miner.analyze_customer("CUST001")
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

# 統一使用 src.config 中的映射，避免重複定義
from src.config import INTENT_TO_DOMAIN

logger = logging.getLogger(__name__)


# ============================================================
# 風控規則與產品推薦規則
# ============================================================
@dataclass
class MiningRule:
    """挖掘規則"""
    rule_id: str
    name: str
    description: str
    trigger_intents: List[str]     # 需要出現的意圖組合
    min_occurrences: int = 1       # 每個意圖至少出現幾次
    time_window_days: int = 30     # 時間窗口（天）
    category: str = "general"      # risk_alert / cross_sell / retention
    priority: int = 1              # 優先度 1(高) ~ 5(低)
    action: str = ""               # 建議動作


# --- 風控預警規則 ---
RISK_RULES: List[MiningRule] = [
    MiningRule(
        rule_id="RISK_001",
        name="旅途遺失風險",
        description="客戶近期出現行李遺失 + 卡片掛失，可能遭遇旅途盜竊",
        trigger_intents=["lost_luggage", "report_lost_card"],
        min_occurrences=1,
        time_window_days=7,
        category="risk_alert",
        priority=1,
        action="自動調降刷卡額度 → 啟動詐欺防護 → 通知風控團隊",
    ),
    MiningRule(
        rule_id="RISK_002",
        name="帳戶異常活動",
        description="客戶短時間內多次查詢交易紀錄 + 凍結帳戶，可能帳戶被盜",
        trigger_intents=["transactions", "freeze_account"],
        min_occurrences=1,
        time_window_days=3,
        category="risk_alert",
        priority=1,
        action="立即凍結帳戶 → 發送簡訊驗證 → 啟動身份確認流程",
    ),
    MiningRule(
        rule_id="RISK_003",
        name="信用卡盜刷風險",
        description="客戶回報詐欺 + 卡片被拒，高度懷疑盜刷",
        trigger_intents=["report_fraud", "card_declined"],
        min_occurrences=1,
        time_window_days=7,
        category="risk_alert",
        priority=1,
        action="暫停卡片 → 追蹤可疑交易 → 安排專人聯繫",
    ),
    MiningRule(
        rule_id="RISK_004",
        name="盜刷後大額轉帳",
        description="客戶掛失信用卡 + 大額轉帳，可能遭竊後緊急搬移資金或被騙",
        trigger_intents=["report_lost_card", "transfer"],
        min_occurrences=1,
        time_window_days=7,
        category="risk_alert",
        priority=1,
        action="人工審核轉帳 → 確認是否為客戶本人操作 → 啟動防盜刷流程",
    ),
    MiningRule(
        rule_id="RISK_005",
        name="帳戶關閉前異常",
        description="客戶要求關戶 + 查餘額 + 轉帳，可能資金被盜正在清空帳戶",
        trigger_intents=["close_account", "balance"],
        min_occurrences=1,
        time_window_days=7,
        category="risk_alert",
        priority=1,
        action="暫緩關戶 → 聯繫客戶確認意圖 → 風控團隊介入",
    ),
    MiningRule(
        rule_id="RISK_006",
        name="卡片遺失併保險取消",
        description="客戶掛失卡片 + 取消保險，可能遭受旅途損失後連環止損",
        trigger_intents=["report_lost_card", "insurance_change"],
        min_occurrences=1,
        time_window_days=14,
        category="risk_alert",
        priority=2,
        action="主動聯繫了解損失情況 → 啟動理賠流程 → 提供緊急援助",
    ),
]

# --- 交叉銷售規則 ---
CROSS_SELL_RULES: List[MiningRule] = [
    MiningRule(
        rule_id="SELL_001",
        name="外幣理財潛在客戶",
        description="客戶頻繁查詢匯率 + 轉帳，可能有外幣理財需求",
        trigger_intents=["exchange_rate", "transfer"],
        min_occurrences=2,
        time_window_days=30,
        category="cross_sell",
        priority=2,
        action="推播雙幣理財產品 → 高利外幣定存專案",
    ),
    MiningRule(
        rule_id="SELL_002",
        name="信貸需求客戶",
        description="客戶查詢信用分數 + 利率，可能在評估貸款",
        trigger_intents=["credit_score", "interest_rate"],
        min_occurrences=1,
        time_window_days=14,
        category="cross_sell",
        priority=2,
        action="推薦低利率信貸方案 → 安排理財專員諮詢",
    ),
    MiningRule(
        rule_id="SELL_003",
        name="旅遊金融服務",
        description="客戶頻繁查詢機票 + 旅遊相關，可能需要旅遊金融產品",
        trigger_intents=["book_flight", "travel_suggestion"],
        min_occurrences=1,
        time_window_days=14,
        category="cross_sell",
        priority=3,
        action="推薦旅平險 → 旅遊刷卡回饋方案 → 機場接送優惠",
    ),
    MiningRule(
        rule_id="SELL_004",
        name="保險升級需求",
        description="客戶查詢保險 + 詢問保險變更，可能在比較方案",
        trigger_intents=["insurance", "insurance_change"],
        min_occurrences=1,
        time_window_days=30,
        category="cross_sell",
        priority=3,
        action="安排保險顧問 → 提供保單健檢服務",
    ),
    MiningRule(
        rule_id="SELL_005",
        name="退休規劃客戶",
        description="客戶查詢 401k 轉帳 + 收入，可能在規劃退休",
        trigger_intents=["rollover_401k", "income"],
        min_occurrences=1,
        time_window_days=30,
        category="cross_sell",
        priority=3,
        action="推薦退休理財規劃 → 年金保險方案",
    ),
    MiningRule(
        rule_id="SELL_006",
        name="資金管理需求",
        description="客戶查詢餘額 + 進行轉帳，可能有資金調度或理財需求",
        trigger_intents=["balance", "transfer"],
        min_occurrences=1,
        time_window_days=14,
        category="cross_sell",
        priority=3,
        action="推薦自動理財帳戶 → 餘額通知服務 → 定期定額投資方案",
    ),
]

# --- 客戶留存規則 ---
RETENTION_RULES: List[MiningRule] = [
    MiningRule(
        rule_id="RET_001",
        name="卡片不滿意信號",
        description="客戶查詢信用額度 + 國際手續費，可能考慮換卡",
        trigger_intents=["credit_limit", "international_fees"],
        min_occurrences=1,
        time_window_days=14,
        category="retention",
        priority=2,
        action="主動提供升等方案 → 手續費減免優惠",
    ),
    MiningRule(
        rule_id="RET_002",
        name="高頻客訴信號",
        description="客戶多次出現帳戶被鎖 + 卡片被拒，體驗不佳",
        trigger_intents=["account_blocked", "card_declined"],
        min_occurrences=2,
        time_window_days=30,
        category="retention",
        priority=2,
        action="安排客服主管回訪 → 提供補償方案",
    ),
    MiningRule(
        rule_id="RET_003",
        name="客戶流失預警",
        description="客戶要求關閉帳戶 + 取消保險，極高流失風險",
        trigger_intents=["close_account", "insurance_change"],
        min_occurrences=1,
        time_window_days=30,
        category="retention",
        priority=1,
        action="立即轉接主管 → 了解不滿原因 → 提供挽留方案（手續費減免/利率優惠）",
    ),
    MiningRule(
        rule_id="RET_004",
        name="服務不滿信號",
        description="客戶投訴服務 + 查詢帳戶/轉帳，可能準備轉銀行",
        trigger_intents=["complaint", "balance"],
        min_occurrences=1,
        time_window_days=14,
        category="retention",
        priority=2,
        action="安排資深客服回訪 → 記錄客訴內容 → 追蹤改善",
    ),
]

ALL_RULES = RISK_RULES + CROSS_SELL_RULES + RETENTION_RULES


# ============================================================
# 產品推薦對照表
# ============================================================
PRODUCT_CATALOG = {
    "SELL_001": [
        {"product": "雙幣理財帳戶", "description": "美元/台幣自動轉換，享高利定存", "target_return": "年化 3.5%~5.2%"},
        {"product": "外幣定存專案", "description": "美元/日圓/歐元 3-6 個月期定存", "target_return": "年化 4.0%+"},
    ],
    "SELL_002": [
        {"product": "低利信貸方案", "description": "年利率 2.5% 起，最高額度 300 萬", "target_return": "月付 $8,333 起"},
        {"product": "信用卡分期 0 利率", "description": "消費滿 3,000 可分 12 期 0 利率", "target_return": "0% 利率"},
    ],
    "SELL_003": [
        {"product": "旅遊平安險", "description": "全球旅遊保障，含班機延誤理賠", "target_return": "保費 $299 起/趟"},
        {"product": "旅遊聯名卡", "description": "海外消費 3% 回饋 + 機場貴賓室", "target_return": "年費 $2,000 回饋 $6,000+"},
    ],
    "SELL_004": [
        {"product": "保單健檢服務", "description": "免費專業保單檢視，找出保障缺口", "target_return": "免費"},
        {"product": "投資型保單", "description": "壽險保障 + 基金投資雙重效益", "target_return": "年化 4%~8%"},
    ],
    "SELL_005": [
        {"product": "退休理財規劃", "description": "量身打造退休金目標與資產配置", "target_return": "免費諮詢"},
        {"product": "年金保險", "description": "保證每月領取，活到老領到老", "target_return": "月領 $15,000 起"},
    ],
    "SELL_006": [
        {"product": "智能理財帳戶", "description": "閒置資金自動轉入高利活存", "target_return": "年化 2.5%~3.0%"},
        {"product": "定期定額基金", "description": "每月自動扣款投資，分散風險", "target_return": "長期年化 6%~8%"},
    ],
}


# ============================================================
# IntentLogger — 意圖紀錄器
# ============================================================
class IntentLogger:
    """
    將每次意圖分類結果寫入 SQLite 資料庫

    Schema:
        intent_logs (
            id INTEGER PRIMARY KEY,
            customer_id TEXT,
            timestamp TEXT (ISO 8601),
            intent TEXT,
            domain TEXT,
            confidence REAL,
            original_text TEXT,
            routing_tier TEXT
        )
    """

    def __init__(self, db_path: str = "intent_history.db"):
        self.db_path = db_path
        # Ensure parent directory exists (e.g. data/ may not exist on Render)
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self._init_db()

    def _init_db(self):
        """建立資料表"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS intent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                intent TEXT NOT NULL,
                domain TEXT,
                confidence REAL,
                original_text TEXT,
                routing_tier TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_customer_time
            ON intent_logs (customer_id, timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_intent
            ON intent_logs (intent)
        """)
        conn.commit()
        conn.close()

    def log(
        self,
        customer_id: str,
        intent: str,
        confidence: float = 0.0,
        original_text: str = "",
        routing_tier: str = "svm",
        timestamp: Optional[str] = None,
    ) -> int:
        """
        紀錄一筆意圖分類結果

        Returns:
            新紀錄的 ID
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        domain = INTENT_TO_DOMAIN.get(intent, "Unknown")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """INSERT INTO intent_logs
               (customer_id, timestamp, intent, domain, confidence, original_text, routing_tier)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (customer_id, timestamp, intent, domain, confidence, original_text, routing_tier),
        )
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(
            f"[IntentLog] {customer_id} → {intent} ({domain}, conf={confidence:.3f})"
        )
        return record_id

    def get_customer_history(
        self,
        customer_id: str,
        days: int = 30,
    ) -> List[Dict]:
        """取得客戶在指定天數內的意圖歷史"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM intent_logs
               WHERE customer_id = ? AND timestamp >= ?
               ORDER BY timestamp DESC""",
            (customer_id, cutoff),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_all_customers(self) -> List[str]:
        """取得所有有紀錄的客戶 ID"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT DISTINCT customer_id FROM intent_logs ORDER BY customer_id"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        """取得整體統計"""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute("SELECT COUNT(*) FROM intent_logs").fetchone()[0]
        n_customers = conn.execute(
            "SELECT COUNT(DISTINCT customer_id) FROM intent_logs"
        ).fetchone()[0]
        top_intents = conn.execute(
            """SELECT intent, COUNT(*) as cnt
               FROM intent_logs GROUP BY intent
               ORDER BY cnt DESC LIMIT 10"""
        ).fetchall()
        conn.close()
        return {
            "total_logs": total,
            "unique_customers": n_customers,
            "top_intents": [{"intent": r[0], "count": r[1]} for r in top_intents],
        }


# ============================================================
# IntentMiner — 意圖挖掘引擎
# ============================================================
@dataclass
class MiningInsight:
    """挖掘結果"""
    rule_id: str
    rule_name: str
    category: str        # risk_alert / cross_sell / retention
    priority: int
    description: str
    matched_intents: List[str]
    action: str
    products: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class IntentMiner:
    """
    意圖挖掘引擎

    根據客戶的意圖歷史，比對預定義規則，
    產出風控預警、交叉銷售、客戶留存等 insights
    """

    def __init__(self, intent_logger: IntentLogger, rules: Optional[List[MiningRule]] = None):
        self.logger = intent_logger
        self.rules = rules or ALL_RULES

    def analyze_customer(
        self,
        customer_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        完整分析一位客戶的隱藏意圖

        Returns:
            {
                "customer_id": str,
                "analysis_period_days": int,
                "total_interactions": int,
                "intent_profile": {intent: count, ...},
                "domain_profile": {domain: count, ...},
                "insights": [MiningInsight, ...],
                "risk_alerts": [...],
                "cross_sell_opportunities": [...],
                "retention_signals": [...],
            }
        """
        history = self.logger.get_customer_history(customer_id, days)

        if not history:
            return {
                "customer_id": customer_id,
                "analysis_period_days": days,
                "total_interactions": 0,
                "intent_profile": {},
                "domain_profile": {},
                "insights": [],
                "risk_alerts": [],
                "cross_sell_opportunities": [],
                "retention_signals": [],
            }

        # 建立意圖/領域 Profile
        intent_counter = Counter(h["intent"] for h in history)
        domain_counter = Counter(h["domain"] for h in history if h["domain"])

        # 比對規則
        insights = self._match_rules(intent_counter, history)

        # 分類 insights
        risk_alerts = [i for i in insights if i.category == "risk_alert"]
        cross_sell = [i for i in insights if i.category == "cross_sell"]
        retention = [i for i in insights if i.category == "retention"]

        return {
            "customer_id": customer_id,
            "analysis_period_days": days,
            "total_interactions": len(history),
            "intent_profile": dict(intent_counter.most_common()),
            "domain_profile": dict(domain_counter.most_common()),
            "insights": [i.to_dict() for i in insights],
            "risk_alerts": [i.to_dict() for i in risk_alerts],
            "cross_sell_opportunities": [i.to_dict() for i in cross_sell],
            "retention_signals": [i.to_dict() for i in retention],
        }

    def _match_rules(
        self,
        intent_counter: Counter,
        history: List[Dict],
    ) -> List[MiningInsight]:
        """比對所有規則"""
        matched = []

        for rule in self.rules:
            # 檢查是否所有 trigger intents 都達到最低出現次數
            all_triggered = all(
                intent_counter.get(intent, 0) >= rule.min_occurrences
                for intent in rule.trigger_intents
            )

            if all_triggered:
                # 檢查時間窗口
                if self._check_time_window(history, rule):
                    products = PRODUCT_CATALOG.get(rule.rule_id, [])
                    insight = MiningInsight(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        category=rule.category,
                        priority=rule.priority,
                        description=rule.description,
                        matched_intents=rule.trigger_intents,
                        action=rule.action,
                        products=products,
                    )
                    matched.append(insight)

        # 依優先度排序
        matched.sort(key=lambda x: x.priority)
        return matched

    def _check_time_window(self, history: List[Dict], rule: MiningRule) -> bool:
        """檢查觸發意圖是否都在時間窗口內"""
        cutoff = (datetime.now() - timedelta(days=rule.time_window_days)).isoformat()
        recent_intents = set()
        for h in history:
            if h["timestamp"] >= cutoff:
                recent_intents.add(h["intent"])
        return all(intent in recent_intents for intent in rule.trigger_intents)

    def batch_analyze(self, days: int = 30) -> List[Dict]:
        """
        批次分析所有客戶

        Returns:
            有 insights 的客戶清單，依風控優先度排序
        """
        customers = self.logger.get_all_customers()
        results = []

        for cid in customers:
            analysis = self.analyze_customer(cid, days)
            if analysis["insights"]:
                results.append(analysis)

        # 依最高優先度排序
        results.sort(
            key=lambda x: min((i["priority"] for i in x["insights"]), default=99)
        )
        return results


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    import tempfile

    # 用暫存資料庫測試
    db_path = os.path.join(tempfile.gettempdir(), "test_intent_mining.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    log = IntentLogger(db_path)

    # 模擬客戶行為
    print("=" * 60)
    print("  Hidden Intent Mining Demo")
    print("=" * 60)

    # 客戶 A：旅途遺失風險
    print("\n--- 模擬客戶 A（旅途遺失風險）---")
    now = datetime.now()
    log.log("CUST_A", "book_flight", 0.92, "book a flight to tokyo",
            timestamp=(now - timedelta(days=5)).isoformat())
    log.log("CUST_A", "lost_luggage", 0.88, "my luggage is lost",
            timestamp=(now - timedelta(days=2)).isoformat())
    log.log("CUST_A", "report_lost_card", 0.95, "I lost my credit card",
            timestamp=(now - timedelta(days=1)).isoformat())

    # 客戶 B：外幣理財潛在客戶
    print("--- 模擬客戶 B（外幣理財需求）---")
    log.log("CUST_B", "exchange_rate", 0.91, "what's the USD exchange rate",
            timestamp=(now - timedelta(days=10)).isoformat())
    log.log("CUST_B", "exchange_rate", 0.89, "JPY to TWD rate",
            timestamp=(now - timedelta(days=7)).isoformat())
    log.log("CUST_B", "transfer", 0.93, "transfer money to savings",
            timestamp=(now - timedelta(days=5)).isoformat())
    log.log("CUST_B", "transfer", 0.90, "wire transfer to USD account",
            timestamp=(now - timedelta(days=3)).isoformat())
    log.log("CUST_B", "exchange_rate", 0.87, "EUR exchange rate today",
            timestamp=(now - timedelta(days=1)).isoformat())

    # 客戶 C：信貸需求
    print("--- 模擬客戶 C（信貸需求）---")
    log.log("CUST_C", "credit_score", 0.94, "what's my credit score",
            timestamp=(now - timedelta(days=8)).isoformat())
    log.log("CUST_C", "interest_rate", 0.91, "current interest rates",
            timestamp=(now - timedelta(days=6)).isoformat())
    log.log("CUST_C", "credit_limit_change", 0.87, "can I increase my credit limit",
            timestamp=(now - timedelta(days=3)).isoformat())

    # 分析
    miner = IntentMiner(log)

    for cid in ["CUST_A", "CUST_B", "CUST_C"]:
        result = miner.analyze_customer(cid)
        print(f"\n{'='*60}")
        print(f"  客戶: {cid}")
        print(f"  互動次數: {result['total_interactions']}")
        print(f"  意圖分布: {result['intent_profile']}")
        print(f"  領域分布: {result['domain_profile']}")

        if result["risk_alerts"]:
            print(f"\n  🚨 風控預警 ({len(result['risk_alerts'])} 筆):")
            for a in result["risk_alerts"]:
                print(f"    [{a['rule_id']}] {a['rule_name']}")
                print(f"      觸發意圖: {a['matched_intents']}")
                print(f"      建議動作: {a['action']}")

        if result["cross_sell_opportunities"]:
            print(f"\n  💰 交叉銷售機會 ({len(result['cross_sell_opportunities'])} 筆):")
            for o in result["cross_sell_opportunities"]:
                print(f"    [{o['rule_id']}] {o['rule_name']}")
                print(f"      觸發意圖: {o['matched_intents']}")
                print(f"      建議動作: {o['action']}")
                if o.get("products"):
                    for p in o["products"]:
                        print(f"      → 推薦: {p['product']} ({p['description']})")

    # 批次分析
    print(f"\n{'='*60}")
    print("  批次分析結果")
    print("=" * 60)
    batch = miner.batch_analyze()
    print(f"  共 {len(batch)} 位客戶有 insights")
    for b in batch:
        n_risk = len(b["risk_alerts"])
        n_sell = len(b["cross_sell_opportunities"])
        n_ret = len(b["retention_signals"])
        print(f"    {b['customer_id']}: 風控={n_risk}, 銷售={n_sell}, 留存={n_ret}")

    # 清理
    os.remove(db_path)
    print("\nDemo 完成！")
