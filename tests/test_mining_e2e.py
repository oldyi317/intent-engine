"""
隱藏意圖挖掘 — 端對端測試腳本

不需要啟動 API server，直接用模組測試完整流程：
  1. 模擬不同類型客戶的對話行為
  2. 用 IntentLogger 紀錄意圖歷史
  3. 用 IntentMiner 分析隱藏需求
  4. 驗證風控預警、交叉銷售、客戶留存是否正確觸發

執行方式：
    cd fubon_intent_project
    python tests/test_mining_e2e.py
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.intent_mining import IntentLogger, IntentMiner


def create_test_db():
    db_path = os.path.join(tempfile.gettempdir(), "e2e_mining_test.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    return db_path


def simulate_customers(logger: IntentLogger):
    """模擬 5 類客戶行為"""
    now = datetime.now()

    # ─────────────────────────────────────────
    # 客戶 A：出國旅途遭竊（風控預警）
    # 預期觸發：RISK_001 旅途遺失風險
    # ─────────────────────────────────────────
    print("\n  [模擬] 客戶 A — 出國旅途遭竊")
    steps = [
        (-7, "book_flight",       0.93, "book a flight to paris"),
        (-5, "travel_suggestion",  0.88, "what to do in paris"),
        (-2, "lost_luggage",       0.91, "my luggage is missing at the airport"),
        (-1, "report_lost_card",   0.95, "I think my credit card was stolen"),
        ( 0, "freeze_account",     0.90, "please freeze my account immediately"),
    ]
    for day_offset, intent, conf, text in steps:
        ts = (now + timedelta(days=day_offset)).isoformat()
        logger.log("CUST_A", intent, conf, text, timestamp=ts)
        print(f"    Day {day_offset:+d}: {intent} ({text})")

    # ─────────────────────────────────────────
    # 客戶 B：頻繁查匯率+轉帳（外幣理財需求）
    # 預期觸發：SELL_001 外幣理財潛在客戶
    # ─────────────────────────────────────────
    print("\n  [模擬] 客戶 B — 外幣理財需求")
    steps = [
        (-20, "exchange_rate", 0.89, "what is the USD to TWD rate"),
        (-15, "exchange_rate", 0.91, "JPY exchange rate today"),
        (-12, "transfer",      0.93, "transfer money to my USD savings"),
        (-8,  "exchange_rate", 0.87, "EUR to TWD exchange rate"),
        (-5,  "transfer",      0.90, "wire transfer to USD account"),
        (-2,  "exchange_rate", 0.92, "current USD exchange rate"),
    ]
    for day_offset, intent, conf, text in steps:
        ts = (now + timedelta(days=day_offset)).isoformat()
        logger.log("CUST_B", intent, conf, text, timestamp=ts)
        print(f"    Day {day_offset:+d}: {intent} ({text})")

    # ─────────────────────────────────────────
    # 客戶 C：信貸比較中（信貸需求）
    # 預期觸發：SELL_002 信貸需求客戶
    # ─────────────────────────────────────────
    print("\n  [模擬] 客戶 C — 信貸評估中")
    steps = [
        (-10, "credit_score",        0.94, "what is my credit score"),
        (-7,  "interest_rate",       0.91, "what are current interest rates"),
        (-4,  "credit_limit_change", 0.87, "can I increase my credit limit"),
        (-2,  "improve_credit_score",0.85, "how to improve my credit score"),
    ]
    for day_offset, intent, conf, text in steps:
        ts = (now + timedelta(days=day_offset)).isoformat()
        logger.log("CUST_C", intent, conf, text, timestamp=ts)
        print(f"    Day {day_offset:+d}: {intent} ({text})")

    # ─────────────────────────────────────────
    # 客戶 D：卡片不滿意（留存風險）
    # 預期觸發：RET_001 卡片不滿意信號
    # ─────────────────────────────────────────
    print("\n  [模擬] 客戶 D — 卡片不滿意")
    steps = [
        (-8, "credit_limit",       0.92, "what is my credit limit"),
        (-5, "international_fees",  0.88, "how much are international fees"),
        (-3, "rewards_balance",     0.90, "check my rewards points"),
    ]
    for day_offset, intent, conf, text in steps:
        ts = (now + timedelta(days=day_offset)).isoformat()
        logger.log("CUST_D", intent, conf, text, timestamp=ts)
        print(f"    Day {day_offset:+d}: {intent} ({text})")

    # ─────────────────────────────────────────
    # 客戶 E：正常用戶（不應觸發任何規則）
    # ─────────────────────────────────────────
    print("\n  [模擬] 客戶 E — 一般查詢（無隱藏意圖）")
    steps = [
        (-5, "weather",   0.95, "what is the weather today"),
        (-3, "greeting",  0.88, "hello"),
        (-1, "balance",   0.93, "check my balance"),
    ]
    for day_offset, intent, conf, text in steps:
        ts = (now + timedelta(days=day_offset)).isoformat()
        logger.log("CUST_E", intent, conf, text, timestamp=ts)
        print(f"    Day {day_offset:+d}: {intent} ({text})")


def verify_results(miner: IntentMiner):
    """驗證挖掘結果"""
    print("\n" + "=" * 60)
    print("  驗證挖掘結果")
    print("=" * 60)

    all_pass = True

    # --- 客戶 A：應有風控預警 ---
    print("\n  [驗證] 客戶 A — 風控預警")
    result_a = miner.analyze_customer("CUST_A", days=30)
    risk_ids = [a["rule_id"] for a in result_a["risk_alerts"]]

    if "RISK_001" in risk_ids:
        print("    ✅ RISK_001 旅途遺失風險 — 已觸發")
    else:
        print("    ❌ RISK_001 旅途遺失風險 — 未觸發")
        all_pass = False

    # 應該也觸發 RISK_002（transactions 沒有，但有 freeze_account）
    # 注意：RISK_002 需要 transactions + freeze_account，但客戶 A 沒有 transactions
    # 所以 RISK_002 不應觸發
    if "RISK_002" not in risk_ids:
        print("    ✅ RISK_002 帳戶異常 — 正確未觸發（缺少 transactions）")
    else:
        print("    ⚠️  RISK_002 帳戶異常 — 意外觸發")

    print(f"    📊 風控預警數: {len(result_a['risk_alerts'])}")
    print(f"    📊 意圖分佈: {result_a['intent_profile']}")
    for alert in result_a["risk_alerts"]:
        print(f"    🚨 [{alert['rule_id']}] {alert['rule_name']}")
        print(f"       動作: {alert['action']}")

    # --- 客戶 B：應有交叉銷售 ---
    print("\n  [驗證] 客戶 B — 交叉銷售")
    result_b = miner.analyze_customer("CUST_B", days=30)
    sell_ids = [o["rule_id"] for o in result_b["cross_sell_opportunities"]]

    if "SELL_001" in sell_ids:
        print("    ✅ SELL_001 外幣理財潛在客戶 — 已觸發")
        sell_001 = [o for o in result_b["cross_sell_opportunities"] if o["rule_id"] == "SELL_001"][0]
        if sell_001.get("products"):
            print(f"    📦 推薦產品 ({len(sell_001['products'])} 項):")
            for p in sell_001["products"]:
                print(f"       → {p['product']}: {p['description']} ({p['target_return']})")
        else:
            print("    ❌ 無推薦產品")
            all_pass = False
    else:
        print("    ❌ SELL_001 外幣理財 — 未觸發")
        all_pass = False

    print(f"    📊 意圖分佈: {result_b['intent_profile']}")

    # --- 客戶 C：信貸需求 ---
    print("\n  [驗證] 客戶 C — 信貸需求")
    result_c = miner.analyze_customer("CUST_C", days=30)
    sell_ids_c = [o["rule_id"] for o in result_c["cross_sell_opportunities"]]

    if "SELL_002" in sell_ids_c:
        print("    ✅ SELL_002 信貸需求客戶 — 已觸發")
        sell_002 = [o for o in result_c["cross_sell_opportunities"] if o["rule_id"] == "SELL_002"][0]
        print(f"       動作: {sell_002['action']}")
    else:
        print("    ❌ SELL_002 信貸需求 — 未觸發")
        all_pass = False

    # --- 客戶 D：留存風險 ---
    print("\n  [驗證] 客戶 D — 客戶留存")
    result_d = miner.analyze_customer("CUST_D", days=30)
    ret_ids = [s["rule_id"] for s in result_d["retention_signals"]]

    if "RET_001" in ret_ids:
        print("    ✅ RET_001 卡片不滿意信號 — 已觸發")
        ret_001 = [s for s in result_d["retention_signals"] if s["rule_id"] == "RET_001"][0]
        print(f"       動作: {ret_001['action']}")
    else:
        print("    ❌ RET_001 卡片不滿意 — 未觸發")
        all_pass = False

    # --- 客戶 E：不應觸發 ---
    print("\n  [驗證] 客戶 E — 正常用戶")
    result_e = miner.analyze_customer("CUST_E", days=30)
    total_insights = len(result_e["insights"])
    if total_insights == 0:
        print("    ✅ 無 insights — 正確（一般查詢不觸發規則）")
    else:
        print(f"    ❌ 意外觸發 {total_insights} 條 insights")
        all_pass = False

    # --- 批次分析 ---
    print("\n  [驗證] 批次分析")
    batch = miner.batch_analyze(days=30)
    batch_ids = [b["customer_id"] for b in batch]
    print(f"    共 {len(batch)} 位客戶有 insights")
    for b in batch:
        n_risk = len(b.get("risk_alerts", []))
        n_sell = len(b.get("cross_sell_opportunities", []))
        n_ret = len(b.get("retention_signals", []))
        print(f"    → {b['customer_id']}: 風控={n_risk}, 銷售={n_sell}, 留存={n_ret}")

    # CUST_E 不應出現在批次結果
    if "CUST_E" not in batch_ids:
        print("    ✅ CUST_E 未出現（正確）")
    else:
        print("    ❌ CUST_E 不應出現在批次結果中")
        all_pass = False

    # 風控客戶應排在最前面
    if len(batch) >= 2 and batch[0]["customer_id"] == "CUST_A":
        print("    ✅ 風控客戶（CUST_A）排在最前面（正確優先度排序）")
    elif len(batch) >= 2:
        print(f"    ⚠️  排序第一位是 {batch[0]['customer_id']}，預期 CUST_A")

    return all_pass


def main():
    print("=" * 60)
    print("  隱藏意圖挖掘 — 端對端測試")
    print("=" * 60)

    db_path = create_test_db()
    logger = IntentLogger(db_path)
    miner = IntentMiner(logger)

    # Step 1: 模擬客戶行為
    print("\n" + "=" * 60)
    print("  Step 1: 模擬客戶行為")
    print("=" * 60)
    simulate_customers(logger)

    # Step 2: 檢查紀錄統計
    print("\n" + "=" * 60)
    print("  Step 2: 紀錄統計")
    print("=" * 60)
    stats = logger.get_stats()
    print(f"  總紀錄數: {stats['total_logs']}")
    print(f"  客戶數: {stats['unique_customers']}")
    print(f"  Top 意圖:")
    for item in stats["top_intents"][:5]:
        print(f"    {item['intent']}: {item['count']} 次")

    # Step 3: 驗證挖掘結果
    all_pass = verify_results(miner)

    # 結論
    print("\n" + "=" * 60)
    if all_pass:
        print("  ✅ 所有驗證通過！隱藏意圖挖掘功能運作正常。")
    else:
        print("  ❌ 部分驗證失敗，請檢查規則設定。")
    print("=" * 60)

    # 清理
    os.remove(db_path)
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
