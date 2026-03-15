"""
測試文本集 — 英文文本，對應 150 個意圖標籤
執行方式：
  1. 啟動伺服器: python -m uvicorn app.main:app --reload
  2. 執行測試:   python tests/test_texts.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

# ============================================================
# 第一部分：單一意圖測試（預期由 SVM 輕量層處理）
# 格式：(輸入文本, 預期意圖標籤)
# ============================================================
SINGLE_INTENT_TEXTS = [
    # Finance
    ("What is my account balance?", "balance"),
    ("I want to transfer money to another account", "transfer"),
    ("Show me my recent transactions", "transactions"),
    ("How much do I owe on my bill?", "bill_balance"),
    ("When is my bill due?", "bill_due"),
    ("I want to pay my bill", "pay_bill"),
    ("What is my credit score?", "credit_score"),
    ("I need to change my credit limit", "credit_limit_change"),
    ("What is my current interest rate?", "interest_rate"),
    ("What is the APR on my card?", "apr"),
    ("I want to report a fraudulent charge", "report_fraud"),
    ("I lost my credit card", "report_lost_card"),
    ("I need to freeze my account", "freeze_account"),
    ("My card was declined at the store", "card_declined"),
    ("I need a new card", "new_card"),
    ("How do I change my PIN?", "pin_change"),
    ("What is my routing number?", "routing"),
    ("How do I set up direct deposit?", "direct_deposit"),
    ("Show my spending history", "spending_history"),
    ("How do I redeem my rewards points?", "redeem_rewards"),
    ("What is my rewards balance?", "rewards_balance"),
    ("How can I improve my credit score?", "improve_credit_score"),
    ("Are there international fees on my card?", "international_fees"),
    ("When does my card expire?", "expiration_date"),

    # Travel
    ("I want to book a flight to Tokyo", "book_flight"),
    ("Book me a hotel in New York", "book_hotel"),
    ("What is the status of my flight?", "flight_status"),
    ("What can I bring as carry-on?", "carry_on"),
    ("I lost my luggage at the airport", "lost_luggage"),
    ("I need to rent a car", "car_rental"),
    ("Set a travel alert on my account", "travel_alert"),
    ("Can you suggest a travel destination?", "travel_suggestion"),
    ("Do I need a visa to visit Japan?", "international_visa"),

    # Shopping
    ("I want to place an order", "order"),
    ("What is the status of my order?", "order_status"),
    ("What is the exchange rate for USD to EUR?", "exchange_rate"),
    ("What is my application status?", "application_status"),

    # Food
    ("Give me a recipe for pasta", "recipe"),
    ("What should I eat for dinner?", "meal_suggestion"),
    ("Suggest a good restaurant nearby", "restaurant_suggestion"),
    ("I want to make a restaurant reservation", "restaurant_reservation"),
    ("How many calories in a banana?", "calories"),

    # Assistant
    ("Set an alarm for 7 AM", "alarm"),
    ("Set a timer for 10 minutes", "timer"),
    ("Remind me to call mom tomorrow", "reminder"),
    ("What is on my calendar today?", "calendar"),
    ("Add buy groceries to my todo list", "todo_list"),
    ("What is the weather like today?", "weather"),
    ("What time is it now?", "time"),
    ("What is the definition of photosynthesis?", "definition"),
    ("How do you spell necessary?", "spelling"),
    ("Translate hello to Spanish", "translate"),

    # Vehicle
    ("I need to schedule maintenance for my car", "schedule_maintenance"),
    ("How do I change my oil?", "oil_change_how"),
    ("When should I change my tire?", "tire_change"),
    ("What is the tire pressure for my car?", "tire_pressure"),
    ("Where is the nearest gas station?", "gas"),

    # Entertainment
    ("Play some music", "play_music"),
    ("Skip to the next song", "next_song"),
    ("What song is playing right now?", "what_song"),
    ("Turn up the volume", "change_volume"),

    # Insurance
    ("Tell me about insurance options", "insurance"),
    ("I want to change my insurance plan", "insurance_change"),

    # Chitchat
    ("Hello!", "greeting"),
    ("Thank you so much", "thank_you"),
    ("Are you a robot?", "are_you_a_bot"),
    ("Tell me a joke", "tell_joke"),
    ("Goodbye!", "goodbye"),
]

# ============================================================
# 第二部分：複合意圖測試（預期觸發 BERT/LLM 重量層）
# ============================================================
COMPOUND_INTENT_TEXTS = [
    "I want to book a flight and also check my account balance",
    "My card was declined and I think someone stole my identity, I need to report fraud",
    "What is the exchange rate for euros, and can you also transfer 500 dollars to my savings?",
    "I lost my luggage and my credit card is missing too",
    "Can you set a reminder for my meeting and also tell me the weather?",
]

# ============================================================
# 第三部分：模擬客戶歷史（測試隱藏意圖挖掘）
# 每位客戶的文本會依序送入 /predict/smart 建立意圖歷史
# ============================================================
CUSTOMER_SCENARIOS = {
    # 客戶A：行李遺失 + 信用卡掛失 → 應觸發 RISK_001（旅途遺失風險）
    "TEST_RISK_A": {
        "texts": [
            "I lost my luggage at the airport",        # → lost_luggage
            "Can I get travel insurance?",              # → insurance
            "I need to report my card as lost",         # → report_lost_card
            "What is the status of my flight?",         # → flight_status
        ],
        "expected_rule": "RISK_001",
        "expected_category": "risk_alerts",
    },

    # 客戶B：頻繁查匯率 + 轉帳 → 應觸發 SELL_001（外幣理財）
    # 注意：SELL_001 要求 min_occurrences=2，所以每個意圖要出現 2 次以上
    "TEST_SELL_B": {
        "texts": [
            "What is the exchange rate for USD to JPY?",    # → exchange_rate
            "I want to transfer money to my other account", # → transfer
            "Check the exchange rate for euros please",     # → exchange_rate
            "Transfer 1000 dollars to savings",             # → transfer
            "What is the current exchange rate?",           # → exchange_rate
        ],
        "expected_rule": "SELL_001",
        "expected_category": "cross_sell_opportunities",
    },

    # 客戶C：查信用分數 + 利率 → 應觸發 SELL_002（信貸需求）
    "TEST_SELL_C": {
        "texts": [
            "What is my credit score?",             # → credit_score
            "What is the current interest rate?",   # → interest_rate
            "How can I improve my credit score?",   # → improve_credit_score
            "Tell me about the APR on my card",     # → apr
        ],
        "expected_rule": "SELL_002",
        "expected_category": "cross_sell_opportunities",
    },

    # 客戶D：查信用額度 + 國際手續費 → 應觸發 RET_001（卡片不滿意）
    "TEST_RET_D": {
        "texts": [
            "What is my credit limit?",                 # → credit_limit
            "Are there international fees on my card?", # → international_fees
            "I want to know about international fees",  # → international_fees
            "Can I change my credit limit?",            # → credit_limit_change
        ],
        "expected_rule": "RET_001",
        "expected_category": "retention_signals",
    },

    # 客戶E：查交易 + 凍結帳戶 → 應觸發 RISK_002（帳戶異常）
    "TEST_RISK_E": {
        "texts": [
            "Show me my recent transactions",   # → transactions
            "I see charges I did not make",      # → transactions (or report_fraud)
            "I need to freeze my account now",   # → freeze_account
            "Check my transaction history",      # → transactions
        ],
        "expected_rule": "RISK_002",
        "expected_category": "risk_alerts",
    },

    # 客戶F：訂機票 + 旅遊建議 → 應觸發 SELL_003（旅遊金融服務）
    "TEST_SELL_F": {
        "texts": [
            "I want to book a flight to Paris",         # → book_flight
            "Can you suggest travel destinations?",     # → travel_suggestion
            "Book me a flight to London next month",    # → book_flight
        ],
        "expected_rule": "SELL_003",
        "expected_category": "cross_sell_opportunities",
    },

    # 客戶G：正常使用者 → 不應觸發任何規則
    "TEST_NORMAL_G": {
        "texts": [
            "What is my balance?",          # → balance
            "What is the weather today?",   # → weather
            "Set a timer for 5 minutes",    # → timer
        ],
        "expected_rule": None,
        "expected_category": None,
    },
}


def test_single_intents():
    """測試單一意圖（使用 /predict/smart）"""
    print("=" * 70)
    print("【測試一】單一意圖 — 雙層路由 (共 {} 題)".format(len(SINGLE_INTENT_TEXTS)))
    print("=" * 70)

    results = {"svm": 0, "bert": 0, "llm": 0, "fallback": 0, "error": 0}
    correct = 0

    for text, expected_intent in SINGLE_INTENT_TEXTS:
        try:
            resp = requests.post(f"{BASE_URL}/predict/smart", json={
                "text": text,
                "customer_id": None
            }, timeout=10)
            data = resp.json()
            # API 回傳欄位名稱是 "tier"
            tier = data.get("tier", "unknown")
            intent = data.get("intent", "unknown")
            conf = data.get("confidence", 0)
            latency = data.get("latency_ms", 0)

            results[tier] = results.get(tier, 0) + 1
            match = "✅" if intent == expected_intent else "❌"
            if intent == expected_intent:
                correct += 1

            tier_icon = "🟢" if tier == "svm" else "🟡" if tier == "bert" else "🔵" if tier == "llm" else "🔴"
            print(f"  {match} {tier_icon}[{tier:8s}] \"{text[:50]:50s}\"")
            print(f"         預期: {expected_intent:25s} | 實際: {intent:25s} (conf={conf:.3f}, {latency:.0f}ms)")
        except Exception as e:
            results["error"] += 1
            print(f"  ❌ \"{text[:50]}\" → 錯誤: {e}")

    total_valid = sum(v for k, v in results.items() if k != "error")
    print(f"\n  {'='*50}")
    print(f"  📊 路由統計: SVM={results['svm']}, BERT={results['bert']}, "
          f"LLM={results['llm']}, FALLBACK={results['fallback']}")
    print(f"  🎯 意圖準確率: {correct}/{len(SINGLE_INTENT_TEXTS)} "
          f"({correct/len(SINGLE_INTENT_TEXTS)*100:.1f}%)")
    if total_valid > 0:
        print(f"  💰 SVM 處理比例: {results['svm']}/{total_valid} "
              f"({results['svm']/total_valid*100:.1f}%)")


def test_compound_intents():
    """測試複合意圖"""
    print("\n" + "=" * 70)
    print("【測試二】複合意圖 — 預期觸發重量層 (共 {} 題)".format(len(COMPOUND_INTENT_TEXTS)))
    print("=" * 70)

    for text in COMPOUND_INTENT_TEXTS:
        try:
            resp = requests.post(f"{BASE_URL}/predict/smart", json={
                "text": text,
                "customer_id": None
            }, timeout=30)
            data = resp.json()
            tier = data.get("tier", "unknown")
            intent = data.get("intent", "unknown")
            conf = data.get("confidence", 0)
            latency = data.get("latency_ms", 0)

            icon = "🟡" if tier in ("bert", "llm") else "🟢"
            print(f"  {icon} [{tier:8s}] \"{text[:60]}\"")
            print(f"         → intent: {intent} (conf={conf:.3f}, {latency:.0f}ms)")

            # 如果有子意圖
            sub = data.get("sub_intents", [])
            if sub:
                for s in sub:
                    if isinstance(s, dict):
                        print(f"         └─ sub: {s.get('intent','?')} [{s.get('tier','?')}] conf={s.get('confidence',0):.3f}")
                    else:
                        print(f"         └─ sub: {s}")
        except Exception as e:
            print(f"  ❌ \"{text[:60]}\" → 錯誤: {e}")


def test_hidden_intent_mining():
    """測試隱藏意圖挖掘"""
    print("\n" + "=" * 70)
    print("【測試三】隱藏意圖挖掘 — 模擬 {} 位客戶歷程".format(len(CUSTOMER_SCENARIOS)))
    print("=" * 70)

    # 步驟1：送入每位客戶的意圖歷史
    for cust_id, scenario in CUSTOMER_SCENARIOS.items():
        texts = scenario["texts"]
        expected = scenario["expected_rule"]
        print(f"\n  📝 {cust_id} — 送入 {len(texts)} 筆查詢 (預期觸發: {expected or '無'})")

        for text in texts:
            try:
                resp = requests.post(f"{BASE_URL}/predict/smart", json={
                    "text": text,
                    "customer_id": cust_id
                }, timeout=10)
                data = resp.json()
                intent = data.get("intent", "?")
                tier = data.get("tier", "?")
                conf = data.get("confidence", 0)
                print(f"     → \"{text[:45]:45s}\" | {intent:25s} [{tier}] conf={conf:.3f}")
            except Exception as e:
                print(f"     ❌ \"{text[:45]}\": {e}")

        time.sleep(0.2)

    # 步驟2：逐一挖掘分析
    print("\n  " + "-" * 60)
    print("  🔍 開始挖掘隱藏意圖...\n")

    pass_count = 0
    fail_count = 0

    for cust_id, scenario in CUSTOMER_SCENARIOS.items():
        expected_rule = scenario["expected_rule"]
        expected_cat = scenario["expected_category"]

        try:
            resp = requests.get(f"{BASE_URL}/intent-mining/{cust_id}?days=30", timeout=10)
            data = resp.json()

            # API 回傳結構：頂層有 risk_alerts, cross_sell_opportunities, retention_signals
            risk = data.get("risk_alerts", [])
            sell = data.get("cross_sell_opportunities", [])
            ret = data.get("retention_signals", [])
            all_triggered = risk + sell + ret
            all_rule_ids = [r.get("rule_id") if isinstance(r, dict) else str(r) for r in all_triggered]

            total = len(all_triggered)

            # 檢查是否觸發預期規則
            if expected_rule is None:
                passed = total == 0
            else:
                passed = expected_rule in all_rule_ids

            status = "✅ PASS" if passed else "❌ FAIL"
            if passed:
                pass_count += 1
            else:
                fail_count += 1

            if total == 0:
                print(f"  {status} {cust_id}: 無觸發 (預期: {expected_rule or '無'})")
            else:
                print(f"  {status} {cust_id}: 觸發 {all_rule_ids} (預期: {expected_rule})")
                for r in risk:
                    print(f"         🔴 風險: {r.get('rule_id')} — {r.get('action', '')[:60]}")
                for s in sell:
                    print(f"         🟡 銷售: {s.get('rule_id')} — {s.get('action', '')[:60]}")
                    products = s.get("products", [])
                    if products:
                        names = [p.get("product", str(p)) if isinstance(p, dict) else str(p) for p in products]
                        print(f"            📦 推薦: {', '.join(names)}")
                for t in ret:
                    print(f"         🔵 留存: {t.get('rule_id')} — {t.get('action', '')[:60]}")
        except Exception as e:
            fail_count += 1
            print(f"  ❌ FAIL {cust_id}: {e}")

    print(f"\n  {'='*50}")
    print(f"  🎯 挖掘驗證: {pass_count}/{pass_count+fail_count} 通過")

    # 步驟3：批次掃描
    print("\n  " + "-" * 60)
    print("  📊 批次掃描所有客戶...\n")
    try:
        resp = requests.get(f"{BASE_URL}/intent-mining?days=30", timeout=10)
        data = resp.json()
        # 批次結果：API 回傳 {"customers": [...], "total_customers_with_insights": N}
        if isinstance(data, list):
            results_list = data
        else:
            results_list = data.get("customers", data.get("results", []))

        print(f"  共 {len(results_list)} 位客戶有觸發規則:")
        for r in results_list:
            if not isinstance(r, dict):
                continue
            cid = r.get("customer_id", "?")
            risk_n = len(r.get("risk_alerts", []))
            sell_n = len(r.get("cross_sell_opportunities", []))
            ret_n = len(r.get("retention_signals", []))
            print(f"     {cid:15s}: 風險={risk_n}, 銷售={sell_n}, 留存={ret_n}")
    except Exception as e:
        print(f"  ❌ 批次掃描失敗: {e}")


def test_routing_stats():
    """查看路由統計"""
    print("\n" + "=" * 70)
    print("【統計】路由使用統計")
    print("=" * 70)
    try:
        resp = requests.get(f"{BASE_URL}/routing-stats", timeout=5)
        stats = resp.json()
        print(f"  總請求數:    {stats.get('total_requests', 0)}")
        print(f"  SVM 處理:    {stats.get('svm_handled', 0)}")
        print(f"  BERT 處理:   {stats.get('bert_handled', 0)}")
        print(f"  LLM 處理:    {stats.get('llm_handled', 0)}")
        print(f"  累計成本:    ${stats.get('total_cost', 0):.4f}")
        print(f"  平均延遲:    {stats.get('avg_latency_ms', 0):.1f}ms")

        total = stats.get('total_requests', 1)
        svm = stats.get('svm_handled', 0)
        if total > 0:
            print(f"  SVM 佔比:    {svm/total*100:.1f}%")
    except Exception as e:
        print(f"  ❌ {e}")


def main():
    print("🚀 富邦意圖辨識系統 — 整合測試")
    print("=" * 70)

    # 先檢查伺服器
    print("  檢查伺服器狀態...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        health = resp.json()
        print(f"  ✅ 狀態: {health.get('status', '?')}")
        print(f"     版本: {health.get('version', '?')}")
        features = health.get('features', [])
        print(f"     功能: {', '.join(features)}")
        if 'bert_heavy_tier' in features:
            print("     🟢 BERT 重量層已啟用")
        else:
            print("     🟡 BERT 未載入，僅使用 SVM 輕量層")
        print()
    except Exception as e:
        print(f"  ❌ 伺服器未啟動！")
        print(f"     請先執行: python -m uvicorn app.main:app --reload")
        print(f"     錯誤: {e}")
        return

    test_single_intents()
    test_compound_intents()
    test_hidden_intent_mining()
    test_routing_stats()

    print("\n" + "=" * 70)
    print("✅ 全部測試完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
