"""
共享設定模組 (Shared Configuration)

統一管理 intent → domain 的映射關係，
避免 evaluator.py 與 intent_mining.py 各自定義造成不一致。

Usage:
    from src.config import INTENT_TO_DOMAIN, INTENT_DOMAINS
"""

# ============================================================
# 意圖領域映射（Domain → Intent List）
# 9 個領域，150 個意圖
# ============================================================
INTENT_DOMAINS = {
    'Finance': [
        'balance', 'transfer', 'transactions', 'bill_balance', 'bill_due',
        'pay_bill', 'credit_score', 'credit_limit', 'credit_limit_change',
        'interest_rate', 'apr', 'min_payment', 'report_fraud',
        'report_lost_card', 'freeze_account', 'account_blocked',
        'card_declined', 'damaged_card', 'new_card',
        'replacement_card_duration', 'pin_change', 'routing',
        'direct_deposit', 'spending_history', 'income', 'taxes', 'w2',
        'insurance', 'insurance_change', 'rewards_balance',
        'redeem_rewards', 'rollover_401k', 'improve_credit_score',
        'international_fees', 'expiration_date', 'order_checks', 'payday',
    ],
    'Travel': [
        'book_flight', 'book_hotel', 'flight_status', 'carry_on',
        'lost_luggage', 'car_rental', 'travel_alert', 'travel_notification',
        'travel_suggestion', 'international_visa', 'vaccines', 'uber',
        'directions', 'distance', 'traffic',
    ],
    'Food': [
        'recipe', 'meal_suggestion', 'restaurant_suggestion',
        'restaurant_reservation', 'restaurant_reviews',
        'accept_reservations', 'confirm_reservation',
        'cancel_reservation', 'how_busy', 'calories', 'nutrition_info',
        'cook_time', 'food_last', 'ingredient_substitution',
        'ingredients_list',
    ],
    'Assistant': [
        'alarm', 'timer', 'reminder', 'reminder_update', 'calendar',
        'calendar_update', 'todo_list', 'todo_list_update', 'shopping_list',
        'shopping_list_update', 'weather', 'time', 'date', 'timezone',
        'next_holiday', 'meeting_schedule', 'schedule_meeting',
        'calculator', 'measurement_conversion', 'definition', 'spelling',
        'translate', 'find_phone', 'make_call', 'text', 'share_location',
        'current_location',
    ],
    'Vehicle': [
        'schedule_maintenance', 'oil_change_how', 'oil_change_when',
        'last_maintenance', 'tire_change', 'tire_pressure', 'gas',
        'gas_type', 'mpg', 'jump_start',
    ],
    'Entertainment': [
        'play_music', 'next_song', 'what_song', 'update_playlist',
        'smart_home', 'change_volume', 'change_speed', 'change_accent',
        'change_language', 'whisper_mode', 'reset_settings', 'sync_device',
    ],
    'Chitchat': [
        'greeting', 'goodbye', 'thank_you', 'yes', 'no', 'maybe',
        'cancel', 'repeat', 'are_you_a_bot', 'what_is_your_name',
        'who_made_you', 'who_do_you_work_for', 'where_are_you_from',
        'how_old_are_you', 'what_are_your_hobbies', 'do_you_have_pets',
        'meaning_of_life', 'what_can_i_ask_you', 'fun_fact', 'tell_joke',
        'flip_coin', 'roll_dice', 'change_ai_name', 'change_user_name',
        'user_name',
    ],
    'Work_HR': [
        'pto_balance', 'pto_request', 'pto_request_status', 'pto_used',
    ],
    'Shopping': [
        'order', 'order_status', 'plug_type', 'exchange_rate',
        'application_status',
    ],
}

# 反向映射：intent → domain
INTENT_TO_DOMAIN = {}
for _domain, _intents in INTENT_DOMAINS.items():
    for _intent in _intents:
        INTENT_TO_DOMAIN[_intent] = _domain
