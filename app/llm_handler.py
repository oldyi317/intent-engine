"""
多意圖拆解模組 (Multi-Intent Decomposition via LLM / Rules)

負責：
  - 偵測複合句（含多個意圖的查詢）
  - 規則式拆分子句
  - 對接 LLM（可選）做更精準的拆分
  - 對每個子句獨立分類

Usage:
    from app.llm_handler import MultiIntentHandler

    handler = MultiIntentHandler(classifier_fn=my_classify_fn)
    result = handler.analyze("check my balance and also book a flight to tokyo")
    # result = {
    #   "is_compound": True,
    #   "sub_queries": [
    #     {"text": "check my balance", "intent": "balance", "confidence": 0.95},
    #     {"text": "book a flight to tokyo", "intent": "book_flight", "confidence": 0.92},
    #   ]
    # }
"""

import re
from typing import List, Dict, Callable, Optional, Tuple


class MultiIntentHandler:
    """
    多意圖拆解處理器

    策略：
      Phase 1 — 偵測是否為複合句（關鍵連接詞 + 句長）
      Phase 2 — 用正則規則拆分子句
      Phase 3 — 對每個子句呼叫分類器
    """

    # 常見動作動詞（用於偵測 "and + 動詞" 的複合句模式）
    _ACTION_VERBS = (
        r'(?:check|book|set|find|show|tell|get|make|cancel|pay|transfer|send|call|'
        r'play|change|update|create|add|remove|delete|report|freeze|order|search|'
        r'remind|schedule|turn|look|help|give|see|list|open|close|reset|start|stop|'
        r'calculate|convert|translate|share|track|review|compare|save|apply|request|'
        r'i\s+(?:want|need|would\s+like|also)\b)'
    )

    # 連接詞 / 分隔模式（依優先順序排列，越上面越優先）
    SPLIT_PATTERNS = [
        r'\band\s+(?:also|then|please|can you|could you|i(?:\s+also)?\s+(?:want|need|would like))\b',
        r'\b(?:also|additionally|plus|furthermore|moreover)\b',
        r'\bas\s+well\s+as\b',
        r'\b(?:then|after\s+that|next|afterwards)\b',
        r'\b(?:oh\s+)?and\s+(?:by\s+the\s+way|one\s+more\s+thing)\b',
        r'[;]',
        r'\.\s+(?:Also|And|Then|Plus|Can|Could|I\s+also|Please)\b',
        # "and" + 動詞：最後才嘗試，避免誤拆 "bread and butter" 這類名詞片語
        # 用 lookahead 讓動詞保留在第二段
        r'\band\s+(?=' + _ACTION_VERBS + r')',
    ]

    COMPOUND_INDICATORS = [
        r'\band\b.*\band\b',
        r'\balso\b',
        r'\bplus\b',
        r'\bas\s+well\b',
        r'\bthen\b',
        r'\bafter\s+that\b',
        r'\bone\s+more\s+thing\b',
        r'\bby\s+the\s+way\b',
        r'\badditionally\b',
        r'[;]',
        # "and" + 動詞 → 高機率是多意圖
        r'\band\s+' + _ACTION_VERBS,
    ]

    def __init__(self, classifier_fn: Optional[Callable] = None):
        """
        Args:
            classifier_fn: 分類函式，輸入 text (str) → 輸出 (intent, confidence)
                           如果不提供，只做拆分不做分類
        """
        self.classifier_fn = classifier_fn

    def is_compound(self, text: str) -> bool:
        """偵測是否為複合句"""
        text_lower = text.lower()
        for pattern in self.COMPOUND_INDICATORS:
            if re.search(pattern, text_lower):
                return True
        if len(text.split()) > 20:
            return True
        return False

    def split_compound(self, text: str) -> List[str]:
        """將複合句拆分為子句"""
        if not self.is_compound(text):
            return [text]

        segments = [text]
        for pattern in self.SPLIT_PATTERNS:
            new_segments = []
            for seg in segments:
                parts = re.split(pattern, seg, flags=re.IGNORECASE)
                parts = [p.strip() for p in parts if p and p.strip()]
                new_segments.extend(parts)
            segments = new_segments

        # 過濾太短的片段
        segments = [s for s in segments if len(s.split()) >= 2]
        return segments if len(segments) > 1 else [text]

    def analyze(self, text: str) -> Dict:
        """
        完整多意圖分析

        Returns:
            {
                "original_text": str,
                "is_compound": bool,
                "sub_queries": [
                    {"text": str, "intent": str, "confidence": float},
                    ...
                ]
            }
        """
        compound = self.is_compound(text)
        segments = self.split_compound(text) if compound else [text]

        sub_queries = []
        for seg in segments:
            if self.classifier_fn:
                intent, conf = self.classifier_fn(seg)
                sub_queries.append({
                    'text': seg,
                    'intent': intent,
                    'confidence': round(float(conf), 4),
                })
            else:
                sub_queries.append({
                    'text': seg,
                    'intent': None,
                    'confidence': 0.0,
                })

        return {
            'original_text': text,
            'is_compound': compound,
            'sub_queries': sub_queries,
        }


# ============================================================
# Demo
# ============================================================
if __name__ == '__main__':
    handler = MultiIntentHandler()

    examples = [
        "what's the weather like today",
        "check my balance and also book a flight to tokyo",
        "report my lost card and also freeze my account",
        "book a hotel in tokyo and then find me a good restaurant; also check if i need a visa",
        "i want to check my credit score and also see my reward points balance",
    ]

    print("=" * 60)
    print("  Multi-Intent Decomposition Demo")
    print("=" * 60)

    for text in examples:
        result = handler.analyze(text)
        print(f"\n  Input: \"{text}\"")
        print(f"  Compound: {result['is_compound']}")
        for i, sq in enumerate(result['sub_queries'], 1):
            print(f"    {i}. \"{sq['text']}\"")
