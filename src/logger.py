"""
共用 Logging 模組

功能：
  - 同時輸出到 console（即時觀看）和 log 檔案（永久記錄）
  - 每個模組各自一個 log 檔，存放在 logs/ 目錄
  - 支援 timestamp、模組名稱、log level

Log 檔案位置：
  logs/
  ├── data_preprocess.log
  ├── model_trainer.log
  ├── train_bert.log
  └── evaluator.log

Usage:
    from src.logger import setup_logger
    logger = setup_logger("data_preprocess")
    logger.info("開始前處理...")
"""

import os
import sys
import logging
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: str = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    建立 Logger，同時輸出到 console 和檔案

    Args:
        name: 模組名稱（也是 log 檔名）
        log_dir: log 存放目錄，預設為專案根目錄下的 logs/
        level: logging level

    Returns:
        配置好的 Logger
    """
    # 決定 log 目錄
    if log_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(project_root, 'logs')

    os.makedirs(log_dir, exist_ok=True)

    # 檔名加上日期，方便多次執行不覆蓋
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重複加 handler（模組被 import 多次時）
    if logger.handlers:
        return logger

    # 格式
    fmt = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler（即時顯示）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler（永久記錄）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Log 檔案: {log_file}")
    return logger
