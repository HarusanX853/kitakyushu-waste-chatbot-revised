import logging, os

def setup_logger(name: str):
    os.makedirs("./logs", exist_ok=True)
    log_file = "./logs/app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler()]
    )
    return logging.getLogger(name)

def get_logger(name: str):
    """ロガーを取得する関数"""
    return logging.getLogger(name)
