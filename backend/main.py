"""
北九州市ごみ分別チャットボット バックエンドサーバー
要件定義書対応: FastAPI + RAG + GPU監視
"""

import os
# ChromaDBテレメトリを無効化
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime

# API ルーター
from backend.api.chat import router as chat_router
from backend.api.upload import router as upload_router
from backend.api.monitor import router as monitor_router

# サービス
from backend.services.logger import setup_logger

# ログ設定
logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # スタートアップ処理
    logger.info("北九州市ごみ分別チャットボット API 起動中...")
    
    # 必要なディレクトリ作成
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_db", exist_ok=True)
    
    logger.info("API サーバー起動完了")
    
    yield  # アプリケーション実行
    
    # シャットダウン処理
    logger.info("API サーバーを終了します")

# アプリケーション初期化
app = FastAPI(
    title="北九州市ごみ分別チャットボット API",
    description="LangChain + Ollama Llama 3.1 Swallow による RAG チャットボット",
    version="1.0.0",
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーター登録
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(monitor_router, prefix="/api", tags=["monitor"])

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "北九州市ごみ分別チャットボット API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    try:
        # RAGサービスの状態確認
        from backend.services.rag_service import get_rag_service
        rag_service = get_rag_service()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "rag": "operational",
                "database": "connected"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """グローバル例外ハンドラ"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    logger.info("サーバーを起動します...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
