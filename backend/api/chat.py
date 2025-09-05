"""
チャットAPI:
- /api/chat/blocking  : 同期応答
- /api/chat/streaming : SSE でストリーミング応答
- /api/bot/respond    : 後半課題の blocking API
- /api/bot/stream     : 後半課題の streaming API
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json, time, os
from datetime import datetime
import asyncio


from services.rag_service import get_rag_service
from services.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

# ====== 追加: チャット履歴保存（JSONL） ======
# backend/api/chat.py → (.. / ..) → プロジェクトルート → frontend/logs/chat_log.json
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOG_DIR = os.path.join(_BASE_DIR, "frontend", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "chat_log.json")

def _append_chat_log(entry: dict) -> None:
    """チャットログを1行JSONで追記。失敗しても本処理は止めない。"""
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # ログ書き込み失敗は警告に留める
        logger.warning(f"failed to write chat log: {e}")

# ====== スキーマ ======
class ChatRequest(BaseModel):
    prompt: str
    use_hybrid: bool = True  # ハイブリッド検索を使用するかどうか

class BotRequest(BaseModel):
    prompt: str
    use_hybrid: bool = True  # ハイブリッド検索を使用するかどうか

class BotResponse(BaseModel):
    reply: str

# 設定定数
QUESTION_MAX_LENGTH = 500  # 質問文字数の上限

def validate_prompt_length(prompt: str) -> None:
    """質問文字数をチェックし、超過時は例外を発生"""
    if len(prompt) > QUESTION_MAX_LENGTH:
        raise HTTPException(
            status_code=400, 
            detail=f"質問が長すぎます。{QUESTION_MAX_LENGTH}文字以内で入力してください。（現在: {len(prompt)}文字）"
        )

@router.get("/health")
async def health():
    return {"status": "ok", "service": "kitakyushu-waste-chatbot"}

@router.get("/search/hybrid-stats")
async def get_hybrid_search_stats():
    """ハイブリッド検索の統計情報を取得"""
    try:
        rag = get_rag_service()
        stats = rag.get_hybrid_search_stats()
        return stats
    except Exception as e:
        logger.error(f"hybrid stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== Blocking ======
@router.post("/chat/blocking")
async def chat_blocking(req: ChatRequest):
    try:
        # 文字数制限チェック
        validate_prompt_length(req.prompt)
        
        rag = get_rag_service()
        res = rag.blocking_query(req.prompt, k=5, use_hybrid=req.use_hybrid)

        payload = {
            "response": res["response"],
            "latency": res["latency"],
            "timestamp": res["timestamp"],
            "context_found": res.get("documents", 0) > 0,
            "source_documents": res.get("documents", 0),
            "search_method": res.get("search_method", "unknown"),
            "hybrid_enabled": res.get("hybrid_enabled", False),
            "mode": "blocking",
        }

        # 追加: ログ保存
        _append_chat_log({
            "timestamp": res["timestamp"],
            "mode": "blocking",
            "prompt": req.prompt,
            "response": res["response"],
            "latency": res["latency"],
            "context_found": payload["context_found"],
            "source_documents": payload["source_documents"],
            "search_method": payload["search_method"],
            "hybrid_enabled": payload["hybrid_enabled"],
            "use_hybrid_request": req.use_hybrid,
        })

        return payload
    except Exception as e:
        logger.error(f"blocking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== Streaming (SSE) ======
@router.post("/chat/streaming")
async def chat_streaming(req: ChatRequest):
    try:
        # 文字数制限チェック
        validate_prompt_length(req.prompt)
        
        rag = get_rag_service()
        start = time.time()

        async def gen():
            full = ""
            async for chunk in rag.streaming_query(req.prompt, k=5, use_hybrid=req.use_hybrid):
                full += chunk
                yield f"data: {json.dumps({'type':'chunk','content':chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            done = {
                "type": "complete",
                "response": full,
                "latency": time.time() - start,
                "timestamp": datetime.now().isoformat(),
                "hybrid_enabled": rag.hybrid_retriever is not None,
                "search_method": "hybrid" if req.use_hybrid and rag.hybrid_retriever else "vector_only"
            }

            # 追加: ログ保存（完了時にまとめて1件分を保存）
            _append_chat_log({
                "timestamp": done["timestamp"],
                "mode": "streaming",
                "prompt": req.prompt,
                "response": full,
                "latency": done["latency"],
                # streaming は docs 数を取得しないので None/0 を入れておく
                "context_found": None,
                "source_documents": None,
            })

            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control":"no-cache",
                "Connection":"keep-alive",
                "Content-Type":"text/event-stream; charset=utf-8",
                "X-Accel-Buffering": "no", 
            },
        )
    except Exception as e:
        logger.error(f"streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== 後半課題の Blocking API 仕様 ======
@router.post("/bot/respond", response_model=BotResponse)
async def bot_respond(req: BotRequest):
    # 文字数制限チェック
    validate_prompt_length(req.prompt)
    
    rag = get_rag_service()
    res = rag.blocking_query(req.prompt, k=5)

    # 追加: ログ保存
    _append_chat_log({
        "timestamp": res["timestamp"],
        "mode": "bot_blocking",
        "prompt": req.prompt,
        "response": res["response"],
        "latency": res["latency"],
        "context_found": (res.get("documents", 0) > 0),
        "source_documents": res.get("documents", 0),
    })

    return BotResponse(reply=res["response"])

# ====== 後半課題の Streaming API 仕様 ======
@router.post("/bot/stream")
async def bot_stream(req: BotRequest):
    # 文字数制限チェック
    validate_prompt_length(req.prompt)
    
    rag = get_rag_service()
    start = time.time()

    async def gen():
        full = ""
        async for chunk in rag.streaming_query(req.prompt, k=5):
            full += chunk
            yield f"data: {json.dumps({'type':'chunk','content':chunk}, ensure_ascii=False)}\n\n"
        done = {
            "type":"complete",
            "reply": full,
            "latency": time.time() - start,
            "timestamp": datetime.now().isoformat()
        }

        # 追加: ログ保存
        _append_chat_log({
            "timestamp": done["timestamp"],
            "mode": "bot_streaming",
            "prompt": req.prompt,
            "response": full,
            "latency": done["latency"],
            "context_found": None,
            "source_documents": None,
        })

        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control":"no-cache",
            "Connection":"keep-alive",
            "Content-Type":"text/event-stream; charset=utf-8"
        },
    )

# ====== データソース管理API ======
@router.get("/data/sources")
async def get_data_sources():
    """データソース一覧を取得"""
    try:
        rag = get_rag_service()
        result = rag.get_data_sources()
        return result
    except Exception as e:
        logger.error(f"データソース取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"データソース取得エラー: {str(e)}")

@router.get("/data/samples")
async def get_sample_documents(source: Optional[str] = None, limit: int = 10):
    """サンプルドキュメントを取得"""
    try:
        if limit > 50:  # 上限設定
            limit = 50
        rag = get_rag_service()
        result = rag.get_sample_documents(source=source, limit=limit)
        return result
    except Exception as e:
        logger.error(f"サンプルドキュメント取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"サンプルドキュメント取得エラー: {str(e)}")

class DataSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

@router.post("/data/search")
async def search_documents(request: DataSearchRequest):
    """ドキュメント検索（RAG検索と同じエンジンを使用）"""
    try:
        rag = get_rag_service()
        docs = rag.similarity_search(request.query, k=request.limit)
        
        # ドキュメントを整理して返す
        formatted_docs = []
        for doc in docs:
            lines = doc.page_content.split('\n')
            parsed_doc = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    parsed_doc[key.strip()] = value.strip()
            
            formatted_docs.append({
                'source': doc.metadata.get('source', 'unknown'),
                'content': parsed_doc,
                'raw_content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            })
        
        return {
            'success': True,
            'query': request.query,
            'documents': formatted_docs,
            'count': len(formatted_docs)
        }
    except Exception as e:
        logger.error(f"ドキュメント検索エラー: {e}")
        raise HTTPException(status_code=500, detail=f"ドキュメント検索エラー: {str(e)}")
