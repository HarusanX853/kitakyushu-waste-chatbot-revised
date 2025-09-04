"""
北九州市ごみ分別チャットボット RAG サービス
- Embeddings: Ollama (bge-m3:latest)
- Vector DB: Chroma (./chroma_db に永続化)
- LLM: Ollama (Swallow v0.5 gguf)
"""

import os
import re
import json
import shutil
import time
import unicodedata
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator
import asyncio

# ChromaDBテレメトリを無効化
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import pandas as pd
import ollama
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from backend.services.logger import setup_logger

# ===== 環境変数 / 既定値 =====
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")
LLM_MODEL   = os.getenv("LLM_MODEL", "hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:latest")
CHROMA_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")
DATA_DIR    = os.getenv("DATA_DIR", "./data")

# 召回強度（環境変数で可調整）
DEFAULT_K   = int(os.getenv("RETRIEVER_K", "10"))
K_MAX       = int(os.getenv("RETRIEVER_K_MAX", "12"))
K_MIN       = int(os.getenv("RETRIEVER_K_MIN", "5"))

# ===== 軽量クレンジング =====
_ZERO_WIDTH_TRANS = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF], None)

# ===== 表記揺れ・同義語辞書 =====
SYNONYMS_MAP = {
    # アルミ関連
    "アルミ缶": ["アルミかん", "アルミカン", "あるみかん", "あるみ缶"],
    "アルミかん": ["アルミ缶", "アルミカン", "あるみかん", "あるみ缶"],
    
    # カタカナ・ひらがな揺れ
    "ペットボトル": ["ペット", "ぺっと", "ペットボトル"],
    "プラスチック": ["プラ", "ぷら"],
    
    # 略語・省略形
    "テレビ": ["TV", "tv", "ティーブイ", "てれび"],
    "エアコン": ["エアーコンディショナー", "クーラー", "えあこん"],
    
    # カン・びん関連
    "缶": ["かん", "カン"],
    "瓶": ["びん", "ビン", "ガラス瓶"],
    
    # 家電製品
    "冷蔵庫": ["れいぞうこ", "冷凍庫"],
    "洗濯機": ["せんたくき", "洗濯機"],
    
    # 一般的な表記揺れ
    "携帯電話": ["携帯", "スマホ", "スマートフォン", "けいたい"],
    "乾電池": ["電池", "でんち", "バッテリー"],
}

def expand_query_with_synonyms(query: str) -> List[str]:
    """
    クエリを同義語で拡張
    """
    queries = [query]
    query_lower = query.lower()
    
    # 完全一致での同義語展開
    for key, synonyms in SYNONYMS_MAP.items():
        if key in query:
            for synonym in synonyms:
                new_query = query.replace(key, synonym)
                if new_query not in queries:
                    queries.append(new_query)
        
        # 同義語からキーへの展開
        for synonym in synonyms:
            if synonym in query:
                new_query = query.replace(synonym, key)
                if new_query not in queries:
                    queries.append(new_query)
    
    return queries

def normalize_query(query: str) -> str:
    """
    クエリの正規化（カタカナ・ひらがな統一等）
    """
    # 全角→半角
    query = unicodedata.normalize("NFKC", query)
    
    # カタカナをひらがなに変換（より寛容な検索のため）
    normalized = ""
    for char in query:
        if 'ァ' <= char <= 'ヶ':
            # カタカナをひらがなに
            normalized += chr(ord(char) - ord('ァ') + ord('ぁ'))
        else:
            normalized += char
    
    return normalized

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_ZERO_WIDTH_TRANS)
    return s.strip()

def strip_quotes_and_brackets(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^[「『“"（(]+', '', s)
    s = re.sub(r'[」』”"）)]+$', '', s)
    return s.strip()

def extract_item_like(query: str) -> str:
    q = clean_text(query)
    m = re.search(r'「(.+?)」', q)
    if m:
        return strip_quotes_and_brackets(m.group(1))
    tmp = re.split(r'[はをにでがともへ]|[?？。!！、]', q, maxsplit=1)[0]
    tmp = strip_quotes_and_brackets(tmp)
    return tmp[:32].strip() if tmp else q

# ===== Embeddings ラッパ（BGE向けに query/passsage 前置詞を付ける）=====
class BGEInstructEmbeddings:
    """
    LangChain の埋め込みIF互換:
    - embed_documents(texts: List[str]) -> List[List[float]]
    - embed_query(text: str) -> List[float]
    """
    def __init__(self, model: str):
        self.inner = OllamaEmbeddings(model=model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts2 = [f"passage: {clean_text(t)}" for t in texts]
        return self.inner.embed_documents(texts2)

    def embed_query(self, text: str) -> List[float]:
        return self.inner.embed_query(f"query: {clean_text(text)}")

# ===== Index manifest （埋め込みの一貫性チェック）=====
MANIFEST_FILE = "manifest.json"
MANIFEST_STRATEGY = "bge_query_passage_prefix_v1"

def _manifest_path() -> str:
    return os.path.join(CHROMA_DIR, MANIFEST_FILE)

def _load_manifest() -> Dict[str, Any] | None:
    try:
        with open(_manifest_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_manifest() -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(
            {
                "embed_model": EMBED_MODEL,
                "strategy": MANIFEST_STRATEGY,
                "created_at": datetime.now().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

def _manifest_mismatch() -> bool:
    m = _load_manifest()
    if not m:
        return True
    return not (
        m.get("embed_model") == EMBED_MODEL
        and m.get("strategy") == MANIFEST_STRATEGY
    )

# ===== 本体 =====
class KitakyushuWasteRAGService:
    """RAGの初期化・CSV取り込み・検索・応答生成"""

    def __init__(self):
        self.logger = setup_logger(__name__)

        self.embeddings = BGEInstructEmbeddings(model=EMBED_MODEL)

        if os.path.isdir(CHROMA_DIR) and _manifest_mismatch():
            self.logger.warning("Embedding 設定が既存インデックスと不一致のため、再構築します。")
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )

        os.makedirs(DATA_DIR, exist_ok=True)
        self._load_csv_dir()
        _write_manifest()

        self.logger.info(
            f"RAG ready | EMBED_MODEL={EMBED_MODEL} | LLM_MODEL={LLM_MODEL} | "
            f"CHROMA_DIR={CHROMA_DIR} | DATA_DIR={DATA_DIR}"
        )

    # ========= CSV 読み込み =========
    def _row_to_text(self, row: pd.Series) -> str:
        item  = row.get("品名") or row.get("品目") or row.get("item") or ""
        how   = row.get("出し方") or row.get("処理方法") or row.get("how") or ""
        note  = row.get("備考") or row.get("注意") or row.get("note") or ""
        area  = row.get("エリア") or row.get("地区") or row.get("area") or ""
        return "\n".join([
            f"品目: {str(item).strip()}",
            f"出し方: {str(how).strip()}",
            f"備考: {str(note).strip()}",
            f"エリア: {str(area).strip()}",
        ]).strip()

    def add_csv(self, filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            return {"success": False, "error": f"CSVが見つかりません: {filepath}"}
            
        # 重複チェック: 既存データソースを確認
        filename = os.path.basename(filepath)
        existing_sources = self.get_data_sources()
        
        if existing_sources.get("success"):
            for source in existing_sources.get("sources", []):
                if source.get("file_name") == filename:
                    self.logger.warning(f"CSV重複スキップ: {filename} は既に登録済み")
                    return {"success": True, "count": 0, "message": f"ファイル '{filename}' は既に登録済みです"}
        
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="cp932")

        docs: List[Document] = []
        for _, row in df.iterrows():
            text = self._row_to_text(row)
            if text:
                docs.append(Document(page_content=text, metadata={"source": filename}))
        if docs:
            self.vectorstore.add_documents(docs)
            self.logger.info(f"CSV取り込み: {filepath} | 文書数={len(docs)}")
            return {"success": True, "count": len(docs)}
        return {"success": True, "count": 0}

    def _load_csv_dir(self) -> None:
        total = 0
        for fn in os.listdir(DATA_DIR):
            if fn.lower().endswith(".csv"):
                res = self.add_csv(os.path.join(DATA_DIR, fn))
                total += int(res.get("count", 0))
        self.logger.info(f"初期CSV取り込み完了: {total} 文書")

    # ========= 検索（強化版） =========
    def _format_docs(self, docs: List[Document], limit_each: int = 320, max_docs: int = 8) -> str:
        if not docs:
            return "関連情報が見つかりませんでした。"
        chunks = []
        for i, d in enumerate(docs[:max_docs], start=1):
            txt = (d.page_content or "").strip()
            if len(txt) > limit_each:
                txt = txt[:limit_each] + "…"
            chunks.append(f"[候補{i}]\n{txt}")
        return "\n\n".join(chunks)

    def similarity_search(self, query: str, k: int = DEFAULT_K) -> List[Document]:
        k = max(K_MIN, min(k or DEFAULT_K, K_MAX))
        q_clean = clean_text(query)
        
        # 同義語拡張クエリを生成
        expanded_queries = expand_query_with_synonyms(q_clean)
        self.logger.info(f"Expanded queries: {expanded_queries}")
        
        item_q = extract_item_like(q_clean)

        def item_match_score(txt: str, item: str) -> int:
            if not txt:
                return 0
            m = re.search(r"品目:\s*(.+)", txt)
            name = (m.group(1) if m else "").strip()
            if not name:
                return 0
            n1 = clean_text(name)
            n2 = clean_text(item)
            
            # 同義語も考慮したマッチング
            if n1 == n2:
                return 3
            
            # 同義語チェック
            for key, synonyms in SYNONYMS_MAP.items():
                if (n1 == key and n2 in synonyms) or (n2 == key and n1 in synonyms):
                    return 2
                if n1 in synonyms and n2 in synonyms:
                    return 2
            
            return 1 if (n2 and (n2 in n1 or n1 in n2)) else 0

        def rerank_by_item(docs, item):
            return sorted(docs, key=lambda d: item_match_score((d.page_content or ""), item), reverse=True)

        def poor(dlist: List[Document], item_hint: str) -> bool:
            if not dlist:
                return True
            heads = sum(1 for d in dlist if "品目:" in (d.page_content or ""))
            has_item = any(item_match_score((d.page_content or ""), item_hint) > 0 for d in dlist)
            return heads < max(1, len(dlist)//3) or not has_item

        def merge_dedup(lists):
            seen = set()
            out = []
            for lst in lists:
                for d in lst or []:
                    key = ((d.page_content or "").strip(), json.dumps(getattr(d, "metadata", {}), ensure_ascii=False, sort_keys=True))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(d)
            return out

        all_docs = []
        
        # 各拡張クエリで検索を実行
        for expanded_query in expanded_queries:
            try:
                docs_main = self.vectorstore.similarity_search(expanded_query, k=k)
                all_docs.extend(docs_main)
                self.logger.info(f"Found {len(docs_main)} docs for query: {expanded_query}")
            except Exception as e:
                self.logger.warning(f"similarity_search failed for '{expanded_query}': {e}")

        # 重複を除去
        all_docs = merge_dedup([all_docs])
        all_docs = rerank_by_item(all_docs, item_q)
        
        # 十分な結果が得られた場合はここで終了
        if not poor(all_docs, item_q) and len(all_docs) >= k//2:
            return all_docs[:k]

        # 追加検索（アイテム名での検索）
        if item_q:
            # アイテム名も同義語拡張
            expanded_items = expand_query_with_synonyms(item_q)
            for expanded_item in expanded_items:
                try:
                    docs_item = self.vectorstore.similarity_search(expanded_item, k=k)
                    all_docs.extend(docs_item)
                except Exception as e:
                    self.logger.warning(f"similarity_search failed for item '{expanded_item}': {e}")

                # 最終的な重複除去とランキング
        final_docs = merge_dedup([all_docs])
        final_docs = rerank_by_item(final_docs, item_q)
        
        return final_docs[:k]

    def format_documents(self, docs: List[Document], limit_each: int = 150) -> str:

        # 3) 品目抽出クエリで再検索（通常→MMR）
        try:
            docs2 = self.vectorstore.similarity_search(item_q, k=k)
        except Exception as e:
            self.logger.warning(f"similarity_search failed (item): {e}")
            docs2 = []
        docs2 = rerank_by_item(docs2, item_q)
        if not poor(docs2, item_q):
            return docs2[:k]

        try:
            docs2_mmr = self.vectorstore.max_marginal_relevance_search(item_q, k=k, fetch_k=max(k*2, 8))
        except Exception as e:
            self.logger.warning(f"MMR search failed (item): {e}")
            docs2_mmr = []
        docs2_mmr = rerank_by_item(docs2_mmr, item_q)
        return docs2_mmr[:k]

    # ========= LLM 呼び出し =========
    def _call_llm(self, prompt: str) -> str:
        res = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_ctx": 4096},
        )
        return (res.get("message", {}) or {}).get("content", "")

    # ========= ユーザーAPI =========
    def blocking_query(self, query: str, k: int = DEFAULT_K) -> Dict[str, Any]:
        t0 = time.time()
        docs = self.similarity_search(query, k=k)
        ctx = self._format_docs(docs)

        prompt = (
            "あなたは北九州市のごみ分別案内の専門AIアシスタントです。"
            "北九州市民の皆様に正確で分かりやすいごみ分別情報を提供することが使命です。"
            "以下の参照データの範囲内で、日本語で簡潔かつ正確に回答してください。"
            "\n回答形式:"
            "「出し方: [具体的な分別方法]」を最初に明記し、必要に応じて「備考: [追加情報]」を補足してください。"
            "\n重要なルール:"
            "1. 質問された品目に関連する情報のみを回答してください（無関係な品目情報は含めない）"
            "2. データベースに該当情報がない場合は「申し訳ございませんが、該当する情報がありません。北九州市のホームページ（https://www.city.kitakyushu.lg.jp/）でご確認いただくか、お住まいの区役所環境課にお問い合わせください。」と回答してください"
            "3. 回答は簡潔で分かりやすく、必ず「出し方」を含めてください"
            "4. 推測や一般的なアドバイスは厳禁です。データベースに基づいた正確な情報のみを提供してください"
            "5. 複数の関連品目がある場合は、質問に最も適合するもののみを選んで回答してください"
            "6. エリア情報がある場合は、該当する地区名も併記してください"
            "\n\n質問:\n"
            f"{clean_text(query)}\n\n参照データ:\n{ctx}\n\n回答:"
        )
        try:
            answer = self._call_llm(prompt).strip()
            return {
                "response": answer,
                "documents": len(docs),
                "latency": time.time() - t0,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"response": f"エラー: {e}", "documents": len(docs), "latency": time.time() - t0}

    async def streaming_query(self, query: str, k: int = DEFAULT_K) -> AsyncGenerator[str, None]:
        docs = self.similarity_search(query, k=k)
        ctx = self._format_docs(docs)

        prompt = (
            "あなたは北九州市のごみ分別案内の専門AIアシスタントです。"
            "北九州市民の皆様に正確で分かりやすいごみ分別情報を提供することが使命です。"
            "以下の参照データの範囲内で、日本語で簡潔かつ正確に回答してください。"
            "\n回答形式:"
            "「出し方: [具体的な分別方法]」を最初に明記し、必要に応じて「備考: [追加情報]」を補足してください。"
            "\n重要なルール:"
            "1. 質問された品目に関連する情報のみを回答してください（無関係な品目情報は含めない）"
            "2. データベースに該当情報がない場合は「申し訳ございませんが、該当する情報がありません。北九州市のホームページ（https://www.city.kitakyushu.lg.jp/）でご確認いただくか、お住まいの区役所環境課にお問い合わせください。」と回答してください"
            "3. 回答は簡潔で分かりやすく、必ず「出し方」を含めてください"
            "4. 推測や一般的なアドバイスは厳禁です。データベースに基づいた正確な情報のみを提供してください"
            "5. 複数の関連品目がある場合は、質問に最も適合するもののみを選んで回答してください"
            "6. エリア情報がある場合は、該当する地区名も併記してください"
            f"\n\n質問:\n{clean_text(query)}\n\n参照データ:\n{ctx}\n\n回答:"
        )

        try:
        # 注意：ollama.chat 是同步生成器；在 async 函数内迭代它会阻塞事件循环，
        # 但每次 yield 都会把已生成内容刷给客户端（SSE/StreamingResponse 可正常工作）。
        # 如果你希望完全不阻塞事件循环，可再升级为线程生产者 + asyncio.Queue 桥接（我也可以给你那版）。
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": 0.1, "num_ctx": 4096},
            )
            for chunk in stream:
                # 正确处理流结束信号
                if chunk.get("done"):
                    break
                msg = chunk.get("message") or {}
                content = msg.get("content", "")
                if content:
                    # 只把非空文本片段发给前端
                    yield content
        except Exception as e:
            yield f"エラー: {e}"

    def get_data_sources(self) -> Dict[str, Any]:
        """データソースの一覧と統計情報を取得"""
        try:
            # ChromaDBから全ドキュメントのメタデータを取得
            collection = self.vectorstore._collection
            all_data = collection.get(include=['metadatas', 'documents'])
            
            # ソース別の統計を作成
            source_stats = {}
            total_documents = len(all_data['documents'])
            
            for i, metadata in enumerate(all_data['metadatas']):
                source = metadata.get('source', 'unknown')
                if source not in source_stats:
                    source_stats[source] = {
                        'file_name': source,
                        'document_count': 0,
                        'sample_content': [],
                        'last_updated': metadata.get('last_updated', 'N/A')
                    }
                
                source_stats[source]['document_count'] += 1
                
                # サンプルコンテンツを保存（最初の3つまで）
                if len(source_stats[source]['sample_content']) < 3:
                    doc_content = all_data['documents'][i]
                    if doc_content:
                        # 品目名を抽出
                        lines = doc_content.split('\n')
                        item_line = next((line for line in lines if line.startswith('品目:')), '')
                        if item_line:
                            item_name = item_line.replace('品目:', '').strip()
                            source_stats[source]['sample_content'].append(item_name)
            
            return {
                'success': True,
                'total_documents': total_documents,
                'total_sources': len(source_stats),
                'sources': list(source_stats.values()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"データソース取得エラー: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_documents': 0,
                'total_sources': 0,
                'sources': []
            }

    def get_sample_documents(self, source: str = None, limit: int = 10) -> Dict[str, Any]:
        """指定されたソースからサンプルドキュメントを取得"""
        try:
            collection = self.vectorstore._collection
            
            if source:
                # 特定のソースからドキュメントを取得
                all_data = collection.get(
                    where={"source": source},
                    include=['metadatas', 'documents'],
                    limit=limit
                )
            else:
                # 全ソースからサンプルを取得
                all_data = collection.get(
                    include=['metadatas', 'documents'],
                    limit=limit
                )
            
            documents = []
            for i, doc_content in enumerate(all_data['documents']):
                metadata = all_data['metadatas'][i]
                
                # ドキュメントから情報を抽出
                lines = doc_content.split('\n')
                parsed_doc = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        parsed_doc[key.strip()] = value.strip()
                
                documents.append({
                    'source': metadata.get('source', 'unknown'),
                    'content': parsed_doc,
                    'raw_content': doc_content[:200] + '...' if len(doc_content) > 200 else doc_content
                })
            
            return {
                'success': True,
                'documents': documents,
                'count': len(documents),
                'source_filter': source
            }
            
        except Exception as e:
            self.logger.error(f"サンプルドキュメント取得エラー: {e}")
            return {
                'success': False,
                'error': str(e),
                'documents': [],
                'count': 0
            }

# ======= シングルトン =======
_rag = None
def get_rag_service() -> KitakyushuWasteRAGService:
    global _rag
    if _rag is None:
        _rag = KitakyushuWasteRAGService()
    return _rag
