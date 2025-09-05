"""
北九州市ごみ分別チャットボット RAG サービス
- Embeddings: BGE-M3 (bge-m3:latest)
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
from langchain_huggingface import HuggingFaceEmbeddings

from .logger import setup_logger, get_logger

# BM25ハイブリッド検索用インポート
from rank_bm25 import BM25Okapi
import jieba
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Union

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

# ===== Embeddings ラッパ（複数モデル対応）=====
class FlexibleEmbeddings:
    """
    LangChain の埋め込みIF互換:
    - embed_documents(texts: List[str]) -> List[List[float]]
    - embed_query(text: str) -> List[float]
    """
    def __init__(self, model: str):
        self.model = model
        self.logger = get_logger(__name__)
        
        # Ruriモデルを使用する場合はHuggingFaceEmbeddingsを試行
        if "ruri" in model.lower():
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.inner = HuggingFaceEmbeddings(
                    model_name=model,
                    model_kwargs={'device': 'cuda'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.is_ruri = True
                self.logger.info(f"Ruriモデル {model} を正常に読み込みました")
            except Exception as e:
                self.logger.warning(f"Ruriモデル読み込み失敗、Ollamaにフォールバック: {e}")
                # フォールバック: nomic-embed-textを使用
                fallback_model = "nomic-embed-text"
                self.inner = OllamaEmbeddings(model=fallback_model)
                self.is_ruri = False
                self.logger.info(f"フォールバックモデル {fallback_model} を使用します")
        else:
            # 従来のOllamaモデルの場合
            self.inner = OllamaEmbeddings(model=model)
            self.is_ruri = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.is_ruri:
            # Ruriモデルは前置詞不要、直接クリーンテキストを使用
            texts2 = [clean_text(t) for t in texts]
        else:
            # BGE系モデルには前置詞を付ける（nomic-embed-textは不要）
            if "bge" in self.model.lower():
                texts2 = [f"passage: {clean_text(t)}" for t in texts]
            else:
                texts2 = [clean_text(t) for t in texts]
        return self.inner.embed_documents(texts2)

    def embed_query(self, text: str) -> List[float]:
        if self.is_ruri:
            # Ruriモデルは前置詞不要
            return self.inner.embed_query(clean_text(text))
        else:
            # BGE系モデルには前置詞を付ける（nomic-embed-textは不要）
            if "bge" in self.model.lower():
                return self.inner.embed_query(f"query: {clean_text(text)}")
            else:
                return self.inner.embed_query(clean_text(text))

# ===== ハイブリッド検索クラス =====
class HybridRetriever:
    """
    BM25（キーワードベース）とBGE-M3（セマンティック）を組み合わせたハイブリッド検索
    """
    
    def __init__(self, chroma_retriever, documents: List[Document], alpha: float = 0.5):
        """
        Args:
            chroma_retriever: ChromaDBのベクトル検索インスタンス
            documents: 全てのドキュメント
            alpha: ハイブリッド重み (0.0=BM25のみ, 1.0=ベクトル検索のみ)
        """
        self.chroma_retriever = chroma_retriever
        self.documents = documents
        self.alpha = alpha
        self.logger = get_logger(__name__)  # ロガーを追加
        
        # BM25用の日本語トークナイザー設定
        self.tokenizer = self._setup_tokenizer()
        
        # BM25インデックス構築
        self.bm25 = self._build_bm25_index()
        
        # ドキュメントIDマッピング
        self.doc_id_to_index = {doc.metadata.get('id', i): i for i, doc in enumerate(documents)}
        
        self.logger = setup_logger(__name__)
    
    def _setup_tokenizer(self):
        """日本語用トークナイザーの設定"""
        def japanese_tokenizer(text: str) -> List[str]:
            # jiebaによる日本語分割
            tokens = list(jieba.cut(text, cut_all=False))
            
            # 追加の前処理
            processed_tokens = []
            for token in tokens:
                token = token.strip()
                if len(token) >= 1 and not re.match(r'^[、。！？\s]+$', token):
                    processed_tokens.append(token.lower())
            
            return processed_tokens
        
        return japanese_tokenizer
    
    def _build_bm25_index(self) -> BM25Okapi:
        """BM25インデックスの構築"""
        self.logger.info("BM25インデックスを構築中...")
        
        # 全ドキュメントをトークン化
        tokenized_docs = []
        for doc in self.documents:
            # ページコンテンツと メタデータを結合
            content = doc.page_content
            if doc.metadata:
                # 重要なメタデータ（アイテム名、分別方法など）を内容に追加
                item_name = doc.metadata.get('item_name', '')
                category = doc.metadata.get('category', '')
                notes = doc.metadata.get('notes', '')
                
                combined_content = f"{content} {item_name} {category} {notes}"
            else:
                combined_content = content
                
            tokens = self.tokenizer(combined_content)
            tokenized_docs.append(tokens)
        
        bm25 = BM25Okapi(tokenized_docs)
        self.logger.info(f"BM25インデックス構築完了: {len(tokenized_docs)} ドキュメント")
        
        return bm25
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """スコアの正規化 (0-1)"""
        if not scores:
            return scores
            
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Document]:
        """
        ハイブリッド検索の実行
        
        Args:
            query: 検索クエリ
            k: 返すドキュメント数
            
        Returns:
            スコア順にソートされたドキュメントリスト
        """
        self.logger.info(f"ハイブリッド検索実行: '{query}' (k={k})")
        
        # 1. BM25検索
        bm25_scores = self._bm25_search(query, k * 2)  # より多く取得して多様性確保
        
        # 2. ベクトル検索 (BGE-M3)
        vector_results = self._vector_search(query, k * 2)
        
        # 3. スコアの正規化と結合
        hybrid_scores = self._combine_scores(bm25_scores, vector_results, k)
        
        # 4. 上位k件を返す
        top_docs = [doc for doc, score in hybrid_scores[:k]]
        
        self.logger.info(f"ハイブリッド検索完了: {len(top_docs)} 件取得")
        return top_docs
    
    def _bm25_search(self, query: str, k: int) -> Dict[int, float]:
        """BM25検索"""
        # クエリの同義語拡張
        expanded_queries = expand_query_with_synonyms(query)
        
        # 各拡張クエリでスコア計算
        combined_scores = defaultdict(float)
        
        for expanded_query in expanded_queries:
            tokenized_query = self.tokenizer(expanded_query)
            if not tokenized_query:
                continue
                
            scores = self.bm25.get_scores(tokenized_query)
            
            # 重み付き加算（元クエリに最大重み）
            weight = 1.0 if expanded_query == query else 0.7
            
            for doc_idx, score in enumerate(scores):
                combined_scores[doc_idx] += score * weight
        
        # 上位k件のインデックスとスコア
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_scores[:k])
    
    def _vector_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """ベクトル検索 (BGE-M3)"""
        try:
            # ChromaDBのsimilarity_search_with_scoreを使用
            results = self.chroma_retriever.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            self.logger.error(f"ベクトル検索エラー: {e}")
            return []
    
    def _combine_scores(self, bm25_scores: Dict[int, float], vector_results: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        """BM25とベクトルスコアの結合"""
        
        # BM25スコアの正規化
        if bm25_scores:
            bm25_values = list(bm25_scores.values())
            normalized_bm25 = self._normalize_scores(bm25_values)
            bm25_normalized = dict(zip(bm25_scores.keys(), normalized_bm25))
        else:
            bm25_normalized = {}
        
        # ベクトル検索スコアの正規化（距離スコアを類似度スコアに変換）
        vector_scores = {}
        if vector_results:
            # ChromaDBは距離を返すので、類似度に変換 (1 / (1 + distance))
            distances = [score for doc, score in vector_results]
            similarities = [1 / (1 + dist) if dist >= 0 else 1.0 for dist in distances]
            normalized_vector = self._normalize_scores(similarities)
            
            for (doc, _), norm_score in zip(vector_results, normalized_vector):
                # ドキュメントIDまたはインデックスを取得
                doc_id = doc.metadata.get('id')
                if doc_id is not None:
                    doc_idx = self.doc_id_to_index.get(doc_id)
                else:
                    # IDがない場合、内容でマッチング（非効率だが仕方なし）
                    doc_idx = None
                    for i, candidate_doc in enumerate(self.documents):
                        if candidate_doc.page_content == doc.page_content:
                            doc_idx = i
                            break
                
                if doc_idx is not None:
                    vector_scores[doc_idx] = norm_score
        
        # ハイブリッドスコア計算
        hybrid_scores = {}
        all_doc_indices = set(bm25_normalized.keys()) | set(vector_scores.keys())
        
        for doc_idx in all_doc_indices:
            bm25_score = bm25_normalized.get(doc_idx, 0.0)
            vector_score = vector_scores.get(doc_idx, 0.0)
            
            # 重み付き結合
            hybrid_score = (1 - self.alpha) * bm25_score + self.alpha * vector_score
            hybrid_scores[doc_idx] = hybrid_score
        
        # ドキュメントとスコアのペアを作成
        doc_score_pairs = []
        for doc_idx, score in hybrid_scores.items():
            if 0 <= doc_idx < len(self.documents):
                doc_score_pairs.append((self.documents[doc_idx], score))
        
        # スコア順にソート
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:k]
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """検索統計情報"""
        return {
            "total_documents": len(self.documents),
            "bm25_ready": self.bm25 is not None,
            "vector_ready": self.chroma_retriever is not None,
            "alpha": self.alpha,
            "tokenizer": "jieba"
        }

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

        self.embeddings = FlexibleEmbeddings(model=EMBED_MODEL)

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

        # ハイブリッド検索の初期化
        self.hybrid_retriever = None
        self._initialize_hybrid_search()

        self.logger.info(
            f"RAG ready | EMBED_MODEL={EMBED_MODEL} | LLM_MODEL={LLM_MODEL} | "
            f"CHROMA_DIR={CHROMA_DIR} | DATA_DIR={DATA_DIR} | Hybrid Search: {'Enabled' if self.hybrid_retriever else 'Disabled'}"
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

    def _initialize_hybrid_search(self):
        """ハイブリッド検索の初期化"""
        try:
            # 全ドキュメントを取得
            all_docs = []
            collection = self.vectorstore._collection
            if collection is not None:
                # ChromaDBから全ドキュメントを取得
                result = collection.get()
                if result and 'documents' in result:
                    for i, content in enumerate(result['documents']):
                        metadata = result.get('metadatas', [{}])[i] if i < len(result.get('metadatas', [])) else {}
                        doc = Document(page_content=content, metadata=metadata)
                        all_docs.append(doc)
            
            if all_docs:
                # ChromaDBのretrieverを作成
                chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": DEFAULT_K * 2})
                
                # ハイブリッドリトリーバーを初期化
                self.hybrid_retriever = HybridRetriever(
                    chroma_retriever=chroma_retriever,
                    documents=all_docs,
                    alpha=0.6  # ベクトル検索を60%、BM25を40%の重みで結合
                )
                self.logger.info(f"ハイブリッド検索初期化完了: {len(all_docs)} ドキュメント")
            else:
                self.logger.warning("ドキュメントが見つからないため、ハイブリッド検索を無効化")
                self.hybrid_retriever = None
                
        except Exception as e:
            self.logger.error(f"ハイブリッド検索初期化エラー: {e}")
            self.hybrid_retriever = None

    def hybrid_similarity_search(self, query: str, k: int = DEFAULT_K, use_hybrid: bool = True) -> List[Document]:
        """
        ハイブリッド検索（BM25 + BGE-M3）またはベクトル検索のみを実行
        
        Args:
            query: 検索クエリ
            k: 返すドキュメント数
            use_hybrid: Trueならハイブリッド検索、Falseならベクトル検索のみ
        """
        k = max(K_MIN, min(k or DEFAULT_K, K_MAX))
        
        self.logger.info(f"検索リクエスト: query='{query}', k={k}, use_hybrid={use_hybrid}, hybrid_retriever={self.hybrid_retriever is not None}")
        
        if use_hybrid and self.hybrid_retriever:
            self.logger.info(f"🔥 ハイブリッド検索実行: '{query}' (k={k})")
            try:
                results = self.hybrid_retriever.hybrid_search(query, k)
                self.logger.info(f"✅ ハイブリッド検索成功: {len(results)} 件取得")
                return results
            except Exception as e:
                self.logger.error(f"❌ ハイブリッド検索エラー、ベクトル検索にフォールバック: {e}")
                return self.similarity_search(query, k)
        else:
            self.logger.info(f"📊 ベクトル検索実行: '{query}' (k={k})")
            return self.similarity_search(query, k)

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
    def blocking_query(self, query: str, k: int = DEFAULT_K, use_hybrid: bool = True) -> Dict[str, Any]:
        """
        ブロッキング検索クエリ
        
        Args:
            query: 検索クエリ
            k: 検索するドキュメント数
            use_hybrid: ハイブリッド検索を使用するかどうか
        """
        t0 = time.time()
        
        # ハイブリッド検索または従来のベクトル検索を選択
        if use_hybrid and self.hybrid_retriever:
            docs = self.hybrid_similarity_search(query, k=k, use_hybrid=True)
            search_method = "hybrid"
        else:
            docs = self.similarity_search(query, k=k)  
            search_method = "vector_only"
            
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
                "search_method": search_method,
                "hybrid_enabled": self.hybrid_retriever is not None,
                "latency": time.time() - t0,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "response": f"エラー: {e}", 
                "documents": len(docs), 
                "search_method": search_method,
                "latency": time.time() - t0
            }

    async def streaming_query(self, query: str, k: int = DEFAULT_K, use_hybrid: bool = True) -> AsyncGenerator[str, None]:
        """
        ストリーミング検索クエリ
        
        Args:
            query: 検索クエリ
            k: 検索するドキュメント数  
            use_hybrid: ハイブリッド検索を使用するかどうか
        """
        # ハイブリッド検索または従来のベクトル検索を選択
        if use_hybrid and self.hybrid_retriever:
            docs = self.hybrid_similarity_search(query, k=k, use_hybrid=True)
        else:
            docs = self.similarity_search(query, k=k)
            
        ctx = self._format_docs(docs)

        prompt = (
            "あなたは北九州市の優秀なごみ分別案内の専門AIアシスタントです。"
            "北九州市民の皆様に正確で分かりやすいごみ分別情報を提供することが使命です。"
            "以下の参照データの範囲内で、日本語で簡潔かつ正確に回答してください。"
            "\n回答形式:"
            "「出し方: [具体的な分別方法]」を最初に明記し、必要に応じて「備考: [追加情報]」を補足してください。"
            "\n重要なルール:"
            "1. 質問された品目に関連する情報を回答してください（無関係な品目情報は含めない）"
            "2. データベースに該当情報がない場合は「申し訳ございませんが、該当する情報がありません。北九州市のホームページ（https://www.city.kitakyushu.lg.jp/）でご確認いただくか、お住まいの区役所環境課にお問い合わせください。」と回答してください"
            "3. 回答は簡潔で分かりやすく、必ず「出し方」を含めてください"
            "4. 推測や一般的なアドバイスは厳禁です。データベースに基づいた正確な情報のみを提供してください"
            "5. 複数の関連品目がある場合は、質問に最も適合するものを選んで回答してください、必要に応じて類似の品目も補足してください"
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

    def get_hybrid_search_stats(self) -> Dict[str, Any]:
        """ハイブリッド検索の統計情報を取得"""
        if not self.hybrid_retriever:
            return {
                "enabled": False,
                "reason": "ハイブリッド検索が初期化されていません"
            }
        
        hybrid_stats = self.hybrid_retriever.get_retrieval_stats()
        return {
            "enabled": True,
            "stats": hybrid_stats,
            "alpha": hybrid_stats.get("alpha", 0.5),
            "description": {
                "alpha": "ハイブリッド重み (0.0=BM25のみ, 1.0=ベクトル検索のみ)",
                "bm25": "キーワードベース検索 (TF-IDF風の語彙マッチング)",
                "vector": "セマンティック検索 (BGE-M3埋め込み)",
                "tokenizer": "日本語分割器 (jieba)"
            }
        }

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
