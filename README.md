# 北九州市ごみ分別チャットボット

本プロジェクトは **FastAPI + LangChain + Ollama + Streamlit** を活用した「ごみ分別チャットボット」です。  
バックエンドでは **RAG（検索拡張生成）** を用い、CSV ファイルを知識ベースとして登録できます。  
フロントエンドは **Streamlit** により、Web ブラウザから簡単に操作できます。  

---

## ✨ 主な機能

- 📂 **CSV アップロード機能**  
  ごみ分別に関する CSV データをアップロードし、知識ベースとして利用可能。  

- 🤖 **大規模言語モデル（LLM）**  
  サーバーにインストールされた **Ollama** のモデル（例：`llama3.1-swallow`、`bge-m3`）を利用。  

- 🔍 **RAG による回答生成**  
  質問に対して、知識ベースを参照した自然な日本語で回答。  

- 📊 **データ管理機能**  
  検索対象データの一覧表示、サンプルデータ閲覧、データ検索機能を提供。

- 🖥 **レスポンシブWeb UI**  
  Streamlit による多機能チャット画面。モバイル対応、QRコード生成、GPU監視機能付き。  

- 🔍 **同義語検索対応**  
  「アルミ缶」「アルミかん」など表記揺れに対応した高精度検索。

- 📱 **モバイル対応**  
  QRコード生成によるスマートフォンでのアクセス対応。

- ⚡ **GPU 監視機能**  
  リアルタイムGPU使用状況監視とパフォーマンス最適化。

---

## 📦 セットアップ手順

### 1. リポジトリをクローン
```bash
git clone <your-repo-url>
cd kitakyushu-waste-chatbot-revised
```

### 2. Python 環境の準備
```bash
conda create -n chatbot python=3.11 -y
conda activate chatbot
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 3. Ollama の準備
サーバーに Ollama をインストールし、以下のモデルを用意してください：

- **llama3.1-swallow** （対話用モデル）
- **bge-m3:latest** （埋め込みモデル）

確認：
```bash
ollama list
```

### 4. バックエンド起動
```bash
# プロジェクトルートディレクトリから実行
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

API が起動したら次の URL で確認できます：
```
http://<サーバーIP>:8000
```

### 5. フロントエンド起動
```bash
cd frontend
streamlit run app.py --server.port 8002 --server.address 0.0.0.0
```

Web UI にアクセス：
```
http://<サーバーIP>:8001
```

**注意**: ポート8001が使用中の場合は、別のポート（8002など）を使用してください：
```bash
streamlit run app.py --server.port 8002 --server.address 0.0.0.0
```

---

## 📤 CSV アップロード例
```bash
curl -F "file=@/path/to/your/data.csv" \
     http://127.0.0.1:8000/api/upload
```

## 🩺 API チェック

### ヘルスチェック
```bash
curl http://127.0.0.1:8000/health
```

### GPU モニタリング
```bash
curl http://127.0.0.1:8000/api/monitor/gpu
```

### データソース一覧
```bash
curl http://127.0.0.1:8000/api/data/sources
```

### データ検索
```bash
curl -X POST "http://127.0.0.1:8000/api/data/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "アルミ缶", "limit": 5}'
```

## 🚀 利用イメージ

1. **データ登録**: CSV をアップロードして知識ベースを登録
2. **チャット**: フロントエンドで質問（例：「アルミ缶はどう捨てればいいですか？」）
3. **AI回答**: LLM が CSV 知識ベースを参照して回答
4. **データ管理**: WebUIでデータ一覧や検索機能を利用

## 🔧 新機能詳細

### データ管理機能
- **📊 データ管理セクション**: WebUI下部に追加
- **🗃️ 検索対象データ一覧**: 登録されたCSVファイルの概要表示
- **🔍 データ詳細表示**: サンプルデータの詳細閲覧
- **🔍 データ検索**: RAGエンジンと同じ検索機能

### モバイル対応
- **📱 QRコード生成**: モバイルアクセス用QRコード自動生成
- **📐 レスポンシブデザイン**: スマートフォン最適化UI

### パフォーマンス機能
- **⚡ GPU監視**: リアルタイムVRAM使用量とGPU使用率表示
- **🕐 生成時間表示**: 回答生成時間のリアルタイム表示
- **💾 チャット履歴管理**: 自動履歴制限とメモリ最適化



