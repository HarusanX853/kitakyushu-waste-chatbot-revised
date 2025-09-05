#!/bin/bash

# 北九州市ごみ分別チャットボット 起動スクリプト v2

echo "🚀 北九州市ごみ分別チャットボット システム起動"
echo "=============================================="

# プロジェクトルートディレクトリに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "� プロジェクトディレクトリ: $(pwd)"

# 必要なディレクトリ作成
mkdir -p logs data chroma_db

# Ollamaサービス確認
echo ""
echo "� Ollama モデル確認..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama コマンドが利用可能です"
    
    # Ollamaサーバー起動確認
    if ! pgrep -f "ollama serve" > /dev/null; then
        echo "📡 Ollamaサーバーを起動しています..."
        ollama serve > logs/ollama.log 2>&1 &
        sleep 5
    fi
    
    # 必要なモデルの確認
    if ollama list | grep -q "hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf"; then
        echo "✅ LLMモデル (Llama 3.1 Swallow) が利用可能です"
    else
        echo "❌ LLMモデルが見つかりません。インストールしています..."
        ollama pull hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf
    fi
    
    if ollama list | grep -q "nomic-embed-text"; then
        echo "✅ Embeddingモデル (nomic-embed-text) が利用可能です"
    else
        echo "❌ Embeddingモデルが見つかりません。インストールしています..."
        ollama pull nomic-embed-text
    fi
else
    echo "❌ Ollama がインストールされていません"
    echo "   https://ollama.ai/ からインストールしてください"
    exit 1
fi

# ==================================================
# 2.5. Ollamaサーバー起動確認
# ==================================================
echo "🔄 Ollamaサーバー起動確認中..."
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "⚠️  Ollamaサーバーが起動していません。起動中..."
    ollama serve &
    
    # 起動待機
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            echo "✅ Ollamaサーバーが起動しました"
            break
        fi
        echo "⏳ Ollamaサーバー起動待機中... ($i/10)"
        sleep 2
    done
else
    echo "✅ Ollamaサーバーは既に起動中"
fi

# ==================================================
# 3. Swallowモデル確認
# ==================================================
echo "🤖 Llama 3.1 Swallowモデル確認中..."
MODEL_NAME="hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf"

if ! ollama list | grep -q "Llama-3.1-Swallow"; then
    echo "⬇️ Swallowモデルをダウンロードします..."
    ollama pull "$MODEL_NAME"
    
    if [ $? -eq 0 ]; then
        echo "✅ Swallowモデルダウンロード完了"
    else
        echo "❌ Swallowモデルダウンロード失敗"
        exit 1
    fi
else
    echo "✅ Swallowモデルは既に利用可能"
fi

# ==================================================
# 4. GPU確認
# ==================================================
echo "🔍 GPU環境確認中..."
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU情報:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️ nvidia-smiが見つかりません（Mac環境の可能性）"
fi

# ==================================================
# 5. Python依存関係確認
# ==================================================
echo "📦 Python依存関係確認中..."
if [ -f "backend/requirements.txt" ]; then
    pip install -r backend/requirements.txt
    echo "✅ バックエンド依存関係インストール完了"
fi

if [ -f "frontend/requirements.txt" ]; then
    pip install -r frontend/requirements.txt
    echo "✅ フロントエンド依存関係インストール完了"
fi

# ==================================================
# 6. Docker確認・起動
# ==================================================
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "🐳 Dockerコンテナ起動中..."
    docker-compose down --remove-orphans
    docker-compose up -d --build
    
    # 起動確認
    sleep 10
    for service in backend frontend; do
        if docker-compose ps | grep -q "${service}.*Up"; then
            echo "✅ ${service}: 起動中"
        else
            echo "❌ ${service}: 起動失敗"
        fi
    done
else
    echo "📱 ローカルモードで起動..."
    
    # バックエンド起動
    cd backend
    python3 main.py &
    BACKEND_PID=$!
    cd ..
    
    # フロントエンド起動
    cd frontend
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
    FRONTEND_PID=$!
    cd ..
    
    echo "✅ バックエンド起動 (PID: $BACKEND_PID)"
    echo "✅ フロントエンド起動 (PID: $FRONTEND_PID)"
fi

# ==================================================
# 7. GPU監視開始
# ==================================================
echo "📊 GPU監視開始..."
if [ -f "monitor_console.py" ]; then
    python3 monitor_console.py &
    MONITOR_PID=$!
    echo "✅ GPU監視プロセス起動 (PID: $MONITOR_PID)"
else
    echo "⚠️ monitor_console.py が見つかりません"
fi

# ==================================================
# 8. 起動完了
# ==================================================
echo ""
echo "🎉 ===== 起動完了 ===== 🎉"
echo ""
echo "🌐 アクセス情報:"
echo "   フロントエンド: http://localhost:8501"
echo "   バックエンドAPI: http://localhost:8000"
echo "   API仕様書: http://localhost:8000/docs"
echo ""
echo "📋 要件定義書対応機能:"
echo "   ✅ Llama 3.1 Swallow"
echo "   ✅ ChromaDB (RAG)"
echo "   ✅ Streaming/Blocking モード"
echo "   ✅ JSONログ出力"
echo "   ✅ VRAM監視"
echo "   ✅ レスポンシブデザイン"
echo ""
echo "⛔ 停止: docker-compose down (または Ctrl+C)"
echo "🔄 再起動: ./start.sh"
echo ""

# 最終確認
sleep 3
echo "🔍 サービス動作確認中..."

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ バックエンドAPI: 正常"
else
    echo "⚠️ バックエンドAPI: 応答なし"
fi

if curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ フロントエンド: 正常"
else
    echo "⚠️ フロントエンド: 応答なし"
fi

echo ""
echo "🚀 システムが正常に起動しました！"