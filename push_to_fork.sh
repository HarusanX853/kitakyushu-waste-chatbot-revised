#!/bin/bash

echo "=== GitHub Fork & Push 手順 ==="
echo ""
echo "1. 以下のURLにアクセスしてForkしてください："
echo "   https://github.com/HarusanX853/kitakyushu-waste-chatbot-revised"
echo ""
echo "2. Forkが完了したら、あなたのGitHubユーザー名を入力してください"
echo -n "GitHubユーザー名: "
read GITHUB_USERNAME
echo ""

# 現在の変更をステージング
echo "=== 変更ファイルをステージング ==="
git add README.md
git add backend/api/chat.py
git add backend/main.py
git add backend/services/rag_service.py
git add frontend/app.py
git add "data/sample2 - シート1 - コピー - コピー.csv"
git add "data/sample2 - シート1 - コピー.csv"

echo "=== 変更をコミット ==="
git commit -m "Enhanced chatbot with improved system prompts, duplicate prevention, and knowledge file listing

- Enhanced system prompts with better error handling and context
- Implemented duplicate data prevention in CSV upload
- Added knowledge file listing functionality in frontend
- Fixed ChromaDB telemetry issues
- Improved health check endpoints
- Added comprehensive data management APIs
- Optimized RAG system performance"

echo ""
echo "=== フォーク先リモートを追加 ==="
git remote add fork https://github.com/$GITHUB_USERNAME/kitakyushu-waste-chatbot-revised.git

echo ""
echo "=== フォーク先にpush ==="
git push fork main

echo ""
echo "=== 次のステップ ==="
echo "1. https://github.com/$GITHUB_USERNAME/kitakyushu-waste-chatbot-revised にアクセス"
echo "2. 'Contribute' → 'Open pull request' をクリック"
echo "3. Pull Requestのタイトルと説明を入力"
echo "4. 'Create pull request' をクリック"
echo ""
echo "Pull Request説明例："
echo "---"
echo "# 北九州市ごみ分別チャットボット 機能強化"
echo ""
echo "## 主な変更点"
echo "- システムプロンプトの最適化とエラーハンドリング改善"
echo "- CSVアップロード時の重複データ防止機能実装"
echo "- ナレッジファイル一覧表示機能追加"
echo "- ChromaDBテレメトリエラーの解決"
echo "- ヘルスチェック機能の改善"
echo "- データ管理API機能の実装"
echo "- RAGシステムのパフォーマンス最適化"
echo ""
echo "## テスト済み機能"
echo "- ✅ CSV重複防止機能"
echo "- ✅ ナレッジファイル表示"
echo "- ✅ システム安定性"
echo "- ✅ API応答性能"
echo "---"
