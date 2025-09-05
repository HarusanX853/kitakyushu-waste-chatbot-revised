#!/bin/bash

echo "📁 北九州市ごみ分別チャットボット - プロジェクト構成"
echo "=========================================="

find . -type f | grep -v __pycache__ | grep -v .git | sort | while read file; do
    # ファイルの種類に応じてアイコンを表示
    case "$file" in
        *.py) echo "🐍 $file";;
        *.md) echo "📝 $file";;
        *.txt) echo "📄 $file";;
        *.csv) echo "📊 $file";;
        *.yml|*.yaml) echo "⚙️ $file";;
        *.sh) echo "🔧 $file";;
        *.log) echo "📋 $file";;
        *) echo "📂 $file";;
    esac
done

echo ""
echo "📊 ファイル統計:"
echo "Python files: $(find . -name '*.py' | wc -l)"
echo "Documentation: $(find . -name '*.md' -o -name '*.txt' | wc -l)"
echo "Configuration: $(find . -name '*.yml' -o -name '*.yaml' -o -name '*.sh' | wc -l)"
echo "Data files: $(find . -name '*.csv' -o -name '*.log' | wc -l)"
echo "Total files: $(find . -type f | grep -v __pycache__ | grep -v .git | wc -l)"
