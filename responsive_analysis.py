#!/usr/bin/env python3
"""
詳細レスポンシブデザイン分析ツール
ファイルベースでの分析も含む
"""
import re
import os
from pathlib import Path

def analyze_css_content(css_content):
    """CSSコンテンツを分析してレスポンシブ要素を検出"""
    score = 0
    details = []
    
    # メディアクエリの検出
    media_queries = re.findall(r'@media[^{]*\{[^}]*\}', css_content, re.IGNORECASE | re.DOTALL)
    if media_queries:
        score += 30
        details.append(f"✅ メディアクエリ: {len(media_queries)}個発見")
        for i, mq in enumerate(media_queries[:3]):  # 最初の3つを表示
            breakpoint = re.search(r'max-width:\s*(\d+px)', mq)
            if breakpoint:
                details.append(f"   📱 ブレークポイント {i+1}: {breakpoint.group(1)}")
    else:
        details.append("❌ メディアクエリ: 見つからず")
    
    # フレックスボックスの検出
    flex_patterns = re.findall(r'display:\s*flex|flex-direction|justify-content|align-items', css_content, re.IGNORECASE)
    if flex_patterns:
        score += 20
        details.append(f"✅ フレックスボックス: {len(flex_patterns)}個のプロパティ")
    else:
        details.append("❌ フレックスボックス: 見つからず")
    
    # グリッドレイアウトの検出
    grid_patterns = re.findall(r'display:\s*grid|grid-template|grid-gap', css_content, re.IGNORECASE)
    if grid_patterns:
        score += 15
        details.append(f"✅ CSSグリッド: {len(grid_patterns)}個のプロパティ")
    else:
        details.append("❌ CSSグリッド: 見つからず")
    
    # レスポンシブ単位の検出
    responsive_units = re.findall(r'\d+(?:vw|vh|%|em|rem)', css_content)
    if responsive_units:
        score += 15
        details.append(f"✅ レスポンシブ単位: {len(set(responsive_units))}種類")
    else:
        details.append("❌ レスポンシブ単位: 見つからず")
    
    return score, details

def analyze_streamlit_code(python_content):
    """Streamlitコードを分析してレスポンシブ要素を検出"""
    score = 0
    details = []
    
    # st.columns の使用
    columns_usage = re.findall(r'st\.columns\([^)]+\)', python_content)
    if columns_usage:
        score += 20
        details.append(f"✅ Streamlitカラム: {len(columns_usage)}個")
    else:
        details.append("❌ Streamlitカラム: 見つからず")
    
    # レスポンシブ設定
    if 'layout="wide"' in python_content:
        score += 10
        details.append("✅ ワイドレイアウト: 設定済み")
    else:
        details.append("❌ ワイドレイアウト: 未設定")
    
    # モバイル対応の分岐処理
    mobile_patterns = re.findall(r'is_mobile|mobile.*mode|モバイル', python_content, re.IGNORECASE)
    if mobile_patterns:
        score += 25
        details.append(f"✅ モバイル対応分岐: {len(mobile_patterns)}箇所")
    else:
        details.append("❌ モバイル対応分岐: 見つからず")
    
    # サイドバー設定
    if 'initial_sidebar_state' in python_content:
        score += 10
        details.append("✅ サイドバー制御: 設定済み")
    else:
        details.append("❌ サイドバー制御: 未設定")
    
    # コンテナやメトリクスの使用
    container_usage = re.findall(r'st\.container\(\)|st\.metric\(', python_content)
    if container_usage:
        score += 10
        details.append(f"✅ レスポンシブコンテナ: {len(container_usage)}個")
    else:
        details.append("❌ レスポンシブコンテナ: 見つからず")
    
    return score, details

def main():
    print("🔍 ファイルベース レスポンシブデザイン分析")
    print("=" * 60)
    
    # アプリケーションファイルを分析
    frontend_file = Path("/home/hisayuki/test-LLM/kitakyushu-waste-chatbot-revised/frontend/app.py")
    
    if not frontend_file.exists():
        print("❌ フロントエンドファイルが見つかりません")
        return
    
    # Pythonコードを読み込み
    with open(frontend_file, 'r', encoding='utf-8') as f:
        python_content = f.read()
    
    print(f"📁 分析対象: {frontend_file}")
    print(f"📏 ファイルサイズ: {len(python_content):,} 文字")
    print()
    
    # CSS分析（埋め込みスタイル）
    css_matches = re.findall(r'st\.markdown\(\s*"""(.*?)"""\s*,\s*unsafe_allow_html=True\)', python_content, re.DOTALL)
    css_content = "\n".join(css_matches)
    
    if css_content:
        print("🎨 CSS分析結果:")
        css_score, css_details = analyze_css_content(css_content)
        for detail in css_details:
            print(f"   {detail}")
        print(f"   💯 CSS スコア: {css_score}/100")
        print()
    else:
        print("❌ 埋め込みCSSが見つかりませんでした")
        css_score = 0
    
    # Streamlitコード分析
    print("🐍 Streamlitコード分析結果:")
    streamlit_score, streamlit_details = analyze_streamlit_code(python_content)
    for detail in streamlit_details:
        print(f"   {detail}")
    print(f"   💯 Streamlit スコア: {streamlit_score}/100")
    print()
    
    # 総合スコア計算
    total_score = min(100, (css_score * 0.6) + (streamlit_score * 0.4))
    print("📊 総合評価:")
    print(f"   CSS貢献度: {css_score * 0.6:.1f}点 (60%)")
    print(f"   Streamlit貢献度: {streamlit_score * 0.4:.1f}点 (40%)")
    print(f"   🏆 総合スコア: {total_score:.1f}/100")
    print()
    
    # 評価とアドバイス
    if total_score >= 80:
        print("🎉 優秀! 非常に良いレスポンシブデザインです")
    elif total_score >= 60:
        print("✅ 良好! レスポンシブ対応ができています")
    elif total_score >= 40:
        print("⚠️  改善の余地あり。追加対応を推奨")
    else:
        print("❌ 不十分。レスポンシブ対応の強化が必要")
    
    # 改善提案
    print("\n💡 改善提案:")
    if css_score < 50:
        print("   - より多くのメディアクエリを追加")
        print("   - フレックスボックスやグリッドレイアウトの活用")
    if streamlit_score < 50:
        print("   - モバイル向けの分岐処理を追加")
        print("   - st.columns()の効果的な活用")
    if total_score < 70:
        print("   - ブレークポイントの最適化")
        print("   - タッチデバイス向けのUI改善")

if __name__ == "__main__":
    main()
