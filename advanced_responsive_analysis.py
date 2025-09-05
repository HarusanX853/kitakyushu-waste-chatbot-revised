#!/usr/bin/env python3
"""
高度なレスポンシブデザイン分析ツール（80-90点台対応）
"""
import re
import os

def advanced_responsive_analysis(file_path):
    """高度なレスポンシブデザイン分析"""
    print(f"🔍 高度なレスポンシブデザイン分析")
    print("=" * 60)
    print(f"📁 分析対象: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ ファイルが見つかりません: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📏 ファイルサイズ: {len(content):,} 文字")
    
    # === CSS分析 ===
    css_score = analyze_css_advanced(content)
    
    # === Streamlit分析 ===
    streamlit_score = analyze_streamlit_advanced(content)
    
    # === JavaScript分析 ===
    js_score = analyze_javascript(content)
    
    # === アクセシビリティ分析 ===
    a11y_score = analyze_accessibility(content)
    
    # === パフォーマンス分析 ===
    perf_score = analyze_performance(content)
    
    # === 総合スコア計算 ===
    weights = {
        'css': 0.30,
        'streamlit': 0.25,
        'javascript': 0.15,
        'accessibility': 0.15,
        'performance': 0.15
    }
    
    total_score = (
        css_score * weights['css'] +
        streamlit_score * weights['streamlit'] +
        js_score * weights['javascript'] +
        a11y_score * weights['accessibility'] +
        perf_score * weights['performance']
    )
    
    print(f"\n📊 総合評価:")
    print(f"   🎨 CSS: {css_score:.1f}点 ({weights['css']*100:.0f}%)")
    print(f"   🐍 Streamlit: {streamlit_score:.1f}点 ({weights['streamlit']*100:.0f}%)")
    print(f"   📱 JavaScript: {js_score:.1f}点 ({weights['javascript']*100:.0f}%)")
    print(f"   ♿ アクセシビリティ: {a11y_score:.1f}点 ({weights['accessibility']*100:.0f}%)")
    print(f"   ⚡ パフォーマンス: {perf_score:.1f}点 ({weights['performance']*100:.0f}%)")
    print(f"   🏆 総合スコア: {total_score:.1f}/100")
    
    # 評価ランク
    if total_score >= 90:
        rank = "🥇 Excellence (最優秀)"
    elif total_score >= 80:
        rank = "🥈 Advanced (上級)"
    elif total_score >= 70:
        rank = "🥉 Good (良好)"
    elif total_score >= 60:
        rank = "📈 Acceptable (許容範囲)"
    else:
        rank = "🔧 Needs Improvement (要改善)"
    
    print(f"\n🎯 評価ランク: {rank}")
    
    # 改善提案
    print(f"\n💡 改善提案:")
    suggest_improvements(css_score, streamlit_score, js_score, a11y_score, perf_score)

def analyze_css_advanced(content):
    """高度なCSS分析"""
    score = 0
    print(f"\n🎨 高度なCSS分析:")
    
    # メディアクエリ（詳細分析）
    media_queries = re.findall(r'@media[^{]*\{[^}]*\}', content, re.IGNORECASE | re.DOTALL)
    breakpoints = re.findall(r'(?:max-width|min-width):\s*(\d+)px', content, re.IGNORECASE)
    
    print(f"   📱 メディアクエリ: {len(media_queries)}個")
    print(f"   📏 ブレークポイント: {len(set(breakpoints))}種類 {set(breakpoints)}")
    
    if len(media_queries) >= 3: score += 15
    elif len(media_queries) >= 2: score += 10
    elif len(media_queries) >= 1: score += 5
    
    # CSS Grid
    grid_usage = len(re.findall(r'display:\s*grid|grid-template', content, re.IGNORECASE))
    print(f"   🔲 CSS Grid: {grid_usage}箇所")
    if grid_usage >= 3: score += 15
    elif grid_usage >= 1: score += 10
    
    # Flexbox
    flex_usage = len(re.findall(r'display:\s*flex|flex-direction|flex-wrap', content, re.IGNORECASE))
    print(f"   📐 Flexbox: {flex_usage}箇所")
    if flex_usage >= 5: score += 10
    elif flex_usage >= 2: score += 5
    
    # CSS Variables
    css_vars = len(re.findall(r'--[\w-]+:', content))
    print(f"   🎯 CSS変数: {css_vars}個")
    if css_vars >= 5: score += 10
    elif css_vars >= 3: score += 5
    
    # レスポンシブ単位
    responsive_units = len(re.findall(r'\d+(?:vw|vh|vmin|vmax|%|em|rem)', content, re.IGNORECASE))
    print(f"   📊 レスポンシブ単位: {responsive_units}箇所")
    if responsive_units >= 20: score += 10
    elif responsive_units >= 10: score += 5
    
    # アニメーション・トランジション
    animations = len(re.findall(r'transition:|animation:|@keyframes', content, re.IGNORECASE))
    print(f"   ✨ アニメーション: {animations}箇所")
    if animations >= 5: score += 10
    elif animations >= 2: score += 5
    
    # ダークモード対応
    dark_mode = len(re.findall(r'prefers-color-scheme:\s*dark', content, re.IGNORECASE))
    print(f"   🌙 ダークモード: {'対応' if dark_mode > 0 else '未対応'}")
    if dark_mode > 0: score += 10
    
    # 高解像度対応
    retina = len(re.findall(r'min-device-pixel-ratio|min-resolution', content, re.IGNORECASE))
    print(f"   🖥️  高解像度対応: {'対応' if retina > 0 else '未対応'}")
    if retina > 0: score += 5
    
    # 画面向き対応
    orientation = len(re.findall(r'orientation:\s*(landscape|portrait)', content, re.IGNORECASE))
    print(f"   🔄 画面向き対応: {'対応' if orientation > 0 else '未対応'}")
    if orientation > 0: score += 5
    
    print(f"   💯 CSS スコア: {score}/100")
    return min(score, 100)

def analyze_streamlit_advanced(content):
    """高度なStreamlit分析"""
    score = 0
    print(f"\n🐍 高度なStreamlit分析:")
    
    # カラムレイアウト
    columns = len(re.findall(r'st\.columns?\(', content))
    print(f"   📊 カラムレイアウト: {columns}箇所")
    if columns >= 10: score += 20
    elif columns >= 5: score += 15
    elif columns >= 2: score += 10
    
    # レスポンシブ分岐
    responsive_branches = len(re.findall(r'if\s+is_(mobile|tablet|desktop)', content))
    print(f"   📱 デバイス分岐: {responsive_branches}箇所")
    if responsive_branches >= 15: score += 20
    elif responsive_branches >= 10: score += 15
    elif responsive_branches >= 5: score += 10
    
    # Expanderの使用（モバイル対応）
    expanders = len(re.findall(r'st\.expander\(', content))
    print(f"   📂 Expander: {expanders}箇所")
    if expanders >= 3: score += 10
    elif expanders >= 1: score += 5
    
    # サイドバー制御
    sidebar_control = len(re.findall(r'initial_sidebar_state', content))
    print(f"   📋 サイドバー制御: {'設定済み' if sidebar_control > 0 else '未設定'}")
    if sidebar_control > 0: score += 10
    
    # ワイドレイアウト
    wide_layout = len(re.findall(r'layout="wide"', content))
    print(f"   📐 ワイドレイアウト: {'設定済み' if wide_layout > 0 else '未設定'}")
    if wide_layout > 0: score += 5
    
    # セッション状態管理
    session_states = len(re.findall(r'st\.session_state\.\w+', content))
    print(f"   💾 セッション状態: {session_states}箇所")
    if session_states >= 10: score += 10
    elif session_states >= 5: score += 5
    
    # エラーハンドリング
    error_handling = len(re.findall(r'try:|except|st\.error|st\.warning', content))
    print(f"   🛡️  エラーハンドリング: {error_handling}箇所")
    if error_handling >= 10: score += 10
    elif error_handling >= 5: score += 5
    
    # use_container_width使用
    container_width = len(re.findall(r'use_container_width=True', content))
    print(f"   📏 コンテナ幅調整: {container_width}箇所")
    if container_width >= 3: score += 10
    elif container_width >= 1: score += 5
    
    print(f"   💯 Streamlit スコア: {score}/100")
    return min(score, 100)

def analyze_javascript(content):
    """JavaScript分析"""
    score = 0
    print(f"\n📱 JavaScript分析:")
    
    # デバイス検出
    device_detection = len(re.findall(r'detectDevice|innerWidth|innerHeight', content, re.IGNORECASE))
    print(f"   📱 デバイス検出: {device_detection}箇所")
    if device_detection >= 3: score += 30
    elif device_detection >= 1: score += 20
    
    # イベントリスナー
    event_listeners = len(re.findall(r'addEventListener|on\w+', content, re.IGNORECASE))
    print(f"   🎯 イベントリスナー: {event_listeners}箇所")
    if event_listeners >= 3: score += 25
    elif event_listeners >= 1: score += 15
    
    # メッセージ通信
    messaging = len(re.findall(r'postMessage|message', content, re.IGNORECASE))
    print(f"   📞 メッセージ通信: {messaging}箇所")
    if messaging >= 1: score += 25
    
    # モダンJavaScript機能
    modern_js = len(re.findall(r'const |let |=>', content))
    print(f"   🆕 モダンJS機能: {modern_js}箇所")
    if modern_js >= 5: score += 20
    elif modern_js >= 2: score += 10
    
    print(f"   💯 JavaScript スコア: {score}/100")
    return min(score, 100)

def analyze_accessibility(content):
    """アクセシビリティ分析"""
    score = 0
    print(f"\n♿ アクセシビリティ分析:")
    
    # フォーカス管理
    focus_management = len(re.findall(r':focus|outline:', content, re.IGNORECASE))
    print(f"   🎯 フォーカス管理: {focus_management}箇所")
    if focus_management >= 3: score += 25
    elif focus_management >= 1: score += 15
    
    # 視覚的改善
    visual_aids = len(re.findall(r'prefers-reduced-motion|high-contrast', content, re.IGNORECASE))
    print(f"   👁️  視覚補助: {visual_aids}箇所")
    if visual_aids >= 2: score += 25
    elif visual_aids >= 1: score += 15
    
    # セマンティックHTML
    semantic_html = len(re.findall(r'aria-|role=|alt=', content, re.IGNORECASE))
    print(f"   📝 セマンティック要素: {semantic_html}箇所")
    if semantic_html >= 5: score += 25
    elif semantic_html >= 2: score += 15
    
    # ヘルプテキスト
    help_text = len(re.findall(r'help=|placeholder=|caption', content))
    print(f"   💬 ヘルプテキスト: {help_text}箇所")
    if help_text >= 5: score += 25
    elif help_text >= 2: score += 15
    
    print(f"   💯 アクセシビリティ スコア: {score}/100")
    return min(score, 100)

def analyze_performance(content):
    """パフォーマンス分析"""
    score = 0
    print(f"\n⚡ パフォーマンス分析:")
    
    # 条件分岐による最適化
    conditional_rendering = len(re.findall(r'if\s+\w+.*:', content))
    print(f"   🔀 条件付きレンダリング: {conditional_rendering}箇所")
    if conditional_rendering >= 20: score += 25
    elif conditional_rendering >= 10: score += 15
    
    # キャッシュ使用
    caching = len(re.findall(r'@st\.cache|st\.cache', content))
    print(f"   💾 キャッシュ使用: {caching}箇所")
    if caching >= 3: score += 25
    elif caching >= 1: score += 15
    
    # 遅延読み込み
    lazy_loading = len(re.findall(r'expander|tabs', content))
    print(f"   🐌 遅延読み込み: {lazy_loading}箇所")
    if lazy_loading >= 3: score += 25
    elif lazy_loading >= 1: score += 15
    
    # エラーハンドリング
    error_prevention = len(re.findall(r'timeout=|try:|except', content))
    print(f"   🛡️  エラー予防: {error_prevention}箇所")
    if error_prevention >= 5: score += 25
    elif error_prevention >= 2: score += 15
    
    print(f"   💯 パフォーマンス スコア: {score}/100")
    return min(score, 100)

def suggest_improvements(css_score, streamlit_score, js_score, a11y_score, perf_score):
    """改善提案"""
    if css_score < 80:
        print("   🎨 CSS Grid レイアウトの追加")
        print("   🌙 ダークモード対応の実装")
        print("   📱 より多くのブレークポイント設定")
    
    if streamlit_score < 80:
        print("   📱 デバイス分岐ロジックの拡充")
        print("   📂 Expanderによるモバイル最適化")
        print("   🔧 use_container_width の活用")
    
    if js_score < 80:
        print("   📱 より詳細なデバイス検出機能")
        print("   🎯 インタラクティブ要素の追加")
    
    if a11y_score < 80:
        print("   ♿ ARIA属性の追加")
        print("   🎯 フォーカス管理の改善")
        print("   💬 より多くのヘルプテキスト")
    
    if perf_score < 80:
        print("   💾 キャッシュ機能の活用")
        print("   🐌 遅延読み込みの実装")

if __name__ == "__main__":
    file_path = "/home/hisayuki/test-LLM/kitakyushu-waste-chatbot-revised/frontend/app.py"
    advanced_responsive_analysis(file_path)
