import os, time, json, requests
import streamlit as st
import qrcode
from io import BytesIO
import base64
import threading
from datetime import datetime, timedelta

# ページ設定（最初に実行）
st.set_page_config(
    page_title="北九州市ごみ分別チャットボット", 
    page_icon="♻️", 
    layout="wide",
    initial_sidebar_state="collapsed"  # モバイルでサイドバーを折りたたみ
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://160.251.239.159:8000/api")

CHAT_BLOCK = f"{BACKEND_URL}/chat/blocking"
CHAT_STREAM = f"{BACKEND_URL}/chat/streaming"
UPLOAD_API = f"{BACKEND_URL}/upload"
GPU_API = f"{BACKEND_URL}/monitor/gpu"
HEALTH_API = f"{BACKEND_URL}/health"
DATA_SOURCES_API = f"{BACKEND_URL}/data/sources"

# 会话状態の初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "generation_stats" not in st.session_state:
    st.session_state.generation_stats = []
if "device_mode" not in st.session_state:
    st.session_state.device_mode = "auto"
if "gpu_monitoring" not in st.session_state:
    st.session_state.gpu_monitoring = False
if "gpu_data" not in st.session_state:
    st.session_state.gpu_data = None
if "last_gpu_update" not in st.session_state:
    st.session_state.last_gpu_update = None
if "gpu_update_interval" not in st.session_state:
    st.session_state.gpu_update_interval = 10  # デフォルト10秒
if "gpu_auto_update" not in st.session_state:
    st.session_state.gpu_auto_update = True

# チャット制限設定
CHAT_HISTORY_LIMIT = 50  # 履歴保存の上限
CHAT_DISPLAY_LIMIT = 10  # 表示件数の上限
STATS_HISTORY_LIMIT = 100  # 統計履歴の上限
QUESTION_MAX_LENGTH = 500  # 質問文字数の上限

# チャット履歴の自動クリーンアップ
def cleanup_chat_history():
    """チャット履歴が上限を超えた場合、古いものから削除"""
    if len(st.session_state.history) > CHAT_HISTORY_LIMIT:
        # 古い履歴を削除（最新50件を保持）
        st.session_state.history = st.session_state.history[-CHAT_HISTORY_LIMIT:]
        return True
    return False

def cleanup_stats_history():
    """統計履歴が上限を超えた場合、古いものから削除"""
    if len(st.session_state.generation_stats) > STATS_HISTORY_LIMIT:
        # 古い統計を削除（最新100件を保持）
        st.session_state.generation_stats = st.session_state.generation_stats[-STATS_HISTORY_LIMIT:]
        return True
    return False

# GPU監視機能
@st.cache_data(ttl=3)  # 3秒間キャッシュ
def get_gpu_status():
    """GPU状態を取得（キャッシュ付き）"""
    try:
        response = requests.get(GPU_API, timeout=3)
        return response.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ナレッジファイル一覧を取得する関数
@st.cache_data(ttl=60)  # 60秒間キャッシュ
def get_knowledge_files():
    """ナレッジファイル一覧を取得"""
    try:
        response = requests.get(DATA_SOURCES_API, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('sources', [])
    except Exception as e:
        st.error(f"ナレッジファイル一覧の取得に失敗しました: {e}")
        return []

def display_knowledge_files():
    """ナレッジファイル一覧を表示"""
    files = get_knowledge_files()
    if files:
        st.markdown("**📋 登録済みナレッジファイル**")
        for i, file_info in enumerate(files):
            if isinstance(file_info, dict):
                filename = file_info.get('file_name', 'Unknown')
                doc_count = file_info.get('document_count', 0)
                last_updated = file_info.get('last_updated', 'N/A')
                sample_content = file_info.get('sample_content', [])
                
                # カード形式で表示
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px 0; background-color: #f9f9f9;">
                    <h4 style="margin: 0; color: #333;">📄 {filename}</h4>
                    <p style="margin: 5px 0; color: #666;">
                        <strong>文書数:</strong> {doc_count} | 
                        <strong>更新日:</strong> {last_updated}
                    </p>
                    {f'<p style="margin: 5px 0; color: #666;"><strong>サンプル:</strong> {", ".join(sample_content[:3])}</p>' if sample_content else ''}
                </div>
                """, unsafe_allow_html=True)
            else:
                # 文字列の場合（後方互換性）
                st.caption(f"📄 {file_info}")
    else:
        st.info("📭 まだナレッジファイルが登録されていません")

def should_update_gpu():
    """GPU情報を更新すべきかチェック（5秒間隔）"""
    if st.session_state.last_gpu_update is None:
        return True
    
    current_time = datetime.now()
    last_update = st.session_state.last_gpu_update
    return (current_time - last_update).total_seconds() >= 5

def update_gpu_data():
    """GPU情報を更新"""
    if st.session_state.gpu_monitoring and should_update_gpu():
        st.session_state.gpu_data = get_gpu_status()
        st.session_state.last_gpu_update = datetime.now()

def render_gpu_monitor():
    """GPU監視UI表示（改善版：チャット機能と共存可能）"""
    st.header("🖥️ GPU リアルタイムモニタ")
    
    # 監視の開始/停止ボタン
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        monitor_button = st.button("🚀 監視開始" if not st.session_state.gpu_monitoring else "⏹️ 監視停止")
        if monitor_button:
            st.session_state.gpu_monitoring = not st.session_state.gpu_monitoring
            if st.session_state.gpu_monitoring:
                st.session_state.gpu_data = None
                st.session_state.last_gpu_update = None
    
    with col2:
        if st.session_state.gpu_monitoring:
            manual_update = st.button("🔄 手動更新")
            if manual_update:
                st.session_state.gpu_data = get_gpu_status()
                st.session_state.last_gpu_update = datetime.now()
    
    with col3:
        # 自動更新間隔の設定
        update_interval = st.selectbox("更新間隔(秒)", [5, 10, 15, 30], index=1, key="gpu_update_interval")
    
    with col4:
        # 自動更新の有効/無効切り替え
        auto_update = st.checkbox("自動更新", value=True, key="gpu_auto_update", 
                                 help="オフにすると手動更新のみになり、チャット機能が快適に使えます")
    
    # GPU情報表示
    if st.session_state.gpu_monitoring:
        # 更新条件をより厳密に制御
        should_update = False
        
        if st.session_state.gpu_data is None or st.session_state.last_gpu_update is None:
            should_update = True
        elif auto_update and (datetime.now() - st.session_state.last_gpu_update).total_seconds() >= update_interval:
            should_update = True
        
        # データ更新（必要な場合のみ）
        if should_update:
            with st.spinner("🔄 GPU情報を更新中..."):
                st.session_state.gpu_data = get_gpu_status()
                st.session_state.last_gpu_update = datetime.now()
        
        # GPU情報表示
        if st.session_state.gpu_data and st.session_state.gpu_data.get("ok"):
            gpu_data = st.session_state.gpu_data
            
            # ステータス表示
            status_col1, status_col2, status_col3 = st.columns(3)
            with status_col1:
                update_time = st.session_state.last_gpu_update.strftime("%H:%M:%S")
                st.success(f"🟢 監視中 - {update_time}")
            with status_col2:
                if auto_update:
                    next_update = st.session_state.last_gpu_update + timedelta(seconds=update_interval)
                    remaining = max(0, (next_update - datetime.now()).total_seconds())
                    st.info(f"⏱️ 次回更新: {remaining:.0f}秒後")
                else:
                    st.warning("⏸️ 自動更新オフ")
            with status_col3:
                st.metric("🔄 更新間隔", f"{update_interval}秒")
            
            # GPU メトリクス
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            used_mb = gpu_data.get("memory_used_mb", 0)
            total_mb = gpu_data.get("memory_total_mb", 1)
            usage_percent = (used_mb / total_mb) * 100
            util_percent = gpu_data.get("utilization_percent", 0)
            gpu_count = gpu_data.get("gpu_count", 1)
            
            with metric_col1:
                st.metric(
                    "🗄️ VRAM使用量", 
                    f"{used_mb:,} MB",
                    delta=f"{usage_percent:.1f}%"
                )
            
            with metric_col2:
                st.metric("⚡ GPU利用率", f"{util_percent}%")
            
            with metric_col3:
                st.metric("🖥️ GPU数", f"{gpu_count}")
            
            # 視覚的プログレスバー
            st.markdown("### 📊 使用状況")
            
            # VRAM使用率
            vram_col1, vram_col2 = st.columns([4, 1])
            with vram_col1:
                st.markdown("**VRAM使用率**")
                vram_bar = st.progress(min(usage_percent / 100, 1.0))
            with vram_col2:
                if usage_percent >= 90:
                    st.error(f"⚠️ {usage_percent:.1f}%")
                elif usage_percent >= 70:
                    st.warning(f"📈 {usage_percent:.1f}%")
                else:
                    st.success(f"✅ {usage_percent:.1f}%")
            
            st.caption(f"使用中: {used_mb:,} MB / 総容量: {total_mb:,} MB")
            
            # GPU利用率
            gpu_col1, gpu_col2 = st.columns([4, 1])
            with gpu_col1:
                st.markdown("**GPU利用率**")
                gpu_bar = st.progress(min(util_percent / 100, 1.0))
            with gpu_col2:
                if util_percent >= 90:
                    st.error(f"🔥 {util_percent}%")
                elif util_percent >= 50:
                    st.warning(f"🔶 {util_percent}%")
                else:
                    st.info(f"💙 {util_percent}%")
            
            # 詳細情報
            with st.expander("🔧 GPU詳細情報"):
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.write("**基本情報:**")
                    st.write(f"- GPU数: {gpu_count}")
                    st.write(f"- 更新間隔: {update_interval}秒")
                    st.write(f"- 自動更新: {'有効' if auto_update else '無効'}")
                    st.write(f"- タイムスタンプ: {gpu_data.get('timestamp', 'N/A')}")
                with detail_col2:
                    st.write("**Raw データ:**")
                    st.code(gpu_data.get('raw', 'N/A'))
            
        elif st.session_state.gpu_data:
            st.error(f"❌ GPU情報取得エラー: {st.session_state.gpu_data.get('error', '不明なエラー')}")
        else:
            st.info("🔄 GPU情報を取得中...")
        
        # 自動更新の制御（チャット機能を阻害しないように改善）
        if auto_update and st.session_state.gpu_monitoring:
            # 更新が必要かつ最後の更新から十分時間が経過した場合のみリロード
            time_since_update = (datetime.now() - st.session_state.last_gpu_update).total_seconds()
            
            if time_since_update >= update_interval:
                # 段階的リロード：即座にリロードせず、少し待つ
                if time_since_update >= update_interval + 2:  # 2秒のバッファ
                    st.rerun()
                else:
                    # 次回のリロードを予約（JavaScript使用）
                    remaining_time = update_interval + 2 - time_since_update
                    st.markdown(f"""
                    <script>
                        setTimeout(function() {{
                            // 条件付きリロード予約
                            if (document.hasFocus()) {{
                                window.location.reload();
                            }}
                        }}, {int(remaining_time * 1000)});
                    </script>
                    """, unsafe_allow_html=True)
        
    else:
        st.info("🚀 「監視開始」ボタンをクリックしてGPU監視を開始してください")
        
        # 機能説明
        with st.expander("📖 監視機能について"):
            st.write("**リアルタイムGPU監視機能（改善版）:**")
            st.write("- ⏱️ 設定可能な間隔で更新（5〜30秒、デフォルト10秒）")
            st.write("- 🔄 自動更新のオン/オフ切り替え可能")
            st.write("- 💬 **自動更新オフでチャット機能と快適に併用可能**")
            st.write("- 📊 VRAM使用量とGPU利用率をグラフィカル表示")
            st.write("- 🎨 使用率に応じた色分けアラート")
            st.write("- 🔧 詳細なGPU情報の表示")
            st.write("")
            st.info("💡 **推奨設定**: チャット使用時は「自動更新」をオフにして、必要に応じて「手動更新」を使用")
        
        # テスト用ボタン
        if st.button("🧪 GPU接続テスト"):
            with st.spinner("GPU接続をテスト中..."):
                test_data = get_gpu_status()
                if test_data.get("ok"):
                    st.success("✅ GPU接続成功！監視を開始できます。")
                    st.json(test_data)
                else:
                    st.error(f"❌ GPU接続失敗: {test_data.get('error', '不明なエラー')}")  # auto, mobile, tablet, desktop

# デバイス検出とレスポンシブモード設定
st.markdown("""
<script>
// デバイス検出用JavaScript
function detectDevice() {
    const width = window.innerWidth;
    if (width <= 480) return 'mobile';
    if (width <= 768) return 'tablet';
    if (width <= 1024) return 'desktop';
    return 'large';
}

// Streamlitに結果を送信
window.addEventListener('load', () => {
    const device = detectDevice();
    window.parent.postMessage({type: 'device_detected', device: device}, '*');
});
</script>
""", unsafe_allow_html=True)

# 高度なレスポンシブCSS（80-90点台対応）
st.markdown("""
<style>
/* === ベースライン設定 === */
:root {
    --primary-color: #2e7d32;
    --secondary-color: #4caf50;
    --accent-color: #1976d2;
    --border-radius: 12px;
    --transition: all 0.3s ease;
    --shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* === レスポンシブグリッドシステム === */
.responsive-grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    margin: 1rem 0;
}

.grid-item {
    background: rgba(255,255,255,0.8);
    padding: 1rem;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    transition: var(--transition);
}

.grid-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
}

/* === スマートフォン対応 (320px-480px) === */
@media (max-width: 480px) {
    .stApp > div:first-child {
        padding: 0.5rem !important;
    }
    
    .main-title {
        font-size: 1.2rem !important;
        text-align: center;
        line-height: 1.3;
        margin-bottom: 0.8rem;
    }
    
    /* タッチフレンドリーボタン */
    .stButton > button {
        min-height: 48px !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: var(--border-radius) !important;
        transition: var(--transition) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: var(--shadow);
    }
    
    /* 入力フィールドの改善 */
    .stTextInput > div > div > input {
        font-size: 16px !important; /* iOS ズーム防止 */
        padding: 0.75rem !important;
        border-radius: var(--border-radius) !important;
        width: 100% !important;
    }
    
    /* モバイル専用スタイル */
    .mobile-friendly {
        -webkit-tap-highlight-color: rgba(0,0,0,0.1);
        -webkit-touch-callout: none;
        -webkit-user-select: none;
        user-select: none;
    }
    
    /* スワイプ対応 */
    .swipeable {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scroll-snap-type: x mandatory;
    }
    
    /* メトリクスを縦並び */
    .metric-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* チャット入力の改善 */
    .chat-input-mobile {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #ddd;
        z-index: 1000;
    }
}

/* === タブレット対応 (481px-768px) === */
@media (min-width: 481px) and (max-width: 768px) {
    .main-title {
        font-size: 1.8rem !important;
        text-align: center;
    }
    
    .metric-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
    }
    
    /* タブレット用ナビゲーション */
    .tablet-nav {
        display: flex;
        justify-content: space-around;
        background: linear-gradient(135deg, #e3f2fd, #f0f8ff);
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
    }
}

/* === デスクトップ対応 (769px-1024px) === */
@media (min-width: 769px) and (max-width: 1024px) {
    .main-title {
        font-size: 2.2rem !important;
    }
    
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }
}

/* === 大画面対応 (1025px+) === */
@media (min-width: 1025px) {
    .main-title {
        font-size: 2.5rem !important;
    }
    
    .desktop-layout {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 2rem;
        align-items: start;
    }
}

/* === 追加のレスポンシブ機能 === */

/* ダークモード対応 */
@media (prefers-color-scheme: dark) {
    .upload-box, .grid-item {
        background: rgba(30, 30, 30, 0.9);
        color: #ffffff;
        border-color: #555;
    }
}

/* 高解像度ディスプレイ対応 */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
    .main-title {
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
    }
}

/* 横向き対応 */
@media (orientation: landscape) and (max-height: 500px) {
    .stApp {
        padding-top: 0.5rem !important;
    }
    
    .main-title {
        font-size: 1.3rem !important;
        margin-bottom: 0.5rem;
    }
}

/* アクセシビリティ改善 */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* フォーカス可視性改善 */
.stButton > button:focus,
.stTextInput > div > div > input:focus {
    outline: 3px solid var(--accent-color) !important;
    outline-offset: 2px !important;
}

/* === 共通スタイル === */
.main-title {
    color: var(--primary-color);
    font-weight: bold;
    border-bottom: 3px solid var(--secondary-color);
    padding-bottom: 0.5rem;
    transition: var(--transition);
}

.upload-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%);
    border: 2px dashed var(--accent-color);
    border-radius: var(--border-radius);
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: var(--shadow);
    transition: var(--transition);
}

.upload-box:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}

.chat-input {
    font-size: 1rem;
    padding: 0.75rem;
    border-radius: var(--border-radius);
    border: 1px solid #ddd;
    transition: var(--transition);
}

.chat-input:focus {
    border-color: var(--accent-color);
    box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1);
}

/* チャットメッセージの改善 */
.stChatMessage {
    margin-bottom: 1rem;
    max-width: 100%;
    word-wrap: break-word;
    animation: fadeInUp 0.3s ease;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* プログレス表示の改善 */
.stProgress > div > div {
    border-radius: var(--border-radius) !important;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">♻️ 北九州市ごみ分別チャットボット（RAG + Ollama）</h1>', unsafe_allow_html=True)

# 会话状态の初期化（最初に実行）
if "history" not in st.session_state:
    st.session_state.history = []
if "generation_stats" not in st.session_state:
    st.session_state.generation_stats = []

# 高度なレスポンシブレイアウト：自動デバイス検出対応
device_override = st.sidebar.selectbox(
    "� レイアウトモード", 
    ["auto", "mobile", "tablet", "desktop", "large"],
    help="デバイスに応じたレイアウトを選択。通常は'auto'を推奨"
)

# デバイス判定ロジック
if device_override == "auto":
    # JavaScript結果がない場合は画面幅で推定
    is_mobile = True  # デフォルトはモバイルファースト
    is_tablet = False
    is_desktop = False
else:
    is_mobile = device_override == "mobile"
    is_tablet = device_override == "tablet" 
    is_desktop = device_override in ["desktop", "large"]

# レスポンシブレイアウトの実装
if is_mobile:
    # === モバイルレイアウト ===
    st.markdown('<div class="responsive-grid mobile-layout">', unsafe_allow_html=True)
    
    # チャット優先表示
    st.markdown("### 💬 ごみ分別質問")
    
    # CSV アップロードを折りたたみ可能に
    with st.expander("📤 CSV アップロード", expanded=False):
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.caption("ナレッジ登録用CSVファイル")
        up = st.file_uploader("CSV ファイル", type=["csv"], key="csv_mobile", label_visibility="collapsed")
        if up:
            if st.button("🚀 アップロード", type="primary", key="upload_mobile", use_container_width=True):
                try:
                    files = {"file": (up.name, up.getvalue(), "text/csv")}
                    r = requests.post(UPLOAD_API, files=files, timeout=120)
                    r.raise_for_status()
                    st.success(f"✅ アップロード成功!")
                    st.json(r.json())
                    # キャッシュをクリアして最新のファイル一覧を取得
                    get_knowledge_files.clear()
                except Exception as e:
                    st.error(f"❌ アップロード失敗: {e}")
        
        # ナレッジファイル一覧を表示
        st.markdown("---")
        display_knowledge_files()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif is_tablet:
    # === タブレットレイアウト ===
    st.markdown('<div class="tablet-nav">', unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 💬 ごみ分別質問")
    
    with col2:
        with st.container():
            st.markdown('<div class="upload-box">', unsafe_allow_html=True)
            st.markdown("**📤 CSV アップロード**")
            up = st.file_uploader("CSV ファイル", type=["csv"], key="csv_tablet", label_visibility="collapsed")
            if up:
                if st.button("🚀 アップロード", type="primary", key="upload_tablet"):
                    try:
                        files = {"file": (up.name, up.getvalue(), "text/csv")}
                        r = requests.post(UPLOAD_API, files=files, timeout=120)
                        r.raise_for_status()
                        st.success(f"✅ アップロード成功!")
                        st.json(r.json())
                        # キャッシュをクリアして最新のファイル一覧を取得
                        get_knowledge_files.clear()
                    except Exception as e:
                        st.error(f"❌ アップロード失敗: {e}")
            
            # ナレッジファイル一覧を表示
            st.markdown("---")
            display_knowledge_files()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # === デスクトップレイアウト ===
    st.markdown('<div class="desktop-layout">', unsafe_allow_html=True)
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        st.markdown("### 💬 ごみ分別質問")
    
    with col2:
        with st.container():
            st.markdown('<div class="upload-box">', unsafe_allow_html=True)
            st.markdown("**📤 CSV アップロード**")
            st.caption("ナレッジ登録用CSVファイル")
            up = st.file_uploader("CSV ファイル", type=["csv"], key="csv_desktop", label_visibility="collapsed")
            if up:
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.info(f"📄 {up.name}")
                with col_b:
                    if st.button("🚀 アップロード", type="primary", key="upload_desktop"):
                        try:
                            files = {"file": (up.name, up.getvalue(), "text/csv")}
                            r = requests.post(UPLOAD_API, files=files, timeout=120)
                            r.raise_for_status()
                            st.success(f"✅ アップロード成功!")
                            st.json(r.json())
                            # キャッシュをクリアして最新のファイル一覧を取得
                            get_knowledge_files.clear()
                        except Exception as e:
                            st.error(f"❌ アップロード失敗: {e}")
            
            # ナレッジファイル一覧を表示
            st.markdown("---")
            display_knowledge_files()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# 侧栏：健康检查 & GPU
with st.sidebar:
    # スマホアクセス用QRコード
    st.header("📱 スマホでアクセス")
    
    # サーバーIP取得
    server_ip = "160.251.239.159"  # 実際のサーバーIP
    app_url = f"http://{server_ip}:8002"
    
    # QRコード生成（キャッシュ化で高速化）
    @st.cache_data
    def generate_qr_code(url):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.getvalue()
    
    # QRコード表示
    qr_image = generate_qr_code(app_url)
    st.image(qr_image, caption=f"📱 スマホでスキャン\n{app_url}", width=200)
    
    # アクセス方法の説明
    with st.expander("📖 スマホでのアクセス方法", expanded=False):
        st.markdown("""
        **方法1: QRコードでアクセス**
        1. スマホのカメラでQRコードをスキャン
        2. 表示されたリンクをタップ
        
        **方法2: URLを直接入力**
        1. スマホのブラウザを開く
        2. アドレスバーに以下を入力:
        ```
        http://160.251.239.159:8002
        ```
        
        **方法3: 同じWiFiネットワークの場合**
        - 同じWiFiに接続してからアクセス
        - より高速で安定した接続が可能
        
        ⚠️ **注意事項**
        - モバイルデータ通信料が発生する場合があります
        - セキュリティのため、信頼できるネットワークでのみ使用してください
        """)
    
    st.divider()
    st.header("サーバーステータス")
    
    # 状態チェックボタン（オンデマンド実行）
    if st.button("🔄 ステータス更新", help="サーバー状態を更新"):
        try:
            with st.spinner("接続中..."):
                r = requests.get(HEALTH_API, timeout=5)
                if r.status_code == 200:
                    st.success("API: Healthy")
                    if isinstance(r.json(), dict) and "details" in r.json():
                        st.json(r.json()["details"])
                else:
                    st.warning("API: Unhealthy")
        except Exception as e:
            st.error(f"Health error: {e}")

    st.divider()
    # 新しいGPUリアルタイムモニタ
    render_gpu_monitor()

    # 生成時間統計サマリー
    if st.session_state.generation_stats:
        st.divider()
        st.header("📈 生成時間統計")
        avg_time = sum(st.session_state.generation_stats) / len(st.session_state.generation_stats)
        current_stats_count = len(st.session_state.generation_stats)
        
        # 統計情報の表示
        col1, col2 = st.columns(2)
        with col1:
            st.metric("平均時間", f"{avg_time:.2f}秒")
            st.metric("最速時間", f"{min(st.session_state.generation_stats):.2f}秒")
        with col2:
            st.metric("記録数", f"{current_stats_count}/{STATS_HISTORY_LIMIT}")
            st.metric("最遅時間", f"{max(st.session_state.generation_stats):.2f}秒")
        
        # プログレスバー（容量使用率）
        usage_rate = current_stats_count / STATS_HISTORY_LIMIT
        st.progress(usage_rate, text=f"統計容量使用率: {usage_rate:.1%}")
        
        if st.button("🗑️ 統計をリセット"):
            st.session_state.generation_stats = []
            st.rerun()
    else:
        st.divider()
        st.header("📈 生成時間統計")
        st.info("まだ統計データがありません")
    
    # システム制限情報
    st.divider()
    st.header("⚙️ システム制限")
    with st.expander("📊 詳細制限情報", expanded=False):
        st.markdown(f"""
        **チャット履歴制限:**
        - 保存上限: {CHAT_HISTORY_LIMIT // 2}会話 ({CHAT_HISTORY_LIMIT}メッセージ)
        - 表示上限: {CHAT_DISPLAY_LIMIT // 2}会話 ({CHAT_DISPLAY_LIMIT}メッセージ)
        
        **質問文字数制限:**
        - 1回の質問: {QUESTION_MAX_LENGTH}文字以内
        
        **統計履歴制限:**
        - 保存上限: {STATS_HISTORY_LIMIT}件
        
        **自動整理:**
        - 上限を超えると古いデータから自動削除
        - 手動整理ボタンでいつでも整理可能
        
        **メモリ効率:**
        - セッション終了時に全データクリア
        - ブラウザ再起動でリセット
        """)
        
        # 現在の使用状況
        current_chats = len([m for m in st.session_state.history if m["role"] == "user"])
        current_stats = len(st.session_state.generation_stats)
        
        st.markdown("**現在の使用状況:**")
        st.metric("会話数", f"{current_chats}/{CHAT_HISTORY_LIMIT // 2}")
        st.metric("統計数", f"{current_stats}/{STATS_HISTORY_LIMIT}")

# 阻塞式调用
def chat_blocking(prompt: str):
    r = requests.post(CHAT_BLOCK, json={"prompt": prompt}, timeout=120)
    r.raise_for_status()
    return r.json()

# 流式调用（改良版：リアルタイム時間表示付き）
def chat_streaming(prompt: str, placeholder, time_placeholder):
    start_time = time.time()
    with requests.post(
        CHAT_STREAM, json={"prompt": prompt}, stream=True, timeout=300,
        headers={"Accept": "text/event-stream"}
    ) as resp:
        resp.raise_for_status()
        full = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line: 
                continue
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                    if obj.get("type") == "chunk":
                        full += obj.get("content", "")
                        placeholder.markdown(full + "▌")
                        # リアルタイム時間更新
                        elapsed = time.time() - start_time
                        time_placeholder.metric("⏱️ 生成時間", f"{elapsed:.1f}秒", delta="生成中...")
                except Exception:
                    pass
        return {"response": full}

# 页面：聊天
st.subheader("💬 ごみ分別質問")

# 生成統計情報の表示（高度なレスポンシブ対応）
if st.session_state.generation_stats:
    st.markdown('<div class="responsive-grid">', unsafe_allow_html=True)
    
    avg_time = sum(st.session_state.generation_stats) / len(st.session_state.generation_stats)
    
    if is_mobile:
        # モバイル：2x2 グリッド（タッチフレンドリー）
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="grid-item">', unsafe_allow_html=True)
            st.metric("📊 平均時間", f"{avg_time:.2f}秒")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="grid-item">', unsafe_allow_html=True)
            st.metric("🔢 回答数", f"{len(st.session_state.generation_stats)}回")
            st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="grid-item">', unsafe_allow_html=True)
            st.metric("🚀 最速", f"{min(st.session_state.generation_stats):.2f}秒")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="grid-item">', unsafe_allow_html=True)
            st.metric("🐌 最遅", f"{max(st.session_state.generation_stats):.2f}秒")
            st.markdown('</div>', unsafe_allow_html=True)
            
    elif is_tablet:
        # タブレット：2x2 グリッド（バランス重視）
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        metrics = [
            ("📊 平均生成時間", f"{avg_time:.2f}秒"),
            ("🚀 最速時間", f"{min(st.session_state.generation_stats):.2f}秒"),
            ("🐌 最遅時間", f"{max(st.session_state.generation_stats):.2f}秒"),
            ("🔢 回答数", f"{len(st.session_state.generation_stats)}回")
        ]
        
        for col, (label, value) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown('<div class="grid-item">', unsafe_allow_html=True)
                st.metric(label, value)
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # デスクトップ：1x4 グリッド（情報密度重視）
        cols = st.columns(4)
        metrics = [
            ("📊 平均生成時間", f"{avg_time:.2f}秒"),
            ("🚀 最速時間", f"{min(st.session_state.generation_stats):.2f}秒"),
            ("🐌 最遅時間", f"{max(st.session_state.generation_stats):.2f}秒"),
            ("🔢 回答数", f"{len(st.session_state.generation_stats)}回")
        ]
        
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.markdown('<div class="grid-item">', unsafe_allow_html=True)
                st.metric(label, value)
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

mode = st.radio("回答モード", ["blocking", "streaming"], horizontal=True)

# レスポンシブ入力フィールド
st.markdown('<div class="chat-input">', unsafe_allow_html=True)
q = st.text_input(
    "質問（例：アルミ缶はどう捨てますか？）",
    placeholder="ごみの分別について質問してください...",
    help="🔍 ヒント: 「ペットボトル」「生ごみ」「電池」などで検索できます",
    max_chars=QUESTION_MAX_LENGTH
)

# 文字数カウンター表示
if q:
    char_count = len(q)
    remaining = QUESTION_MAX_LENGTH - char_count
    if remaining >= 0:
        st.caption(f"文字数: {char_count}/{QUESTION_MAX_LENGTH} (残り{remaining}文字)")
    else:
        st.error(f"⚠️ 文字数超過: {char_count}/{QUESTION_MAX_LENGTH} ({-remaining}文字超過)")

st.markdown('</div>', unsafe_allow_html=True)

# 送信ボタンをレスポンシブに
if is_mobile:
    # モバイル：大きなボタン
    send_button = st.button("📤 質問を送信", type="primary", use_container_width=True)
else:
    # デスクトップ：通常サイズ
    send_button = st.button("送信", type="primary")

if send_button and q.strip():
    # 文字数制限チェック
    if len(q) > QUESTION_MAX_LENGTH:
        st.error(f"❌ 質問が長すぎます。{QUESTION_MAX_LENGTH}文字以内で入力してください。（現在: {len(q)}文字）")
    else:
        # 生成時間表示用のプレースホルダー（レスポンシブ対応）
        if is_mobile:
            # モバイル：縦配置
            response_area = st.empty()
            time_display = st.empty()
        else:
            # デスクトップ：横配置
            col1, col2 = st.columns([3, 1])
            with col1:
                response_area = st.empty()
            with col2:
                time_display = st.empty()
        
        t0 = time.time()
        try:
            if mode == "blocking":
                # ブロッキングモードでも時間表示
                time_display.metric("⏱️ 生成時間", "計測中...", delta="処理開始")
                data = chat_blocking(q)
                ans = data.get("response", "")
                response_area.markdown(ans)
            else:
                data = chat_streaming(q, response_area, time_display)
            ans = data.get("response", "")
            
            # 最終時間計算と表示
            final_time = time.time() - t0
            time_display.metric("✅ 生成完了", f"{final_time:.2f}秒", delta="完了!")
            
            # 統計に追加
            st.session_state.generation_stats.append(final_time)
            
            # 履歴に追加
            st.session_state.history.append({"role": "user", "text": q})
            st.session_state.history.append({"role": "assistant", "text": ans, "latency": final_time})
            
            # 履歴と統計のクリーンアップ
            history_cleaned = cleanup_chat_history()
            stats_cleaned = cleanup_stats_history()
            
            # 成功メッセージ
            success_msg = f"✨ 回答生成完了！ 生成時間: {final_time:.2f}秒"
            if history_cleaned or stats_cleaned:
                success_msg += f"\n📝 履歴管理: 保存上限により古いデータを整理しました"
            st.success(success_msg)
            
        except Exception as e:
            time_display.error("❌ エラー発生")
            st.error(f"エラー: {e}")

# 質問履歴（レスポンシブ対応）
if st.session_state.history:
    # 履歴統計情報の表示
    total_conversations = len([m for m in st.session_state.history if m["role"] == "user"])
    displayed_conversations = min(CHAT_DISPLAY_LIMIT // 2, total_conversations)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💬 総会話数", f"{total_conversations}回")
    with col2:
        st.metric("👁️ 表示中", f"{displayed_conversations}回")
    with col3:
        st.metric("💾 保存上限", f"{CHAT_HISTORY_LIMIT // 2}回")
    
    # 履歴管理ボタン
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("�️ 履歴をクリア", help="すべての会話履歴を削除"):
            st.session_state.history = []
            st.rerun()
    
    with col2:
        if st.button("📊 統計をリセット", help="生成時間統計をリセット"):
            st.session_state.generation_stats = []
            st.rerun()
    
    with col3:
        if st.button("💾 手動整理", help="古い履歴を手動で整理"):
            history_cleaned = cleanup_chat_history()
            stats_cleaned = cleanup_stats_history()
            if history_cleaned or stats_cleaned:
                st.success("📝 履歴を整理しました")
            else:
                st.info("📝 整理の必要はありません")
            st.rerun()
    
    st.subheader(f"📝 質問履歴（最新{CHAT_DISPLAY_LIMIT}件表示）")
    
    # 履歴表示
    display_history = st.session_state.history[-CHAT_DISPLAY_LIMIT:]
    for i, m in enumerate(display_history):
        if m["role"] == "user":
            st.chat_message("user").write(m["text"])
        else:
            if is_mobile:
                # モバイル：縦配置で時間情報も表示
                with st.chat_message("assistant"):
                    st.write(m["text"])
                    if "latency" in m:
                        if m["latency"] < 5:
                            st.success(f"⚡ 生成時間: {m['latency']:.2f}秒")
                        elif m["latency"] < 10:
                            st.info(f"⏱️ 生成時間: {m['latency']:.2f}秒")
                        else:
                            st.warning(f"🐌 生成時間: {m['latency']:.2f}秒")
            else:
                # デスクトップ：横配置
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.chat_message("assistant").write(m["text"])
                with col2:
                    if "latency" in m:
                        if m["latency"] < 5:
                            st.success(f"🚀 {m['latency']:.2f}秒")
                        elif m["latency"] < 10:
                            st.info(f"⏱️ {m['latency']:.2f}秒")
                        else:
                            st.warning(f"🐌 {m['latency']:.2f}秒")
