"""
GPU監視機能の最適化版
チャット機能を妨げない設計
"""
import streamlit as st
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Any

def create_optimized_gpu_monitor():
    """
    最適化されたGPU監視機能
    - チャット機能と共存可能
    - 効率的な更新制御
    - ユーザーフレンドリーな設定
    """
    
    st.header("🖥️ GPU監視（最適化版）")
    
    # 設定パネル
    with st.expander("⚙️ 監視設定", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            monitor_enabled = st.checkbox(
                "GPU監視を有効にする", 
                value=st.session_state.get("gpu_monitoring", False),
                key="gpu_monitor_enabled"
            )
        
        with col2:
            auto_refresh = st.checkbox(
                "自動更新", 
                value=st.session_state.get("gpu_auto_update", False),
                key="gpu_auto_refresh",
                help="オフにするとチャット機能がより快適になります"
            )
        
        with col3:
            update_interval = st.selectbox(
                "更新間隔",
                [10, 15, 30, 60],
                index=0,
                key="gpu_refresh_interval",
                help="長い間隔ほどチャット機能への影響が少なくなります"
            )
    
    # 監視状態の更新
    st.session_state.gpu_monitoring = monitor_enabled
    st.session_state.gpu_auto_update = auto_refresh
    st.session_state.gpu_update_interval = update_interval
    
    if not monitor_enabled:
        st.info("GPU監視が無効です。上記の設定で有効にしてください。")
        return
    
    # 手動更新ボタン
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 今すぐ更新", key="gpu_manual_update"):
            st.session_state.gpu_data = get_gpu_status_cached()
            st.session_state.last_gpu_update = datetime.now()
    
    with col2:
        if st.button("🧹 データクリア", key="gpu_clear_data"):
            st.session_state.gpu_data = None
            st.session_state.last_gpu_update = None
    
    with col3:
        # ステータス表示
        if st.session_state.get("last_gpu_update"):
            last_update = st.session_state.last_gpu_update
            elapsed = (datetime.now() - last_update).total_seconds()
            if auto_refresh:
                next_update_in = max(0, update_interval - elapsed)
                st.caption(f"📊 最終更新: {elapsed:.0f}秒前 | 次回: {next_update_in:.0f}秒後")
            else:
                st.caption(f"📊 最終更新: {elapsed:.0f}秒前 | 手動更新モード")
        else:
            st.caption("📊 データ未取得")
    
    # 自動更新の制御（条件付き）
    should_auto_update = (
        auto_refresh and 
        monitor_enabled and 
        (st.session_state.get("last_gpu_update") is None or 
         (datetime.now() - st.session_state.last_gpu_update).total_seconds() >= update_interval)
    )
    
    if should_auto_update:
        st.session_state.gpu_data = get_gpu_status_cached()
        st.session_state.last_gpu_update = datetime.now()
    
    # GPU情報表示
    display_gpu_info()
    
    # 自動更新のための再実行（条件付き）
    if auto_refresh and monitor_enabled:
        # より長い間隔で再実行（チャット機能への影響を最小化）
        time.sleep(2)  # 2秒待機
        if should_auto_update:
            st.rerun()

@st.cache_data(ttl=5)  # 5秒間キャッシュ
def get_gpu_status_cached():
    """キャッシュ付きGPU状態取得"""
    try:
        BACKEND_URL = "http://160.251.239.159:8000/api"
        response = requests.get(f"{BACKEND_URL}/monitor/gpu", timeout=3)
        return response.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def display_gpu_info():
    """GPU情報表示（軽量化版）"""
    gpu_data = st.session_state.get("gpu_data")
    
    if not gpu_data:
        st.info("🔄 GPU情報を取得するには「今すぐ更新」をクリックしてください")
        return
    
    if not gpu_data.get("ok"):
        st.error(f"❌ GPU情報取得エラー: {gpu_data.get('error', '不明なエラー')}")
        return
    
    # 軽量なメトリクス表示
    col1, col2, col3 = st.columns(3)
    
    used_mb = gpu_data.get("memory_used_mb", 0)
    total_mb = gpu_data.get("memory_total_mb", 1)
    usage_percent = (used_mb / total_mb) * 100
    util_percent = gpu_data.get("utilization_percent", 0)
    
    with col1:
        st.metric("VRAM使用量", f"{used_mb:,} MB", f"{usage_percent:.1f}%")
    
    with col2:
        st.metric("GPU利用率", f"{util_percent}%")
    
    with col3:
        st.metric("総VRAM", f"{total_mb:,} MB")
    
    # シンプルなプログレスバー
    st.subheader("📊 使用状況")
    
    # VRAM
    st.write("**VRAM使用率**")
    progress_col, status_col = st.columns([4, 1])
    with progress_col:
        st.progress(min(usage_percent / 100, 1.0))
    with status_col:
        if usage_percent >= 90:
            st.error(f"{usage_percent:.1f}%")
        elif usage_percent >= 70:
            st.warning(f"{usage_percent:.1f}%")
        else:
            st.success(f"{usage_percent:.1f}%")
    
    # GPU利用率
    st.write("**GPU利用率**")
    progress_col2, status_col2 = st.columns([4, 1])
    with progress_col2:
        st.progress(min(util_percent / 100, 1.0))
    with status_col2:
        if util_percent >= 80:
            st.error(f"{util_percent}%")
        elif util_percent >= 50:
            st.warning(f"{util_percent}%")
        else:
            st.info(f"{util_percent}%")

if __name__ == "__main__":
    create_optimized_gpu_monitor()
