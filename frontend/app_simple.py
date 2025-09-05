import os, time, json, requests
import streamlit as st
import qrcode
from io import BytesIO
import base64

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

# 会话状態の初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "generation_stats" not in st.session_state:
    st.session_state.generation_stats = []
if "device_mode" not in st.session_state:
    st.session_state.device_mode = "auto"

# 簡単なタイトル表示（問題診断用）
st.title("♻️ 北九州市ごみ分別チャットボット")

# テスト用の簡単なコンテンツ
st.write("🔄 アプリが正常に読み込まれました！")

# 簡単なテスト機能
if st.button("🧪 接続テスト"):
    try:
        response = requests.get(HEALTH_API, timeout=5)
        if response.status_code == 200:
            st.success("✅ バックエンド接続成功")
        else:
            st.error("❌ バックエンド接続失敗")
    except Exception as e:
        st.error(f"❌ エラー: {e}")

# 質問入力（簡易版）
user_input = st.text_input("質問してください：")
if user_input:
    st.write(f"あなたの質問: {user_input}")

st.write("📱 この画面が表示されれば、基本的なアプリは動作しています。")
