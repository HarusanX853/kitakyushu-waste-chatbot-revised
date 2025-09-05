#!/usr/bin/env python3
"""
レスポンシブデザインチェック用簡易ツール
"""
import requests
from bs4 import BeautifulSoup
import re

def check_responsive_design(url):
    """レスポンシブデザインの基本チェック"""
    print(f"🔍 レスポンシブデザイン分析: {url}")
    print("=" * 50)
    
    try:
        # HTMLを取得
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # BeautifulSoupでパース
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Viewportメタタグの確認
        viewport = soup.find('meta', attrs={'name': 'viewport'})
        if viewport:
            print("✅ Viewport メタタグ: 見つかりました")
            print(f"   内容: {viewport.get('content', 'N/A')}")
        else:
            print("❌ Viewport メタタグ: 見つかりませんでした")
        
        # 2. レスポンシブCSS（メディアクエリ）の確認
        style_tags = soup.find_all('style')
        media_queries = []
        for style in style_tags:
            if style.string:
                matches = re.findall(r'@media[^{]*\{[^}]*\}', style.string, re.IGNORECASE | re.DOTALL)
                media_queries.extend(matches)
        
        if media_queries:
            print(f"✅ メディアクエリ: {len(media_queries)}個見つかりました")
        else:
            print("❌ メディアクエリ: 見つかりませんでした")
        
        # 3. フレキシブルレイアウト要素の確認
        flex_elements = soup.find_all(attrs={"style": re.compile(r"display:\s*flex|flex-direction", re.I)})
        grid_elements = soup.find_all(attrs={"style": re.compile(r"display:\s*grid", re.I)})
        
        print(f"📐 フレックスボックス要素: {len(flex_elements)}個")
        print(f"📐 グリッド要素: {len(grid_elements)}個")
        
        # 4. 画像の確認
        images = soup.find_all('img')
        responsive_images = 0
        for img in images:
            if 'max-width' in str(img.get('style', '')) or 'width' in str(img.get('style', '')):
                responsive_images += 1
        
        print(f"🖼️  画像総数: {len(images)}個")
        print(f"🖼️  レスポンシブ対応画像: {responsive_images}個")
        
        # 5. Streamlit固有の要素チェック
        st_elements = soup.find_all(attrs={"class": re.compile(r"stColumn|stContainer", re.I)})
        print(f"📱 Streamlitコンテナ: {len(st_elements)}個")
        
        # 総合評価
        score = 0
        if viewport: score += 25
        if media_queries: score += 25
        if flex_elements or grid_elements: score += 25
        if responsive_images > 0: score += 25
        
        print(f"\n📊 レスポンシブ対応スコア: {score}/100")
        
        if score >= 75:
            print("🎉 良好なレスポンシブデザインです！")
        elif score >= 50:
            print("⚠️  改善の余地があります")
        else:
            print("❌ レスポンシブ対応が不十分です")
            
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    # Streamlitアプリをチェック
    check_responsive_design("http://160.251.239.159:8002")
