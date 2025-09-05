import os
from fastapi import APIRouter, UploadFile, File, HTTPException

from services.rag_service import get_rag_service

router = APIRouter()

# プロジェクトルート基準で data ディレクトリを作る
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
os.makedirs(DATA_DIR, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    CSV/TXT をアップロードして知識ベースに取り込む
    - .csv: 行を文書化して Chroma に追加
    - .txt: 全文を 1 文書として追加
    """
    try:
        save_path = os.path.join(DATA_DIR, file.filename)
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

        rag = get_rag_service()

        if file.filename.lower().endswith(".csv"):
            res = rag.add_csv(save_path)
            return {"status": "ok", "filename": file.filename, "ingested": res.get("count", 0)}
        else:
            # テキスト等はそのまま 1 文書として追加
            text = content.decode("utf-8", errors="ignore")
            rag.vectorstore.add_texts([text], metadatas=[{"source": file.filename}])
            return {"status": "ok", "filename": file.filename, "ingested": 1}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
