#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import time
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api/chat/blocking"  # 换成你的服务器IP:端口
INPUT_TXT = "questions.csv"
OUTPUT_CSV = "answers.csv"
TIMEOUT = 120
SLEEP = 0.2

def ask_one(q: str) -> dict:
    resp = requests.post(
        BASE_URL,
        json={"prompt": q},
        timeout=TIMEOUT,
        headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()
    return resp.json()

def main():
    lines = Path(INPUT_TXT).read_text(encoding="utf-8").splitlines()
    rows = [ln.strip() for ln in lines if ln.strip()]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["question","answer","latency","timestamp","context_found","source_documents","mode","error"])
        writer.writeheader()

        for i, q in enumerate(rows, start=1):
            try:
                data = ask_one(q)
                writer.writerow({
                    "question": q,
                    "answer": data.get("response", ""),
                    "latency": data.get("latency", ""),
                    "timestamp": data.get("timestamp", ""),
                    "context_found": data.get("context_found", ""),
                    "source_documents": data.get("source_documents", ""),
                    "mode": data.get("mode", "blocking"),
                    "error": ""
                })
                print(f"[{i}] OK: {q[:30]}")
            except requests.RequestException as e:
                writer.writerow({"question": q, "answer": "", "latency": "", "timestamp": "", "context_found": "", "source_documents": "", "mode": "blocking", "error": f"HTTP error: {e}"})
                print(f"[{i}] HTTP ERROR: {q[:30]} => {e}")
            except Exception as e:
                writer.writerow({"question": q, "answer": "", "latency": "", "timestamp": "", "context_found": "", "source_documents": "", "mode": "blocking", "error": f"Other error: {e}"})
                print(f"[{i}] ERROR: {q[:30]} => {e}")

            time.sleep(SLEEP)

    print(f"\nDone. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
