import subprocess
from datetime import datetime

class GPUMonitor:
    """
    nvidia-smi を使って GPU のメモリ/利用率を取得
    """

    @staticmethod
    def get_status():
        try:
            # 取内存占用(MB) / 总内存(MB) / GPU利用率(%)
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

            # 多卡时每行一块GPU，这里先返回第1块；需要可扩展为列表
            line = out.splitlines()[0]
            mem_used, mem_total, util = [x.strip() for x in line.split(",")]

            return {
                "ok": True,
                "timestamp": datetime.now().isoformat(),
                "gpu_count": len(out.splitlines()),
                "memory_used_mb": int(mem_used),
                "memory_total_mb": int(mem_total),
                "utilization_percent": int(util),
                "raw": out,
            }
        except Exception as e:
            return {
                "ok": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }
