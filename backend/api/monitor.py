from fastapi import APIRouter
from services.gpu_moniter import GPUMonitor  # 相対インポートに修正

router = APIRouter()

@router.get("/monitor/gpu")
async def monitor_gpu():
    """GPU の使用状況"""
    return GPUMonitor.get_status()
