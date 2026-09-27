"""
config.py —— 路径与常量（统一改这里即可）。
"""
from __future__ import annotations
import json
from pathlib import Path

# ---------- 根目录 ----------
# 以本文件位置推断仓库根：src/rag/config.py → 上两级
RAG_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = RAG_ROOT / "src" / "rag"
DATA_DIR = RAG_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
AUTO_DIR = RAW_DIR / "auto"      # Agent（Cursor）自动采集目录
MANUAL_DIR = RAW_DIR / "manual"  # 用户手动放入（知网PDF/教材/说明书）
EVAL_DIR = DATA_DIR / "eval"
REPORTS_DIR = RAG_ROOT / "reports"
CHROMA_DIR = RAG_ROOT / "chroma_db"

# ---------- 模型 ----------
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
COLLECTION_NAME = "t2dm_medical_kb"
DEFAULT_BATCH_SIZE = 32

# ---------- 检索 ----------
DEFAULT_MATCHING_MODE = "dense"
DEFAULT_HIER_MODE = "off"
DEFAULT_USE_PARALLEL_FUSION = True
HIER_TOP_K = 10
HIER_MAX_HIGH = 6
HIER_MAX_LOW = 4


# ---------- 知识库目标（与项目计划一致）----------
TARGET_DOC_COUNT = (50, 80)        # 50-80 份文档
TARGET_CHUNK_COUNT = (3000, 8000)  # 3000-8000 chunk

# ---------- DeepSeek ----------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
# 注意：API key 不要硬编码，运行时从环境变量 DEEPSEEK_API_KEY 读取

LOCKED_LAMBDAS_PATH = EVAL_DIR / "locked_lambdas.json"
LOCKED_SELECTIVE_RAG_PATH = EVAL_DIR / "locked_selective_rag.json"


def locked_passage_count(default: int = HIER_TOP_K) -> int:
    """验证集锁定的段数；文件不存在时用默认值。"""
    path = EVAL_DIR / "locked_passage_count.json"
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data["top_k"])


def load_locked_lambdas():
    """读验证集锁死的 λ。文件不存在则返回默认权威≥适用≥一致。"""
    from .parallel_fusion import FusionLambdas

    if not LOCKED_LAMBDAS_PATH.exists():
        return FusionLambdas(1.5, 0.5, 0.2)
    data = json.loads(LOCKED_LAMBDAS_PATH.read_text(encoding="utf-8"))
    return FusionLambdas(
        float(data["lam1"]),
        float(data["lam2"]),
        float(data["lam3"]),
    ).validated()


def load_locked_selective_thresholds():
    """Selective RAG v1 正式锁参（参赛主配置）。"""
    from .selective_rag import SelectiveThresholds

    if not LOCKED_SELECTIVE_RAG_PATH.exists():
        return SelectiveThresholds(0.5, 0.8, 0.6, 0.9)
    data = json.loads(LOCKED_SELECTIVE_RAG_PATH.read_text(encoding="utf-8"))
    th = data.get("thresholds") or {}
    return SelectiveThresholds(
        relevance_min=float(th.get("relevance_min", 0.5)),
        sufficiency_min=float(th.get("sufficiency_min", 0.8)),
        risk_relevance_min=float(th.get("risk_relevance_min", 0.6)),
        risk_sufficiency_min=float(th.get("risk_sufficiency_min", 0.9)),
    )


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, AUTO_DIR, MANUAL_DIR, EVAL_DIR, REPORTS_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)
