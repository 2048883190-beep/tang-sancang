"""公共工具：Chroma 结果标准化。"""
from typing import List, Dict


def normalize_results(results: dict) -> List[dict]:
    """将 Chroma 返回的原始结果标准化为统一的 dict 列表。"""
    if not results.get("ids") or not results["ids"][0]:
        return []
    chunks = []
    ids = results["ids"][0]
    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    dists = results.get("distances", [[]])[0] or []
    for i in range(len(ids)):
        chunks.append({
            "id": ids[i],
            "content": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
        })
    return chunks
