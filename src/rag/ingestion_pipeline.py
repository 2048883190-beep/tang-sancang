"""
ingestion_pipeline.py — 2型糖尿病知识库数据采集-解析-入库标准管线

管线的核心思想：
  不依赖文档格式（PDF/网页/纯文本），
  统一经过「清洗 → 标注元数据 → 切块 → 向量化入库」四步，
  确保入库的每一块chunk都带着完整溯源信息。
"""

import os
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# ============================================================
# 第一部分：元数据标准 — 这是你的「新表头」
# ============================================================

@dataclass
class DocMeta:
    """
    每条知识块的统一元数据结构。
    入库时随向量一起存入 Chroma，检索时可据此过滤/溯源。
    """
    # ---- 必填字段 ----
    doc_id: str                  # 文档唯一ID（建议用来源缩写+日期，如 cds_2024_01）
    title: str                   # 文档标题（指南名称/论文标题）
    source: str                  # 来源类型：guideline / textbook / drug_label / paper / wiki / nmpa
    specialty: str = "endocrinology"  # 科室（固定糖尿病=内分泌科）
    topic: str = ""              # 主题标签，逗号分隔：diagnosis, medication, complication, etc.

    # ---- 发布信息 ----
    publisher: str = ""          # 制定机构（中华医学会糖尿病学分会 / ADA / 卫健委）
    publish_date: str = ""       # 发布日期 YYYY-MM-DD
    version: str = ""            # 版本号（如 2024版）

    # ---- 文档本身 ----
    language: str = "zh"         # zh / en
    original_format: str = ""    # pdf / html / txt
    file_path: str = ""          # 原始文件路径（本地）

    # ---- 知识定位 ----
    keywords: List[str] = field(default_factory=list)  # 关键词列表
    abstract: str = ""           # 摘要（如有）

    def to_chroma_metadata(self) -> dict:
        """转为 Chroma 能存的 metadata（必须全是 str/int/float/bool）"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "specialty": self.specialty,
            "topic": self.topic,
            "publisher": self.publisher,
            "publish_date": self.publish_date,
            "version": self.version,
            "language": self.language,
            "original_format": self.original_format,
            "file_path": self.file_path or "",
            "abstract": self.abstract[:500],  # 截断，Chroma metadata 不宜过长
        }


# ============================================================
# 第二部分：文档源清单 — 这是你的「数据采集计划表」
# ============================================================

"""
你需要收集的文档清单，按来源分类。
每一条对应一个 DocMeta，下面是示例。
"""

SOURCE_PLAN = {
    # ------ 诊疗指南（核心） ------
    "guideline": [
        {
            "title": "中国2型糖尿病防治指南（2024版）",
            "publisher": "中华医学会糖尿病学分会",
            "url": "待补充",
            "topic": "diagnosis,treatment,complication,screening",
            "priority": "P0",  # P0=必须有, P1=重要, P2=锦上添花
        },
        {
            "title": "中国老年2型糖尿病防治临床指南（2024版）",
            "publisher": "中华医学会老年医学分会",
            "url": "待补充",
            "topic": "elderly,treatment,medication",
            "priority": "P0",
        },
        {
            "title": "中国2型糖尿病合并慢性肾脏病管理专家共识",
            "publisher": "中华医学会糖尿病学分会",
            "url": "待补充",
            "topic": "ckd,medication,complication",
            "priority": "P0",
        },
        {
            "title": "ADA Standards of Care in Diabetes 2025",
            "publisher": "American Diabetes Association",
            "url": "https://diabetesjournals.org/care/issue/48/Supplement_1",
            "topic": "diagnosis,treatment,medication,complication",
            "priority": "P1",
        },
        # 继续补充...总共目标 15-20 份核心指南
    ],

    # ------ 药品说明书 ------
    "drug_label": [
        {"title": "盐酸二甲双胍片说明书", "generic": "metformin", "priority": "P0"},
        {"title": "达格列净片说明书", "generic": "dapagliflozin", "priority": "P0"},
        {"title": "利拉鲁肽注射液说明书", "generic": "liraglutide", "priority": "P0"},
        {"title": "司美格鲁肽注射液说明书", "generic": "semaglutide", "priority": "P0"},
        {"title": "西格列汀片说明书", "generic": "sitagliptin", "priority": "P1"},
        {"title": "恩格列净片说明书", "generic": "empagliflozin", "priority": "P1"},
        # 目标 8-12 种降糖药
    ],

    # ------ 医学教材 ------
    "textbook": [
        {"title": "内科学 第9版 — 糖尿病章节", "publisher": "人民卫生出版社", "priority": "P1"},
        {"title": "药理学 第9版 — 降糖药物章节", "publisher": "人民卫生出版社", "priority": "P1"},
    ],

    # ------ PubMed 论文（摘要即可）------
    "paper": [
        # 高引用综述和RCT证据
        {"title": "待搜索补充", "priority": "P2"},
    ],
}


# ============================================================
# 第三部分：文档解析器 — 不管什么格式，统一变成干净文本
# ============================================================

class DocumentParser:
    """
    统一解析入口：PDF / TXT / HTML → 纯文本 + 元数据
    """

    @staticmethod
    def parse_pdf(filepath: str) -> str:
        """PDF → 纯文本"""
        text = ""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except ImportError:
            # 备选方案：pdfplumber
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        return text.strip()

    @staticmethod
    def parse_txt(filepath: str) -> str:
        """纯文本 → 纯文本（顺便处理编码）"""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()

    @staticmethod
    def parse_html(filepath: str) -> str:
        """HTML → 纯文本"""
        from bs4 import BeautifulSoup
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        # 去掉 script 和 style
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n").strip()

    @classmethod
    def parse(cls, filepath: str) -> str:
        """自动判断文件类型并解析"""
        ext = Path(filepath).suffix.lower()
        parsers = {
            ".pdf": cls.parse_pdf,
            ".txt": cls.parse_txt,
            ".html": cls.parse_html,
            ".htm": cls.parse_html,
        }
        parser = parsers.get(ext)
        if not parser:
            raise ValueError(f"不支持的文件格式: {ext}")
        return parser(filepath)


# ============================================================
# 第四部分：文本清洗 — 去掉页码、多余空行、乱码
# ============================================================

class TextCleaner:
    """医学文本专用清洗"""

    @staticmethod
    def clean(text: str) -> str:
        # 1. 去掉孤立的页码（" - 342 - " → ""）
        import re
        text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text)

        # 2. 合并 3 个以上连续换行为 2 个
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 3. 去掉行首行尾空白
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 4. 去掉全空行但保留段落间隔
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text.strip()


# ============================================================
# 第五部分：医学文本切块 — 语义边界切分
# ============================================================

class MedicalChunker:
    """
    医学文本切块器。
    不是按固定字数硬切，而是优先按章节标题/段落边界切。
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_by_sections(self, text: str) -> List[str]:
        """
        按章节标题分割。
        识别模式：
          - 「一、」「二、」中文序号
          - 「1.」「2.」数字序号
          - 「第X章」
        """
        import re
        # 章节分割点
        patterns = [
            r'\n[一二三四五六七八九十]、',    # 一、概述
            r'\n（[一二三四五六七八九十]）',   # （一）
            r'\n\d+\.\s+[A-Z\u4e00-\u9fff]',  # 1. 诊断标准
            r'\n第[一二三四五六七八九十\d]+章', # 第3章
            r'\n[ABCDEFG]、',                     # A、定义
        ]
        combined = '|'.join(f'({p})' for p in patterns)
        sections = re.split(combined, text)
        # 过滤空值
        return [s.strip() for s in sections if s and len(s.strip()) > 20]

    def chunk(self, text: str) -> List[str]:
        """
        主切块逻辑：
        1. 先按章节标题粗切
        2. 超过 chunk_size 的章节再用 RecursiveCharacterTextSplitter 细切
        3. 保证 overlap
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:  # 旧包路径兜底
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        final_chunks = []
        sections = self.split_by_sections(text)

        # 如果没切出章节（非结构化文本），直接按固定大小切
        if len(sections) <= 1:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "；", "，", " "],
            )
            return [chunk.page_content for chunk in splitter.create_documents([text])]

        for section in sections:
            if len(section) <= self.chunk_size * 1.2:
                final_chunks.append(section)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", "。", "；", "，", " "],
                )
                sub_chunks = splitter.create_documents([section])
                final_chunks.extend([c.page_content for c in sub_chunks])

        return final_chunks


# ============================================================
# 第六部分：完整入库管线 — 把上面所有步骤串起来
# ============================================================

class IngestionPipeline:
    """
    一条命令完成：源文件 → 解析 → 清洗 → 切块 → 向量化 → 入库

    用法:
        pipeline = IngestionPipeline(chroma_client, embedding_fn)
        pipeline.ingest_one(
            filepath="guidelines/糖尿病防治指南2024.pdf",
            meta=DocMeta(
                doc_id="cds_2024",
                title="中国2型糖尿病防治指南（2024版）",
                source="guideline",
                topic="diagnosis,treatment,complication",
                publisher="中华医学会糖尿病学分会",
                publish_date="2024-12",
            )
        )
    """

    def __init__(self, chroma_client, embedding_fn, collection_name="t2dm_knowledge"):
        self.chroma = chroma_client
        self.embed = embedding_fn
        self.chunker = MedicalChunker(chunk_size=512, chunk_overlap=80)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """确保 collection 存在"""
        try:
            self.chroma.get_collection(self.collection_name)
        except Exception:
            self.chroma.create_collection(
                name=self.collection_name,
                metadata={"description": "2型糖尿病知识库 — BSCV-RAG实验"}
            )

    def ingest_one(self, filepath: str, meta: DocMeta) -> Dict:
        """
        入库一份文档的完整流程。

        Returns:
            {"status": "ok", "doc_id": "...", "chunk_count": 42, "tokens_est": 8500}
        """
        collection = self.chroma.get_collection(self.collection_name)
        meta.original_format = Path(filepath).suffix.lower().lstrip(".")
        meta.file_path = filepath

        # Step 1: 解析
        raw_text = DocumentParser.parse(filepath)

        # Step 2: 清洗
        clean_text = TextCleaner.clean(raw_text)
        meta.abstract = clean_text[:300]  # 前300字当摘要

        # Step 3: 切块
        chunks = self.chunker.chunk(clean_text)

        # Step 4: 生成唯一ID（doc_id + chunk序号），方便后面交集去重
        doc_hash = hashlib.md5(f"{meta.doc_id}_{meta.title}".encode()).hexdigest()[:8]
        chunk_ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
        metadatas = [meta.to_chroma_metadata() for _ in chunks]

        # Step 5: 向量化 + 入库（支持 batch_embed(list)->list 或单条 embed(str)->vec）
        if hasattr(self.embed, "__call__") and getattr(self, "_batch_embed", None):
            embeddings = self._batch_embed(chunks)
        else:
            try:
                embeddings = self.embed(chunks)  # type: ignore[arg-type]
                if embeddings and isinstance(embeddings[0], (float, int)):
                    embeddings = [embeddings]
            except Exception:
                embeddings = [self.embed(c) for c in chunks]

        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        total_chars = sum(len(c) for c in chunks)
        stats = {
            "status": "ok",
            "doc_id": meta.doc_id,
            "title": meta.title,
            "filepath": filepath,
            "chunk_count": len(chunks),
            "total_chars": total_chars,
            "tokens_est": total_chars // 2,  # 中文大约2字符=1token
        }

        print(f"  入库完成: {meta.title}")
        print(f"    切块数: {len(chunks)}, 估计token: {stats['tokens_est']}")
        return stats

    def ingest_batch(self, file_list: List[tuple]) -> List[Dict]:
        """
        批量入库: [(filepath, DocMeta), ...]
        """
        results = []
        for filepath, meta in file_list:
            try:
                r = self.ingest_one(filepath, meta)
                results.append(r)
            except Exception as e:
                results.append({"status": "error", "doc_id": meta.doc_id, "error": str(e)})

        # 打印汇总
        ok = sum(1 for r in results if r.get("status") == "ok")
        total_chunks = sum(r.get("chunk_count", 0) for r in results if r.get("status") == "ok")
        print(f"\n{'='*50}")
        print(f"批量入库完成: {ok}/{len(results)} 份成功, 共 {total_chunks} 个chunk")
        return results


# ============================================================
# 第七部分：数据采集清单文件 — 替代你原来的「表头模板」
# ============================================================

def generate_collection_checklist(output_path="collection_checklist.md"):
    """
    生成一份 Markdown 采集清单。
    你可以打印出来或放在项目里，每收集到一份就勾掉。
    """
    lines = [
        "# 2型糖尿病知识库 — 数据采集清单",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 采集状态标记",
        "- [ ] 待采集",
        "- [~] 已找到源文件，待入库",
        "- [x] 已入库",
        "",
        "---",
        "",
    ]

    for source_type, items in SOURCE_PLAN.items():
        lines.append(f"## {source_type}")
        lines.append("")
        for i, item in enumerate(items):
            status = "- [ ]"
            priority_mark = {"P0": "⚠️必收", "P1": "重要", "P2": "可选"}
            lines.append(
                f"{status} **{item['title']}** "
                f"| {priority_mark.get(item['priority'], '')} "
                f"| 发布方: {item.get('publisher', '')}"
            )
            lines.append(f"  - 主题: {item.get('topic', '')}")
            lines.append("")
        lines.append("")

    lines.append("---")
    lines.append(f"## 采集进度")
    lines.append(f"总文档数: {sum(len(v) for v in SOURCE_PLAN.values())}")
    lines.append(f"已入库: 0")
    lines.append(f"完成率: 0%")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"采集清单已生成: {output_path}")


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # -------- 生成采集清单 --------
    generate_collection_checklist()

    # -------- 初始化管线 --------
    # chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # def my_embed(text): ...
    # pipeline = IngestionPipeline(chroma_client, my_embed)

    # -------- 入库一份示例文档 --------
    # meta = DocMeta(
    #     doc_id="cds_2024",
    #     title="中国2型糖尿病防治指南（2024版）",
    #     source="guideline",
    #     topic="diagnosis,treatment,complication",
    #     publisher="中华医学会糖尿病学分会",
    #     publish_date="2024-12",
    # )
    # pipeline.ingest_one("downloads/糖尿病指南2024.pdf", meta)

    print("\n管线代码就绪。")
    print("你需要的标准已经内置在 DocMeta 和 SOURCE_PLAN 里了。")
