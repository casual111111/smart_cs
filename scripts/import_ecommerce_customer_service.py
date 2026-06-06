import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


REPO_ID = "rjac/e-commerce-customer-support-qa"
OUT_DIR = Path("data/knowledge_base")

# 可选：如果你以后手动下载成 json/jsonl，可以放这里
LOCAL_DATA_PATH = Path("data/ecommerce_customer_support_qa.json")

# 是否把完整 conversation 也写入知识库
# 建议先设 False，避免 RAG 检索时被长对话干扰
INCLUDE_CONVERSATION = False

# 如果 INCLUDE_CONVERSATION=True，最多写入多少字符
MAX_CONVERSATION_CHARS = 1800


def safe_filename(name: str) -> str:
    name = str(name or "unknown").strip().lower()
    name = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", name)
    name = name.strip("_")
    return name or "unknown"


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def load_local_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "train", "rows"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported local JSON format: {path}")


def load_rows() -> list[dict]:
    """
    优先读本地 JSON；
    本地没有时，从 Hugging Face 加载 rjac/e-commerce-customer-support-qa。
    """
    if LOCAL_DATA_PATH.exists():
        try:
            rows = load_local_json(LOCAL_DATA_PATH)
            print(f"Loaded local rows: {len(rows)}")
            return rows
        except Exception as exc:
            print(f"Ignoring invalid local data file: {LOCAL_DATA_PATH} ({exc})")

    ds = load_dataset(REPO_ID, split="train")
    rows = [dict(row) for row in ds]
    print(f"Loaded Hugging Face rows: {len(rows)}")
    print(f"Columns: {list(rows[0].keys()) if rows else []}")
    return rows


def parse_qa_field(qa_raw: Any) -> list[dict]:
    """
    这个数据集的 qa 字段通常是字符串形式的 JSON，例如：
    {
      "knowledge": [
        {
          "customer_summary_question": "...",
          "agent_summary_solution": "..."
        }
      ]
    }

    这里把它解析成 list[dict]。
    """
    if not qa_raw:
        return []

    if isinstance(qa_raw, dict):
        qa_obj = qa_raw
    else:
        text = str(qa_raw).strip()
        try:
            qa_obj = json.loads(text)
        except json.JSONDecodeError:
            return []

    knowledge = qa_obj.get("knowledge", [])

    if isinstance(knowledge, list):
        return [item for item in knowledge if isinstance(item, dict)]

    return []


def build_markdown_block(row: dict, index: int) -> str:
    issue_area = normalize_text(row.get("issue_area")) or "General"
    issue_category = normalize_text(row.get("issue_category")) or "General"
    issue_sub_category = normalize_text(row.get("issue_sub_category")) or "General"
    issue_category_sub_category = normalize_text(row.get("issue_category_sub_category"))

    customer_sentiment = normalize_text(row.get("customer_sentiment"))
    product_category = normalize_text(row.get("product_category"))
    product_sub_category = normalize_text(row.get("product_sub_category"))
    issue_complexity = normalize_text(row.get("issue_complexity"))
    agent_experience_level = normalize_text(row.get("agent_experience_level"))

    conversation = normalize_text(row.get("conversation"))
    qa_items = parse_qa_field(row.get("qa"))

    lines = []

    title = issue_category_sub_category or f"{issue_category} -> {issue_sub_category}"

    lines.append(f"## Case {index}: {title}")
    lines.append("")

    lines.append("### Metadata")
    lines.append("")
    lines.append(f"- Issue area: {issue_area}")
    lines.append(f"- Issue category: {issue_category}")
    lines.append(f"- Issue sub category: {issue_sub_category}")

    if customer_sentiment:
        lines.append(f"- Customer sentiment: {customer_sentiment}")
    if product_category:
        lines.append(f"- Product category: {product_category}")
    if product_sub_category:
        lines.append(f"- Product sub category: {product_sub_category}")
    if issue_complexity:
        lines.append(f"- Issue complexity: {issue_complexity}")
    if agent_experience_level:
        lines.append(f"- Required agent level: {agent_experience_level}")

    lines.append("")

    if qa_items:
        lines.append("### Customer Questions and Agent Solutions")
        lines.append("")

        for qa_index, qa in enumerate(qa_items, start=1):
            question = normalize_text(qa.get("customer_summary_question"))
            solution = normalize_text(qa.get("agent_summary_solution"))

            if not question and not solution:
                continue

            lines.append(f"#### Q{qa_index}: {question or 'Customer support question'}")
            lines.append("")
            lines.append(solution or "No solution provided.")
            lines.append("")
    else:
        lines.append("### Customer Support Knowledge")
        lines.append("")
        lines.append("No structured QA summary found for this case.")
        lines.append("")

    if INCLUDE_CONVERSATION and conversation:
        lines.append("### Example Conversation")
        lines.append("")
        if len(conversation) > MAX_CONVERSATION_CHARS:
            conversation = conversation[:MAX_CONVERSATION_CHARS] + "..."
        lines.append(conversation)
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()

    if not rows:
        print("No rows loaded.")
        return

    grouped = defaultdict(list)

    for row in rows:
        issue_area = normalize_text(row.get("issue_area")) or "General"
        grouped[issue_area].append(row)

    print(f"Grouped issue areas: {list(grouped.keys())}")

    for issue_area, items in grouped.items():
        file_name = f"ecommerce_{safe_filename(issue_area)}.md"
        out_path = OUT_DIR / file_name

        lines = []
        lines.append(f"# {issue_area} Customer Service Knowledge Base")
        lines.append("")
        lines.append("> Source dataset: rjac/e-commerce-customer-support-qa")
        lines.append("> Usage: smart-cs e-commerce customer service RAG knowledge base")
        lines.append("")
        lines.append("This document contains customer support cases, summarized customer questions, agent solutions, issue categories, product categories, and customer sentiment metadata.")
        lines.append("")

        for index, row in enumerate(items, start=1):
            lines.append(build_markdown_block(row, index))

        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Written: {out_path} ({len(items)} cases)")

    print("Done. Restart FastAPI to reload knowledge base.")


if __name__ == "__main__":
    main()