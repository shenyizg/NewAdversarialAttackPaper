import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import arxiv
import requests
from lxml import etree
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

SEARCH_QUERIES = [
    "large language model attack",
    "llm jailbreak",
    "prompt injection attack",
    "multimodal model attack",
    "vision language model attack",
    "diffusion model attack",
    "text to image adversarial",
    "text to video diffusion attack",
    "text to speech adversarial",
    "physical adversarial attack",
    "adversarial defense",
]

MODEL_TYPE_ORDER = ["LLM", "Generative", "Traditional"]
MODEL_TYPE_LABELS_EN = {
    "LLM": "LLM / MLLM (Language Models)",
    "Generative": "General Generative Models",
    "Traditional": "Traditional Deep Learning Models",
}
MODEL_TYPE_LABELS_CN = {
    "LLM": "LLM / MLLM（语言模型）",
    "Generative": "泛生成模型",
    "Traditional": "传统深度学习模型",
}

CATEGORY_ORDER = {
    "LLM": ["Base/Interpretation", "Red Teaming/Jailbreak", "Safety Alignment", "Other"],
    "Generative": ["Base/Interpretation", "Red Teaming/Jailbreak", "Safety Alignment", "Other"],
    "Traditional": ["Digital Attack", "Physical Attack", "Adversarial Defense"],
}
CATEGORY_LABELS_CN = {
    "Base/Interpretation": "基础/可解释性",
    "Red Teaming/Jailbreak": "红队/越狱",
    "Safety Alignment": "安全对齐",
    "Other": "其他攻防技术",
    "Digital Attack": "数字攻击",
    "Physical Attack": "物理攻击",
    "Adversarial Defense": "对抗防御",
}
CATEGORY_LABELS_EN = {
    "Base/Interpretation": "Base/Interpretation",
    "Red Teaming/Jailbreak": "Red Teaming/Jailbreak",
    "Safety Alignment": "Safety Alignment",
    "Other": "Other",
    "Digital Attack": "Digital Attack",
    "Physical Attack": "Physical Attack",
    "Adversarial Defense": "Adversarial Defense",
}

CATEGORY_FILES = {
    "LLM": {
        "Base/Interpretation": "llm-base.md",
        "Red Teaming/Jailbreak": "llm-red-teaming-jailbreak.md",
        "Safety Alignment": "llm-safety-alignment.md",
        "Other": "llm-other.md",
    },
    "Generative": {
        "Base/Interpretation": "generative-base.md",
        "Red Teaming/Jailbreak": "generative-red-teaming-jailbreak.md",
        "Safety Alignment": "generative-safety-alignment.md",
        "Other": "generative-other.md",
    },
    "Traditional": {
        "Digital Attack": "traditional-digital-attack.md",
        "Physical Attack": "traditional-physical-attack.md",
        "Adversarial Defense": "traditional-adversarial-defense.md",
    },
}

MODEL_TYPE_ALIASES = {
    "llm": "LLM",
    "language model": "LLM",
    "mllm": "LLM",
    "vlm": "LLM",
    "generative": "Generative",
    "diffusion": "Generative",
    "t2i": "Generative",
    "t2v": "Generative",
    "i2v": "Generative",
    "tts": "Generative",
    "traditional": "Traditional",
    "non-generative": "Traditional",
    "classifier": "Traditional",
    "classification": "Traditional",
}
LLM_CATEGORY_ALIASES = {
    "base/interpretation": "Base/Interpretation",
    "interpretation": "Base/Interpretation",
    "base": "Base/Interpretation",
    "red teaming/jailbreak": "Red Teaming/Jailbreak",
    "red teaming": "Red Teaming/Jailbreak",
    "jailbreak": "Red Teaming/Jailbreak",
    "safety alignment": "Safety Alignment",
    "alignment": "Safety Alignment",
    "other": "Other",
}
TRAD_CATEGORY_ALIASES = {
    "digital attack": "Digital Attack",
    "digital": "Digital Attack",
    "physical attack": "Physical Attack",
    "physical": "Physical Attack",
    "adversarial defense": "Adversarial Defense",
    "defense": "Adversarial Defense",
    "defence": "Adversarial Defense",
}

CLASSIFY_SYSTEM_PROMPT = """You are an expert in adversarial machine learning.
Classify the paper only. Return JSON with keys: model_type, attack_category, confidence.

Model types (use exact labels):
- LLM: large language models, instruction-tuned LMs, MLLM/VLM, LLM-based agents.
- Generative: non-LLM generative models like diffusion, T2I/T2V/I2V, Text to Speech Models.
- Traditional: non-generative deep learning models (e.g., ResNet/CNN classifiers, Automatic Speech Recognition Models).

Attack categories (use exact labels):
- For LLM or Generative: Base/Interpretation, Red Teaming/Jailbreak, Safety Alignment, Other.
- For Traditional: Digital Attack, Physical Attack, Adversarial Defense.

Rules:
- Use the exact labels above.
- Provide confidence as a float between 0 and 1 (1 = most confident).
- Output JSON only.
"""

TRANSLATE_SYSTEM_PROMPT = """You are a professional technical translator.
Translate the title and abstract into Simplified Chinese.
Return JSON only with keys: title_zh, abstract_zh.

Rules:
- Keep technical terms and acronyms in English when appropriate.
- Do not add commentary.
- Keep translations concise.
"""

_last_llm_call_ts = 0.0


def load_config(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as fo:
        data = json.load(fo)
    llm_cfg = data.get("llm_api") or {}
    required = ["base_url", "api_key", "model"]
    missing = [key for key in required if not llm_cfg.get(key)]
    if missing:
        raise ValueError(f"Missing llm_api config keys: {', '.join(missing)}")
    return {
        "base_url": llm_cfg["base_url"],
        "api_key": llm_cfg["api_key"],
        "model": llm_cfg["model"],
        "temperature": llm_cfg.get("temperature", 0),
        "max_tokens": llm_cfg.get("max_tokens", 800),
        "min_interval_seconds": llm_cfg.get("min_interval_seconds", 1.0),
        "timeout_seconds": llm_cfg.get("timeout_seconds", 60),
    }


LLM_CONFIG = load_config(CONFIG_PATH)
LLM_BASE_URL = LLM_CONFIG["base_url"]
LLM_API_KEY = LLM_CONFIG["api_key"]
LLM_MODEL = LLM_CONFIG["model"]
LLM_TEMPERATURE = float(LLM_CONFIG["temperature"])
LLM_MAX_TOKENS = int(LLM_CONFIG["max_tokens"])
LLM_MIN_INTERVAL_SECONDS = float(LLM_CONFIG["min_interval_seconds"])
LLM_TIMEOUT_SECONDS = float(LLM_CONFIG["timeout_seconds"])


def build_api_url(base_url):
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def throttle_llm():
    global _last_llm_call_ts
    now = time.time()
    elapsed = now - _last_llm_call_ts
    if elapsed < LLM_MIN_INTERVAL_SECONDS:
        time.sleep(LLM_MIN_INTERVAL_SECONDS - elapsed)


def call_llm(messages):
    throttle_llm()
    url = build_api_url(LLM_BASE_URL)
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    _last_llm_call_ts = time.time()
    return data["choices"][0]["message"]["content"]


def call_llm_with_retry(messages, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            return call_llm(messages)
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise last_error


def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def normalize_model_type(raw_value):
    if not raw_value:
        return "Traditional"
    normalized = raw_value.strip().lower()
    for key, value in MODEL_TYPE_ALIASES.items():
        if key in normalized:
            return value
    if raw_value in MODEL_TYPE_ORDER:
        return raw_value
    return "Traditional"


def normalize_attack_category(model_type, raw_value):
    normalized = (raw_value or "").strip().lower()
    if model_type in ("LLM", "Generative"):
        for key, value in LLM_CATEGORY_ALIASES.items():
            if key in normalized:
                return value
        return "Other"
    for key, value in TRAD_CATEGORY_ALIASES.items():
        if key in normalized:
            return value
    return "Digital Attack"


def normalize_confidence(raw_value):
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, value))


def classify_paper(title, abstract, primary_category, comment):
    user_prompt = (
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        f"Primary category: {primary_category}\n"
        f"Comment: {comment or 'N/A'}"
    )
    try:
        response = call_llm_with_retry(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:
        print(f"LLM classify call failed: {exc}", file=sys.stderr)
        return "Traditional", "Digital Attack", 0.0

    data = extract_json(response)
    if not data:
        print("LLM classify response is not valid JSON, using fallbacks.", file=sys.stderr)
        return "Traditional", "Digital Attack", 0.0

    model_type = normalize_model_type(data.get("model_type", ""))
    attack_category = normalize_attack_category(model_type, data.get("attack_category", ""))
    confidence = normalize_confidence(data.get("confidence"))
    return model_type, attack_category, confidence


def translate_paper(title, abstract):
    user_prompt = f"Title: {title}\nAbstract: {abstract}"
    try:
        response = call_llm_with_retry(
            [
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:
        print(f"LLM translate call failed: {exc}", file=sys.stderr)
        return title, abstract

    data = extract_json(response)
    if not data:
        print("LLM translate response is not valid JSON, using fallbacks.", file=sys.stderr)
        return title, abstract

    title_zh = (data.get("title_zh") or "").strip() or title
    abstract_zh = (data.get("abstract_zh") or "").strip() or abstract
    return title_zh, abstract_zh


def build_search_url(query):
    return (
        "https://arxiv.org/search/?query="
        f"{quote_plus(query)}&searchtype=all&source=header&order=-submitted_date"
    )


def fetch_arxiv_ids(queries):
    ids = []
    for query in queries:
        url = build_search_url(query)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        html = etree.HTML(response.text)
        page_ids = html.xpath('//p[@class="list-title is-inline-block"]/a/text()')
        for arxiv_id in page_ids:
            cleaned = arxiv_id.replace("arXiv:", "").strip()
            if cleaned:
                ids.append(cleaned)
    seen = set()
    unique_ids = []
    for arxiv_id in ids:
        if arxiv_id not in seen:
            unique_ids.append(arxiv_id)
            seen.add(arxiv_id)
    return unique_ids


def build_paper_stub(result):
    authors = ", ".join(str(author) for author in result.authors)
    comment = result.comment.replace("\n", " ").strip() if result.comment else ""
    subdate = result.updated.strftime("%Y-%m-%d")
    if subdate == datetime.now().strftime("%Y-%m-%d"):
        subdate = f"**NEW** {subdate}"

    title = result.title.replace("\n", " ").strip()
    abstract = result.summary.replace("\n", " ").strip()

    return {
        "title_en": title,
        "title_zh": "",
        "primary_category": result.primary_category,
        "comment": comment,
        "submit_date": subdate,
        "abs_url": result.entry_id,
        "pdf_url": result.pdf_url,
        "authors": authors,
        "abstract_en": abstract,
        "abstract_zh": "",
        "model_type": "",
        "attack_category": "",
        "confidence": 0.0,
    }


def classify_papers(papers):
    for paper in tqdm(papers, desc="Classifying"):
        model_type, attack_category, confidence = classify_paper(
            paper["title_en"],
            paper["abstract_en"],
            paper["primary_category"],
            paper["comment"],
        )
        paper["model_type"] = model_type
        paper["attack_category"] = attack_category
        paper["confidence"] = confidence
    return papers


def translate_papers(papers):
    for paper in tqdm(papers, desc="Translating"):
        title_zh, abstract_zh = translate_paper(paper["title_en"], paper["abstract_en"])
        paper["title_zh"] = title_zh
        paper["abstract_zh"] = abstract_zh
    return papers


def bucket_papers(papers):
    buckets = {
        model_type: {category: [] for category in CATEGORY_ORDER[model_type]}
        for model_type in MODEL_TYPE_ORDER
    }
    for paper in papers:
        model_type = paper["model_type"] or "Traditional"
        attack_category = paper["attack_category"] or CATEGORY_ORDER[model_type][-1]
        if model_type not in buckets:
            model_type = "Traditional"
        if attack_category not in buckets[model_type]:
            attack_category = CATEGORY_ORDER[model_type][-1]
        buckets[model_type][attack_category].append(paper)

    for model_type, categories in buckets.items():
        for category in categories:
            categories[category].sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
    return buckets


def format_paper(paper, index, lang="en"):
    comment = f"{paper['comment']}\n\n" if paper["comment"] else ""
    confidence = f"{paper.get('confidence', 0.0):.2f}"
    if lang == "cn":
        return (
            f"## **{index}. {paper['title_en']}**\n\n"
            f"{paper['title_zh']} {paper['primary_category']}\n\n"
            f"{comment}"
            f"**SubmitDate**: {paper['submit_date']}    [abs]({paper['abs_url']}) "
            f"[paper-pdf]({paper['pdf_url']})\n\n"
            f"**Confidence**: {confidence}\n\n"
            f"**Authors**: {paper['authors']}\n\n"
            f"**Abstract**: {paper['abstract_en']}\n\n"
            f"摘要: {paper['abstract_zh']}\n\n\n\n"
        )
    return (
        f"## **{index}. {paper['title_en']}**\n\n"
        f"{paper['primary_category']}\n\n"
        f"{comment}"
        f"**SubmitDate**: {paper['submit_date']}    [abs]({paper['abs_url']}) "
        f"[paper-pdf]({paper['pdf_url']})\n\n"
        f"**Confidence**: {confidence}\n\n"
        f"**Authors**: {paper['authors']}\n\n"
        f"**Abstract**: {paper['abstract_en']}\n\n\n\n"
    )


def category_file_path(model_type, category, lang):
    base_name = CATEGORY_FILES[model_type][category]
    if lang == "cn":
        base = Path(base_name)
        return BASE_DIR / f"{base.stem}-cn{base.suffix}"
    return BASE_DIR / base_name


def write_category_files(buckets, lang="en", now_str=None):
    if now_str is None:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for model_type in MODEL_TYPE_ORDER:
        for category in CATEGORY_ORDER[model_type]:
            papers = buckets[model_type][category]
            path = category_file_path(model_type, category, lang)
            model_label = MODEL_TYPE_LABELS_EN[model_type] if lang == "en" else MODEL_TYPE_LABELS_CN[model_type]
            category_label = (
                CATEGORY_LABELS_EN[category] if lang == "en" else CATEGORY_LABELS_CN[category]
            )
            title = f"{model_label} - {category_label}"
            with open(path, "w", encoding="utf-8") as fo:
                fo.write(f"# {title}\n**update at {now_str}**\n\n")
                if lang == "en":
                    fo.write("Sorted by classifier confidence (high to low).\n\n")
                else:
                    fo.write("按分类器置信度从高到低排序。\n\n")
                index = 1
                for paper in papers:
                    fo.write(format_paper(paper, index, lang=lang))
                    index += 1


def write_index_readme(buckets, now_str=None):
    if now_str is None:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = BASE_DIR / "README.md"
    with open(path, "w", encoding="utf-8") as fo:
        fo.write("# Latest Adversarial Attack Papers\n")
        fo.write(f"**update at {now_str}**\n\n")
        fo.write("This index groups papers by model family and attack/defense focus.\n")
        fo.write("Each category has both English and Simplified Chinese lists.\n")
        fo.write("Entries are sorted by LLM classifier confidence (high to low).\n\n")

        fo.write("## Taxonomy\n")
        fo.write("- Model types: LLM/MLLM, General Generative Models, Traditional Deep Learning Models\n")
        fo.write(
            "- LLM/Generative categories: Base/Interpretation, Red Teaming/Jailbreak, "
            "Safety Alignment, Other\n"
        )
        fo.write("- Traditional categories: Digital Attack, Physical Attack, Adversarial Defense\n\n")

        for model_type in MODEL_TYPE_ORDER:
            fo.write(f"## {MODEL_TYPE_LABELS_EN[model_type]}\n\n")
            for category in CATEGORY_ORDER[model_type]:
                en_path = category_file_path(model_type, category, "en").name
                cn_path = category_file_path(model_type, category, "cn").name
                fo.write(
                    f"- {CATEGORY_LABELS_EN[category]}: "
                    f"[EN]({en_path}) [中文]({cn_path})\n"
                )
            fo.write("\n")


def collect_papers(queries):
    ids = fetch_arxiv_ids(queries)
    if not ids:
        return []
    papers = []
    client = arxiv.Client()
    chunk_size = 50
    with tqdm(total=len(ids), desc="Fetching arXiv") as progress:
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            search = arxiv.Search(
                id_list=chunk,
                max_results=len(chunk),
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            try:
                for result in client.results(search):
                    papers.append(build_paper_stub(result))
                    progress.update(1)
            except Exception as exc:
                print(f"arXiv batch failed ({len(chunk)} ids): {exc}", file=sys.stderr)
                # Fallback to per-id to isolate bad entries.
                for arxiv_id in chunk:
                    single = arxiv.Search(
                        id_list=[arxiv_id],
                        max_results=1,
                        sort_by=arxiv.SortCriterion.SubmittedDate,
                    )
                    try:
                        for result in client.results(single):
                            papers.append(build_paper_stub(result))
                            progress.update(1)
                    except Exception as inner_exc:
                        print(f"arXiv id failed ({arxiv_id}): {inner_exc}", file=sys.stderr)
    return papers


def main():
    papers = collect_papers(SEARCH_QUERIES)
    if not papers:
        print("No papers found.", file=sys.stderr)
        return
    classify_papers(papers)
    translate_papers(papers)
    buckets = bucket_papers(papers)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_category_files(buckets, lang="en", now_str=now_str)
    write_category_files(buckets, lang="cn", now_str=now_str)
    write_index_readme(buckets, now_str=now_str)


if __name__ == "__main__":
    main()
