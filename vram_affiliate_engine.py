import os
import re
import json
import time
import hashlib
from datetime import datetime, timezone
from urllib.parse import quote_plus
import requests

# Konfigurasi Environment Variables
AMAZON_TAG = os.environ.get("AMAZON_TAG", "yourtag-20").strip()
SITE_URL = os.environ.get("SITE_URL", "https://username.github.io/repo-name").rstrip("/")
REPO_NAME = os.environ.get("GITHUB_REPOSITORY", "user/repo")

# Kunci verifikasi IndexNow deterministik
INDEXNOW_KEY = hashlib.md5(f"{REPO_NAME}_{AMAZON_TAG}_vram".encode("utf-8")).hexdigest()

DB_PATH = "data/models_db.json"
DOCS_DIR = "docs"
MODELS_DIR = os.path.join(DOCS_DIR, "models")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "ai-model"


def load_db() -> dict:
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_db(db: dict):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def make_high_ticket_amazon_url(query: str, min_price_cents: int = 20000) -> str:
    """
    Membuat URL afiliasi Amazon US dengan parameter filter harga minimum (default: 20000 cents = $200).
    Memastikan pengunjung hanya melihat produk bernilai di atas $200.
    """
    encoded_q = quote_plus(query)
    return f"https://www.amazon.com/s?k={encoded_q}&rh=p_36%3A{min_price_cents}-&tag={AMAZON_TAG}"


def fetch_trending_hf_models() -> list:
    """Mengambil daftar model AI/LLM populer dan trending dari API publik Hugging Face."""
    endpoints = [
        "https://huggingface.co/api/models?sort=likes7d&direction=-1&limit=45",
        "https://huggingface.co/api/models?sort=downloads&direction=-1&filter=text-generation&limit=45"
    ]
    models = []
    seen = set()
    for url in endpoints:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            for item in resp.json():
                model_id = item.get("id", "")
                if model_id and model_id not in seen:
                    seen.add(model_id)
                    models.append(model_id)
        except Exception as e:
            print(f"[WARN] Gagal mengambil daftar model dari {url}: {e}")
    return models


def extract_parameter_billions(model_id: str, detail_data: dict) -> float:
    """Mengekstrak ukuran parameter model (dalam Miliar / Billions) dari metadata safetensors atau nama model."""
    safetensors = detail_data.get("safetensors", {})
    if isinstance(safetensors, dict):
        total_params = safetensors.get("total", 0)
        if isinstance(total_params, (int, float)) and total_params > 100_000_000:
            return round(total_params / 1_000_000_000, 1)

    # Fallback: deteksi pola seperti 70B, 32B, 14B, 8B, 7B, 8x7B dari model_id atau tags
    combined_text = model_id + " " + " ".join(detail_data.get("tags", []))
    moe_match = re.search(r"(\d+)x(\d+(?:\.\d+)?)[bB]", combined_text)
    if moe_match:
        experts = float(moe_match.group(1))
        size = float(moe_match.group(2))
        return round(experts * size * 0.85, 1)

    param_match = re.search(r"(?:^|[-_/ ])(\d+(?:\.\d+)?)[bB](?:$|[-_/ ])", combined_text)
    if param_match:
        val = float(param_match.group(1))
        if 0.5 <= val <= 1000:
            return round(val, 1)

    # Default konservatif untuk model LLM/Diffusion modern
    return 8.0


def build_hardware_tiers(params_b: float, vram_q4: float, vram_fp16: float) -> list:
    """Menentukan daftar perangkat keras Amazon di atas $200 berdasarkan kebutuhan VRAM."""
    if vram_q4 <= 8.5:
        gpu_primary = {
            "category": "Recommended 16GB VRAM GPU ($449 - $620)",
            "badge": "Best Value Local AI Card",
            "price_range": "$449 – $620",
            "title": "NVIDIA GeForce RTX 4060 Ti 16GB / RTX 4070 Super 12GB GDDR6X",
            "desc": f"Provides ample VRAM headroom to run {params_b}B models in 4-bit (Q4_K_M) and 8-bit quantization with large context windows at 60+ tokens/sec.",
            "query": "NVIDIA GeForce RTX 4060 Ti 16GB graphics card",
            "min_cents": 40000,
            "cta": "Compare 16GB GPUs ($400+) on Amazon"
        }
    elif vram_q4 <= 16.5:
        gpu_primary = {
            "category": "High-Bandwidth 16GB-20GB GPU ($749 - $1,099)",
            "badge": "Optimal 14B-32B AI Accelerator",
            "price_range": "$749 – $1,099",
            "title": "NVIDIA GeForce RTX 4070 Ti Super 16GB / RTX 4080 Super 16GB GDDR6X",
            "desc": f"High memory bandwidth (up to 736 GB/s) required to load {params_b}B weights into VRAM without slow PCIe system RAM offloading.",
            "query": "NVIDIA GeForce RTX 4070 Ti Super 16GB graphics card",
            "min_cents": 70000,
            "cta": "View RTX 4070 Ti / 4080 Super ($700+) on Amazon"
        }
    else:
        gpu_primary = {
            "category": "Flagship 24GB VRAM Workstation GPU ($999 - $1,950)",
            "badge": "Maximum Local LLM Throughput",
            "price_range": "$999 – $1,950",
            "title": "NVIDIA GeForce RTX 4090 24GB / RTX 3090 24GB GDDR6X GPU",
            "desc": f"1,008 GB/s memory bandwidth and 24GB VRAM capacity tailored for heavy {params_b}B parameter inference, LoRA fine-tuning, and multi-GPU scaling.",
            "query": "NVIDIA GeForce RTX 4090 24GB graphics card",
            "min_cents": 95000,
            "cta": "Check 24GB Flagship GPUs ($950+) on Amazon"
        }

    return [
        gpu_primary,
        {
            "category": "Unified Memory Local AI Mini Workstation ($699 - $1,599)",
            "badge": f"Fits {vram_fp16}GB FP16 / Large Context",
            "price_range": "$699 – $1,599",
            "title": "Apple Mac Mini M4 Pro / Ryzen AI 9 HX 370 Mini PC (32GB - 64GB Unified RAM)",
            "desc": f"Run large quantized models silently at under 65W power draw using high-capacity unified memory architecture.",
            "query": "Apple Mac Mini M4 Pro OR Ryzen AI 9 HX 370 64GB Mini PC",
            "min_cents": 60000,
            "cta": "Explore AI Mini Workstations ($600+) on Amazon"
        },
        {
            "category": "Turnkey Prebuilt AI Desktop Workstation ($1,499 - $2,999)",
            "badge": "Plug-and-Play CUDA System",
            "price_range": "$1,499 – $2,999",
            "title": "Liquid-Cooled RTX 4080 Super / RTX 4090 AI Workstation Desktop PC (64GB DDR5)",
            "desc": f"Pre-configured high-airflow desktop system ready for Ollama, vLLM, LM Studio, and PyTorch out of the box.",
            "query": "RTX 4080 Super 64GB DDR5 gaming workstation desktop PC",
            "min_cents": 140000,
            "cta": "See Prebuilt AI Desktops ($1,400+) on Amazon"
        },
        {
            "category": "CPU Layer-Offload Memory & Gen4 NVMe Kit ($240 - $420)",
            "badge": "High-Speed llama.cpp Offloading",
            "price_range": "$240 – $420",
            "title": "96GB / 128GB DDR5 6000MHz CL30 Dual-Channel RAM + 4TB NVMe M.2 SSD",
            "desc": f"Essential for splitting {params_b}B GGUF model layers across system RAM and loading multi-gigabyte safetensors weights instantly.",
            "query": "96GB 2x48GB DDR5 6000MHz desktop memory kit",
            "min_cents": 21000,
            "cta": "Check 96GB/128GB DDR5 Kits ($210+) on Amazon"
        },
        {
            "category": "ATX 3.1 PCIe 5.0 Multi-GPU Power Supply ($215 - $340)",
            "badge": "Native 12V-2x6 600W Connector",
            "price_range": "$215 – $340",
            "title": "1200W / 1300W 80+ Platinum ATX 3.1 Full-Modular Workstation Power Supply",
            "desc": "Handles transient GPU power spikes during full-load LLM token generation and dual-GPU workstation builds.",
            "query": "1200W 80 Plus Platinum ATX 3.1 PCIe 5.0 power supply",
            "min_cents": 20000,
            "cta": "View 1200W+ Platinum PSUs ($200+) on Amazon"
        }
    ]


def fetch_model_details(model_id: str) -> dict | None:
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        params_b = extract_parameter_billions(model_id, data)
        vram_fp16 = round(params_b * 2.25, 1)
        vram_q8 = round(params_b * 1.18, 1)
        vram_q4 = round(params_b * 0.68, 1)

        author = model_id.split("/")[0] if "/" in model_id else "OpenSource"
        short_name = model_id.split("/")[-1]
        pipeline = data.get("pipeline_tag", "text-generation")
        downloads = int(data.get("downloads", 0))
        likes = int(data.get("likes", 0))

        slug = f"{slugify(model_id)}-vram-requirements-gpu-workstation-guide"

        return {
            "model_id": model_id,
            "author": author,
            "short_name": short_name,
            "slug": slug,
            "pipeline": pipeline,
            "downloads": downloads,
            "likes": likes,
            "params_b": params_b,
            "vram_fp16": vram_fp16,
            "vram_q8": vram_q8,
            "vram_q4": vram_q4,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d")
        }
    except Exception as e:
        print(f"[WARN] Gagal memproses model {model_id}: {e}")
        return None


def build_model_page(model: dict, all_models: list):
    os.makedirs(MODELS_DIR, exist_ok=True)
    m_id = model["model_id"]
    short = model["short_name"]
    params_b = model["params_b"]
    vram_fp16 = model["vram_fp16"]
    vram_q8 = model["vram_q8"]
    vram_q4 = model["vram_q4"]
    page_url = f"{SITE_URL}/models/{model['slug']}.html"

    tiers = build_hardware_tiers(params_b, vram_q4, vram_fp16)

    cards_html = ""
    for t in tiers:
        amz_url = make_high_ticket_amazon_url(t["query"], t["min_cents"])
        cards_html += f"""
        <div class="card">
            <div class="card-top">
                <span class="cat">{t['category']}</span>
                <span class="badge">{t['badge']}</span>
            </div>
            <h3>{t['title']}</h3>
            <p>{t['desc']}</p>
            <div class="card-footer">
                <span class="price-pill">Est. Price: {t['price_range']}</span>
                <a href="{amz_url}" target="_blank" rel="nofollow sponsored noopener" class="btn">{t['cta']} &rarr;</a>
            </div>
        </div>
        """

    # Internal linking ke model lain untuk memperkuat crawlability SEO
    related_html = ""
    for rel in all_models[:8]:
        if rel["model_id"] == m_id:
            continue
        related_html += f"""
        <a href="{SITE_URL}/models/{rel['slug']}.html" class="rel-item">
            <strong>{rel['short_name']} ({rel['params_b']}B)</strong>
            <span>Q4 VRAM: {rel['vram_q4']} GB &bull; FP16: {rel['vram_fp16']} GB &rarr;</span>
        </a>
        """

    schema_data = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f"How much VRAM is required to run {m_id} locally?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"Running {m_id} ({params_b}B parameters) requires approximately {vram_q4} GB of VRAM in 4-bit quantization (Q4_K_M), {vram_q8} GB in 8-bit (Q8_0), and {vram_fp16} GB in full 16-bit (FP16) precision."
                }
            },
            {
                "@type": "Question",
                "name": f"What is the best GPU to run {short} at high tokens per second?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": f"For {short} ({params_b}B), we recommend {tiers[0]['title']} or an Apple Silicon / Ryzen AI Unified Memory workstation with at least {max(16, int(vram_q8)+4)} GB of memory."
                }
            }
        ]
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{short} ({params_b}B) VRAM Requirements, Context Calculator & Best GPU Workstation Builds</title>
    <meta name="description" content="Exact VRAM requirements for {m_id} ({params_b}B): {vram_q4}GB (4-bit Q4_K_M), {vram_q8}GB (8-bit), and {vram_fp16}GB (FP16). Compare compatible GPUs, Mini PCs, and AI workstations.">
    <link rel="canonical" href="{page_url}">
    <script type="application/ld+json">{json.dumps(schema_data)}</script>
    <style>
        :root {{ --bg: #090d16; --panel: #111827; --border: #1f2937; --text: #f9fafb; --muted: #9ca3af; --accent: #10b981; --accent-dark: #059669; --highlight: #38bdf8; }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; padding-bottom: 60px; }}
        header {{ background: #030712; border-bottom: 1px solid var(--border); padding: 18px 24px; }}
        .nav {{ max-width: 1120px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; }}
        .logo {{ color: var(--text); font-weight: 800; font-size: 1.25rem; text-decoration: none; }}
        .logo span {{ color: var(--accent); }}
        .container {{ max-width: 1120px; margin: 32px auto; padding: 0 20px; }}
        .hero {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 28px; margin-bottom: 28px; }}
        .pills {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }}
        .pill {{ background: #030712; border: 1px solid var(--border); color: var(--highlight); font-size: 0.8rem; padding: 4px 12px; border-radius: 999px; font-weight: 600; }}
        h1 {{ font-size: 1.9rem; line-height: 1.25; margin-bottom: 12px; }}
        .vram-matrix {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 24px 0; }}
        .vram-box {{ background: #030712; border: 1px solid var(--border); border-radius: 10px; padding: 18px; text-align: center; }}
        .vram-box span {{ display: block; color: var(--muted); font-size: 0.85rem; margin-bottom: 6px; }}
        .vram-box strong {{ font-size: 1.8rem; color: var(--accent); }}
        .calc-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 24px; margin-bottom: 36px; }}
        .calc-controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; margin-top: 16px; }}
        label {{ display: block; font-size: 0.88rem; color: var(--muted); margin-bottom: 6px; }}
        select {{ width: 100%; padding: 11px; border-radius: 8px; background: #030712; color: var(--text); border: 1px solid var(--border); font-size: 0.95rem; }}
        .calc-output {{ margin-top: 20px; padding: 16px; background: #030712; border-left: 4px solid var(--accent); border-radius: 8px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
        .section-h2 {{ font-size: 1.45rem; margin: 36px 0 18px; border-left: 4px solid var(--accent); padding-left: 12px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }}
        .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 22px; display: flex; flex-direction: column; justify-content: space-between; }}
        .card-top {{ display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
        .cat {{ font-size: 0.76rem; text-transform: uppercase; color: var(--accent); font-weight: 700; letter-spacing: 0.5px; }}
        .badge {{ background: #030712; color: var(--highlight); font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; border: 1px solid #0284c7; }}
        .card h3 {{ font-size: 1.12rem; margin-bottom: 10px; color: #fff; }}
        .card p {{ color: var(--muted); font-size: 0.92rem; margin-bottom: 18px; }}
        .card-footer {{ margin-top: auto; }}
        .price-pill {{ display: block; font-size: 0.85rem; color: #e2e8f0; margin-bottom: 10px; font-weight: 600; }}
        .btn {{ display: block; text-align: center; background: var(--accent); color: #030712; font-weight: 800; text-decoration: none; padding: 13px 16px; border-radius: 8px; transition: background 0.15s; }}
        .btn:hover {{ background: var(--accent-dark); color: #fff; }}
        .rel-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; margin-top: 16px; }}
        .rel-item {{ background: var(--panel); border: 1px solid var(--border); padding: 14px; border-radius: 10px; text-decoration: none; color: var(--text); display: block; }}
        .rel-item strong {{ display: block; font-size: 0.95rem; margin-bottom: 4px; color: #fff; }}
        .rel-item span {{ font-size: 0.82rem; color: var(--muted); }}
        footer {{ max-width: 1120px; margin: 50px auto 0; padding: 24px 20px 0; border-top: 1px solid var(--border); text-align: center; color: var(--muted); font-size: 0.82rem; }}
    </style>
</head>
<body>
    <header>
        <div class="nav">
            <a href="{SITE_URL}/" class="logo">AI<span>VRAM</span>Bench</a>
            <a href="{SITE_URL}/" style="color: var(--muted); text-decoration: none; font-size: 0.9rem;">&larr; All AI Model Benchmarks</a>
        </div>
    </header>
    <main class="container">
        <section class="hero">
            <div class="pills">
                <span class="pill">Model: {m_id}</span>
                <span class="pill">Parameters: {params_b} Billion</span>
                <span class="pill">Task: {model['pipeline']}</span>
                <span class="pill">Verified: {model['updated_at']}</span>
            </div>
            <h1>{short} ({params_b}B) Local VRAM Requirements & Workstation Hardware Guide</h1>
            <p style="color: var(--muted);">Complete memory footprint breakdown for running <strong>{m_id}</strong> locally via Ollama, vLLM, LM Studio, or llama.cpp, paired with compatible high-bandwidth GPUs and AI workstations.</p>
            <div class="vram-matrix">
                <div class="vram-box">
                    <span>4-Bit Quantization (Q4_K_M / AWQ)</span>
                    <strong>{vram_q4} GB VRAM</strong>
                </div>
                <div class="vram-box">
                    <span>8-Bit Quantization (Q8_0 / EXL2)</span>
                    <strong>{vram_q8} GB VRAM</strong>
                </div>
                <div class="vram-box">
                    <span>Full Half-Precision (FP16 / BF16)</span>
                    <strong>{vram_fp16} GB VRAM</strong>
                </div>
            </div>
        </section>

        <section class="calc-box">
            <h2 style="font-size: 1.25rem;">Interactive {short} Context Window & KV-Cache VRAM Calculator</h2>
            <div class="calc-controls">
                <div>
                    <label for="quantSelect">Select Quantization Precision</label>
                    <select id="quantSelect" onchange="updateVramCalc()">
                        <option value="{vram_q4}">4-Bit (Q4_K_M / GGUF) - Recommended</option>
                        <option value="{vram_q8}">8-Bit (Q8_0) - Near-Lossless</option>
                        <option value="{vram_fp16}">16-Bit (FP16 / BF16) - Full Weights</option>
                    </select>
                </div>
                <div>
                    <label for="ctxSelect">Target Context Length (Tokens)</label>
                    <select id="ctxSelect" onchange="updateVramCalc()">
                        <option value="1.0">4,096 Tokens (Standard Chat)</option>
                        <option value="2.2">16,384 Tokens (RAG / Coding)</option>
                        <option value="4.5">32,768 Tokens (Long Document Analysis)</option>
                        <option value="9.0">65,536+ Tokens (Full Repo Context)</option>
                    </select>
                </div>
            </div>
            <div class="calc-output">
                <div>
                    <span style="color: var(--muted); font-size: 0.85rem;">Total Estimated Memory (Weights + KV Cache):</span>
                    <div id="totalVramText" style="font-size: 1.4rem; font-weight: 800; color: var(--accent);">{round(vram_q4 + 1.0, 1)} GB VRAM Required</div>
                </div>
                <a id="dynamicAmzBtn" href="{make_high_ticket_amazon_url(tiers[0]['query'], tiers[0]['min_cents'])}" target="_blank" rel="nofollow sponsored noopener" class="btn" style="padding: 10px 18px;">View Matching Hardware on Amazon &rarr;</a>
            </div>
        </section>

        <h2 class="section-h2">Compatible High-Performance Hardware Builds ($200+ Tier) for {short}</h2>
        <div class="grid">
            {cards_html}
        </div>

        <h2 class="section-h2">Compare Other Trending AI Model Hardware Specs</h2>
        <div class="rel-grid">
            {related_html}
        </div>
    </main>
    <footer>
        <p>As an Amazon Associate, this platform earns from qualifying purchases. All hardware links automatically filter for workstation-grade components ($200+). Data synced with Hugging Face Hub API on {model['updated_at']}.</p>
    </footer>
    <script>
        function updateVramCalc() {{
            const base = parseFloat(document.getElementById('quantSelect').value);
            const ctxOverhead = parseFloat(document.getElementById('ctxSelect').value) * ({params_b} / 8.0);
            const total = (base + ctxOverhead).toFixed(1);
            document.getElementById('totalVramText').innerText = total + ' GB VRAM Required';
        }}
    </script>
</body>
</html>"""

    filepath = os.path.join(MODELS_DIR, f"{model['slug']}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)


def build_index_and_seo_assets(db: dict):
    os.makedirs(DOCS_DIR, exist_ok=True)
    models_sorted = sorted(db.values(), key=lambda x: x.get("likes", 0), reverse=True)

    rows_html = ""
    for m in models_sorted:
        rows_html += f"""
        <article class="m-card">
            <a href="{SITE_URL}/models/{m['slug']}.html" style="text-decoration: none; color: inherit;">
                <div class="m-top">
                    <span class="m-author">{m['author']}</span>
                    <span class="m-size">{m['params_b']}B Params</span>
                </div>
                <h2>{m['short_name']}</h2>
                <div class="m-specs">
                    <span>4-Bit: <strong>{m['vram_q4']} GB</strong></span>
                    <span>8-Bit: <strong>{m['vram_q8']} GB</strong></span>
                    <span>FP16: <strong>{m['vram_fp16']} GB</strong></span>
                </div>
                <span class="m-cta">View GPU & Workstation Matches &rarr;</span>
            </a>
        </article>
        """

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIVRAMBench - Local LLM & AI Model VRAM Calculator & Workstation Hardware Finder</title>
    <meta name="description" content="Calculate exact VRAM requirements (Q4_K_M, Q8_0, FP16) for trending Hugging Face LLMs and find compatible 16GB/24GB GPUs, Unified Memory Mini PCs, and AI Workstations.">
    <link rel="canonical" href="{SITE_URL}/">
    <style>
        :root {{ --bg: #090d16; --panel: #111827; --border: #1f2937; --text: #f9fafb; --muted: #9ca3af; --accent: #10b981; --highlight: #38bdf8; }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
        header {{ background: #030712; border-bottom: 1px solid var(--border); padding: 36px 20px; text-align: center; }}
        header h1 {{ font-size: 2.1rem; margin-bottom: 10px; }}
        header h1 span {{ color: var(--accent); }}
        header p {{ color: var(--muted); max-width: 720px; margin: 0 auto 20px; }}
        #filterInput {{ width: 100%; max-width: 540px; padding: 13px 16px; border-radius: 8px; border: 1px solid var(--border); background: #090d16; color: #fff; font-size: 1rem; }}
        .container {{ max-width: 1120px; margin: 36px auto; padding: 0 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .m-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; transition: transform 0.15s, border-color 0.15s; }}
        .m-card:hover {{ transform: translateY(-4px); border-color: var(--accent); }}
        .m-top {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.8rem; }}
        .m-author {{ color: var(--muted); }}
        .m-size {{ color: var(--highlight); font-weight: 700; }}
        .m-card h2 {{ font-size: 1.15rem; margin-bottom: 12px; color: #fff; word-break: break-word; }}
        .m-specs {{ display: flex; justify-content: space-between; background: #030712; padding: 10px 12px; border-radius: 8px; font-size: 0.82rem; color: var(--muted); margin-bottom: 14px; }}
        .m-specs strong {{ color: var(--text); }}
        .m-cta {{ font-size: 0.88rem; color: var(--accent); font-weight: 700; }}
        footer {{ text-align: center; padding: 40px 20px; color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--border); margin-top: 60px; }}
    </style>
</head>
<body>
    <header>
        <h1>AI<span>VRAM</span>Bench Hardware Matrix</h1>
        <p>Automated VRAM Sizing (4-Bit GGUF, 8-Bit, FP16) & Workstation GPU Upgrade Finder for Trending Open-Source AI Models</p>
        <input type="text" id="filterInput" placeholder="Search AI model (e.g. Llama, DeepSeek, Qwen, Mistral, 70B, 32B)..." onkeyup="filterModels()">
    </header>
    <main class="container">
        <div class="grid" id="modelsGrid">
            {rows_html}
        </div>
    </main>
    <footer>
        <p>As an Amazon Associate, this site earns from qualifying purchases. Automatically updated every 24 hours via Hugging Face Hub API.</p>
    </footer>
    <script>
        function filterModels() {{
            const q = document.getElementById('filterInput').value.toLowerCase();
            document.querySelectorAll('.m-card').forEach(el => {{
                el.style.display = el.innerText.toLowerCase().includes(q) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>"""

    with open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    # Sitemap XML
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    urls_xml = f"""  <url>
    <loc>{SITE_URL}/</loc>
    <lastmod>{today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>\n"""

    for m in models_sorted:
        urls_xml += f"""  <url>
    <loc>{SITE_URL}/models/{m['slug']}.html</loc>
    <lastmod>{m.get('updated_at', today)}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>\n"""

    sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{urls_xml}</urlset>"""

    with open(os.path.join(DOCS_DIR, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write(sitemap)

    with open(os.path.join(DOCS_DIR, "robots.txt"), "w", encoding="utf-8") as f:
        f.write(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n")

    with open(os.path.join(DOCS_DIR, f"{INDEXNOW_KEY}.txt"), "w", encoding="utf-8") as f:
        f.write(INDEXNOW_KEY)


def ping_indexnow(new_urls: list):
    if not new_urls or "username.github.io" in SITE_URL:
        return
    host = SITE_URL.replace("https://", "").replace("http://", "").split("/")[0]
    payload = {
        "host": host,
        "key": INDEXNOW_KEY,
        "keyLocation": f"{SITE_URL}/{INDEXNOW_KEY}.txt",
        "urlList": new_urls
    }
    try:
        r = requests.post(
            "https://api.indexnow.org/indexnow",
            json=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=15
        )
        print(f"[INFO] IndexNow HTTP Status: {r.status_code}")
    except Exception as e:
        print(f"[WARN] IndexNow Ping Error: {e}")


def main():
    db = load_db()
    candidates = fetch_trending_hf_models()
    print(f"[INFO] Ditemukan {len(candidates)} kandidat model AI dari Hugging Face API.")

    new_urls = []
    added = 0
    # Tambahkan maksimal 20 model baru setiap hari agar pertumbuhan halaman konsisten
    for model_id in candidates:
        if model_id in db:
            continue
        if added >= 20:
            break
        info = fetch_model_details(model_id)
        if info:
            db[model_id] = info
            new_urls.append(f"{SITE_URL}/models/{info['slug']}.html")
            added += 1
            print(f"[OK] Diproses: {model_id} ({info['params_b']}B)")
        time.sleep(0.8)

    all_models_list = list(db.values())
    for m_id, m_data in db.items():
        build_model_page(m_data, all_models_list)

    build_index_and_seo_assets(db)
    save_db(db)

    if new_urls:
        new_urls.append(f"{SITE_URL}/")
        ping_indexnow(new_urls)
    print(f"[DONE] Total halaman aktif: {len(db)} | Halaman baru ditambahkan: {added}")


if __name__ == "__main__":
    main()
