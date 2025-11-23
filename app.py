# ai_research_copilot_hackathon_ui_fixed.py
"""
AI Research Copilot - Hackathon Edition
Enhanced UI, Multi-Citation Styles, Modern Colors

⚠ IMPORTANT SCIENTIFIC NOTE:
This app is designed to generate **conceptual, AI-generated drafts only**.
- No real data are collected or analyzed by this UI.
- The "Results" section is treated as **"Expected Outcomes (Hypothetical)"**.
- Any described studies, methods, or outcomes are hypothetical unless you
  replace them with real data and verified analysis.
"""

import os
import time
import json
import requests
import nest_asyncio
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import gradio as gr

nest_asyncio.apply()

# ===================== CONFIG =====================
# In Colab you can either:
# 1) Set HF_TOKEN in a cell:  HF_TOKEN = "hf_..."
# 2) Or set os.environ["HF_TOKEN"] = "hf_..."

# Try to read from global (if set in notebook), otherwise from env
try:
    HF_TOKEN  # type: ignore[name-defined]
except NameError:
    HF_TOKEN = os.getenv("", None)

try:
    HF_MODEL  # type: ignore[name-defined]
except NameError:
    HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-70B-Instruct")

# Groq not used in this HF-only setup, but keep for future extension
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
HF_VERIFY_MODEL = os.getenv("HF_VERIFY_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None
REQUEST_TIMEOUT = 30

# ===================== UTILS =====================

def safe_get(d, *keys, default=None):
    """Safe nested getter for dict/list structures."""
    try:
        for k in keys:
            if isinstance(d, list) and isinstance(k, int):
                d = d[k]
            else:
                d = d.get(k, {})
        return d if d != {} else default
    except Exception:
        return default

# -------------------- Citation formatting --------------------
def format_authors(authors_list):
    names = []
    for a in authors_list or []:
        if isinstance(a, dict):
            given = a.get("given", "")
            family = a.get("family", "")
        else:
            given, family = str(a), ""
        if given and family:
            names.append(f"{family}, {given[0]}.")
        elif family:
            names.append(family)
    return ", ".join(names)

def format_citation_apa(item):
    authors = format_authors(item.get("author", item.get("authors", [])))
    year = (
        item.get("issued", {}).get("date-parts", [[item.get("year", "")]])[0][0]
        if item.get("issued")
        else item.get("year", "")
    )
    title = (item.get("title") or [""])[0] if isinstance(item.get("title"), list) else item.get("title", "")
    journal = (
        (item.get("container-title") or [""])[0]
        if item.get("container-title")
        else item.get("venue", "")
    )
    doi = item.get("DOI", "") or item.get("doi", "")
    if doi:
        return f"{authors} ({year}). {title}. {journal}. https://doi.org/{doi}"
    return f"{authors} ({year}). {title}. {journal}."

def format_citation_ieee(item):
    authors = format_authors(item.get("author", item.get("authors", [])))
    title = (item.get("title") or [""])[0] if isinstance(item.get("title"), list) else item.get("title", "")
    journal = (
        (item.get("container-title") or [""])[0]
        if item.get("container-title")
        else item.get("venue", "")
    )
    year = (
        item.get("issued", {}).get("date-parts", [[item.get("year", "")]])[0][0]
        if item.get("issued")
        else item.get("year", "")
    )
    return f'{authors}, "{title}," {journal}, {year}.'

def format_citation_mla(item):
    authors = format_authors(item.get("author", item.get("authors", [])))
    title = (item.get("title") or [""])[0] if isinstance(item.get("title"), list) else item.get("title", "")
    journal = (
        (item.get("container-title") or [""])[0]
        if item.get("container-title")
        else item.get("venue", "")
    )
    year = (
        item.get("issued", {}).get("date-parts", [[item.get("year", "")]])[0][0]
        if item.get("issued")
        else item.get("year", "")
    )
    return f'{authors}. "{title}." {journal}, {year}.'

def format_citation(item, style="APA"):
    style = style.upper()
    if style == "APA":
        return format_citation_apa(item)
    elif style == "IEEE":
        return format_citation_ieee(item)
    elif style == "MLA":
        return format_citation_mla(item)
    return format_citation_apa(item)

# -------------------- Section title helper --------------------
def display_section_title(sec_key: str) -> str:
    """
    Map internal section keys to human-friendly headings.

    IMPORTANT: "results" and "expected_outcomes" are both rendered as
    "Expected Outcomes (Hypothetical)" to make it crystal clear that
    this app is NOT reporting real empirical results.
    """
    key = sec_key.lower()
    if key in ("results", "expected_outcomes"):
        return "Expected Outcomes (Hypothetical)"
    return sec_key.replace("_", " ").title()

# ===================== HF INFERENCE CALL =====================

def call_hf_inference(prompt: str, max_new_tokens: int = 900) -> Optional[str]:
    """
    Lightweight wrapper for Hugging Face text-generation Inference API.
    Returns the generated text or None on error.
    """
    if not HF_TOKEN:
        print("HF_TOKEN is not set. Cannot call Hugging Face Inference API.")
        return None

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.6,
            "top_p": 0.9,
            "return_full_text": False,
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # HF text-gen usually returns a list with {'generated_text': '...'}
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        # Some models may return dict with 'generated_text'
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        print("Unexpected HF response format:", data)
        return None
    except Exception as e:
        print("HF inference error:", e)
        return None

# ===================== OFFLINE FALLBACK GENERATOR =====================

def offline_fallback(topic: str, num_refs: int) -> Dict[str, Any]:
    """
    Simple offline generator so the app still works without HF_TOKEN or internet.
    Everything is clearly conceptual and generic.
    """
    base = topic if topic.strip() else "the chosen research topic"

    sections = {
        "abstract": (
            f"This conceptual paper provides an overview of {base}, outlining key ideas, "
            "debates, and open questions. It does not present empirical findings but "
            "instead sketches a high-level research direction."
        ),
        "introduction": (
            f"{base} has become an increasingly important subject in contemporary research. "
            "This paper introduces the topic, explains why it matters, and highlights the "
            "core issues that motivate further investigation."
        ),
        "literature_review": (
            f"The existing literature on {base} spans multiple disciplines and perspectives. "
            "Researchers have discussed theoretical foundations, practical applications, and "
            "ethical implications, while also identifying unresolved tensions and gaps. "
            "This section summarizes common themes and recurring arguments at a high level."
        ),
        "methodology": (
            f"This paper proposes a **conceptual methodology** for studying {base}. "
            "Rather than describing a completed experiment, it outlines a possible design, "
            "including research questions, qualitative and/or quantitative approaches, and "
            "considerations for sampling, measurement, and analysis. These ideas should be "
            "treated as a starting point for a future empirical study."
        ),
        "expected_outcomes": (
            f"In line with the proposed methodology, we outline **expected outcomes** that are "
            "purely hypothetical. We anticipate that the study would generate insights into "
            f"how {base} influences key variables of interest, reveal patterns in participants' "
            "experiences, and point to new directions for theory and practice. These outcomes "
            "are speculative and must be validated by actual research."
        ),
        "conclusion": (
            f"In conclusion, this conceptual paper organizes current thinking about {base}, "
            "proposes a research design for future work, and sketches possible findings and "
            "implications. By treating all outcomes as hypothetical, it emphasizes the need for "
            "rigorous empirical studies before drawing firm conclusions."
        ),
    }

    # Very generic "references" as placeholders (no DOIs).
    references = []
    for i in range(1, min(num_refs, 5) + 1):
        references.append(
            {
                "title": f"Conceptual perspectives on {base} ({i})",
                "author": [{"given": "A.", "family": f"Researcher{i}"}],
                "year": "",
                "venue": "Unspecified Journal",
                "DOI": "",
            }
        )

    # Simple BibTeX
    bib_entries = []
    for i, r in enumerate(references, start=1):
        key = f"ref{i}"
        title = r.get("title", "").replace("{", "").replace("}", "")
        year = r.get("year", "")
        journal = r.get("venue", "").replace("{", "").replace("}", "")
        doi = r.get("DOI", "")
        auth_objs = r.get("author", [])
        authors_str = " and ".join(
            [
                f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                for a in auth_objs
            ]
        )
        entry = "@article{{{key},\n  title={{ {title} }},\n  author={{ {authors} }},\n  journal={{ {journal} }},\n  year={{ {year} }},\n  doi={{ {doi} }}\n}}".format(
            key=key,
            title=title,
            authors=authors_str,
            journal=journal,
            year=year,
            doi=doi,
        )
        bib_entries.append(entry)

    return {
        "topic": topic or "Untitled",
        "sections": sections,
        "references": references,
        "bibtex": "\n\n".join(bib_entries),
    }

# ===================== PAPER GENERATION PIPELINE =====================

def generate_full_paper_pipeline(topic: str, num_refs: int = 20, use_groq: bool = False) -> Dict[str, Any]:
    """
    Generates a conceptual research paper draft.

    Priority:
    - If HF_TOKEN is present and HF call works → use HF model.
    - Otherwise → use offline_fallback so the app always works.

    SAFETY RULES:
    - No fabricated empirical data, statistics, sample sizes, p-values, etc.
    - All outcomes must be framed as hypothetical / conceptual.
    - The 'results' content is stored under 'expected_outcomes'.
    - References are suggested readings, NOT guaranteed accurate citations.
    """

    # If HF is not configured, go straight to offline fallback
    if not HF_TOKEN:
        print("HF_TOKEN missing → using offline fallback generator.")
        return offline_fallback(topic, num_refs)

    # Prompt for structured JSON response
    system_instructions = f"""
You are an expert academic writing assistant.

TASK:
Generate a **conceptual** research paper draft about the topic: "{topic}".

CRITICAL RULES:
- This is **not** an empirical paper.
- Do NOT fabricate any specific statistics, sample sizes, p-values, accuracies, or numeric results.
- All described studies, methods, or outcomes must be clearly hypothetical or conceptual.
- Treat any "results" as **Expected Outcomes (Hypothetical)** only.
- Do not claim that a real study was run, data were collected, or experiments were performed.

OUTPUT FORMAT:
Reply with VALID JSON ONLY (no extra text), following EXACTLY this schema:

{{
  "topic": "...",
  "sections": {{
    "abstract": "...",
    "introduction": "...",
    "literature_review": "...",
    "methodology": "...",
    "expected_outcomes": "...",
    "conclusion": "..."
  }},
  "references": [
    {{
      "title": "...",
      "authors": [{{"given": "...", "family": "..."}}],
      "year": 2023,
      "venue": "...",
      "doi": ""
    }}
    // up to {num_refs} entries total
  ]
}}

NOTES:
- All sections should be written in formal academic English.
- Make the "methodology" a **proposed study design**, not a description of a completed experiment.
- In "expected_outcomes", describe plausible, hypothetical outcomes in qualitative terms only
  (e.g., "we expect that", "likely", "may lead to"), without numbers.
- For references:
  - Suggest real-sounding scholarly works if you can.
  - If you are not certain of a real DOI, leave "doi" as an empty string "" rather than guessing.
  - It is better to leave "doi" empty than to invent one.
- Do not include comments or explanations outside the JSON.
"""

    raw = call_hf_inference(system_instructions)
    if raw is None:
        print("HF call failed → using offline fallback generator.")
        return offline_fallback(topic, num_refs)

    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            text = text[first:last+1]

    try:
        data = json.loads(text)
    except Exception as e:
        print("JSON parse failed:", e)
        return offline_fallback(topic, num_refs)

    topic_out = data.get("topic", topic or "Untitled")
    sections_raw = data.get("sections", {})
    sections = {
        "abstract": sections_raw.get("abstract", "No abstract generated."),
        "introduction": sections_raw.get("introduction", "No introduction generated."),
        "literature_review": sections_raw.get("literature_review", "No literature review generated."),
        "methodology": sections_raw.get("methodology", "No methodology generated."),
        "expected_outcomes": sections_raw.get("expected_outcomes", sections_raw.get("results", "No expected outcomes generated.")),
        "conclusion": sections_raw.get("conclusion", "No conclusion generated."),
    }

    refs_raw = data.get("references", []) or []
    references: List[Dict[str, Any]] = []
    for r in refs_raw[:num_refs]:
        title = r.get("title", "")
        authors = r.get("authors", r.get("author", [])) or []
        year = r.get("year", "")
        venue = r.get("venue", "")
        doi = r.get("doi", "") or r.get("DOI", "")

        norm_authors = []
        if isinstance(authors, list):
            for a in authors:
                if isinstance(a, dict):
                    norm_authors.append(
                        {
                            "given": a.get("given", ""),
                            "family": a.get("family", ""),
                        }
                    )
                else:
                    parts = str(a).split()
                    given = " ".join(parts[:-1]) if len(parts) > 1 else ""
                    family = parts[-1] if parts else ""
                    norm_authors.append({"given": given, "family": family})
        elif isinstance(authors, dict):
            norm_authors.append(
                {"given": authors.get("given", ""), "family": authors.get("family", "")}
            )

        ref_obj = {
            "title": title,
            "author": norm_authors,
            "year": year,
            "venue": venue,
            "DOI": doi,
        }
        references.append(ref_obj)

    bib_entries = []
    for i, r in enumerate(references, start=1):
        key = f"ref{i}"
        title = r.get("title", "").replace("{", "").replace("}", "")
        year = r.get("year", "")
        journal = r.get("venue", "").replace("{", "").replace("}", "")
        doi = r.get("DOI", "")
        auth_objs = r.get("author", [])
        authors_str = " and ".join(
            [
                f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                for a in auth_objs
            ]
        )
        entry = "@article{{{key},\n  title={{ {title} }},\n  author={{ {authors} }},\n  journal={{ {journal} }},\n  year={{ {year} }},\n  doi={{ {doi} }}\n}}".format(
            key=key,
            title=title,
            authors=authors_str,
            journal=journal,
            year=year,
            doi=doi,
        )
        bib_entries.append(entry)
    bibtex = "\n\n".join(bib_entries)

    return {
        "topic": topic_out,
        "sections": sections,
        "references": references,
        "bibtex": bibtex,
    }

# ===================== PDF Export =====================

def export_pdf_clean(out_json: Dict[str, Any], filename: str = "generated_paper.pdf") -> str:
    try:
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )
        styles = getSampleStyleSheet()
        Story = []

        Story.append(Paragraph(out_json.get("topic", "Untitled"), styles["Title"]))
        Story.append(Spacer(1, 12))

        Story.append(Paragraph("AI Research Copilot — Generated Draft", styles["Normal"]))
        Story.append(Spacer(1, 6))
        Story.append(
            Paragraph(
                "<i>Important: This is an AI-generated conceptual draft. "
                "Any described methods or outcomes are hypothetical unless you "
                "replace them with real data and verified analysis.</i>",
                styles["BodyText"],
            )
        )
        Story.append(Spacer(1, 12))

        sections = out_json.get("sections", {})
        section_order = [
            "abstract",
            "introduction",
            "literature_review",
            "methodology",
            "expected_outcomes",
            "results",
            "conclusion",
        ]
        used = set()

        for sec in section_order:
            t = sections.get(sec, "")
            if t:
                used.add(sec)
                Story.append(Paragraph(display_section_title(sec), styles["Heading2"]))
                Story.append(Paragraph(t.replace("\n", "<br/>"), styles["BodyText"]))
                Story.append(Spacer(1, 12))

        for sec, t in sections.items():
            if sec in used or not t:
                continue
            Story.append(Paragraph(display_section_title(sec), styles["Heading2"]))
            Story.append(Paragraph(t.replace("\n", "<br/>"), styles["BodyText"]))
            Story.append(Spacer(1, 12))

        Story.append(Paragraph("References (Suggested Reading)", styles["Heading2"]))
        Story.append(
            Paragraph(
                "<i>Note: These references are suggested by the model. "
                "Entries with DOIs are marked as more easily verifiable, "
                "but you must still check each source manually.</i>",
                styles["BodyText"],
            )
        )
        Story.append(Spacer(1, 12))

        for i, r in enumerate(out_json.get("references", []), start=1):
            ref_text = format_citation(r, style="APA")
            Story.append(Paragraph(f"{i}. {ref_text}", styles["BodyText"]))
            Story.append(Spacer(1, 6))

        Story.append(Spacer(1, 12))
        Story.append(
            Paragraph(
                "<i>This draft must not be treated as a completed empirical "
                "paper. Replace hypothetical content with real data, methods, "
                "and citations before any academic or professional use.</i>",
                styles["BodyText"],
            )
        )

        doc.build(Story)
        return filename
    except Exception as e:
        print("PDF export failed:", e)
        return ""

# ===================== UI PIPELINE =====================

def run_ui_pipeline(topic: str, num_refs: int, style: str, use_groq: bool):
    start = time.time()
    if not topic.strip():
        return "Please enter a research topic.", "<div>No references</div>", None, None

    out_json = generate_full_paper_pipeline(topic, num_refs=num_refs, use_groq=use_groq)

    if "results" in out_json.get("sections", {}) and "expected_outcomes" not in out_json["sections"]:
        out_json["sections"]["expected_outcomes"] = out_json["sections"].pop("results")

    safe_name = "".join(c for c in topic if c.isalnum() or c in (" ", "")).strip()[:40].replace(" ", "") or "paper"
    pdf_path = export_pdf_clean(out_json, filename=f"{safe_name}_generated.pdf")

    bib_path = f"{safe_name}_references.bib"
    try:
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(out_json.get("bibtex", ""))
    except Exception as e:
        print("Failed to write bib file:", e)
        bib_path = None

    refs_rows = []
    for i, r in enumerate(out_json.get("references", [])):
        title = (r.get("title") or [""])[0] if isinstance(r.get("title"), list) else r.get("title", "")
        authors_list = r.get("author", r.get("authors", []))
        if isinstance(authors_list, list):
            authors_str = ", ".join(
                [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors_list]
            )
        else:
            authors_str = str(authors_list)
        has_doi = bool(r.get("DOI") or r.get("doi"))
        refs_rows.append({"index": i + 1, "title": title, "authors": authors_str, "has_doi": has_doi})

    table_html = (
        "<div class='table-container'>"
        "<p><i>Note: 'Has DOI?' is only a lightweight signal. Always open and check each paper manually.</i></p>"
        "<table><tr><th>#</th><th>Title</th><th>Authors</th><th>Has DOI?</th></tr>"
    )
    for r in refs_rows:
        check = "✔" if r["has_doi"] else "—"
        table_html += (
            f"<tr><td>{r['index']}</td>"
            f"<td>{r['title']}</td>"
            f"<td>{r['authors']}</td>"
            f"<td class='center'>{check}</td></tr>"
        )
    table_html += "</table></div>"

    disclaimer_md = (
        "**Important:** This is an AI-generated *conceptual* draft. "
        "Any described studies, methods, or outcomes are hypothetical unless "
        "you replace them with real data and verified analysis.\n\n"
        "> Do **not** submit this as empirical work without performing and "
        "reporting actual studies."
    )

    sections_md = f"## {topic}\n\n{disclaimer_md}\n\n"

    sections = out_json.get("sections", {})
    ordered_keys = [
        "abstract",
        "introduction",
        "literature_review",
        "methodology",
        "expected_outcomes",
        "conclusion",
    ]
    used = set()

    for sec in ordered_keys:
        text = sections.get(sec, "")
        if text:
            used.add(sec)
            sections_md += f"### {display_section_title(sec)}\n\n{text}\n\n"

    for sec, text in sections.items():
        if sec in used or not text:
            continue
        sections_md += f"### {display_section_title(sec)}\n\n{text}\n\n"

    took = time.time() - start
    sections_md += (
        f"\n_Generated in {took:.1f}s._\n\n"
        "> References shown in the next tab are suggestions only. "
        "Always verify them against publisher or indexing databases before citing."
    )

    return sections_md, table_html, bib_path, pdf_path

# ===================== GRADIO UI =====================

css = """
.gradio-container { 
    font-family:'Arial', sans-serif; 
    background-color:#edf2f7; 
    padding:25px; 
}

/* FORCE ALL OUTPUT TEXT TO BLACK */
.gr-markdown, .gr-html, .markdown-body, .output-markdown, .output-html, .prose * {
    color: #000 !important;
}

.input-card { 
    padding:25px; 
    background-color:#ffffff; 
    border-radius:15px; 
    box-shadow:0 6px 20px rgba(0,0,0,0.1); 
    margin-bottom:25px; 
}
.table-container { overflow-x:auto; margin-top:12px; }
.table-container table { width:100%; border-collapse: collapse; font-size:14px; background-color:#fff; }
.table-container th, .table-container td { border:1px solid #ccc; padding:8px; text-align:left; }
.table-container td.center { text-align:center; }
.table-container tr:hover { background-color:#f1f1f1; }
.gr-button { border-radius:10px; font-weight:bold; padding:10px 20px; background-color:#3498db; color:white; border:none; }
.gr-button:hover { background-color:#2980b9; }
"""

with gr.Blocks(css=css, title="AI Research Copilot (Conceptual Drafts)") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="input-card"):
                topic_in = gr.Textbox(
                    label="Research Topic",
                    placeholder="e.g. 'Explainable AI and Human Trust'",
                )
                refs_in = gr.Slider(
                    label="Number of References",
                    minimum=5,
                    maximum=100,
                    step=5,
                    value=20,
                )
                style_in = gr.Dropdown(
                    choices=["APA", "IEEE", "MLA"],
                    value="APA",
                    label="Citation Style (for display/export)",
                )
                groq_in = gr.Checkbox(
                    label="Use Groq for generation (ignored in this HF/offline version)",
                    value=False,
                )
                run_btn = gr.Button("Generate Conceptual Paper", variant="primary")

            with gr.Group():
                bib_file = gr.File(label="Download BibTeX", file_types=[".bib"])
                pdf_file = gr.File(label="Download PDF", file_types=[".pdf"])

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Paper Sections (Conceptual)"):
                    abstract_md = gr.Markdown()
                with gr.Tab("References Table"):
                    refs_html = gr.HTML()

    run_btn.click(
        fn=run_ui_pipeline,
        inputs=[topic_in, refs_in, style_in, groq_in],
        outputs=[abstract_md, refs_html, bib_file, pdf_file],
    )

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)

