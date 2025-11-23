# ai_research_copilot_final.py
"""
AI Research Copilot — Groq primary, Hugging Face fallback.

"""

import os
import time
import re
import nest_asyncio
from typing import Dict, Any, List
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import gradio as gr
import requests

nest_asyncio.apply()

# ---------------- CONFIG ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip() or None
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_ROUTER_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"

# Tunables
REQUEST_TIMEOUT = 180
MAX_TOKENS_GROQ = 8000        # allow full paper generation
MAX_NEW_TOKENS_HF = 8000

SECTIONS = ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Conclusion"]

# ---------------- Helpers ----------------
def format_authors(authors_list):
    out=[]
    for a in authors_list or []:
        if isinstance(a, dict):
            given=a.get("given","")
            family=a.get("family","")
            if family and given: out.append(f"{family}, {given[0]}.")
            elif family: out.append(family)
            else: out.append(given)
        else: out.append(str(a))
    return ", ".join(out)

def format_citation_apa(item):
    authors=format_authors(item.get("author",[]))
    year=item.get("issued",{}).get("date-parts",[[None]])[0][0] or ""
    title=(item.get("title") or [""])[0] if isinstance(item.get("title"), list) else item.get("title","")
    journal=item.get("container-title","") or item.get("venue","")
    doi=item.get("DOI","") or item.get("doi","")
    if doi: return f"{authors} ({year}). {title}. {journal}. https://doi.org/{doi}"
    return f"{authors} ({year}). {title}. {journal}."

def export_pdf_clean(out_json: Dict[str,Any], filename: str) -> str:
    try:
        doc=SimpleDocTemplate(filename,pagesize=A4,rightMargin=40,leftMargin=40,topMargin=40,bottomMargin=40)
        styles=getSampleStyleSheet()
        Story=[]
        Story.append(Paragraph(out_json.get("topic","Untitled"),styles["Title"]))
        Story.append(Spacer(1,12))
        Story.append(Paragraph("AI Research Copilot — Generated Draft",styles["Normal"]))
        Story.append(Spacer(1,12))
        for sec in SECTIONS:
            t=out_json.get("sections",{}).get(sec.lower().replace(" ","_"),"")
            if t:
                Story.append(Paragraph(sec,styles["Heading2"]))
                paras=[p.strip() for p in re.split(r'\n\s*\n',t) if p.strip()]
                for p in paras:
                    Story.append(Paragraph(p.replace("\n"," "),styles["BodyText"]))
                    Story.append(Spacer(1,6))
                Story.append(Spacer(1,12))
        Story.append(Paragraph("References",styles["Heading2"]))
        for i,r in enumerate(out_json.get("references",[]),start=1):
            ref_text=format_citation_apa(r)
            Story.append(Paragraph(f"{i}. {ref_text}",styles["BodyText"]))
            Story.append(Spacer(1,6))
        doc.build(Story)
        return filename
    except Exception as e:
        print("PDF export failed:",e)
        return ""

# ---------------- Model calls ----------------
def call_groq_chat(prompt: str, model: str = GROQ_MODEL, max_tokens: int = MAX_TOKENS_GROQ) -> str:
    if not GROQ_API_KEY: return ""
    url=GROQ_ENDPOINT
    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type":"application/json"}
    payload={"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0.2}
    try:
        r=requests.post(url,headers=headers,json=payload,timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data=r.json()
        if isinstance(data,dict) and "choices" in data and len(data["choices"])>0:
            ch=data["choices"][0]
            if "message" in ch and isinstance(ch["message"],dict): return ch["message"].get("content","") or ch.get("text","")
            return ch.get("text","") or ch.get("message",{}).get("content","")
        return data.get("text","") or ""
    except Exception as e:
        print("Groq call failed:",e)
        return ""

def call_hf_router_chat(prompt: str, model: str = HF_MODEL, max_new_tokens: int = MAX_NEW_TOKENS_HF) -> str:
    if not HF_TOKEN: return ""
    url=HF_ROUTER_ENDPOINT
    headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type":"application/json"}
    payload={"model":model,"messages":[{"role":"user","content":prompt}],"max_new_tokens":max_new_tokens,"temperature":0.2}
    try:
        r=requests.post(url,headers=headers,json=payload,timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data=r.json()
        if isinstance(data,dict) and "choices" in data and len(data["choices"])>0:
            return data["choices"][0].get("message",{}).get("content","") or data["choices"][0].get("text","")
        if isinstance(data,list) and len(data)>0: return data[0].get("generated_text","")
        return ""
    except Exception as e:
        print("HF router call failed:",e)
        return ""

# ---------------- CrossRef API to fetch real references ----------------
def fetch_real_refs(topic: str, max_results: int = 10) -> List[Dict[str,Any]]:
    url=f"https://api.crossref.org/works?query={topic}&rows={max_results}"
    try:
        r=requests.get(url,timeout=30)
        items=r.json().get('message',{}).get('items',[])
        refs=[]
        for i in items:
            ref={
                "title": [i.get("title",[""])[0]],
                "author": [{"given":a.get("given",""),"family":a.get("family","")} for a in i.get("author",[])] if i.get("author") else [],
                "issued": {"date-parts":[[i.get("published-print",{}).get("date-parts",[[0]])[0][0] or 0]]},
                "DOI": i.get("DOI","")
            }
            refs.append(ref)
        return refs
    except Exception as e:
        print("CrossRef fetch failed:",e)
        return []

# ---------------- Pipeline ----------------
def generate_full_paper_pipeline(topic: str, num_refs: int = 10, use_groq: bool = True) -> Dict[str,Any]:
    sections_text = {}
    for sec in SECTIONS:
        prompt = f"Write the {sec} section for a research paper on '{topic}'. Use multiple paragraphs. Academic style."
        text = ""
        if use_groq and GROQ_API_KEY: text = call_groq_chat(prompt, model=GROQ_MODEL)
        if (not text or len(text.strip())<200) and HF_TOKEN: text = call_hf_router_chat(prompt, model=HF_MODEL)
        sections_text[sec.lower().replace(" ","_")] = text or f"[{sec} generation failed]"

    # Fetch *real references* from CrossRef
    references = fetch_real_refs(topic, max_results=num_refs)

    return {"topic": topic, "sections": sections_text, "references": references, "bibtex": ""}  # bibtex kept empty; real references used in PDF

# ---------------- UI ----------------
def run_ui_pipeline(topic: str, num_refs: int, style: str, use_groq: bool):
    start=time.time()
    if not topic or not topic.strip(): return "Please enter a research topic.","<div>No references</div>",None,None
    out=generate_full_paper_pipeline(topic.strip(), num_refs=num_refs, use_groq=use_groq)

    safe_name="".join(c for c in topic if c.isalnum() or c in (" ","")).strip()[:60].replace(" ","") or "paper"
    pdf_path=f"{safe_name}_generated.pdf"
    export_pdf_clean(out,pdf_path)
    pdf_file_path=pdf_path if Path(pdf_path).exists() else None

    bib_path=None
    # Optional: save references as .bib using real DOIs
    if out.get("references"):
        bib_path=f"{safe_name}_references.bib"
        with open(bib_path,"w",encoding="utf-8") as f:
            for r in out["references"]:
                f.write(f"@article{{{r['DOI'].replace('/','_')},\n")
                f.write(f"  title={{ {r['title'][0]} }},\n")
                authors = " and ".join([f"{a['family']}, {a['given']}" for a in r['author']])
                f.write(f"  author={{ {authors} }},\n")
                year = r.get("issued",{}).get("date-parts",[[None]])[0][0] or ""
                f.write(f"  year={{ {year} }},\n")
                f.write(f"  doi={{ {r['DOI']} }}\n}}\n\n")

    rows=[]
    for i,r in enumerate(out.get("references",[]),start=1):
        title=(r.get("title") or [""])[0] if isinstance(r.get("title"),list) else r.get("title","")
        authors_list=r.get("author",[])
        authors_str=", ".join([f"{a.get('given','')} {a.get('family','')}".strip() for a in authors_list]) if isinstance(authors_list,list) else str(authors_list)
        valid=bool(r.get("DOI"))
        rows.append({"index":i,"title":title,"authors":authors_str,"valid":valid})

    table_html="<div class='table-container'><table><tr><th>#</th><th>Title</th><th>Authors</th><th>Verified</th></tr>"
    if rows:
        for r in rows:
            check="✔" if r['valid'] else "—"
            table_html+=f"<tr><td>{r['index']}</td><td>{r['title']}</td><td>{r['authors']}</td><td class='center'>{check}</td></tr>"
    else: table_html+="<tr><td colspan='4'>No references found.</td></tr>"
    table_html+="</table></div>"

    sections_md=f"## {topic}\n\n"
    for sec,text in out.get("sections",{}).items():
        sections_md+=f"### {sec.replace('_',' ').title()}\n\n{text}\n\n"
    took=time.time()-start
    sections_md+=f"\n\n_Generated in {took:.1f}s._"
    return sections_md,table_html,(bib_path if bib_path else None),(pdf_file_path if pdf_file_path else None)

# ---------------- Gradio UI ----------------
css="""
.gradio-container { font-family:'Arial', sans-serif; background-color:#f7fafc; padding:18px; }
.input-card { padding:18px; background-color:#ffffff; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.06); margin-bottom:18px; }
.table-container { overflow-x:auto; margin-top:12px; }
.table-container table { width:100%; border-collapse: collapse; font-size:14px; background-color:#fff; }
.table-container th, .table-container td { border:1px solid #e2e8f0; padding:8px; text-align:left; }
.table-container td.center { text-align:center; }
"""

with gr.Blocks(css=css, title="AI Research Copilot (Groq + HF)") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="input-card"):
                topic_in = gr.Textbox(label="Research Topic", placeholder="e.g. AI in Education")
                refs_in = gr.Slider(label="Number of References", minimum=5, maximum=100, step=1, value=10)
                style_in = gr.Dropdown(choices=['APA','IEEE','MLA'], value='APA', label='Citation Style')
                groq_in = gr.Checkbox(label="Use Groq primary (fallback to HF)", value=True)
                run_btn = gr.Button("Generate Paper", variant="primary")
            with gr.Group():
                bib_file = gr.File(label="Download BibTeX", file_types=[".bib"])
                pdf_file = gr.File(label="Download PDF", file_types=[".pdf"])
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Paper Sections"):
                    abstract_md = gr.Markdown()
                with gr.Tab("References Table"):
                    refs_html = gr.HTML()
    run_btn.click(fn=run_ui_pipeline,
                  inputs=[topic_in, refs_in, style_in, groq_in],
                  outputs=[abstract_md, refs_html, bib_file, pdf_file])

if _name=="main_":
    demo.launch(share=True,inbrowser=True)
