import os
import json
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI
import re

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

# ============================================================
# LEGAL DETECTIVE - ROBUST DOCUMENT ANALYZER
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

# Common legal document formats. The AI receives the original file,
# so scanned/image PDFs can also be examined instead of relying only
# on pypdf text extraction.
ALLOWED_EXTENSIONS = {
    "pdf", "docx", "txt", "md", "rtf"
}

MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")
API_KEY = os.environ.get("OPENAI_API_KEY")
# Classroom/demo mode is ON by default. It requires no API key or card.
# Set DEMO_MODE=false later if you intentionally connect a paid AI API.
DEMO_MODE = os.environ.get("DEMO_MODE", "true").lower() in {"1", "true", "yes", "on"}

client = OpenAI(api_key=API_KEY) if API_KEY else None

# Temporary server-side document registry.
# key = our random document id
# value = OpenAI file id + local path
DOCUMENTS = {}


# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def parse_json(text):
    """Safely parse JSON returned by the model."""
    if not text:
        return {}

    text = text.strip()

    # Remove accidental markdown fences.
    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "", 1)
        if text.endswith("```"):
            text = text[:-3]

    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Last-resort object extraction.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {}


def require_client():
    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add OPENAI_API_KEY to Render Environment Variables."
        )


def upload_to_openai(path):
    """Upload the original document so the model can inspect text and pages."""
    require_client()

    with open(path, "rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="user_data"
        )

    return uploaded.id


def ask_model(instructions, content):
    require_client()

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return response.output_text


def ask_model_with_file(instructions, text_prompt, openai_file_id):
    """
    Send the original uploaded file to the model.
    This is important for scanned PDFs because the model can inspect
    the document itself instead of depending only on extracted text.
    """
    require_client()

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": openai_file_id
                    },
                    {
                        "type": "input_text",
                        "text": text_prompt
                    }
                ]
            }
        ]
    )

    return response.output_text



def extract_local_text(path):
    """Extract enough local text for the no-card classroom demo."""
    suffix = Path(path).suffix.lower()
    try:
        if suffix == ".txt" or suffix == ".md":
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        if suffix == ".rtf":
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
            raw = re.sub(r"\\[a-zA-Z]+-?\\d* ?", " ", raw)
            return re.sub(r"[{}]", " ", raw)
        if suffix == ".docx" and DocxDocument:
            doc = DocxDocument(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if suffix == ".pdf" and PdfReader:
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        print("LOCAL TEXT EXTRACTION ERROR:", repr(exc))
    return ""


def _clean_sentence(text, limit=220):
    text = re.sub(r"\\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _find_sentences(text, keywords, limit=5):
    chunks = re.split(r"(?<=[.!?])\\s+|\\n+", text or "")
    out = []
    for chunk in chunks:
        c = _clean_sentence(chunk)
        low = c.lower()
        if c and any(k in low for k in keywords) and c not in out:
            out.append(c)
        if len(out) >= limit:
            break
    return out


def demo_classify(text, filename):
    combined = ((filename or "") + "\n" + (text or "")).lower()
    legal_terms = [
        "agreement", "contract", "lease", "rent", "tenant", "landlord",
        "party", "parties", "whereas", "hereinafter", "shall", "liable",
        "court", "judgment", "order", "petition", "plaint", "affidavit",
        "deed", "notice", "legal notice", "act,", "section ", "statute",
        "regulation", "law", "jurisdiction", "arbitration", "indemnity",
        "confidentiality", "employment", "employee", "employer", "witness",
        "consideration", "termination", "governing law", "power of attorney"
    ]
    hits = [k for k in legal_terms if k in combined]
    is_legal = len(hits) >= 2 or any(x in (filename or "").lower() for x in ["agreement", "lease", "deed", "contract", "notice", "judgment", "petition", "affidavit", "legal"])
    if not is_legal:
        return {
            "is_legal": False,
            "classification": "not_legal",
            "document_type": "General / non-legal document",
            "confidence": 88,
            "reason": "The document does not contain enough clear legal-document indicators for this classroom prototype.",
            "jurisdiction": "Not specified"
        }
    low = combined
    if any(x in low for x in ["rent", "tenant", "landlord", "lease"]): dtype = "Rental / Lease Agreement"
    elif any(x in low for x in ["employment", "employee", "employer", "salary"]): dtype = "Employment Agreement"
    elif any(x in low for x in ["judgment", "court order", "decree"]): dtype = "Judgment / Court Order"
    elif any(x in low for x in ["affidavit", "sworn statement"]): dtype = "Affidavit / Declaration"
    elif any(x in low for x in ["petition", "plaint", "written statement"]): dtype = "Court Pleading / Petition"
    elif "deed" in low: dtype = "Deed / Property Document"
    elif "notice" in low: dtype = "Legal Notice"
    elif any(x in low for x in ["act,", "statute", "section ", "bill"]): dtype = "Statute / Legal Instrument"
    else: dtype = "Legal Agreement / Document"
    return {
        "is_legal": True,
        "classification": "legal",
        "document_type": dtype,
        "confidence": min(98, 78 + min(len(hits) * 2, 20)),
        "reason": "Legal terms and document structure were detected. This classroom prototype will demonstrate the analysis workflow.",
        "jurisdiction": "Not specified in the document"
    }


def demo_analysis(path, filename):
    text = extract_local_text(path)
    low = text.lower()
    classification = demo_classify(text, filename)
    dtype = classification["document_type"]

    parties = _find_sentences(text, ["landlord", "tenant", "employer", "employee", "party", "parties"], 4)
    dates = _find_sentences(text, ["date", "dated", "commence", "effective", "expiry", "termination", "notice period"], 5)
    money = _find_sentences(text, ["rent", "salary", "amount", "fee", "deposit", "payment", "consideration", "penalty"], 6)
    clauses = _find_sentences(text, ["shall", "must", "may", "termination", "notice", "indemn", "confidential", "dispute", "arbitration"], 8)
    laws = _find_sentences(text, ["act,", "section ", "statute", "regulation", "rule ", "article ", "law"], 5)

    rights = []
    obligations = []
    risks = []
    if any(x in low for x in ["tenant", "rent", "lease"]):
        rights.append("The document appears to define rights connected with possession/use of the premises and the rental relationship.")
        obligations.append("The parties appear to have payment, maintenance, notice, or compliance duties described in the document.")
        if "security deposit" in low or "deposit" in low:
            risks.append("Check the security-deposit amount, refund conditions, deductions, and return timeline carefully.")
    if "termination" in low or "notice period" in low:
        risks.append("Termination and notice provisions should be checked for the required notice period and conditions.")
    if "penalty" in low or "late fee" in low:
        risks.append("Payment-default or penalty language deserves attention because it may create additional financial exposure.")
    if "arbitration" in low:
        risks.append("An arbitration provision may affect how disputes are resolved and where proceedings occur.")
    if not risks:
        risks.append("Review the document for unclear obligations, deadlines, payment terms, and dispute provisions before relying on it.")

    summary = f"This classroom prototype identified the upload as a {dtype}. It found legal-style provisions and organized them into a reader-friendly report."
    simple = "In simple terms, Legal Detective is highlighting what the document is about, who appears to be involved, what they may need to do, important money/dates, and areas that deserve closer attention."
    if text:
        summary += " The report below is based on text that could be extracted from the uploaded file."
    else:
        summary += " The file appears to be image-based or difficult to extract locally, so the demo uses a general document workflow rather than pretending to read unavailable text."

    return normalize_analysis({
        "title": filename.rsplit(".", 1)[0].replace("_", " ").title() or "Legal document",
        "document_type": dtype,
        "jurisdiction": classification.get("jurisdiction", "Not specified"),
        "language": "English / detected from document text" if text else "Not detected",
        "summary": summary,
        "simple_explanation": simple,
        "parties": parties or ["Parties are not clearly extractable from this document."],
        "legal_references": laws or ["No specific Act, section, or legal authority was clearly extracted in demo mode."],
        "important_terms": ["Legal document classification", "Rights and obligations", "Important dates", "Financial terms", "Risk / attention points"],
        "financial_information": money or ["No clear financial amount was extracted in demo mode."],
        "important_dates": dates or ["No clear date was extracted in demo mode."],
        "key_clauses": clauses or ["Key legal provisions could not be confidently extracted from the available text."],
        "rights": rights or ["The document may define rights between its parties; review the source wording for exact details."],
        "obligations": obligations or ["The document may create duties for one or more parties; review the source wording for exact details."],
        "risks_and_attention": risks,
        "missing_or_unclear": ["This is a classroom demonstration mode, not a professional legal opinion.", "Exact legal effect should be checked against the complete original document."],
        "termination": "The report checks for termination language; see the key clauses and dates above.",
        "dispute_resolution": "The report checks for arbitration, court, or dispute language in the extracted text.",
        "governing_law": "Not specified unless clearly extracted from the document.",
        "timeline": dates[:6] or ["No clear timeline was extracted in demo mode."],
        "questions_to_ask": ["Which provisions create the most important obligations?", "What deadlines or notice periods should I watch?", "Are there payment, penalty, termination, or dispute terms that need clarification?"],
        "overall_attention_level": "Medium",
        "overall_reason": "The prototype highlights provisions that a reader should review, but it is not a substitute for professional legal advice."
    })


def normalize_analysis(data):
    """Make the frontend resilient if the model omits a field."""
    defaults = {
        "title": "Legal document",
        "document_type": "Unknown",
        "jurisdiction": "Not specified in the document.",
        "language": "Not specified",
        "summary": "Not specified in the document.",
        "simple_explanation": "Not specified in the document.",
        "parties": [],
        "legal_references": [],
        "important_terms": [],
        "financial_information": [],
        "important_dates": [],
        "key_clauses": [],
        "rights": [],
        "obligations": [],
        "risks_and_attention": [],
        "missing_or_unclear": [],
        "termination": "Not specified in the document.",
        "dispute_resolution": "Not specified in the document.",
        "governing_law": "Not specified in the document.",
        "timeline": [],
        "questions_to_ask": [],
        "overall_attention_level": "Unknown",
        "overall_reason": ""
    }

    for key, value in defaults.items():
        if key not in data or data[key] is None:
            data[key] = value

    # Keep list fields as lists.
    list_fields = [
        "parties", "legal_references", "important_terms",
        "financial_information", "important_dates", "key_clauses",
        "rights", "obligations", "risks_and_attention",
        "missing_or_unclear", "timeline", "questions_to_ask"
    ]

    for key in list_fields:
        if not isinstance(data.get(key), list):
            data[key] = [str(data[key])] if data.get(key) else []

    return data


# ============================================================
# LEGAL CLASSIFICATION
# ============================================================

def classify_document(openai_file_id, filename):
    """
    Classify broadly and fairly.

    Important difference from the old classifier:
    - It does NOT require the word 'agreement' or 'clause'.
    - It recognizes legislation, judgments, petitions, deeds,
      affidavits, notices, pleadings, legal correspondence, etc.
    - It can inspect scanned/image PDFs through the original file.
    - It uses a middle category for ambiguous documents instead of
      blindly rejecting useful legal material.
    """

    instructions = """
You are the document-classification engine for Legal Detective.

Your job is to determine whether the uploaded document is meaningfully
LEGAL / LAW-RELATED.

Be accurate, not artificially strict.

A document is legal if its primary or substantial purpose is connected
to law, legal rights, legal duties, legal proceedings, legal relationships,
regulation, compliance, or legally operative records.

LEGAL DOCUMENT EXAMPLES INCLUDE:
- contracts and agreements of all kinds
- rental/lease documents
- employment agreements and offer/employment contracts
- NDAs and confidentiality agreements
- sale deeds, gift deeds, property documents and conveyances
- wills, trusts, probate documents
- affidavits and declarations
- powers of attorney
- court judgments, orders, decrees and opinions
- plaints, petitions, written statements, pleadings and applications
- summons, warrants and court notices
- legal notices and demand notices
- arbitration documents
- settlement agreements
- government notifications, regulations and legal circulars
- statutes, Acts, bills and legislative documents
- rules, regulations and by-laws
- licenses, permits and regulatory approvals
- insurance policies when they create/describe legal rights and duties
- privacy policies, terms of service and commercial terms
- compliance documents
- legal correspondence between parties/lawyers
- certificates or registrations that have clear legal effect
- corporate resolutions and legal filings
- tax/legal notices
- intellectual-property filings and agreements
- immigration/visa legal forms
- family-law documents
- criminal/civil case documents

DO NOT reject a document just because it has few or no explicit "clauses".
A judgment, statute, affidavit, petition, deed or notice may not use
the word "clause" at all.

NON-LEGAL EXAMPLES:
- ordinary school/college timetables
- marksheets and ordinary academic records
- lecture notes and textbooks
- general news
- shopping lists
- restaurant menus
- ordinary invoices/receipts with no meaningful legal purpose
- personal diary/notes
- ordinary advertisements
- travel itineraries
- ordinary event schedules
- generic presentations
- unrelated technical manuals
- ordinary emails with no legal purpose

MIXED/AMBIGUOUS RULE:
If a document contains a substantial legal component, classify it as
legal and explain what makes it legal. Do not require a perfect legal
template.

SCANNED DOCUMENT RULE:
If the document is image-based, inspect the visible pages. Do not say
"no text" merely because machine text extraction would be difficult.

Return ONLY valid JSON:
{
  "is_legal": true,
  "classification": "legal",
  "document_type": "specific type",
  "confidence": 96,
  "reason": "short evidence-based reason",
  "jurisdiction": "jurisdiction if visible, otherwise Not specified"
}

classification must be one of:
"legal", "not_legal", "uncertain".

If classification is "uncertain", set is_legal to false only when
there is genuinely insufficient evidence. Explain exactly what is unclear.
"""

    prompt = f"""
Filename: {filename}

Classify the uploaded document using the rules above.
Do not invent a document type, jurisdiction or legal authority.
"""

    raw = ask_model_with_file(instructions, prompt, openai_file_id)
    result = parse_json(raw)

    if not result:
        return {
            "is_legal": False,
            "classification": "uncertain",
            "document_type": "Unable to verify",
            "confidence": 0,
            "reason": "The document could not be reliably classified.",
            "jurisdiction": "Not specified"
        }

    result["confidence"] = max(
        0, min(100, int(result.get("confidence", 0) or 0))
    )

    return result


# ============================================================
# FULL LEGAL ANALYSIS
# ============================================================

def analyze_legal_document(openai_file_id, filename):
    instructions = """
You are Legal Detective, an advanced legal-document analysis assistant.

Analyze the ORIGINAL uploaded document carefully. You may use the
document's visible pages, text, headings, tables and formatting.

CRITICAL ACCURACY RULES:
1. Never invent a fact, clause, party, date, amount, Act, section,
   court, jurisdiction or legal authority.
2. If something is not present, say "Not specified in the document."
3. A document does not need to contain traditional "clauses" to be
   legal. Analyze whatever legal structure it actually contains.
4. Quote or paraphrase only what the document supports.
5. When citing a page, use the page number visible/associated with
   the document where reasonably available.
6. Distinguish between:
   - "Mentioned in document" = explicitly present.
   - "Potential issue" = an observation about the document.
   Never present a guessed law as an applicable law.
7. Do not provide a definitive legal opinion. This is informational
   document analysis.

LEGAL REFERENCES:
Extract Acts, statutes, regulations, rules, sections, articles,
constitutional provisions, case names, court names and legal authorities
ONLY when they are actually mentioned or clearly identifiable in the
document. Do not manufacture Acts merely because they might apply.

IMPORTANT:
If the uploaded document is a judgment, statute, petition, notice,
affidavit, deed, order, pleading, legal correspondence, policy or other
legal document, adapt the analysis to that type instead of forcing it
into a contract template.

Return ONLY valid JSON with this exact structure:

{
  "title": "",
  "document_type": "",
  "jurisdiction": "",
  "language": "",
  "summary": "",
  "simple_explanation": "",

  "parties": [],
  "legal_references": [],
  "important_terms": [],
  "financial_information": [],
  "important_dates": [],
  "key_clauses": [],
  "rights": [],
  "obligations": [],
  "risks_and_attention": [],
  "missing_or_unclear": [],

  "termination": "",
  "dispute_resolution": "",
  "governing_law": "",

  "timeline": [],
  "questions_to_ask": [],

  "overall_attention_level": "Low",
  "overall_reason": ""
}

LIST QUALITY:
Each list item should be a useful, specific sentence rather than a
single vague word.

KEY CLAUSES:
For contracts, identify major clauses. For non-contract legal documents,
use "Key Legal Provisions" instead conceptually and summarize the most
important provisions.

RIGHTS:
State important rights granted, restricted, reserved or recognized.

OBLIGATIONS:
State what each important party/person/entity is required or expected
to do.

RISKS:
Identify concrete things deserving attention. Do not label every normal
legal provision as a risk. Explain why each item matters.

MISSING/UNCLEAR:
Mention genuinely missing, ambiguous or unreadable information.

TIMELINE:
Create a chronological list of important dates/events if present.

QUESTIONS TO ASK:
Suggest practical questions a reader may want to ask about this document,
but do not answer questions using information that is not in it.

OVERALL ATTENTION:
Low / Medium / High, with a short evidence-based reason.
"""

    prompt = f"""
Analyze this legal document:

Filename: {filename}

Give a comprehensive but readable analysis. Do not skip important
sections merely because the document is not a contract.
"""

    raw = ask_model_with_file(instructions, prompt, openai_file_id)
    return normalize_analysis(parse_json(raw))


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# ANALYZE
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():
    local_path = None

    try:
        if "file" not in request.files:
            return jsonify({
                "status": "error",
                "message": "Please upload a document."
            }), 400

        file = request.files["file"]

        if not file.filename:
            return jsonify({
                "status": "error",
                "message": "No file selected."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "status": "error",
                "message": (
                    "Unsupported file type. Please upload PDF, DOCX, TXT, "
                    "MD or RTF."
                )
            }), 400

        original_filename = secure_filename(file.filename)

        document_id = uuid.uuid4().hex
        unique_name = f"{document_id}_{original_filename}"
        local_path = UPLOAD_FOLDER / unique_name
        file.save(local_path)

        # Classroom mode: no API key, card, or paid AI credits required.
        # The original OpenAI integration remains available for later use.
        if DEMO_MODE:
            text = extract_local_text(str(local_path))
            classification = demo_classify(text, original_filename)
            DOCUMENTS[document_id] = {
                "local_path": str(local_path),
                "filename": original_filename,
                "demo": True,
                "text": text
            }
        else:
            # Upload ORIGINAL file to OpenAI so the model can inspect
            # scanned/image PDFs instead of depending only on pypdf.
            openai_file_id = upload_to_openai(str(local_path))
            DOCUMENTS[document_id] = {
                "openai_file_id": openai_file_id,
                "local_path": str(local_path),
                "filename": original_filename,
                "demo": False
            }
            classification = classify_document(openai_file_id, original_filename)

        is_legal = bool(classification.get("is_legal", False))
        classification_type = classification.get(
            "classification", "uncertain"
        )

        if not is_legal:
            return jsonify({
                "status": "not_legal",
                "mode": "classroom_demo" if DEMO_MODE else "ai_api",
                "document_id": document_id,
                "filename": original_filename,
                "document_type": classification.get(
                    "document_type", "Not a legal document"
                ),
                "classification": classification_type,
                "confidence": classification.get("confidence", 0),
                "reason": classification.get(
                    "reason",
                    "The document does not appear to be primarily legal."
                ),
                "jurisdiction": classification.get(
                    "jurisdiction",
                    "Not specified"
                ),
                "message": (
                    "This document does not appear to be a legal document. "
                    "Legal analysis was not performed."
                )
            })

        if DEMO_MODE:
            analysis = demo_analysis(str(local_path), original_filename)
        else:
            analysis = analyze_legal_document(
                DOCUMENTS[document_id]["openai_file_id"],
                original_filename
            )

        return jsonify({
            "status": "legal",
            "mode": "classroom_demo" if DEMO_MODE else "ai_api",
            "document_id": document_id,
            "filename": original_filename,
            "classification_confidence": classification.get(
                "confidence", 0
            ),
            "document_type": classification.get(
                "document_type",
                analysis.get("document_type", "Legal document")
            ),
            "jurisdiction": classification.get(
                "jurisdiction",
                analysis.get("jurisdiction", "Not specified")
            ),
            "analysis": analysis
        })

    except Exception as e:
        print("ANALYZE ERROR:", repr(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ============================================================
# ASK THE DOCUMENT
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}

        question = (data.get("question") or "").strip()
        document_id = (data.get("document_id") or "").strip()

        if not question:
            return jsonify({
                "status": "error",
                "message": "Please enter a question."
            }), 400

        document = DOCUMENTS.get(document_id)

        if not document:
            return jsonify({
                "status": "error",
                "message": (
                    "Document session expired. Please upload the document again."
                )
            }), 404

        instructions = """
You are Legal Detective's document Q&A assistant.

Answer ONLY from the uploaded document.

Rules:
- Do not invent facts.
- If the answer is not in the document, say:
  "I cannot find that information in this document."
- If possible, mention the relevant page number.
- If the user asks whether something is legal, do not give a
  definitive legal opinion. Explain what the document says and
  recommend professional advice where appropriate.
- Keep the answer clear.
"""

        prompt = f"""
User's question:
{question}

Answer using only the uploaded document.
"""

        if DEMO_MODE or document.get("demo"):
            text = document.get("text", "")
            q = question.lower()
            relevant = _find_sentences(text, [w for w in re.findall(r"[a-zA-Z]{4,}", q)[:6]], 4) if text else []
            if relevant:
                answer = "Demo document answer based on extracted text:\n\n" + "\n".join("• " + x for x in relevant)
            else:
                answer = (
                    "In classroom demo mode, I could not find a specific answer in the extracted text. "
                    "Please check the original document for the exact wording."
                )
        else:
            answer = ask_model_with_file(
                instructions,
                prompt,
                document["openai_file_id"]
            )

        return jsonify({
            "status": "success",
            "mode": "classroom_demo" if DEMO_MODE else "ai_api",
            "answer": answer
        })

    except Exception as e:
        print("ASK ERROR:", repr(e))

        return jsonify({
            "status": "error",
            "message": "Could not answer the question."
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Legal Detective"
    })


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "status": "error",
        "message": "File is too large. Maximum size is 25 MB."
    }), 413


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )

