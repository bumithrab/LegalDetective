from flask import Flask, request, jsonify, render_template
from pypdf import PdfReader
import os
import re

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# ANALYZE DOCUMENT
# ==========================================

@app.route("/analyze", methods=["POST"])
def analyze():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    text = re.sub(r"\s+", " ", text).strip()

    clauses = []


    # MONTHLY RENT
    rent_match = re.search(
        r"monthly rent of\s*[^\d]*([\d,]+)",
        text,
        re.IGNORECASE
    )

    if rent_match:
        clauses.append({
            "title": "Monthly Rent",
            "value": "₹" + rent_match.group(1)
        })


    # SECURITY DEPOSIT
    deposit_match = re.search(
        r"security deposit of\s*[^\d]*([\d,]+)",
        text,
        re.IGNORECASE
    )

    if deposit_match:
        clauses.append({
            "title": "Security Deposit",
            "value": "₹" + deposit_match.group(1)
        })


    # NOTICE PERIOD
    notice_match = re.search(
        r"notice period of\s*(\d+\s*(?:months?|days?))",
        text,
        re.IGNORECASE
    )

    if notice_match:
        clauses.append({
            "title": "Notice Period",
            "value": notice_match.group(1)
        })


    # EARLY TERMINATION
    termination_match = re.search(
        r"7\.\s*EARLY TERMINATION(.*?)(?=8\.\s*LANDLORD RESPONSIBILITIES)",
        text,
        re.IGNORECASE
    )

    if termination_match:
        clauses.append({
            "title": "Early Termination",
            "value": termination_match.group(1).strip()
        })


    return jsonify({
        "message": "Document analyzed successfully!",
        "clauses": clauses
    })


# ==========================================
# ASK THE DOCUMENT
# ==========================================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No question received."
        }), 400

    question = data.get("question", "").lower().strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400


    # Find uploaded PDF
    files = [
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not files:
        return jsonify({
            "error": "Please upload a document first."
        }), 400


    latest_file = max(
        files,
        key=lambda f: os.path.getmtime(
            os.path.join(UPLOAD_FOLDER, f)
        )
    )


    file_path = os.path.join(
        UPLOAD_FOLDER,
        latest_file
    )


    # Read PDF
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    text = re.sub(r"\s+", " ", text).strip()


    # ==========================================
    # SECURITY DEPOSIT QUESTION
    # ==========================================

    if "deposit" in question:

        match = re.search(
            r"security deposit of\s*[^\d]*([\d,]+)",
            text,
            re.IGNORECASE
        )

        if match:

            return jsonify({
                "answer": f"The security deposit is ₹{match.group(1)}.",
                "source": "Section 3 — Security Deposit"
            })


    # ==========================================
    # RENT QUESTION
    # ==========================================

    if "rent" in question and "deposit" not in question:

        match = re.search(
            r"monthly rent of\s*[^\d]*([\d,]+)",
            text,
            re.IGNORECASE
        )

        if match:

            return jsonify({
                "answer": f"The monthly rent is ₹{match.group(1)}.",
                "source": "Section 2 — Monthly Rent"
            })


    # ==========================================
    # NOTICE QUESTION
    # ==========================================

    if "notice" in question:

        match = re.search(
            r"notice period of\s*(\d+\s*(?:months?|days?))",
            text,
            re.IGNORECASE
        )

        if match:

            return jsonify({
                "answer": f"The required notice period is {match.group(1)}.",
                "source": "Section 4 — Notice Period"
            })


    # ==========================================
    # EARLY TERMINATION QUESTION
    # ==========================================

    if (
        "leave early" in question
        or "early termination" in question
        or "terminate" in question
    ):

        match = re.search(
            r"7\.\s*EARLY TERMINATION(.*?)(?=8\.\s*LANDLORD RESPONSIBILITIES)",
            text,
            re.IGNORECASE
        )

        if match:

            return jsonify({
                "answer": match.group(1).strip(),
                "source": "Section 7 — Early Termination"
            })


    # ==========================================
    # NOT FOUND
    # ==========================================

    return jsonify({
        "answer": "I could not find this information in the uploaded document.",
        "source": "Uploaded document"
    })


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)