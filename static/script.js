function goToAnalyze() {
    document.getElementById("analyze").scrollIntoView({
        behavior: "smooth"
    });
}


function chooseFile() {
    document.getElementById("fileInput").click();
}


document.getElementById("fileInput").addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    const result = document.getElementById("fileResult");

    result.innerHTML = `
        <p>✓ ${file.name} selected</p>
        <br>

        <button type="button" class="primary-btn" onclick="analyzeDocument()">
            🔍 Analyze Document
        </button>
    `;
});


async function analyzeDocument() {

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        return;
    }

    const result = document.getElementById("fileResult");

    result.innerHTML = `
        <p>🔍 AI is analyzing your document...</p>
    `;

    const formData = new FormData();
    formData.append("file", file);

    try {

        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Analysis failed");
        }

        let clauseHTML = "";

        data.clauses.forEach(function (clause) {

            clauseHTML += `
                <div class="clause-card">

                    <div class="clause-title">
                        ${clause.title}
                    </div>

                    <div class="clause-value">
                        ${clause.value}
                    </div>

                </div>
            `;

        });

        result.innerHTML = `
            <h3>⚖️ AI Legal Detective</h3>

            <p style="margin-top:15px;">
                Important clauses found in your document:
            </p>

            <div style="margin-top:20px;">
                ${clauseHTML}
            </div>

            <div class="ask-box">

                <p class="small-title">ASK THE DOCUMENT</p>

                <h3>Have a question?</h3>

                <p>
                    Ask a question about the uploaded document.
                </p>

                <input
                    type="text"
                    id="questionInput"
                    placeholder="e.g. What is the security deposit?"
                >

                <button
                    type="button"
                    class="primary-btn"
                    onclick="askQuestion()"
                >
                    🤖 Ask AI
                </button>

                <div id="answerResult"></div>

            </div>
        `;

    } catch (error) {

        result.innerHTML = `
            <p>❌ ${error.message}</p>
        `;

    }
}


async function askQuestion() {

    const questionInput = document.getElementById("questionInput");
    const answerResult = document.getElementById("answerResult");

    if (!questionInput || !answerResult) {
        alert("Question box was not found.");
        return;
    }

    const question = questionInput.value.trim();

    if (!question) {

        answerResult.innerHTML = `
            <p>⚠️ Please type a question first.</p>
        `;

        return;
    }

    answerResult.innerHTML = `
        <p>🤖 AI is thinking...</p>
    `;

    try {

        const response = await fetch("/ask", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Could not get an answer.");
        }

        answerResult.innerHTML = `
            <div class="clause-card">

                <div class="clause-title">
                    🤖 AI ANSWER
                </div>

                <div class="clause-value">
                    ${data.answer}
                </div>

                <div style="margin-top:15px;">
                    📌 <strong>Source:</strong> ${data.source}
                </div>

            </div>
        `;

    } catch (error) {

        answerResult.innerHTML = `
            <p>❌ ${error.message}</p>
        `;

    }
}
