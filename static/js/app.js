document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const textInput = document.getElementById('news-text');
    const urlInput = document.getElementById('news-url');
    const charCount = document.getElementById('char-count');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    const errorMsg = document.getElementById('error-message');
    const resultSection = document.getElementById('result-section');
    const resetBtn = document.getElementById('reset-btn');

    let activeTab = 'text';

    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            activeTab = btn.dataset.tab;
            document.getElementById(`${activeTab}-tab`).classList.add('active');
            errorMsg.classList.add('hidden');
        });
    });

    // Character counter
    textInput.addEventListener('input', () => {
        const len = textInput.value.length;
        charCount.textContent = len;
        if (len > 5000) {
            charCount.style.color = '#ef4444';
        } else {
            charCount.style.color = '#94a3b8';
        }
    });

    // Analyze Click
    analyzeBtn.addEventListener('click', async () => {
        errorMsg.classList.add('hidden');
        resultSection.classList.add('hidden');

        const payload = {};
        if (activeTab === 'text') {
            const val = textInput.value.trim();
            if (val.length < 10) {
                showError("Please enter valid text (at least 10 characters).");
                return;
            }
            if (val.length > 5000) {
                showError("Text is too long (exceeds 5000 characters).");
                return;
            }
            payload.text = val;
        } else {
            const url = urlInput.value.trim();
            if (!url) {
                showError("Please paste an article URL.");
                return;
            }
            payload.url = url;
        }

        // Set Loading State
        analyzeBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        loader.classList.remove('hidden');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                const errDetail = data.detail || "";
                if (errDetail.includes("Failed to extract article")) {
                    throw new Error("Could not read the article from this URL. The site might be blocking bots. Try pasting the text manually!");
                } else if (errDetail.includes("Validation Error") || errDetail.includes("Must provide")) {
                    throw new Error("Invalid request sent to the server. Please check your text/URL.");
                } else {
                    throw new Error("Our servers are experiencing a glitch right now. Please try again.");
                }
            }

            renderResults(data);

        } catch (error) {
            showError(error.message);
        } finally {
            analyzeBtn.disabled = false;
            btnText.textContent = "Run Analysis";
            loader.classList.add('hidden');
        }
    });

    // New Analysis Reset
    resetBtn.addEventListener('click', () => {
        // Show empty state and hide result section
        document.getElementById('empty-state').style.display = 'block';
        resultSection.classList.add('hidden');

        // Reset inputs
        textInput.value = '';
        urlInput.value = '';
        charCount.textContent = '0';
        errorMsg.classList.add('hidden');

        // Reset bars
        document.getElementById('prob-real').style.width = `0%`;
        document.getElementById('prob-fake').style.width = `0%`;

        // Focus text input
        textInput.focus();
    });

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
    }

    function renderResults(data) {
        // Hide Empty State and Show Result Section
        document.getElementById('empty-state').style.display = 'none';
        resultSection.classList.remove('hidden');

        // Main RoBERTa Results
        const isFake = data.prediction === 'fake';
        const badge = document.getElementById('verdict-badge');
        badge.textContent = data.prediction.toUpperCase();
        badge.className = `badge ${data.prediction}`;

        document.getElementById('main-confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;

        // Probability bars
        const probReal = data.probabilities.real * 100;
        const probFake = data.probabilities.fake * 100;

        // Animate bars
        setTimeout(() => {
            document.getElementById('prob-real').style.width = `${probReal}%`;
            document.getElementById('prob-fake').style.width = `${probFake}%`;
        }, 100);

        document.getElementById('val-real').textContent = `${probReal.toFixed(1)}%`;
        document.getElementById('val-fake').textContent = `${probFake.toFixed(1)}%`;

        // LLM Results (Escalation Pipeline)
        const llmSection = document.getElementById('llm-section');
        if (data.method === 'hybrid_with_llm' && data.llm_verification) {
            llmSection.classList.remove('hidden');
            const llm = data.llm_verification;

            const llmBadge = document.getElementById('llm-badge');
            if (llm.verdict === 'real' || llm.verdict === 'fake') {
                llmBadge.textContent = llm.verdict.toUpperCase();
                llmBadge.className = `badge-small ${llm.verdict}`;
            } else {
                llmBadge.textContent = "UNCERTAIN";
                llmBadge.className = "badge-small";
            }

            document.getElementById('llm-confidence').textContent = `${(llm.confidence * 100).toFixed(1)}%`;
            document.getElementById('llm-text').textContent = llm.reasoning || "No reasoning provided by LLM.";

        } else {
            llmSection.classList.add('hidden');
        }
    }
});
