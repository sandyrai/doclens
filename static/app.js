// ===========================================================
// AI Document Agent — Client-side Application
// ===========================================================


// ===========================================================
// SESSION ID
// ===========================================================

const SESSION_ID = crypto.randomUUID();


// ===========================================================
// STATE
// ===========================================================

let documentCount = 0;
let isUploading = false;
let activeDocumentSource = null;
let activeAbortController = null;  // For stop button


// ===========================================================
// INITIALIZATION
// ===========================================================

document.addEventListener("DOMContentLoaded", () => {

    loadDocuments();

    // Enter key sends message (Shift+Enter for newline)
    document.getElementById("question")
        .addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

    // Auto-resize textarea as user types
    document.getElementById("question")
        .addEventListener("input", autoResizeTextarea);

    // File input change handler
    document.getElementById("file-input")
        .addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFiles(e.target.files);
            }
        });

    // Drag and drop on the upload zone
    const zone = document.getElementById("upload-zone");

    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
    });

    zone.addEventListener("dragleave", () => {
        zone.classList.remove("dragover");
    });

    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFiles(files);
        }
    });

    // Hamburger menu (mobile)
    const hamburger = document.getElementById("hamburger-btn");
    const overlay = document.getElementById("sidebar-overlay");

    if (hamburger) {
        hamburger.addEventListener("click", toggleSidebar);
    }
    if (overlay) {
        overlay.addEventListener("click", closeSidebar);
    }
});


// ===========================================================
// TEXTAREA AUTO-RESIZE
// ===========================================================

function autoResizeTextarea() {
    const ta = document.getElementById("question");
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
}


// ===========================================================
// MOBILE SIDEBAR
// ===========================================================

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebar-overlay");

    sidebar.classList.toggle("open");
    overlay.classList.toggle("visible");
}

function closeSidebar() {
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebar-overlay");

    sidebar.classList.remove("open");
    overlay.classList.remove("visible");
}


// ===========================================================
// UPLOAD
// ===========================================================

async function handleFiles(fileList) {

    if (isUploading) return;

    const supportedExts = [".pdf", ".docx", ".txt", ".csv"];

    const validFiles = Array.from(fileList).filter(f => {
        const name = f.name.toLowerCase();
        return supportedExts.some(ext => name.endsWith(ext));
    });

    if (validFiles.length === 0) {
        showUploadStatus(
            "error",
            "Supported formats: PDF, DOCX, TXT, CSV"
        );
        return;
    }

    isUploading = true;

    const zone = document.getElementById("upload-zone");
    const progress = document.getElementById("upload-progress");
    const progressText = document.getElementById("progress-text");

    zone.classList.add("uploading");
    progress.style.display = "block";

    unlockChat();

    for (let i = 0; i < validFiles.length; i++) {

        const file = validFiles[i];

        const fileLabel = validFiles.length > 1
            ? `${file.name} (${i + 1}/${validFiles.length})`
            : file.name;

        progressText.textContent =
            `Uploading ${fileLabel}...`;

        try {

            const formData = new FormData();
            formData.append("file", file);

            progressText.textContent =
                `Extracting & indexing ${fileLabel}...`;

            const processingCard = showProcessingCard(file.name);

            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(
                    err.error || `HTTP ${response.status}`
                );
            }

            const result = await response.json();

            if (processingCard) processingCard.remove();

            if (result.status === "duplicate") {
                progressText.textContent =
                    `${fileLabel}: already uploaded`;

                showSummaryCard(
                    file.name,
                    result.summary || "This file was already uploaded.",
                    [],
                    result.pages,
                    result.chunks
                );
            } else {
                progressText.textContent =
                    `${fileLabel}: ${result.pages || 0} pages indexed`;

                showSummaryCard(
                    file.name,
                    result.summary || "Document indexed successfully.",
                    result.insights || [],
                    result.pages,
                    result.chunks
                );
            }

        } catch (err) {

            showUploadStatus(
                "error",
                `Failed to upload ${file.name}: ${err.message}`
            );
        }
    }

    await loadDocuments();

    zone.classList.remove("uploading");
    zone.classList.add("success");
    progress.style.display = "none";

    setTimeout(() => {
        zone.classList.remove("success");
    }, 2000);

    document.getElementById("file-input").value = "";
    isUploading = false;
}


function showUploadStatus(type, message) {

    const zone = document.getElementById("upload-zone");
    const progress = document.getElementById("upload-progress");
    const progressText = document.getElementById("progress-text");

    zone.classList.remove("uploading", "success", "error");
    zone.classList.add(type);

    progressText.textContent = message;
    progress.style.display = "block";

    setTimeout(() => {
        zone.classList.remove(type);
        progress.style.display = "none";
    }, 4000);
}


// ===========================================================
// DOCUMENTS READY CARD
// ===========================================================

let documentsReadyShown = false;

function showDocumentsReadyCard(docs) {

    if (documentsReadyShown) return;
    documentsReadyShown = true;

    const chat = document.getElementById("chat-messages");

    const card = document.createElement("div");
    card.className = "summary-card";

    const docList = docs.map(
        d => `<div class="insight-item">${escapeHtml(d.filename || d.document_id)} (${d.pages || 0} pages)</div>`
    ).join("");

    card.innerHTML = `
        <div class="summary-card-inner">
            <h3>Documents loaded</h3>
            <div class="summary-text">
                ${docs.length} document${docs.length > 1 ? 's' : ''} ready for analysis.
                Ask any question below.
            </div>
            <div class="insights-title">Available Documents</div>
            ${docList}
        </div>
    `;

    chat.appendChild(card);
}


// ===========================================================
// PROCESSING CARD
// ===========================================================

function showProcessingCard(filename) {

    unlockChat();

    const chat = document.getElementById("chat-messages");

    const card = document.createElement("div");
    card.className = "summary-card processing-card";

    card.innerHTML = `
        <div class="summary-card-inner">
            <h3>${escapeHtml(filename)}</h3>
            <div class="processing-indicator">
                <div class="processing-spinner"></div>
                <span>Processing... extracting text and
                generating embeddings.</span>
            </div>
        </div>
    `;

    chat.appendChild(card);
    chat.scrollTop = chat.scrollHeight;

    return card;
}


// ===========================================================
// SUMMARY CARD
// ===========================================================

function showSummaryCard(
    filename, summary, insights, pages, chunks
) {

    unlockChat();

    const chat = document.getElementById("chat-messages");

    const card = document.createElement("div");
    card.className = "summary-card";

    const cleanSummary = formatText(summary);

    let insightsHtml = "";

    if (insights && insights.length > 0) {
        insightsHtml = `
            <div class="insights-title">Key Insights</div>
            ${insights.map(
                i => `<div class="insight-item">${formatText(i)}</div>`
            ).join("")}
        `;
    }

    card.innerHTML = `
        <div class="summary-card-inner">
            <h3>${escapeHtml(filename)}</h3>
            <div style="font-size: 11px; color: #999; margin-bottom: 10px;">
                ${pages || 0} pages &middot; ${chunks || 0} chunks indexed
            </div>
            <div class="summary-text">${cleanSummary}</div>
            ${insightsHtml}
        </div>
    `;

    chat.appendChild(card);
    chat.scrollTop = chat.scrollHeight;
}


// ===========================================================
// DOCUMENT LIST
// ===========================================================

async function loadDocuments() {

    try {

        const response = await fetch("/documents");
        const data = await response.json();
        const docs = data.documents || [];

        documentCount = docs.length;

        document.getElementById("doc-count")
            .textContent = documentCount;

        const listEl = document.getElementById("doc-list");

        if (docs.length === 0) {

            listEl.innerHTML = `
                <div class="doc-empty">
                    No documents uploaded yet.<br>
                    Upload a PDF to get started.
                </div>
            `;

            lockChat();
            return;
        }

        unlockChat();
        showDocumentsReadyCard(docs);

        listEl.innerHTML = docs.map(doc => `
            <div class="doc-item" id="doc-${doc.document_id}"
                 data-filename="${escapeHtml(doc.filename || doc.document_id)}"
                 onclick="selectDocument(this)">
                <div class="doc-icon">&#128196;</div>
                <div class="doc-info">
                    <div class="doc-name"
                         title="${escapeHtml(doc.filename || doc.document_id)}">
                        ${escapeHtml(doc.filename || doc.document_id)}
                    </div>
                    <div class="doc-meta">
                        ${doc.pages || 0} pages
                    </div>
                </div>
                <button
                    class="doc-delete"
                    onclick="event.stopPropagation(); deleteDocument('${doc.document_id}')"
                    title="Remove document"
                >
                    &#10005;
                </button>
            </div>
        `).join("");

        // Auto-select if only 1 document
        if (docs.length === 1) {
            const firstItem = listEl.querySelector(".doc-item");
            if (firstItem) selectDocument(firstItem);
        } else {
            if (activeDocumentSource) {
                const match = listEl.querySelector(
                    `.doc-item[data-filename="${activeDocumentSource}"]`
                );
                if (match) {
                    match.classList.add("active");
                } else {
                    activeDocumentSource = null;
                    updateSourceFilterChip();
                }
            }
        }

    } catch (err) {
        console.error("Failed to load documents:", err);
    }
}


async function deleteDocument(documentId) {

    if (!confirm("Remove this document?")) return;

    try {
        await fetch(`/documents/${documentId}`, {
            method: "DELETE",
        });

        const deletedEl = document.getElementById(
            "doc-" + documentId
        );
        if (
            deletedEl &&
            deletedEl.dataset.filename === activeDocumentSource
        ) {
            activeDocumentSource = null;
            updateSourceFilterChip();
        }

        await loadDocuments();

    } catch (err) {
        console.error("Delete failed:", err);
    }
}


function selectDocument(el) {

    const filename = el.dataset.filename;

    if (activeDocumentSource === filename) {
        activeDocumentSource = null;
        el.classList.remove("active");
    } else {
        document.querySelectorAll(".doc-item.active")
            .forEach(item => item.classList.remove("active"));

        activeDocumentSource = filename;
        el.classList.add("active");
    }

    updateSourceFilterChip();

    // Close sidebar on mobile after selection
    if (window.innerWidth <= 768) {
        closeSidebar();
    }
}


function updateSourceFilterChip() {

    const chip = document.getElementById("source-filter-chip");
    const label = document.getElementById("source-filter-label");

    if (activeDocumentSource) {
        chip.classList.remove("all-docs");
        label.textContent = "Searching: " + activeDocumentSource;
    } else {
        chip.classList.add("all-docs");
        label.textContent = "Searching: All documents";
    }
}


// ===========================================================
// CHAT LOCK / UNLOCK
// ===========================================================

function lockChat() {

    document.getElementById("chat-blocked")
        .style.display = "flex";

    document.getElementById("chat-messages")
        .style.display = "none";

    document.getElementById("question").disabled = true;

    document.getElementById("question")
        .placeholder = "Upload a PDF first...";

    document.getElementById("send-btn").disabled = true;
    document.getElementById("attach-btn").disabled = true;

    document.querySelector(".input-bar")
        .classList.add("disabled");
}


function unlockChat() {

    document.getElementById("chat-blocked")
        .style.display = "none";

    document.getElementById("chat-messages")
        .style.display = "block";

    document.getElementById("question").disabled = false;

    document.getElementById("question")
        .placeholder = "Ask about your documents...";

    document.getElementById("send-btn").disabled = false;
    document.getElementById("attach-btn").disabled = false;

    document.querySelector(".input-bar")
        .classList.remove("disabled");
}


// ===========================================================
// CHAT — streaming messages
// ===========================================================

async function sendMessage() {

    const input = document.getElementById("question");
    const sendBtn = document.getElementById("send-btn");
    const stopBtn = document.getElementById("stop-btn");
    const chat = document.getElementById("chat-messages");

    const question = input.value.trim();
    if (!question) return;

    // Disable input, show stop button
    input.disabled = true;
    sendBtn.style.display = "none";
    stopBtn.style.display = "flex";

    // Create abort controller for this request
    activeAbortController = new AbortController();

    // Add user message
    const userMsg = document.createElement("div");
    userMsg.className = "message user";

    userMsg.innerHTML = `
        <div class="message-inner">
            <div class="message-avatar">You</div>
            <div class="message-body">
                <div class="message-content"></div>
            </div>
        </div>
    `;

    userMsg.querySelector(".message-content")
        .textContent = question;

    chat.appendChild(userMsg);
    input.value = "";

    // Reset textarea height
    input.style.height = "auto";

    // Create AI response container
    const aiMsg = document.createElement("div");
    aiMsg.className = "message";

    aiMsg.innerHTML = `
        <div class="message-inner">
            <div class="message-avatar">AI</div>
            <div class="message-body">
                <div class="message-content ai-answer"></div>
                <div class="message-status ai-status">
                    Thinking...
                </div>
                <div class="message-timing ai-timing"></div>
            </div>
        </div>
    `;

    chat.appendChild(aiMsg);

    const answerEl = aiMsg.querySelector(".ai-answer");
    const statusEl = aiMsg.querySelector(".ai-status");
    const timingEl = aiMsg.querySelector(".ai-timing");

    // Show typing indicator while waiting for first token
    answerEl.innerHTML = `
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;

    let wasStopped = false;

    try {

        const response = await fetch("/chat/stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                session_id: SESSION_ID,
                source_filter: activeDocumentSource,
            }),
            signal: activeAbortController.signal,
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {

            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;

                const event = JSON.parse(line);

                handleStreamEvent(
                    event,
                    answerEl,
                    statusEl,
                    timingEl
                );
            }

            chat.scrollTop = chat.scrollHeight;
        }

    } catch (err) {

        if (err.name === "AbortError") {
            wasStopped = true;

            // Remove typing indicator if still there
            const indicator = answerEl.querySelector(
                ".typing-indicator"
            );
            if (indicator) indicator.remove();

            // If no content was generated, show stopped msg
            const rawContent = answerEl.getAttribute(
                "data-raw"
            ) || "";

            if (!rawContent.trim()) {
                answerEl.innerHTML = formatText(
                    "*Generation stopped.*"
                );
            }

            statusEl.innerHTML = "&#9724; Stopped";

        } else {
            statusEl.textContent = "Error: " + err.message;
        }

    } finally {
        activeAbortController = null;
        input.disabled = false;
        sendBtn.style.display = "flex";
        stopBtn.style.display = "none";
        input.focus();
    }
}


function stopGeneration() {

    if (activeAbortController) {
        activeAbortController.abort();
    }
}


function handleStreamEvent(
    event, answerEl, statusEl, timingEl
) {

    if (event.type === "status") {

        const icons = {
            thinking: "&#128269;",
            search_complete: "&#128196;",
            decision_completed: "&#10003;",
            tool_call: "&#128295;",
            generating: "&#9997;&#65039;",
            first_token: "&#9997;&#65039;",
        };

        const icon = icons[event.status] || "&#9203;";

        statusEl.innerHTML = icon + " " + escapeHtml(event.message);

        if (event.first_token_seconds) {
            timingEl.textContent =
                `First token: ${event.first_token_seconds}s`;
        }
    }

    if (event.type === "tool_result") {
        statusEl.innerHTML =
            `&#128295; ${escapeHtml(event.tool)} completed`;
        timingEl.textContent =
            `Tool time: ${event.tool_time_seconds}s`;
    }

    if (event.type === "token") {
        // Remove typing indicator on first token
        const indicator = answerEl.querySelector(".typing-indicator");
        if (indicator) indicator.remove();

        const currentRaw = answerEl.getAttribute("data-raw") || "";
        const newRaw = currentRaw + event.content;
        answerEl.setAttribute("data-raw", newRaw);
        answerEl.innerHTML = formatText(newRaw);
    }

    if (event.type === "completed") {
        statusEl.innerHTML = "&#9989; Completed";
        timingEl.textContent =
            `Response time: ${event.response_time_seconds}s`;
    }

    if (event.type === "error") {
        statusEl.innerHTML =
            "&#10060; Error: " + escapeHtml(event.message);
        timingEl.textContent =
            `Failed after ${event.elapsed_seconds}s`;
    }
}


// ===========================================================
// TEXT FORMATTING
// ===========================================================

function formatText(text) {

    if (!text) return "";

    // Split into lines to detect markdown tables
    const lines = text.split("\n");
    const result = [];
    let i = 0;

    while (i < lines.length) {

        // Check if this line looks like a markdown table row
        const line = lines[i].trim();

        if (
            line.startsWith("|") &&
            line.endsWith("|") &&
            line.includes("|", 1)
        ) {

            // Collect all consecutive table lines
            const tableLines = [];

            while (i < lines.length) {
                const tl = lines[i].trim();
                if (
                    tl.startsWith("|") &&
                    tl.endsWith("|")
                ) {
                    tableLines.push(tl);
                    i++;
                } else {
                    break;
                }
            }

            // Build HTML table
            result.push(buildMarkdownTable(tableLines));

        } else {
            // Normal line — apply inline formatting
            result.push(formatLine(escapeHtml(lines[i])));
            i++;
        }
    }

    return result.join("");
}


function buildMarkdownTable(lines) {

    if (lines.length < 2) {
        // Not enough rows for a table, render as text
        return lines.map(
            l => formatLine(escapeHtml(l))
        ).join("");
    }

    // Parse rows: split by | and trim each cell
    function parseCells(line) {
        return line
            .split("|")
            .slice(1, -1)  // remove empty first/last from leading/trailing |
            .map(cell => cell.trim());
    }

    // Check if a row is a separator (like |---|---|---|)
    function isSeparator(line) {
        const cells = parseCells(line);
        return cells.every(
            c => /^[:\-][\-]+[:]*$/.test(c)
        );
    }

    let headerRow = null;
    const bodyRows = [];
    let separatorFound = false;

    for (let i = 0; i < lines.length; i++) {

        if (isSeparator(lines[i])) {
            separatorFound = true;
            // The row before this is the header
            if (i === 1 && bodyRows.length === 0) {
                headerRow = parseCells(lines[0]);
            }
            continue;
        }

        if (i === 0 && !separatorFound) {
            // First row, might be header (checked later)
            continue;
        }

        bodyRows.push(parseCells(lines[i]));
    }

    // If we had a first row but no separator, treat all as body
    if (!separatorFound) {
        headerRow = null;
        bodyRows.length = 0;

        for (const line of lines) {
            bodyRows.push(parseCells(line));
        }
    }

    // Build HTML
    let html = '<div class="table-wrapper"><table class="md-table">';

    if (headerRow) {
        html += "<thead><tr>";
        for (const cell of headerRow) {
            html += `<th>${formatLine(escapeHtml(cell))}</th>`;
        }
        html += "</tr></thead>";
    }

    html += "<tbody>";

    for (const row of bodyRows) {
        html += "<tr>";
        for (const cell of row) {
            html += `<td>${formatLine(escapeHtml(cell))}</td>`;
        }
        html += "</tr>";
    }

    html += "</tbody></table></div>";

    return html;
}


function formatLine(text) {

    // **bold** -> <strong>
    text = text.replace(
        /\*\*(.+?)\*\*/g,
        "<strong>$1</strong>"
    );

    // *italic* -> <em>
    text = text.replace(
        /\*(.+?)\*/g,
        "<em>$1</em>"
    );

    // `code` -> <code>
    text = text.replace(
        /`([^`]+)`/g,
        "<code>$1</code>"
    );

    // Add <br> at end for line break
    if (text.trim() === "") {
        return "<br>";
    }

    return text + "<br>";
}


function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
