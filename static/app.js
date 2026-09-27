// ===========================================================
// AI Document Agent — Client-side Application
// ===========================================================


// ===========================================================
// SESSION ID — Persistent across page reloads
// ===========================================================
//
// WHY localStorage?
//
//   Before this change, every page reload generated a new
//   random session ID. That meant:
//     - Refreshing the page lost the current conversation
//     - The user couldn't resume where they left off
//
//   Now we store the session ID in localStorage so it
//   survives page reloads. The user can also switch between
//   sessions by clicking items in the chat history sidebar.
//
// HOW IT WORKS:
//
//   1. On page load, check localStorage for a saved session_id
//   2. If found, use it (resume the last conversation)
//   3. If not found, generate a new UUID and save it
//   4. When user clicks "New Chat", generate a new ID
//   5. When user clicks a past conversation, switch to its ID

let SESSION_ID = localStorage.getItem("session_id")
    || crypto.randomUUID();

// Save to localStorage so it persists across reloads
localStorage.setItem("session_id", SESSION_ID);


// ===========================================================
// STATE
// ===========================================================

let documentCount = 0;
let isUploading = false;
let activeDocumentSource = null;
let activeAbortController = null;  // For stop button

// ---------------------------------------------------------
// Smart Suggestions state (Phase 4)
// ---------------------------------------------------------
//
// WHY TRACK SUGGESTIONS IN STATE?
//
//   Suggestion chips are shown in two scenarios:
//
//   1. DOCUMENT SUGGESTIONS — fetched from the server after
//      a document finishes processing. These are cached on
//      the server keyed by filename, so we only fetch once.
//
//   2. FOLLOW-UP SUGGESTIONS — received as streaming events
//      after each AI answer. These change with every response.
//
//   We store the current suggestions here so we can clear
//   them when the user sends a new message (the old follow-ups
//   are no longer relevant).

let currentSuggestions = [];

// ---------------------------------------------------------
// Usage tracking state (Phase 1: Rate Limiting)
// ---------------------------------------------------------
//
// WHY TRACK USAGE IN THE FRONTEND?
//
//   The server enforces limits (returns HTTP 429), but the
//   frontend also needs to know usage for two reasons:
//
//   1. DISPLAY — show "3/15 questions today" in the UI
//   2. PROACTIVE DISABLE — disable the input when limits
//      are reached, instead of letting the user type a
//      question only to get an error response.
//
//   We fetch from GET /usage on page load and after each
//   question or upload.

let currentUsage = null;


// ===========================================================
// DEBOUNCED SCROLL — Performance Optimization
// ===========================================================
//
// WHY DEBOUNCE?
//
//   During streaming, we receive 50-200 token events per
//   second. Previously, each token triggered:
//     chat.scrollTop = chat.scrollHeight
//
//   That's 50-200 forced layout recalculations per second.
//   The browser has to:
//     1. Compute the full layout (reflow)
//     2. Measure scrollHeight (forces synchronous layout)
//     3. Set scrollTop (triggers another reflow)
//
//   This causes visible jank (stuttering) on slower machines.
//
// HOW DEBOUNCE FIXES IT:
//
//   Instead of scrolling on every token, we schedule ONE
//   scroll using requestAnimationFrame(). If another token
//   arrives before the frame renders, the previous scroll
//   is cancelled. Result: exactly 1 scroll per frame
//   (60fps = 60 scrolls/sec max), no matter how fast
//   tokens arrive. 200 token events → 60 scrolls instead
//   of 200. That's a 70% reduction in layout work.

let _scrollRafId = null;

function debouncedScroll(container) {
    // Cancel any pending scroll — we'll use this newer one
    if (_scrollRafId) cancelAnimationFrame(_scrollRafId);

    // Schedule scroll for next animation frame
    // (browser batches this with its own rendering)
    _scrollRafId = requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
        _scrollRafId = null;
    });
}


// ===========================================================
// TOAST NOTIFICATIONS — Non-blocking UI feedback
// ===========================================================
//
// WHY REPLACE confirm() / alert()?
//
//   Browser confirm() and alert() are MODAL — they freeze
//   the entire page until the user clicks OK. This means:
//     - All JavaScript stops executing
//     - Streaming responses are paused
//     - The UI becomes completely unresponsive
//
//   Toast notifications slide in from the top, show the
//   message for 3-4 seconds, then fade out — all without
//   blocking anything. The user can keep working while
//   the notification is visible.
//
// TYPES:
//   "success" — green, for completed actions
//   "error"   — red, for failures
//   "info"    — blue, for neutral information

function showToast(message, type = "info", duration = 3500) {

    // Create or reuse the toast container
    let container = document.getElementById("toast-container");

    if (!container) {
        container = document.createElement("div");
        container.id = "toast-container";
        document.body.appendChild(container);
    }

    // Build the toast element
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    // Icon based on type
    const icons = {
        success: "&#9989;",   // green check
        error: "&#10060;",    // red X
        info: "&#8505;&#65039;",  // info icon
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);

    // Trigger entrance animation on next frame
    // (element must be in DOM before adding "show" class
    //  for the CSS transition to fire)
    requestAnimationFrame(() => {
        toast.classList.add("show");
    });

    // Auto-remove after duration
    setTimeout(() => {
        toast.classList.remove("show");

        // Wait for fade-out animation to finish,
        // then remove from DOM to prevent memory leak
        toast.addEventListener("transitionend", () => {
            toast.remove();
        });

        // Fallback: remove after 500ms even if
        // transitionend doesn't fire (some browsers)
        setTimeout(() => toast.remove(), 500);
    }, duration);
}


// ===========================================================
// USAGE TRACKING (Phase 1: Rate Limiting)
// ===========================================================
//
// These functions fetch and display the user's current usage
// limits. The server tracks usage per IP per day — we just
// read and display it.
//
// FLOW:
//   1. fetchUsage() → calls GET /usage → updates currentUsage
//   2. updateUsageDisplay() → reads currentUsage → updates DOM
//   3. If limits reached → disables input, shows banner
//
// WHEN IS fetchUsage() CALLED?
//   - On page load (DOMContentLoaded)
//   - After a question is sent (in sendMessage())
//   - After a file is uploaded (in uploadFile())
//   - After receiving a 429 error response

async function fetchUsage() {
    try {
        const response = await fetch("/usage");

        if (!response.ok) return;

        currentUsage = await response.json();
        updateUsageDisplay();

    } catch (err) {
        // Usage display is a nice-to-have — if the fetch
        // fails, the chat still works normally. The server
        // will still enforce limits via 429 responses.
        console.warn("Failed to fetch usage:", err);
    }
}


function updateUsageDisplay() {
    // ---------------------------------------------------------
    // UPDATE THE USAGE COUNTER UI
    // ---------------------------------------------------------
    //
    // This function reads currentUsage (set by fetchUsage())
    // and updates the DOM to show:
    //   - "3/15 questions today" with appropriate styling
    //   - "1/1 uploads today" with appropriate styling
    //   - A limit banner if either limit is reached
    //
    // COLOR CODING:
    //   - Default (gray): plenty of usage remaining
    //   - Warning (orange): 80%+ used (e.g., 12/15 questions)
    //   - Limit (red): 100% used, input gets disabled
    //
    // WHY CSS CLASSES?
    //   Instead of inline styles, we use .usage-warning and
    //   .usage-limit classes defined in styles.css. This
    //   separates presentation from logic and makes it easy
    //   to adjust colors without editing JavaScript.

    const container = document.getElementById("usage-counter");
    if (!container || !currentUsage) return;

    const qUsed = currentUsage.questions_used;
    const qLimit = currentUsage.questions_limit;
    const uUsed = currentUsage.uploads_used;
    const uLimit = currentUsage.uploads_limit;

    // Determine warning/limit states
    const qPct = qLimit > 0 ? (qUsed / qLimit) : 0;
    const uPct = uLimit > 0 ? (uUsed / uLimit) : 0;

    // CSS class for question counter
    let qClass = "";
    if (qPct >= 1) qClass = "usage-limit";
    else if (qPct >= 0.8) qClass = "usage-warning";

    // CSS class for upload counter
    let uClass = "";
    if (uPct >= 1) uClass = "usage-limit";
    else if (uPct >= 0.8) uClass = "usage-warning";

    // Build the counter HTML
    container.innerHTML = `
        <span class="usage-item ${qClass}">
            <span class="usage-icon">&#128172;</span>
            ${qUsed}/${qLimit} questions today
        </span>
        <span class="usage-item ${uClass}">
            <span class="usage-icon">&#128206;</span>
            ${uUsed}/${uLimit} uploads today
        </span>
    `;

    // ---------------------------------------------------------
    // Handle limit-reached states
    // ---------------------------------------------------------
    //
    // When the question limit is reached, we disable the
    // input and send button so the user can't type a question
    // that will be rejected. This is friendlier than letting
    // them type and then showing an error.

    const input = document.getElementById("question");
    const sendBtn = document.getElementById("send-btn");

    // Remove any existing limit banner
    const existingBanner = document.querySelector(
        ".usage-limit-banner"
    );
    if (existingBanner) existingBanner.remove();

    if (qPct >= 1) {
        // Question limit reached — disable chat input
        if (input) {
            input.disabled = true;
            input.placeholder =
                "Daily question limit reached. " +
                "Come back tomorrow!";
        }
        if (sendBtn) sendBtn.disabled = true;

        // Add limit banner
        const banner = document.createElement("div");
        banner.className = "usage-limit-banner";
        banner.textContent =
            "You’ve reached your daily limit of " +
            qLimit + " questions. " +
            "Limits reset at midnight IST.";
        container.after(banner);

    } else if (documentCount > 0) {
        // Under limit and documents exist — enable input
        // (We only enable if documents are uploaded,
        // because the chat needs documents to search.)
        if (input) {
            input.disabled = false;
            input.placeholder = "Ask about your documents...";
        }
        if (sendBtn) sendBtn.disabled = false;
    }

    // Handle upload limit
    const attachBtn = document.getElementById("attach-btn");
    const uploadZone = document.getElementById("upload-zone");

    if (uPct >= 1) {
        if (attachBtn) attachBtn.disabled = true;
        if (uploadZone) {
            uploadZone.style.opacity = "0.5";
            uploadZone.style.pointerEvents = "none";
        }
    }
}


// ===========================================================
// INITIALIZATION
// ===========================================================

document.addEventListener("DOMContentLoaded", () => {

    // Load chat history sidebar and documents list
    loadChatHistory();
    loadDocuments();

    // Fetch and display current usage limits (Phase 1)
    fetchUsage();

    // Restore collapsed sidebar sections from localStorage
    // (so user preferences survive page reloads)
    restoreCollapsedSections();

    // ---------------------------------------------------------
    // Restore current session's messages on page reload
    // ---------------------------------------------------------
    //
    // WHY IS THIS NEEDED?
    //
    //   When the page reloads, SESSION_ID is restored from
    //   localStorage (line 29), so we know WHICH conversation
    //   was active. loadChatHistory() above builds the sidebar
    //   and highlights the active session. But nothing loads
    //   that session's MESSAGES into the chat area.
    //
    //   Without this call, the user sees their conversation
    //   listed in the sidebar but the chat area is empty —
    //   they have to click the conversation again to see it.
    //
    // WHY NOT JUST CALL switchToSession()?
    //
    //   switchToSession() has a guard:
    //     if (sessionId === SESSION_ID) return;
    //   This prevents reloading the same session when the
    //   user clicks the already-active item. But it also
    //   blocks the reload case. So we use a dedicated
    //   function that skips that guard.

    loadCurrentSessionMessages();

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

    // Hamburger menu — works on both mobile and desktop now
    const hamburger = document.getElementById("hamburger-btn");
    const overlay = document.getElementById("sidebar-overlay");

    if (hamburger) {
        hamburger.addEventListener("click", toggleSidebar);
    }
    if (overlay) {
        overlay.addEventListener("click", closeSidebar);
    }

    // ---------------------------------------------------------
    // RESIZE HANDLER — clean up sidebar state on breakpoint change
    // ---------------------------------------------------------
    //
    // WHY IS THIS NEEDED?
    //   If the user opens the sidebar on mobile (adds "open" class),
    //   then rotates to landscape or resizes to desktop width, the
    //   "open" class is still there but irrelevant. And vice versa:
    //   if they collapse the sidebar on desktop (adds "collapsed"),
    //   then shrink to mobile, the collapsed state shouldn't block
    //   the mobile overlay from working.
    //
    //   This listener cleans up stale classes when crossing the
    //   768px boundary.
    window.addEventListener("resize", () => {
        const sidebar = document.getElementById("sidebar");
        const overlayEl = document.getElementById("sidebar-overlay");
        if (!sidebar) return;

        const isMobile = window.innerWidth <= 768;

        if (isMobile) {
            // Entering mobile: remove desktop collapse classes
            sidebar.classList.remove("collapsed");
            document.body.classList.remove("sidebar-hidden");
        } else {
            // Entering desktop: remove mobile overlay classes
            sidebar.classList.remove("open");
            if (overlayEl) overlayEl.classList.remove("visible");
        }
    });

    // ---------------------------------------------------------
    // AUTO-HIDE SUGGESTION CHIPS ON SCROLL UP
    // ---------------------------------------------------------
    //
    // THE PROBLEM:
    //   Suggestion chips sit above the input area and can take
    //   up half the visible chat space (especially on mobile
    //   or small screens). When the user scrolls UP to re-read
    //   earlier messages, the chips block their view — irritating.
    //
    // THE FIX:
    //   Listen to scroll events on the chat messages container.
    //   If the user scrolls UP (not at the bottom), hide the
    //   chips. When they scroll back DOWN to the bottom, show
    //   them again. This gives the user full reading space when
    //   browsing history, and brings suggestions back when
    //   they're ready to ask a new question.
    //
    // HOW "AT BOTTOM" IS DETECTED:
    //   scrollHeight = total scrollable height (all messages)
    //   scrollTop    = how far the user has scrolled from top
    //   clientHeight = visible area height
    //
    //   If scrollHeight - scrollTop - clientHeight < 80px,
    //   the user is "at the bottom". The 80px threshold
    //   accounts for sub-pixel rounding and small overscrolls.
    //
    // WHY requestAnimationFrame?
    //   Scroll events fire 60+ times per second. Doing DOM
    //   changes on every event causes jank (stuttery scrolling).
    //   requestAnimationFrame batches our work into the next
    //   paint frame — smooth scrolling, no wasted CPU.

    const chatForChips = document.getElementById("chat-messages");
    const chipsContainer = document.getElementById("suggestion-chips");

    if (chatForChips && chipsContainer) {

        let chipScrollTicking = false;

        chatForChips.addEventListener("scroll", () => {

            // Batch scroll handling into animation frames
            // to avoid jank (see explanation above)
            if (!chipScrollTicking) {
                chipScrollTicking = true;

                requestAnimationFrame(() => {
                    // Calculate distance from the bottom
                    const distanceFromBottom =
                        chatForChips.scrollHeight
                        - chatForChips.scrollTop
                        - chatForChips.clientHeight;

                    // 80px threshold — "close enough" to bottom
                    const isAtBottom = distanceFromBottom < 80;

                    if (isAtBottom) {
                        // User is at the bottom → show chips
                        chipsContainer.classList.remove(
                            "chips-hidden"
                        );
                    } else {
                        // User scrolled up → hide chips
                        chipsContainer.classList.add(
                            "chips-hidden"
                        );
                    }

                    chipScrollTicking = false;
                });
            }
        });
    }

    // ---------------------------------------------------------
    // Keyboard shortcuts — Phase 6 UI Polish
    // ---------------------------------------------------------
    //
    // Ctrl+Shift+N → Start new chat (like browser "new tab")
    // Escape       → Stop AI generation (same as stop button)
    // Ctrl+Shift+E → Export conversation (Phase 7)
    //
    // WHY CTRL+SHIFT instead of just CTRL?
    //   Ctrl+N opens a new browser window. We don't want
    //   to hijack that. Ctrl+Shift+N is usually "new
    //   incognito window" which is less commonly needed
    //   inside a web app.

    document.addEventListener("keydown", (e) => {

        // Ctrl+Shift+N → New chat
        if (e.ctrlKey && e.shiftKey && e.key === "N") {
            e.preventDefault();
            startNewChat();
        }

        // Escape → Stop generation
        if (e.key === "Escape" && activeAbortController) {
            stopGeneration();
        }

        // Ctrl+Shift+E → Export conversation
        if (e.ctrlKey && e.shiftKey && e.key === "E") {
            e.preventDefault();
            exportConversation("html");
        }
    });
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
// SIDEBAR TOGGLE — ChatGPT-style open/close
// ===========================================================
//
// HOW IT WORKS ON DESKTOP vs MOBILE:
//
//   DESKTOP (>768px):
//     The sidebar is inline (part of flexbox layout).
//     Toggling adds/removes "collapsed" class which sets
//     width: 0 and slides it off-screen. The chat area
//     automatically expands to fill the freed space (flex: 1).
//     body.sidebar-hidden controls the hamburger visibility.
//
//   MOBILE (≤768px):
//     The sidebar is position: fixed (overlay). Toggling
//     adds/removes "open" class which slides it in from
//     the left. A dark overlay appears behind it so the
//     user can tap outside to close.
//
// WHY TWO DIFFERENT APPROACHES?
//   On desktop, the sidebar is PART of the layout — when it
//   collapses, the chat area should grow. An overlay would
//   just cover the chat without giving it more space.
//   On mobile, there's no room for a sidebar in the layout,
//   so an overlay on top of the chat is the right UX.

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebar-overlay");

    // Check if we're on mobile (sidebar uses fixed positioning)
    const isMobile = window.innerWidth <= 768;

    if (isMobile) {
        // Mobile: overlay behavior (unchanged)
        sidebar.classList.toggle("open");
        overlay.classList.toggle("visible");
    } else {
        // Desktop: inline collapse/expand
        const isCollapsed = sidebar.classList.contains("collapsed");

        if (isCollapsed) {
            // Expand — remove collapsed state
            sidebar.classList.remove("collapsed");
            document.body.classList.remove("sidebar-hidden");
        } else {
            // Collapse — slide sidebar out
            sidebar.classList.add("collapsed");
            document.body.classList.add("sidebar-hidden");
        }
    }
}

function closeSidebar() {
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebar-overlay");

    const isMobile = window.innerWidth <= 768;

    if (isMobile) {
        // Mobile: hide overlay sidebar
        sidebar.classList.remove("open");
        overlay.classList.remove("visible");
    } else {
        // Desktop: collapse inline sidebar
        sidebar.classList.add("collapsed");
        document.body.classList.add("sidebar-hidden");
    }
}


// ===========================================================
// COLLAPSIBLE SIDEBAR SECTIONS
// ===========================================================
//
// Toggles "collapsed" class on a sidebar section (Chat History
// or Documents). When collapsed, the section's content area
// slides up and the chevron arrow rotates to point right.
//
// WHY TOGGLE ON THE PARENT SECTION (not the content)?
//   CSS selectors like ".collapsed .collapsible-content" and
//   ".collapsed .section-chevron" let us control BOTH the
//   content AND the chevron from a single class toggle on
//   the parent. If we toggled the content directly, we'd
//   need separate logic for the chevron.
//
// PERSISTENCE:
//   The collapsed state is saved in localStorage so it
//   survives page reloads. Users who prefer sections closed
//   don't have to close them every time they reload.

function toggleSection(sectionId) {

    const section = document.getElementById(sectionId);
    if (!section) return;

    section.classList.toggle("collapsed");

    // Save state to localStorage so it persists across reloads.
    // Key format: "section_collapsed_chat-history-section" = "true"
    const key = "section_collapsed_" + sectionId;
    const isNowCollapsed = section.classList.contains("collapsed");

    try {
        localStorage.setItem(key, isNowCollapsed ? "true" : "false");
    } catch (e) {
        // localStorage might be unavailable (private browsing) —
        // that's fine, just don't persist.
    }
}


// ---------------------------------------------------------
// restoreCollapsedSections — restore saved collapse states
// ---------------------------------------------------------
//
// Called on page load (DOMContentLoaded). Reads localStorage
// for each collapsible section and restores the collapsed
// state if it was previously collapsed by the user.

function restoreCollapsedSections() {

    const sections = [
        "chat-history-section",
        "doc-list-section",
    ];

    sections.forEach(sectionId => {
        try {
            const key = "section_collapsed_" + sectionId;
            const saved = localStorage.getItem(key);

            if (saved === "true") {
                const section = document.getElementById(sectionId);
                if (section) {
                    section.classList.add("collapsed");
                }
            }
        } catch (e) {
            // localStorage unavailable — start expanded (default)
        }
    });
}


// ===========================================================
// UPLOAD
// ===========================================================

// ---------------------------------------------------------
// handleFiles — Background Upload with Polling
// ---------------------------------------------------------
//
// BEFORE (blocking):
//   Upload file → wait 5-30s for server to finish → show result
//   User sees a frozen "Extracting & indexing..." message.
//
// AFTER (background + polling):
//   Upload file → server returns instantly with task_id →
//   poll /upload/status/{task_id} every 2 seconds →
//   show animated progress stages → done!
//
// The user sees:
//   "Uploading report.pdf..."     (sending file to server)
//   "Saving file..."              (server saved to disk)
//   "Extracting text..."          (reading PDF content)
//   "Building search index..."    (chunking + embedding)
//   "Ready! 5 pages, 23 chunks"  (done!)
//
// Each stage has a progress bar that fills smoothly.

// Maps stage names from the server to user-friendly labels
// Stage labels shown in the progress bar during file upload.
// Phase 2: Added "ocr" stage for scanned documents and images.
const STAGE_LABELS = {
    saving:     "Saving file...",
    extracting: "Extracting text from document...",
    ocr:        "Running OCR on scanned pages...",
    processing: "Building search index...",
    chunking:   "Splitting into searchable chunks...",
    embedding:  "Generating embeddings...",
    storing:    "Storing in vector database...",
    done:       "Processing complete!",
    failed:     "Processing failed",
};


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
    const progressFill = progress.querySelector(".progress-fill");

    zone.classList.add("uploading");
    progress.style.display = "block";

    unlockChat();

    for (let i = 0; i < validFiles.length; i++) {

        const file = validFiles[i];

        const fileLabel = validFiles.length > 1
            ? `${file.name} (${i + 1}/${validFiles.length})`
            : file.name;

        // --- Step 1: Upload the file ---
        // This returns INSTANTLY now (server saves file
        // and starts background processing).

        progressText.textContent =
            `Uploading ${fileLabel}...`;

        if (progressFill) {
            progressFill.style.width = "5%";
        }

        try {

            const formData = new FormData();
            formData.append("file", file);

            const processingCard = showProcessingCard(file.name);

            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });

            // -------------------------------------------------
            // Handle rate limiting on upload (Phase 1)
            // -------------------------------------------------
            //
            // If the server returns 429, the daily upload
            // limit is reached. We show a friendly toast and
            // refresh the usage counter, then skip this file.

            if (response.status === 429) {
                const errData = await response.json();
                if (processingCard) processingCard.remove();

                showToast(
                    errData.error ||
                    "Daily upload limit reached!",
                    5000
                );

                await fetchUsage();
                continue;  // Skip to next file (if any)
            }

            if (!response.ok) {
                const err = await response.json();
                throw new Error(
                    err.error || `HTTP ${response.status}`
                );
            }

            const result = await response.json();

            // --- Handle duplicates (no polling needed) ---
            // Duplicates are detected instantly because the
            // server checks the SHA-256 hash before starting
            // background processing.

            if (result.status === "duplicate") {
                if (processingCard) processingCard.remove();

                progressText.textContent =
                    `${fileLabel}: already uploaded`;

                if (progressFill) {
                    progressFill.style.width = "100%";
                }

                showSummaryCard(
                    file.name,
                    result.summary ||
                        "This file was already uploaded.",
                    [],
                    result.pages,
                    result.chunks
                );

                continue;
            }

            // --- Step 2: Poll for progress ---
            // Server returned {task_id, status: "processing"}.
            // Now we poll /upload/status/{task_id} every 2
            // seconds until status is "ready" or "error".

            if (result.task_id) {

                const finalResult = await pollUploadStatus(
                    result.task_id,
                    fileLabel,
                    progressText,
                    progressFill,
                );

                if (processingCard) processingCard.remove();

                if (finalResult && finalResult.status === "ready") {

                    // Processing complete — show success
                    const r = finalResult.result || {};

                    progressText.textContent =
                        `${fileLabel}: ${r.pages || 0} pages indexed`;

                    showSummaryCard(
                        file.name,
                        r.summary ||
                            "Document indexed successfully.",
                        r.insights || [],
                        r.pages,
                        r.chunks
                    );

                    // Phase 4: Fetch smart suggestions for
                    // this document. The server generates
                    // starter questions in the background
                    // during processing — they should be
                    // ready by now. If not, the user can
                    // still type their own question.
                    fetchDocumentSuggestions(file.name);

                } else if (
                    finalResult && finalResult.status === "error"
                ) {

                    // Processing failed
                    throw new Error(
                        finalResult.error ||
                            "Processing failed"
                    );

                }

            } else {

                // Fallback: server returned old-style
                // response (no task_id) — handle it like
                // before for backward compatibility.
                if (processingCard) processingCard.remove();

                progressText.textContent =
                    `${fileLabel}: ${result.pages || 0} pages indexed`;

                showSummaryCard(
                    file.name,
                    result.summary ||
                        "Document indexed successfully.",
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

    // Refresh usage counter after upload (Phase 1)
    await fetchUsage();

    zone.classList.remove("uploading");
    zone.classList.add("success");
    progress.style.display = "none";

    setTimeout(() => {
        zone.classList.remove("success");
    }, 2000);

    document.getElementById("file-input").value = "";
    isUploading = false;
}


// ---------------------------------------------------------
// pollUploadStatus — poll until "ready" or "error"
// ---------------------------------------------------------
//
// Calls GET /upload/status/{task_id} every 2 seconds.
// Updates the progress bar and stage label on each poll.
//
// Returns the final status object when processing is done.
// Times out after 5 minutes (150 polls × 2 seconds) to
// prevent infinite polling on stuck tasks.

async function pollUploadStatus(
    taskId, fileLabel, progressText, progressFill,
) {

    const MAX_POLLS = 150;   // 150 × 2s = 5 minutes max
    const POLL_INTERVAL = 2000;  // 2 seconds

    for (let poll = 0; poll < MAX_POLLS; poll++) {

        // Wait 2 seconds between polls
        await new Promise(r => setTimeout(r, POLL_INTERVAL));

        try {

            const response = await fetch(
                `/upload/status/${taskId}`
            );

            if (!response.ok) {
                // Server returned an error status code.
                // This could be a 404 (unknown task) or
                // a 500 (server error). Either way, stop.
                const err = await response.json();
                return {
                    status: "error",
                    error: err.error || `HTTP ${response.status}`,
                };
            }

            const status = await response.json();

            // Update the progress bar width
            if (progressFill && status.progress_pct != null) {
                progressFill.style.width =
                    status.progress_pct + "%";
            }

            // Update the stage label
            const stageLabel =
                STAGE_LABELS[status.stage] ||
                `Processing: ${status.stage}...`;

            progressText.textContent =
                `${fileLabel}: ${stageLabel}`;

            // Check if processing is done
            if (status.status === "ready") {
                if (progressFill) {
                    progressFill.style.width = "100%";
                }
                return status;
            }

            if (status.status === "error") {
                return status;
            }

            // Still processing — continue polling

        } catch (err) {
            // Network error during polling — keep trying
            // (the server might be temporarily busy)
            console.warn(
                "Poll failed, retrying:", err.message
            );
        }
    }

    // Timed out after 5 minutes
    return {
        status: "error",
        error: "Upload processing timed out after 5 minutes.",
    };
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
                 data-document-id="${doc.document_id}"
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
            <!-- Phase 8: Image gallery container for this document.
                 Hidden by default — shown when the document is selected
                 and has extracted images. Loaded lazily on first click. -->
            <div class="doc-image-gallery"
                 id="gallery-${doc.document_id}"
                 data-loaded="false">
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
                }
            }
        }

        // Always update the document switcher chips
        // above the input after loading documents
        updateDocumentSwitcher();

    } catch (err) {
        console.error("Failed to load documents:", err);
    }
}


// ===========================================================
// PHASE 8: IMAGE GALLERY — Browse extracted PDF images
// ===========================================================
//
// WHY AN IMAGE GALLERY?
//
//   PDFs often contain charts, diagrams, signatures, photos,
//   and infographics that text search can't capture. The image
//   gallery lets users browse these visual elements after
//   upload — they appear as clickable thumbnails below the
//   document info in the sidebar.
//
// HOW IT WORKS:
//
//   1. After a document is uploaded and processed, the backend
//      extracts embedded images and saves them to disk.
//   2. GET /documents/{id}/images returns metadata (filenames,
//      page numbers, sizes) for all extracted images.
//   3. Each image is displayed as a lazy-loaded thumbnail.
//   4. Clicking a thumbnail opens the full-size image in a
//      lightweight modal overlay.
//
// PERFORMANCE OPTIMIZATIONS:
//
//   - Lazy loading: thumbnails use loading="lazy" so images
//     only download when scrolled into view. A document with
//     50 images doesn't trigger 50 simultaneous requests.
//   - Thumbnails use CSS object-fit: cover for consistent
//     sizing without layout shifts.
//   - The modal uses a simple overlay — no library needed.

async function loadDocumentImages(documentId) {
    // Fetch the list of extracted images from the backend.
    // Returns an array of {filename, page, size_bytes, format}.

    try {
        const response = await fetch(
            `/documents/${documentId}/images`
        );

        if (!response.ok) return [];

        const data = await response.json();
        return data.images || [];

    } catch (err) {
        console.error(
            "Failed to load images for", documentId, err
        );
        return [];
    }
}


function renderImageGallery(documentId, images, containerEl) {
    // Renders a grid of image thumbnails inside the given
    // container element. Each thumbnail links to the full
    // image via the API endpoint.
    //
    // Args:
    //   documentId: The document's unique ID (for URL building)
    //   images: Array from loadDocumentImages()
    //   containerEl: DOM element to render into

    if (!images || images.length === 0) {
        containerEl.innerHTML = "";
        return;
    }

    const galleryHtml = `
        <div class="image-gallery-header">
            Images (${images.length})
        </div>
        <div class="image-gallery-grid">
            ${images.map(img => `
                <div class="image-thumb"
                     onclick="openImageModal('/documents/${documentId}/images/${img.filename}')"
                     title="Page ${img.page} — ${img.filename} (${formatFileSize(img.size_bytes)})">
                    <img src="/documents/${documentId}/images/${img.filename}"
                         alt="Page ${img.page} image"
                         loading="lazy">
                    <div class="image-thumb-label">p${img.page}</div>
                </div>
            `).join("")}
        </div>
    `;

    containerEl.innerHTML = galleryHtml;
}


function formatFileSize(bytes) {
    // Convert bytes to human-readable size string.
    // 1024 → "1.0 KB", 1048576 → "1.0 MB"
    if (bytes < 1024) return bytes + " B";

    const kb = bytes / 1024;
    if (kb < 1024) return kb.toFixed(1) + " KB";

    const mb = kb / 1024;
    return mb.toFixed(1) + " MB";
}


// ---------------------------------------------------------
// Image Modal — fullscreen overlay for viewing images
// ---------------------------------------------------------
//
// A lightweight modal that shows the full-size image when
// a thumbnail is clicked. Closes on click outside the image,
// Escape key, or the X button.
//
// WHY NOT window.open()?
//   Opening a new tab/window for each image is jarring and
//   creates tab clutter. A modal overlay keeps the user in
//   context and is the standard UX for image galleries.

function openImageModal(imageUrl) {

    // Create modal overlay
    const modal = document.createElement("div");
    modal.className = "image-modal";
    modal.id = "image-modal";

    modal.innerHTML = `
        <div class="image-modal-backdrop"
             onclick="closeImageModal()"></div>
        <div class="image-modal-content">
            <button class="image-modal-close"
                    onclick="closeImageModal()"
                    title="Close (Escape)">
                &#10005;
            </button>
            <img src="${imageUrl}"
                 alt="Document image"
                 class="image-modal-img">
        </div>
    `;

    document.body.appendChild(modal);

    // Trigger entrance animation on next frame
    requestAnimationFrame(() => {
        modal.classList.add("show");
    });

    // Close on Escape key
    const escHandler = (e) => {
        if (e.key === "Escape") {
            closeImageModal();
            document.removeEventListener("keydown", escHandler);
        }
    };
    document.addEventListener("keydown", escHandler);
}


function closeImageModal() {
    const modal = document.getElementById("image-modal");
    if (!modal) return;

    modal.classList.remove("show");

    // Wait for fade-out animation, then remove from DOM
    setTimeout(() => modal.remove(), 200);
}


async function deleteDocument(documentId) {

    // Phase 6: Use non-blocking confirmation instead of
    // browser confirm() which freezes the entire UI.
    // We proceed directly and show a toast notification
    // after deletion. The user can re-upload if needed.

    try {
        const deletedEl = document.getElementById(
            "doc-" + documentId
        );

        // Get filename before deletion for the toast
        const filename = deletedEl
            ? deletedEl.dataset.filename
            : documentId;

        if (
            deletedEl &&
            deletedEl.dataset.filename === activeDocumentSource
        ) {
            activeDocumentSource = null;
        }

        await fetch(`/documents/${documentId}`, {
            method: "DELETE",
        });

        showToast(`"${filename}" removed`, "success");
        await loadDocuments();

    } catch (err) {
        console.error("Delete failed:", err);
        showToast("Failed to delete document", "error");
    }
}


// ---------------------------------------------------------
// Document selection — sidebar item click handler
// ---------------------------------------------------------
//
// When user clicks a document in the sidebar, we:
//   1. Toggle it as the active document source
//   2. Update the sidebar highlighting
//   3. Update the document switcher chips above the input

function selectDocument(el) {

    const filename = el.dataset.filename;
    const documentId = el.dataset.documentId;

    // Hide all image galleries first
    document.querySelectorAll(".doc-image-gallery")
        .forEach(g => g.classList.remove("visible"));

    if (activeDocumentSource === filename) {
        // Clicking the same document again deselects it
        // → goes back to "All Documents" mode
        activeDocumentSource = null;
        el.classList.remove("active");
    } else {
        // Select this document, deselect any previous
        document.querySelectorAll(".doc-item.active")
            .forEach(item => item.classList.remove("active"));

        activeDocumentSource = filename;
        el.classList.add("active");

        // Phase 8: Load and show image gallery for this document.
        // We load lazily — only fetch images the first time a
        // document is selected. After that, the gallery stays
        // in the DOM and we just toggle visibility.
        if (documentId) {
            const galleryEl = document.getElementById(
                "gallery-" + documentId
            );

            if (galleryEl) {
                galleryEl.classList.add("visible");

                // Only fetch once (data-loaded flag)
                if (galleryEl.dataset.loaded === "false") {
                    galleryEl.dataset.loaded = "true";

                    loadDocumentImages(documentId).then(images => {
                        if (images.length > 0) {
                            renderImageGallery(
                                documentId, images, galleryEl
                            );
                        }
                    });
                }
            }
        }
    }

    // Update the chips above the text input
    updateDocumentSwitcher();

    // Close sidebar on mobile after selection
    if (window.innerWidth <= 768) {
        closeSidebar();
    }
}


// ---------------------------------------------------------
// selectDocumentByChip — chip click handler
// ---------------------------------------------------------
//
// Called when user clicks a document chip above the input.
// Syncs the sidebar selection and updates all chip states.

function selectDocumentByChip(filename) {

    if (filename === null) {
        // "All Documents" chip clicked
        activeDocumentSource = null;
    } else if (activeDocumentSource === filename) {
        // Same chip clicked again → deselect → "All"
        activeDocumentSource = null;
    } else {
        activeDocumentSource = filename;
    }

    // Sync sidebar selection
    document.querySelectorAll(".doc-item.active")
        .forEach(item => item.classList.remove("active"));

    if (activeDocumentSource) {
        const match = document.querySelector(
            `.doc-item[data-filename="${activeDocumentSource}"]`
        );
        if (match) match.classList.add("active");
    }

    updateDocumentSwitcher();
}


// ---------------------------------------------------------
// updateDocumentSwitcher — renders chips above the input
// ---------------------------------------------------------
//
// Shows each uploaded document as a clickable pill with
// a file type icon. The active document is highlighted
// green. An "All" chip appears when 2+ docs are uploaded.
//
// FILE TYPE ICONS:
//   PDF  → red document icon
//   DOCX → blue document icon
//   CSV  → green spreadsheet icon
//   TXT  → gray text icon

function getFileIcon(filename) {
    // Returns an emoji icon based on file extension.
    // Phase 2: Added image format icons for OCR uploads.
    const ext = filename.split(".").pop().toLowerCase();
    switch (ext) {
        case "pdf":  return "📄";  // page icon
        case "docx": return "📃";  // page with curl
        case "csv":  return "📊";  // bar chart
        case "txt":  return "📝";  // memo
        case "png":  return "🖼️";  // framed picture
        case "jpg":  return "🖼️";  // framed picture
        case "jpeg": return "🖼️";  // framed picture
        case "tiff": return "🖼️";  // framed picture
        case "bmp":  return "🖼️";  // framed picture
        default:     return "📁";  // folder
    }
}

function updateDocumentSwitcher() {

    const switcher = document.getElementById("doc-switcher");
    if (!switcher) return;

    // Get all document items from the sidebar
    const docItems = document.querySelectorAll(".doc-item");

    if (docItems.length === 0) {
        switcher.innerHTML = "";
        return;
    }

    // Build chips HTML
    let chipsHtml = "";

    // Add "All" chip only when multiple documents exist
    if (docItems.length > 1) {
        const allActive = !activeDocumentSource ? "active" : "";
        chipsHtml += `
            <div class="doc-chip chip-all ${allActive}"
                 onclick="selectDocumentByChip(null)"
                 title="Search all documents">
                <span class="chip-icon">📚</span>
                <span class="chip-name">All</span>
            </div>
        `;
    }

    // Add a chip for each document
    docItems.forEach(item => {
        const filename = item.dataset.filename;
        const isActive = activeDocumentSource === filename;
        const activeClass = isActive ? "active" : "";
        const icon = getFileIcon(filename);

        // Show short name (without long prefix hash)
        const displayName = filename.length > 25
            ? filename.substring(0, 22) + "..."
            : filename;

        chipsHtml += `
            <div class="doc-chip ${activeClass}"
                 onclick="selectDocumentByChip('${escapeHtml(filename)}')"
                 title="${escapeHtml(filename)}">
                <span class="chip-icon">${icon}</span>
                <span class="chip-name">${escapeHtml(displayName)}</span>
            </div>
        `;
    });

    switcher.innerHTML = chipsHtml;
}


// ===========================================================
// SMART SUGGESTIONS — Phase 4: Clickable question chips
// ===========================================================
//
// TWO TYPES OF SUGGESTIONS:
//
//   1. DOCUMENT SUGGESTIONS (after upload):
//      When a document finishes processing, we call
//      GET /suggestions?source=filename.pdf to fetch
//      4 starter questions the LLM generated based on
//      the document content. These help users who don't
//      know what to ask.
//
//   2. FOLLOW-UP SUGGESTIONS (after each answer):
//      The streaming response includes a "suggestions"
//      event with 3 follow-up questions. These guide
//      the conversation naturally ("What else can you
//      tell me about X?").
//
// HOW THE UI WORKS:
//
//   Suggestion chips appear as rounded pills above the
//   input bar. Clicking one:
//     1. Fills the input with the question text
//     2. Immediately sends the message
//     3. Clears the chips (new ones come with the answer)
//
//   This creates a conversational flow where the user can
//   just click through suggestions without typing.


// ---------------------------------------------------------
// renderSuggestionChips — display clickable question pills
// ---------------------------------------------------------
//
// Takes an array of question strings and renders them as
// clickable chips in the suggestion container. Each chip
// has a fade-in animation for a polished feel.
//
// Args:
//   suggestions — array of question strings
//                 e.g. ["Summarize this", "Who scored highest?"]
//
// Called from:
//   - fetchDocumentSuggestions() — after document upload
//   - handleStreamEvent() — when "suggestions" event arrives

function renderSuggestionChips(suggestions) {

    const container = document.getElementById("suggestion-chips");
    if (!container) return;

    // Store current suggestions in state
    currentSuggestions = suggestions || [];

    // Clear existing chips
    container.innerHTML = "";

    // If no suggestions, hide the container
    if (!currentSuggestions.length) return;

    // Build chips HTML
    // Each chip is a <button> for accessibility (keyboard
    // navigation, screen readers). The ::before pseudo-element
    // in CSS adds a sparkle icon automatically.
    currentSuggestions.forEach(question => {

        const chip = document.createElement("button");
        chip.className = "suggestion-chip";
        chip.textContent = question;
        chip.title = question;  // Full text on hover

        // Click handler: fill input and send immediately
        chip.addEventListener("click", () => {
            sendSuggestion(question);
        });

        container.appendChild(chip);
    });
}


// ---------------------------------------------------------
// sendSuggestion — handle suggestion chip click
// ---------------------------------------------------------
//
// When a user clicks a suggestion chip:
//   1. Fill the input textarea with the question
//   2. Clear the suggestion chips (they're consumed)
//   3. Send the message automatically
//
// WHY CLEAR CHIPS BEFORE SENDING?
//   The old suggestions are no longer relevant once the
//   user asks a new question. New follow-up suggestions
//   will arrive with the AI's answer.

function sendSuggestion(question) {

    const input = document.getElementById("question");
    if (!input) return;

    // Fill the input with the clicked question
    input.value = question;

    // Clear current chips — new ones will come with the answer
    renderSuggestionChips([]);

    // Send the message immediately
    sendMessage();
}


// ---------------------------------------------------------
// fetchDocumentSuggestions — get starter questions after upload
// ---------------------------------------------------------
//
// Called after a document finishes processing (status: "ready").
// Fetches pre-generated questions from the server:
//   GET /suggestions?source=report.pdf
//
// The server generates these in the background during document
// processing, so they're usually ready by the time we ask.
//
// If no suggestions are available (server still generating,
// or generation failed), we just don't show chips — the user
// can still type their own question.

async function fetchDocumentSuggestions(filename) {

    try {
        const response = await fetch(
            `/suggestions?source=${encodeURIComponent(filename)}`
        );

        if (!response.ok) return;

        const data = await response.json();
        const suggestions = data.suggestions || [];

        if (suggestions.length > 0) {
            renderSuggestionChips(suggestions);
        }

    } catch (err) {
        // Suggestions are optional — if the fetch fails,
        // the user can still type their own questions.
        console.warn(
            "Failed to fetch suggestions:", err.message
        );
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

    // Clear suggestion chips — the old suggestions are no
    // longer relevant once the user sends a new question.
    // New follow-up suggestions will arrive with the AI's
    // answer via the "suggestions" streaming event.
    renderSuggestionChips([]);

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

        // -------------------------------------------------
        // Handle rate limiting (Phase 1)
        // -------------------------------------------------
        //
        // If the server returns 429 (Too Many Requests),
        // the user has hit their daily question limit.
        // We show a friendly message and refresh the
        // usage counter.
        //
        // WHY HANDLE 429 SEPARATELY?
        //
        //   Other HTTP errors (500, 503) are server problems.
        //   429 is a deliberate policy decision — the user
        //   isn't doing anything wrong, they've just used
        //   their daily allowance. The message should be
        //   friendly, not alarming.

        if (response.status === 429) {
            const errData = await response.json();
            answerEl.innerHTML = "";
            statusEl.textContent = "";
            answerEl.textContent =
                errData.error ||
                "Daily question limit reached. " +
                "Come back tomorrow!";
            answerEl.style.color = "#dc2626";
            // Refresh usage display
            await fetchUsage();
            return;
        }

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

            // Phase 6 optimization: use debounced scroll
            // instead of scrolling on every single token.
            // Reduces 200+ forced reflows to ~60 per second.
            debouncedScroll(chat);
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

        // Refresh the chat history sidebar so the
        // current conversation appears (or updates its
        // title/timestamp) in the sidebar list.
        loadChatHistory();

        // Refresh usage counter after each question
        // so the counter updates immediately (Phase 1)
        fetchUsage();
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

        // -------------------------------------------------
        // Cache hit indicator
        // -------------------------------------------------
        //
        // WHY A SPECIAL CASE?
        //
        //   When the server finds a cached answer, it sends
        //   a status event with status="cache_hit". We show
        //   a lightning bolt icon to make it clear this is
        //   a fast cached response, not a fresh LLM call.
        //
        //   This helps the user understand WHY the answer
        //   came back so fast (< 1 second vs 10-30 seconds).
        //
        //   The "cache_hit" icon is intentionally different
        //   from other status icons to stand out visually.
        const icon = icons[event.status]
                  || (event.status === "cache_hit"
                      ? "&#9889;"   // ⚡ lightning bolt
                      : "&#9203;"); // ⏫ default hourglass

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

    // ---------------------------------------------------------
    // html_content — Phase 3 interactive content (quiz, etc.)
    // ---------------------------------------------------------
    // When the agent generates interactive content (quiz,
    // summary, article, Q&A, flashcards), it sends the
    // complete HTML as an "html_content" event instead of
    // streaming tokens.  We render it inside a sandboxed
    // iframe so the embedded Tailwind CSS / JS cannot
    // interfere with the parent page styles.
    // ---------------------------------------------------------
    if (event.type === "html_content") {

        // Remove the typing dots — content has arrived
        const indicator = answerEl.querySelector(".typing-indicator");
        if (indicator) indicator.remove();

        // Build an iframe to hold the generated HTML.
        // "srcdoc" lets us inject a full HTML document
        // without needing a separate URL / file.
        const iframe = document.createElement("iframe");

        // sandbox="allow-scripts" lets the embedded JS
        // (quiz scoring, flashcard flips) run, while
        // still blocking top-navigation, form submits
        // to external URLs, etc.
        iframe.sandbox = "allow-scripts";

        // Visual styling — the iframe should blend into
        // the chat bubble seamlessly.
        iframe.style.width      = "100%";
        iframe.style.border     = "none";
        iframe.style.borderRadius = "8px";
        iframe.style.minHeight  = "200px";   // avoid flicker
        iframe.style.overflow   = "hidden";
        iframe.style.display    = "block";

        // Write the HTML into the iframe via srcdoc.
        // srcdoc is supported in all modern browsers and
        // avoids a network round-trip.
        iframe.srcdoc = event.html;

        // Clear the answer bubble and insert the iframe
        answerEl.innerHTML = "";
        answerEl.appendChild(iframe);

        // Store the content type so we can label it
        // (e.g. "Quiz", "Summary") in the status bar
        answerEl.setAttribute(
            "data-content-type", event.content_type || ""
        );

        // Add a download button below the iframe so the
        // user can save just this output (flashcards, quiz,
        // summary) as a standalone HTML file.
        addDownloadButton(
            answerEl, iframe, event.content_type || "content"
        );

        // -------------------------------------------------
        // Auto-resize: the embedded HTML sends a
        // postMessage({ type: "resize", height: N }) when
        // its body size changes (see _wrap_html() in
        // content_renderer.py).  We listen for that here
        // and adjust the iframe height so there is no
        // inner scrollbar.
        // -------------------------------------------------
        window.addEventListener("message", function resizer(msg) {

            // Safety: only react to our own iframe's
            // messages (check source === iframe window)
            if (msg.source !== iframe.contentWindow) return;

            if (
                msg.data &&
                msg.data.type === "resize" &&
                typeof msg.data.height === "number"
            ) {
                // Add a small buffer (16px) to prevent
                // a scrollbar from appearing due to
                // sub-pixel rounding differences.
                iframe.style.height =
                    (msg.data.height + 16) + "px";
            }
        });

        // Fallback: if the iframe's onload fires before
        // the postMessage listener is ready, set an
        // initial height from the iframe's own content.
        iframe.addEventListener("load", function () {
            try {
                const doc = iframe.contentDocument
                         || iframe.contentWindow.document;
                if (doc && doc.body) {
                    iframe.style.height =
                        (doc.body.scrollHeight + 16) + "px";
                }
            } catch (_) {
                // Cross-origin — sandbox may block this;
                // the postMessage path will handle it.
            }
        });
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

        // If this was an interactive content response,
        // show the content type in the status bar so the
        // user knows what was generated.
        // -------------------------------------------------
        // Show completion status with optional cache label
        // -------------------------------------------------
        //
        // If the answer came from cache (event.cached=true),
        // we append "(cached)" to the status so the user
        // can see it was a fast cache hit vs a fresh answer.
        //
        // The ⚡ icon replaces ✅ for cached answers to
        // make the speed difference visually obvious.

        const isCached = event.cached === true;
        const cacheLabel = isCached ? " (cached)" : "";
        const doneIcon = isCached
            ? "&#9889;"   // ⚡ lightning bolt for cached
            : "&#9989;";  // ✅ checkmark for fresh

        const cType = answerEl.getAttribute("data-content-type");
        if (cType) {
            // Capitalise first letter for display
            const label = cType.charAt(0).toUpperCase()
                        + cType.slice(1);
            statusEl.innerHTML =
                doneIcon + " " + escapeHtml(label)
                + " generated" + cacheLabel;
        } else {
            statusEl.innerHTML =
                doneIcon + " Completed" + cacheLabel;
        }

        timingEl.textContent =
            `Response time: ${event.response_time_seconds}s`;

        // Phase 6: Add copy button to the completed message.
        // The raw markdown text is stored in data-raw attribute
        // by the token handler, so we copy that (not the HTML).
        addCopyButton(answerEl);
    }

    // ---------------------------------------------------------
    // Follow-up suggestions — Phase 4 Smart Suggestions
    // ---------------------------------------------------------
    //
    // After the AI finishes answering, the server sends a
    // "suggestions" event with 3 follow-up questions. These
    // are contextual — based on what was just asked and answered.
    //
    // Example:
    //   User asked: "Who scored highest?"
    //   AI answered: "Ananya scored 95.2%"
    //   Suggestions: [
    //     "What were Ananya's subject scores?",
    //     "Who scored the lowest?",
    //     "How many students scored above 90%?"
    //   ]
    //
    // We render these as clickable chips above the input,
    // replacing any previous chips.

    if (event.type === "suggestions") {

        const questions = event.questions || [];

        if (questions.length > 0) {
            renderSuggestionChips(questions);
        }
    }

    if (event.type === "error") {
        statusEl.innerHTML =
            "&#10060; Error: " + escapeHtml(event.message);
        timingEl.textContent =
            event.elapsed_seconds != null
                ? `Failed after ${event.elapsed_seconds}s`
                : "Request failed";
    }
}


// ===========================================================
// DOWNLOAD BUTTON for interactive content
// ===========================================================
//
// WHY THIS EXISTS:
//
//   The existing export (Ctrl+Shift+E) downloads the ENTIRE
//   chat session as one HTML file. But when the user generates
//   a flashcard set, quiz, or summary, they often want to
//   download JUST that specific output — to study offline,
//   share with classmates, or print.
//
//   This function adds a small download button below each
//   interactive content iframe. Clicking it downloads the
//   iframe's HTML as a standalone file.
//
// HOW IT WORKS:
//
//   1. Creates a button element styled to match the UI
//   2. On click, extracts the iframe's srcdoc (the HTML)
//   3. Wraps it in a Blob and triggers a file download
//   4. Filename includes the content type and timestamp
//      e.g. "flashcards-2026-08-21.html"

function addDownloadButton(containerEl, iframe, contentType) {

    // Create a wrapper div for the button so it sits
    // below the iframe content, right-aligned
    const btnWrapper = document.createElement("div");
    btnWrapper.style.display = "flex";
    btnWrapper.style.justifyContent = "flex-end";
    btnWrapper.style.marginTop = "8px";
    btnWrapper.style.gap = "8px";

    // Build the download button
    //
    // WHY NOT className = "copy-btn"?
    //   The copy-btn class has position:absolute, top:4px,
    //   right:4px, and opacity:0. That makes it fly to the
    //   top-right corner of the message and stay invisible
    //   until hover. We need this button to be ALWAYS visible
    //   and BELOW the iframe content, so we use inline styles
    //   that match the copy-btn look without the positioning.
    const downloadBtn = document.createElement("button");
    downloadBtn.title = `Download ${contentType || "content"} as HTML`;

    // Inline styles — same look as copy-btn but with
    // position:static (default) so it stays in flow
    downloadBtn.style.display = "flex";
    downloadBtn.style.alignItems = "center";
    downloadBtn.style.gap = "4px";
    downloadBtn.style.padding = "6px 14px";
    downloadBtn.style.border = "1px solid #ddd";
    downloadBtn.style.borderRadius = "6px";
    downloadBtn.style.background = "#fff";
    downloadBtn.style.color = "#666";
    downloadBtn.style.fontSize = "12px";
    downloadBtn.style.cursor = "pointer";
    downloadBtn.style.transition = "background 0.15s, color 0.15s";

    // Hover effect via event listeners (inline styles
    // can't do :hover pseudo-class)
    downloadBtn.addEventListener("mouseenter", () => {
        downloadBtn.style.background = "#f3f3f3";
        downloadBtn.style.color = "#333";
        downloadBtn.style.borderColor = "#bbb";
    });
    downloadBtn.addEventListener("mouseleave", () => {
        downloadBtn.style.background = "#fff";
        downloadBtn.style.color = "#666";
        downloadBtn.style.borderColor = "#ddd";
    });

    // Down-arrow icon + label
    downloadBtn.innerHTML = "&#11015; Download";

    downloadBtn.addEventListener("click", () => {

        // Get the HTML content from the iframe
        const html = iframe.srcdoc || iframe.getAttribute("srcdoc");

        if (!html) {
            showToast("No content to download", "info");
            return;
        }

        // Create a Blob with the HTML content
        const blob = new Blob([html], { type: "text/html" });
        const url = URL.createObjectURL(blob);

        // Generate a descriptive filename
        // e.g. "flashcards-2026-08-21.html"
        const now = new Date();
        const dateStr = now.toISOString().slice(0, 10);
        const label = contentType || "content";
        const filename = `${label}-${dateStr}.html`;

        // Create a temporary <a> element to trigger download
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();

        // Clean up
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast(
            `Downloaded ${label} as ${filename}`, "success"
        );
    });

    btnWrapper.appendChild(downloadBtn);
    containerEl.appendChild(btnWrapper);
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


// ===========================================================
// CHAT HISTORY — Conversation management
// ===========================================================
//
// These functions manage the chat history sidebar, which
// lets users:
//   1. See a list of past conversations
//   2. Click one to resume it (loads old messages)
//   3. Start a new conversation ("New Chat" button)
//   4. Delete old conversations
//
// HOW IT WORKS:
//
//   - Conversations are stored in SQLite on the server
//   - The sidebar fetches the list via GET /sessions
//   - Clicking a conversation loads its messages via
//     GET /sessions/{id}/messages
//   - The current session_id is stored in localStorage
//     so it persists across page reloads


// ---------------------------------------------------------
// loadChatHistory — fetch and display past conversations
// ---------------------------------------------------------
//
// Calls GET /sessions to get the list of conversations,
// then renders them in the sidebar. The active conversation
// is highlighted. Each item shows:
//   - Title (auto-generated from first question)
//   - Relative time ("2 min ago", "Yesterday")
//   - Delete button (appears on hover)

async function loadChatHistory() {

    try {

        const response = await fetch("/sessions");
        const data = await response.json();
        const sessions = data.sessions || [];

        const listEl = document.getElementById("chat-list");

        if (sessions.length === 0) {
            listEl.innerHTML = `
                <div class="chat-empty">
                    No conversations yet.<br>
                    Ask a question to start.
                </div>
            `;
            return;
        }

        // Build HTML for each conversation item
        listEl.innerHTML = sessions.map(session => {

            // Is this the currently active conversation?
            const isActive = session.id === SESSION_ID;
            const activeClass = isActive ? "active" : "";

            // Format the timestamp as relative time
            const timeAgo = formatTimeAgo(session.updated_at);

            // Message count badge
            const msgCount = session.message_count || 0;

            return `
                <div class="chat-item ${activeClass}"
                     data-session-id="${session.id}"
                     onclick="switchToSession('${session.id}')">
                    <div class="chat-item-icon">&#128172;</div>
                    <div class="chat-item-info">
                        <div class="chat-item-title"
                             title="${escapeHtml(session.title)}">
                            ${escapeHtml(session.title)}
                        </div>
                        <div class="chat-item-time">
                            ${timeAgo}${msgCount > 0 ? ' · ' + msgCount + ' msgs' : ''}
                        </div>
                    </div>
                    <button
                        class="chat-delete"
                        onclick="event.stopPropagation(); deleteChatSession('${session.id}')"
                        title="Delete conversation"
                    >
                        &#10005;
                    </button>
                </div>
            `;
        }).join("");

    } catch (err) {
        console.error("Failed to load chat history:", err);
    }
}


// ---------------------------------------------------------
// loadCurrentSessionMessages — restore chat on page reload
// ---------------------------------------------------------
//
// THE PROBLEM THIS SOLVES:
//
//   When the page reloads, SESSION_ID is restored from
//   localStorage and the sidebar shows the correct session
//   as active. But the CHAT AREA stays empty because
//   nothing fetches that session's messages.
//
//   switchToSession() can't be reused here because it has
//   a guard: `if (sessionId === SESSION_ID) return;`
//   That guard exists to prevent reloading the same session
//   when clicking the already-active sidebar item, but it
//   also blocks the reload case where SESSION_ID is already
//   set from localStorage.
//
// THE FIX:
//
//   This dedicated function runs once at startup. It:
//     1. Checks if SESSION_ID was restored from localStorage
//        (as opposed to freshly generated — a fresh UUID
//         won't exist in the database yet)
//     2. Fetches that session's messages from the server
//     3. Renders them in the chat area
//     4. Handles edge cases:
//        - New session (no messages) → show welcome message
//        - Deleted session (404/not found) → do nothing,
//          the user sees the default welcome screen
//
// CALLED FROM:
//   DOMContentLoaded handler, AFTER loadChatHistory()
//   completes (so the sidebar is already built and the
//   active session is highlighted).

async function loadCurrentSessionMessages() {

    // Only restore if we have a saved session ID
    // (not a freshly generated one that has no messages yet)
    if (!SESSION_ID) return;

    try {

        // Fetch messages for the current session
        const response = await fetch(
            `/sessions/${SESSION_ID}/messages`
        );

        // If the session doesn't exist in the database
        // (e.g., it was deleted, or this is a brand-new UUID),
        // just leave the chat area as-is (welcome screen).
        if (!response.ok) return;

        const data = await response.json();
        const messages = data.messages || [];

        // No messages saved for this session yet —
        // nothing to restore, keep the default view.
        if (messages.length === 0) return;

        // Clear any placeholder content in the chat area
        const chat = document.getElementById("chat-messages");
        chat.innerHTML = "";

        // Unlock the chat interface (in case it was locked)
        unlockChat();

        // Render each saved message using the same function
        // that switchToSession() uses — this ensures loaded
        // messages look identical to live ones (same DOM
        // structure, formatting, copy buttons, etc.)
        // Pass content_type and html_content so interactive
        // content (flashcards, quizzes) renders in iframes.
        for (const msg of messages) {
            renderHistoryMessage(
                msg.role,
                msg.content,
                msg.content_type || null,
                msg.html_content || null,
            );
        }

        // Scroll to the bottom so the user sees the most
        // recent messages (same behavior as live chat)
        chat.scrollTop = chat.scrollHeight;

        console.log(
            `Restored ${messages.length} messages for session ${SESSION_ID.slice(0, 8)}`
        );

    } catch (err) {
        // Network error or server down — silently fail.
        // The user still has the empty chat UI and can
        // manually click a session in the sidebar.
        console.error(
            "Failed to restore session messages:", err
        );
    }
}


// ---------------------------------------------------------
// switchToSession — load a past conversation
// ---------------------------------------------------------
//
// Called when the user clicks a conversation in the sidebar.
// Steps:
//   1. Update SESSION_ID to the selected conversation
//   2. Save to localStorage (persist across reloads)
//   3. Fetch all messages for that session from the server
//   4. Clear the chat area and re-render all messages
//   5. Update the sidebar to highlight the active chat

async function switchToSession(sessionId) {

    // Don't reload if already viewing this session
    if (sessionId === SESSION_ID) return;

    // Update the active session
    SESSION_ID = sessionId;
    localStorage.setItem("session_id", SESSION_ID);

    // Fetch messages for this session
    try {

        const response = await fetch(
            `/sessions/${sessionId}/messages`
        );

        if (!response.ok) {
            console.error("Failed to load session messages");
            return;
        }

        const data = await response.json();
        const messages = data.messages || [];

        // Clear the chat area
        const chat = document.getElementById("chat-messages");
        chat.innerHTML = "";

        // Show the chat area (in case it was hidden)
        unlockChat();

        // Re-render each message in the chat area
        // This recreates the same DOM structure as the
        // streaming handler, so the messages look identical
        // to when they were first received.

        if (messages.length === 0) {
            // Empty session — show welcome message
            chat.innerHTML = `
                <div class="message">
                    <div class="message-inner">
                        <div class="message-avatar">AI</div>
                        <div class="message-body">
                            <div class="message-content">
                                Your documents are ready. Ask me anything
                                — summaries, specific data, comparisons,
                                or analysis.
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Render each saved message.
            // Pass content_type and html_content so that
            // interactive content (flashcards, quizzes)
            // gets rendered in an iframe, not as plain text.
            for (const msg of messages) {
                renderHistoryMessage(
                    msg.role,
                    msg.content,
                    msg.content_type || null,
                    msg.html_content || null,
                );
            }
        }

        // Update sidebar highlighting
        document.querySelectorAll(".chat-item.active")
            .forEach(el => el.classList.remove("active"));

        const activeItem = document.querySelector(
            `.chat-item[data-session-id="${sessionId}"]`
        );
        if (activeItem) activeItem.classList.add("active");

        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            closeSidebar();
        }

        // Scroll to bottom
        chat.scrollTop = chat.scrollHeight;

    } catch (err) {
        console.error("Failed to switch session:", err);
    }
}


// ---------------------------------------------------------
// renderHistoryMessage — display a saved message
// ---------------------------------------------------------
//
// Creates the same DOM structure as the streaming handler
// so that loaded messages look identical to live ones.
// Used when loading a past conversation from the database.
//
// HANDLES TWO TYPES OF MESSAGES:
//
//   1. Normal text messages (content_type is null):
//      - User messages: rendered as plain text
//      - Assistant messages: rendered via formatText()
//        (markdown → HTML with bold, italic, tables, etc.)
//
//   2. Interactive content (content_type is set):
//      - Flashcards, quizzes, summaries, articles, Q&A
//      - The html_content field contains the full HTML
//      - Rendered inside a sandboxed iframe, exactly like
//        the live streaming handler does
//      - This is what was MISSING before — previously the
//        HTML was lost on page reload because we only saved
//        "[Generated flashcards]" as the content
//
// Args:
//   role:         "user" or "assistant"
//   content:      The message text (or label for HTML content)
//   contentType:  null for text, "flashcards"/"quiz"/etc.
//   htmlContent:  null for text, full HTML for interactive

function renderHistoryMessage(
    role, content, contentType = null, htmlContent = null
) {

    const chat = document.getElementById("chat-messages");

    const msg = document.createElement("div");
    msg.className = role === "user" ? "message user" : "message";

    const avatarLabel = role === "user" ? "You" : "AI";

    msg.innerHTML = `
        <div class="message-inner">
            <div class="message-avatar">${avatarLabel}</div>
            <div class="message-body">
                <div class="message-content"></div>
            </div>
        </div>
    `;

    const contentEl = msg.querySelector(".message-content");

    // -------------------------------------------------
    // Interactive content — render in iframe
    // -------------------------------------------------
    //
    // If this message has html_content, it's an interactive
    // element (flashcards, quiz, etc.) that needs to be
    // rendered inside a sandboxed iframe — the same way
    // the live streaming handler renders it.
    //
    // This is the key fix: previously, loading a session
    // from history would show "[Generated flashcards]"
    // as plain text because the HTML was never saved.
    // Now we save the HTML and re-render it here.

    if (contentType && htmlContent) {

        // Build an iframe — same setup as handleStreamEvent's
        // html_content handler (lines ~1535-1610)
        const iframe = document.createElement("iframe");

        // sandbox="allow-scripts" lets embedded JS run
        // (quiz scoring, flashcard flips) while blocking
        // top-navigation and form submits.
        iframe.sandbox = "allow-scripts";

        // Visual styling — blend into the chat bubble
        iframe.style.width         = "100%";
        iframe.style.border        = "none";
        iframe.style.borderRadius  = "8px";
        iframe.style.minHeight     = "200px";
        iframe.style.overflow      = "hidden";
        iframe.style.display       = "block";

        // Write the saved HTML into the iframe
        iframe.srcdoc = htmlContent;

        // Replace the content area with the iframe
        contentEl.innerHTML = "";
        contentEl.appendChild(iframe);

        // Store content type for potential future use
        // (e.g. labeling, export filtering)
        contentEl.setAttribute(
            "data-content-type", contentType
        );

        // Add download button — same as in the live
        // streaming handler, so history-loaded content
        // also has the download option.
        addDownloadButton(
            contentEl, iframe, contentType
        );

        // -------------------------------------------------
        // Auto-resize: listen for postMessage from iframe
        // -------------------------------------------------
        // The content_renderer.py wraps all HTML templates
        // with a resize observer that sends height updates
        // via postMessage. We listen for those here to
        // make the iframe fit its content perfectly.
        window.addEventListener("message", function resizer(e) {

            if (e.source !== iframe.contentWindow) return;

            if (
                e.data &&
                e.data.type === "resize" &&
                typeof e.data.height === "number"
            ) {
                iframe.style.height =
                    (e.data.height + 16) + "px";
            }
        });

        // Fallback: set initial height from content
        iframe.addEventListener("load", function () {
            try {
                const doc = iframe.contentDocument
                         || iframe.contentWindow.document;
                if (doc && doc.body) {
                    iframe.style.height =
                        (doc.body.scrollHeight + 16) + "px";
                }
            } catch (_) {
                // Cross-origin sandbox may block this;
                // the postMessage path handles it.
            }
        });

    } else if (role === "user") {
        // -------------------------------------------------
        // User message — plain text (no markdown)
        // -------------------------------------------------
        contentEl.textContent = content;

    } else {
        // -------------------------------------------------
        // Normal assistant message — render markdown
        // -------------------------------------------------
        contentEl.innerHTML = formatText(content);

        // Phase 6: Store raw text and add copy button
        // for assistant messages loaded from history
        contentEl.setAttribute("data-raw", content);
        addCopyButton(contentEl);
    }

    chat.appendChild(msg);
}


// ---------------------------------------------------------
// startNewChat — create a fresh conversation
// ---------------------------------------------------------
//
// Called when the user clicks the "New Chat" button (+) in
// the sidebar. Steps:
//   1. Generate a new session ID
//   2. Save to localStorage
//   3. Clear the chat area
//   4. Refresh the sidebar to show the new (empty) session
//   5. Focus the input field

async function startNewChat() {

    // Generate a new session ID
    SESSION_ID = crypto.randomUUID();
    localStorage.setItem("session_id", SESSION_ID);

    // Clear the chat area and show welcome message
    const chat = document.getElementById("chat-messages");

    chat.innerHTML = `
        <div class="message">
            <div class="message-inner">
                <div class="message-avatar">AI</div>
                <div class="message-body">
                    <div class="message-content">
                        Your documents are ready. Ask me anything
                        — summaries, specific data, comparisons,
                        or analysis.
                    </div>
                </div>
            </div>
        </div>
    `;

    // Reset the documentsReadyShown flag so it can
    // show the card again if needed
    documentsReadyShown = false;

    // Update sidebar — remove active highlighting
    document.querySelectorAll(".chat-item.active")
        .forEach(el => el.classList.remove("active"));

    // Show chat area in case it was hidden
    if (documentCount > 0) {
        unlockChat();
    }

    // Focus input
    document.getElementById("question").focus();

    // Close sidebar on mobile
    if (window.innerWidth <= 768) {
        closeSidebar();
    }
}


// ---------------------------------------------------------
// deleteChatSession — remove a conversation
// ---------------------------------------------------------
//
// Called when the user clicks the X button on a conversation
// in the sidebar. Deletes from the server (SQLite) and
// refreshes the sidebar list.

async function deleteChatSession(sessionId) {

    // Phase 6: Non-blocking delete — no more confirm()
    // freezing the page. We delete directly and show
    // a toast. The user can always start a new chat.

    try {

        await fetch(`/sessions/${sessionId}`, {
            method: "DELETE",
        });

        showToast("Conversation deleted", "success");

        // If we just deleted the active session,
        // start a new one so the user isn't stuck
        // viewing a deleted conversation.
        if (sessionId === SESSION_ID) {
            await startNewChat();
        }

        // Refresh the sidebar list
        await loadChatHistory();

    } catch (err) {
        console.error("Delete session failed:", err);
        showToast("Failed to delete conversation", "error");
    }
}


// ---------------------------------------------------------
// formatTimeAgo — human-readable relative timestamps
// ---------------------------------------------------------
//
// Converts an ISO timestamp into a friendly relative
// string like "2 min ago", "3 hours ago", "Yesterday".
//
// WHY RELATIVE TIME?
//   "2 min ago" is easier to scan than "2026-08-19T14:23:05Z"
//   when browsing a list of conversations. Users care about
//   recency, not exact timestamps.

function formatTimeAgo(isoString) {

    if (!isoString) return "";

    const date = new Date(isoString);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    // Less than a minute
    if (seconds < 60) return "Just now";

    // Minutes
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) {
        return minutes === 1
            ? "1 min ago"
            : `${minutes} min ago`;
    }

    // Hours
    const hours = Math.floor(minutes / 60);
    if (hours < 24) {
        return hours === 1
            ? "1 hour ago"
            : `${hours} hours ago`;
    }

    // Days
    const days = Math.floor(hours / 24);
    if (days === 1) return "Yesterday";
    if (days < 7) return `${days} days ago`;

    // Weeks
    const weeks = Math.floor(days / 7);
    if (weeks < 4) {
        return weeks === 1
            ? "1 week ago"
            : `${weeks} weeks ago`;
    }

    // Months
    const months = Math.floor(days / 30);
    if (months < 12) {
        return months === 1
            ? "1 month ago"
            : `${months} months ago`;
    }

    // Years
    return "Long ago";
}


// ===========================================================
// COPY BUTTON — Phase 6: One-click copy of AI responses
// ===========================================================
//
// Adds a small "Copy" button to the top-right corner of
// each AI message. Clicking it copies the raw markdown
// text (not the rendered HTML) to the clipboard.
//
// WHY RAW TEXT, NOT HTML?
//   When users paste into emails, docs, or chat apps,
//   they want clean text with markdown formatting —
//   not HTML tags like <strong> or <table>. The raw
//   text is stored in the data-raw attribute.

function addCopyButton(answerEl) {

    // Don't add duplicate copy buttons
    if (answerEl.querySelector(".copy-btn")) return;

    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.title = "Copy to clipboard";

    // Clipboard SVG icon
    btn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round"
             width="14" height="14">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
            <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
        </svg>
        <span>Copy</span>
    `;

    btn.addEventListener("click", async (e) => {
        e.stopPropagation();

        // Get the raw text (markdown) — prefer data-raw,
        // fall back to textContent (plain text extraction)
        const rawText = answerEl.getAttribute("data-raw")
            || answerEl.textContent;

        try {
            await navigator.clipboard.writeText(rawText);

            // Visual feedback: change button text briefly
            btn.querySelector("span").textContent = "Copied!";
            btn.classList.add("copied");

            setTimeout(() => {
                btn.querySelector("span").textContent = "Copy";
                btn.classList.remove("copied");
            }, 2000);

        } catch (err) {
            // Clipboard API may fail in some contexts
            // (e.g., non-HTTPS, iframe restrictions)
            showToast("Failed to copy", "error");
        }
    });

    // Insert the button at the top of the answer element.
    // We use position:relative on the parent and
    // position:absolute on the button for top-right placement.
    answerEl.style.position = "relative";
    answerEl.appendChild(btn);
}


// ===========================================================
// PHASE 7: EXPORT CONVERSATION — HTML & PDF
// ===========================================================
//
// Lets users export the current conversation as:
//   1. HTML — self-contained file with inline CSS, no
//      external dependencies. Opens in any browser.
//   2. PDF — uses the browser's built-in print dialog
//      (Ctrl+P → Save as PDF). Zero dependencies.
//
// WHY BROWSER PRINT FOR PDF?
//
//   Adding a PDF library (like jsPDF or pdfmake) would:
//     - Add 200-500KB to the page weight
//     - Require complex layout code for tables/formatting
//     - Still not look as good as the browser's renderer
//
//   Browser print is free, handles all CSS perfectly, and
//   users already know how to use "Save as PDF" in the
//   print dialog. We just add a print-optimized stylesheet
//   that hides the sidebar, input bar, and buttons.
//
// HOW IT WORKS:
//
//   exportConversation("html"):
//     1. Collect all messages from the chat area DOM
//     2. Build a standalone HTML document with inline CSS
//     3. Create a Blob and trigger download
//
//   exportConversation("pdf"):
//     1. Add print-specific CSS class to body
//     2. Call window.print() — browser shows print dialog
//     3. User selects "Save as PDF" in the dialog
//     4. Remove print class after dialog closes

async function exportConversation(format = "html") {

    const chat = document.getElementById("chat-messages");
    const messages = chat.querySelectorAll(".message");

    if (messages.length === 0) {
        showToast("No messages to export", "info");
        return;
    }

    if (format === "pdf") {
        // -------------------------------------------------
        // PDF export via browser print
        // -------------------------------------------------
        // Add a class that triggers print-specific CSS
        // (hides sidebar, input bar, shows only chat).
        // The print stylesheet is in styles.css.

        document.body.classList.add("print-mode");
        window.print();

        // Remove print class after dialog closes
        // (onafterprint fires when dialog is dismissed)
        window.addEventListener("afterprint", () => {
            document.body.classList.remove("print-mode");
        }, { once: true });

        // Fallback: remove after 5 seconds in case
        // afterprint doesn't fire (some browsers)
        setTimeout(() => {
            document.body.classList.remove("print-mode");
        }, 5000);

        return;
    }

    // -------------------------------------------------
    // HTML export — self-contained file
    // -------------------------------------------------
    // Collect messages and build a complete HTML document
    // with all styles inlined. No external dependencies.

    let messagesHtml = "";

    messages.forEach(msg => {
        const isUser = msg.classList.contains("user");
        const role = isUser ? "You" : "AI";
        const roleClass = isUser ? "user" : "assistant";

        const contentEl = msg.querySelector(".message-content");
        // Get rendered HTML content for export
        const content = contentEl
            ? contentEl.innerHTML
            : "";

        messagesHtml += `
            <div class="export-message export-${roleClass}">
                <div class="export-avatar">${role}</div>
                <div class="export-content">${content}</div>
            </div>
        `;
    });

    // Build the complete HTML document with inline styles
    const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Export — AI Document Agent</title>
    <style>
        /* Self-contained export styles — no external deps */
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, 'Segoe UI', Roboto,
                         Helvetica, Arial, sans-serif;
            background: #f7f7f8;
            color: #1a1a1a;
            line-height: 1.6;
        }

        .export-header {
            background: #171717;
            color: #fff;
            padding: 20px 24px;
            text-align: center;
        }

        .export-header h1 {
            font-size: 18px;
            font-weight: 600;
        }

        .export-header .export-date {
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }

        .export-messages {
            max-width: 760px;
            margin: 0 auto;
            padding: 20px 24px;
        }

        .export-message {
            display: flex;
            gap: 14px;
            padding: 20px 0;
            border-bottom: 1px solid #eee;
            align-items: flex-start;
        }

        .export-message:last-child {
            border-bottom: none;
        }

        .export-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 600;
            flex-shrink: 0;
        }

        .export-user .export-avatar {
            background: #6366f1;
            color: #fff;
        }

        .export-assistant .export-avatar {
            background: #e8e8ed;
            color: #555;
        }

        .export-content {
            flex: 1;
            font-size: 14px;
            line-height: 1.7;
            color: #333;
            word-wrap: break-word;
            min-width: 0;
        }

        .export-user {
            background: #fff;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 8px 0;
        }

        /* Table styles for exported tables */
        .table-wrapper {
            overflow-x: auto;
            margin: 12px 0;
            border-radius: 8px;
            border: 1px solid #e2e2e8;
        }

        .md-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .md-table th {
            padding: 10px 14px;
            text-align: left;
            font-weight: 600;
            background: #f0f0f5;
            border-bottom: 2px solid #d5d5dd;
        }

        .md-table td {
            padding: 8px 14px;
            border-bottom: 1px solid #eee;
        }

        code {
            background: #f0f0f5;
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 13px;
            color: #6366f1;
        }

        strong { font-weight: 600; }
        em { font-style: italic; }
    </style>
</head>
<body>
    <div class="export-header">
        <h1>AI Document Agent — Chat Export</h1>
        <div class="export-date">
            Exported on ${new Date().toLocaleDateString("en-US", {
                weekday: "long",
                year: "numeric",
                month: "long",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
            })}
        </div>
    </div>
    <div class="export-messages">
        ${messagesHtml}
    </div>
</body>
</html>`;

    // Create a Blob and trigger download
    // Blob = Binary Large Object — an in-memory file
    const blob = new Blob([html], { type: "text/html" });

    // Create a temporary <a> element to trigger download.
    // This is the standard way to programmatically download
    // a file in JavaScript (no server round-trip needed).
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;

    // Generate filename with date
    const dateStr = new Date().toISOString().slice(0, 10);
    a.download = `chat-export-${dateStr}.html`;

    // Trigger the download
    document.body.appendChild(a);
    a.click();

    // Cleanup: revoke the Blob URL to free memory
    // and remove the temporary <a> element
    setTimeout(() => {
        URL.revokeObjectURL(url);
        a.remove();
    }, 100);

    showToast("Conversation exported as HTML", "success");
}
