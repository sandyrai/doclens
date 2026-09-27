"""Content Renderer — HTML template engine for generated content.

PHASE 3: INTERACTIVE CONTENT GENERATION

  This module converts structured JSON data (from the LLM)
  into rich, interactive HTML that renders inside the chat.

  WHY A SEPARATE MODULE?

    The LLM's job is to EXTRACT content from documents
    (questions, summaries, key points). It's good at that.

    But free/small LLMs are BAD at generating valid HTML —
    they forget closing tags, break CSS, produce inconsistent
    styling. So we separate the concerns:

      LLM → generates structured JSON data
      content_renderer.py → wraps JSON in beautiful HTML

    This way:
    1. The UI is consistent every time (same template)
    2. We can change the look without touching the LLM
    3. The LLM focuses on content quality, not markup

  HOW IT WORKS:

    1. agent.py detects a content request ("create a quiz")
    2. agent.py sends a structured prompt to the LLM
    3. LLM returns JSON: {"type": "quiz", "questions": [...]}
    4. agent.py calls render_content(json_data)
    5. This module picks the right template and returns HTML
    6. The HTML is sent to the frontend for rendering

  TEMPLATES:

    1. QUIZ    — MCQ with radio buttons, submit, auto-scoring
    2. SUMMARY — numbered key points in a clean card
    3. ARTICLE — formatted paragraphs with headings
    4. QA      — collapsible question/answer accordion

  STYLING:

    All templates use Tailwind CSS via CDN for consistent,
    responsive styling. A small amount of custom CSS handles
    quiz-specific interactions (correct/wrong highlighting).

  ADDING A NEW TEMPLATE:

    1. Create a new render_xxx(data) function
    2. Add it to the RENDERERS dict at the bottom
    3. Add the content type to CONTENT_PATTERNS in agent.py
    4. Done — the pipeline handles everything else
"""

import logging
import json
from typing import Any


logger = logging.getLogger(__name__)


# =============================================================
# Shared HTML wrapper
# =============================================================
#
# Every template is wrapped in this base HTML structure.
# It loads Tailwind CSS from CDN and adds our custom styles
# for quiz interactions (correct/wrong highlighting).
#
# WHY TAILWIND?
#   - Utility-first: style directly in HTML classes
#   - No custom CSS files to manage
#   - Responsive out of the box
#   - Tiny learning curve: "bg-blue-500" = blue background
#
# The wrapper is a COMPLETE HTML document because the
# frontend renders it inside an iframe (for isolation).

def _wrap_html(title: str, body_html: str, extra_css: str = "", extra_js: str = "") -> str:
    """Wrap template body in a complete HTML document.

    Args:
        title: Page title (shown in iframe if visible).
        body_html: The template's inner HTML content.
        extra_css: Additional CSS specific to this template.
        extra_js: Additional JavaScript (e.g. quiz scoring).

    Returns:
        Complete HTML document string ready for iframe.
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>

    <!-- Tailwind CSS via CDN -->
    <script src="https://cdn.tailwindcss.com"></script>

    <!-- Custom Tailwind config for our color scheme -->
    <script>
    tailwind.config = {{
        theme: {{
            extend: {{
                colors: {{
                    // Our app's color palette — matches the
                    // dark theme of the main DocAgent UI
                    primary: {{
                        50:  '#eef2ff',
                        100: '#e0e7ff',
                        500: '#6366f1',  // indigo
                        600: '#4f46e5',
                        700: '#4338ca',
                    }},
                    surface: {{
                        dark:   '#1e1e2e',  // card background
                        darker: '#181825',  // page background
                        light:  '#313244',  // borders/dividers
                    }},
                    correct: '#22c55e',  // green for right answers
                    wrong:   '#ef4444',  // red for wrong answers
                }}
            }}
        }}
    }}
    </script>

    <style>
        /* -----------------------------------------------
         * Base styles for all content templates
         *
         * These handle the dark theme background, smooth
         * transitions for quiz feedback, and responsive
         * text sizing.
         * ----------------------------------------------- */
        body {{
            background-color: #181825;
            color: #cdd6f4;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 16px;
        }}

        /* Quiz option highlight transitions */
        .quiz-option {{
            transition: all 0.3s ease;
        }}
        .quiz-option.correct {{
            background-color: rgba(34, 197, 94, 0.15) !important;
            border-color: #22c55e !important;
        }}
        .quiz-option.wrong {{
            background-color: rgba(239, 68, 68, 0.15) !important;
            border-color: #ef4444 !important;
        }}

        /* Accordion animation for Q&A */
        .qa-answer {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        .qa-answer.open {{
            max-height: 500px;
        }}

        /* Custom scrollbar for dark theme */
        ::-webkit-scrollbar {{
            width: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: #181825;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #45475a;
            border-radius: 3px;
        }}

        {extra_css}
    </style>
</head>
<body>
    {body_html}

    <script>
    {extra_js}

    // -----------------------------------------------
    // AUTO-RESIZE: Tell parent iframe to resize
    //
    //   The main page (index.html) listens for this
    //   message and adjusts the iframe height to match
    //   the content — no scrollbars inside the iframe.
    // -----------------------------------------------
    function notifyParentSize() {{
        const height = document.body.scrollHeight;
        window.parent.postMessage(
            {{ type: 'resize', height: height }},
            '*'
        );
    }}

    // Notify on load and after any content change
    window.addEventListener('load', () => {{
        // Small delay to let Tailwind finish rendering
        setTimeout(notifyParentSize, 100);
    }});

    // Also notify when images load or content expands
    new ResizeObserver(notifyParentSize)
        .observe(document.body);
    </script>
</body>
</html>"""


# =============================================================
# Template 1: QUIZ
# =============================================================
#
# Interactive multiple-choice quiz with:
#   - Numbered questions with 4 radio-button options
#   - "Submit Quiz" button
#   - Auto-scoring in JavaScript (no server needed)
#   - Green/red highlighting for correct/wrong answers
#   - Score display with percentage
#
# Expected JSON format from LLM:
#   {
#     "type": "quiz",
#     "title": "Data Science MCQ Quiz",
#     "source": "AI_Chapter_2_Class_7.pdf",
#     "questions": [
#       {
#         "q": "What is the key part of AI?",
#         "options": ["Robotics", "Data science", ...],
#         "answer": 1  // 0-based index of correct option
#       },
#       ...
#     ]
#   }

def render_quiz(data: dict) -> str:
    """Render an interactive MCQ quiz from structured data.

    Args:
        data: Quiz JSON with title, source, and questions.

    Returns:
        Complete HTML document string.
    """

    title = data.get("title", "Quiz")
    source = data.get("source", "Document")
    questions = data.get("questions", [])

    if not questions:
        return _wrap_html(title, """
            <div class="text-center py-8">
                <p class="text-gray-400">No questions could be generated.</p>
            </div>
        """)

    # Build each question's HTML
    questions_html = ""

    for i, q in enumerate(questions):
        options_html = ""

        for j, opt in enumerate(q.get("options", [])):
            # Each option is a styled radio button label
            # data-correct marks whether this is the right answer
            is_correct = "true" if j == q.get("answer", -1) else "false"
            options_html += f"""
                <label class="quiz-option block cursor-pointer p-3 mb-2
                              rounded-lg border border-surface-light
                              hover:border-primary-500 hover:bg-surface-dark"
                       data-correct="{is_correct}">
                    <input type="radio" name="q{i}" value="{j}"
                           class="mr-3 accent-primary-500">
                    <span>{_escape_html(opt)}</span>
                </label>
            """

        questions_html += f"""
            <div class="quiz-question mb-6 p-4 rounded-xl bg-surface-dark
                        border border-surface-light"
                 id="question-{i}">
                <p class="text-lg font-semibold mb-3 text-white">
                    {i + 1}. {_escape_html(q.get('q', ''))}
                </p>
                <div class="options-container">
                    {options_html}
                </div>
            </div>
        """

    body = f"""
        <!-- Quiz Header -->
        <div class="max-w-2xl mx-auto">
            <div class="text-center mb-6">
                <h1 class="text-2xl font-bold text-white mb-2">
                    {_escape_html(title)}
                </h1>
                <p class="text-sm text-gray-400">
                    Source: {_escape_html(source)} |
                    {len(questions)} Questions
                </p>
            </div>

            <!-- Questions -->
            <form id="quiz-form" onsubmit="return false;">
                {questions_html}
            </form>

            <!-- Submit Button -->
            <div class="text-center mt-6">
                <button onclick="scoreQuiz()"
                        id="submit-btn"
                        class="px-8 py-3 bg-primary-600 hover:bg-primary-700
                               text-white font-semibold rounded-xl
                               transition-all duration-200
                               shadow-lg hover:shadow-xl">
                    Submit Quiz
                </button>
            </div>

            <!-- Result Box (hidden until submit) -->
            <div id="quiz-result" class="hidden mt-6 p-6 rounded-xl
                                         text-center border">
            </div>
        </div>
    """

    # Quiz scoring JavaScript
    quiz_js = f"""
    // -----------------------------------------------
    // QUIZ SCORING LOGIC
    //
    //   Runs entirely in the browser — no server call.
    //   On submit:
    //   1. Check each question's selected answer
    //   2. Compare with data-correct attribute
    //   3. Highlight correct (green) and wrong (red)
    //   4. Show score with percentage
    //   5. Disable further changes
    // -----------------------------------------------

    const TOTAL_QUESTIONS = {len(questions)};

    function scoreQuiz() {{
        let correct = 0;
        let unanswered = 0;

        for (let i = 0; i < TOTAL_QUESTIONS; i++) {{
            const questionDiv = document.getElementById('question-' + i);
            const options = questionDiv.querySelectorAll('.quiz-option');
            const selected = questionDiv.querySelector('input[type="radio"]:checked');

            if (!selected) {{
                // No answer selected for this question
                unanswered++;

                // Highlight the correct answer in green
                options.forEach(opt => {{
                    if (opt.dataset.correct === 'true') {{
                        opt.classList.add('correct');
                    }}
                }});
                continue;
            }}

            const selectedLabel = selected.closest('.quiz-option');
            const isCorrect = selectedLabel.dataset.correct === 'true';

            if (isCorrect) {{
                // User picked the right answer — green
                correct++;
                selectedLabel.classList.add('correct');
            }} else {{
                // User picked wrong — red on their choice,
                // green on the correct one
                selectedLabel.classList.add('wrong');
                options.forEach(opt => {{
                    if (opt.dataset.correct === 'true') {{
                        opt.classList.add('correct');
                    }}
                }});
            }}

            // Disable all radio buttons for this question
            options.forEach(opt => {{
                const radio = opt.querySelector('input[type="radio"]');
                radio.disabled = true;
                opt.classList.remove('cursor-pointer', 'hover:border-primary-500',
                                   'hover:bg-surface-dark');
                opt.classList.add('cursor-default');
            }});
        }}

        // Calculate percentage
        const percentage = Math.round((correct / TOTAL_QUESTIONS) * 100);

        // Choose result color based on score
        let resultClass, emoji, message;
        if (percentage >= 80) {{
            resultClass = 'border-correct bg-correct/10';
            emoji = '🎉';
            message = 'Excellent!';
        }} else if (percentage >= 60) {{
            resultClass = 'border-primary-500 bg-primary-500/10';
            emoji = '👍';
            message = 'Good job!';
        }} else if (percentage >= 40) {{
            resultClass = 'border-yellow-500 bg-yellow-500/10';
            emoji = '📚';
            message = 'Keep studying!';
        }} else {{
            resultClass = 'border-wrong bg-wrong/10';
            emoji = '💪';
            message = 'Review the material and try again!';
        }}

        // Show result
        const resultDiv = document.getElementById('quiz-result');
        let resultHTML = `
            <p class="text-3xl mb-2">${{emoji}}</p>
            <p class="text-2xl font-bold text-white mb-1">
                ${{correct}} / ${{TOTAL_QUESTIONS}} Correct (${{percentage}}%)
            </p>
            <p class="text-gray-300">${{message}}</p>
        `;

        if (unanswered > 0) {{
            resultHTML += `
                <p class="text-sm text-gray-400 mt-2">
                    ${{unanswered}} question(s) left unanswered
                </p>
            `;
        }}

        resultDiv.innerHTML = resultHTML;
        resultDiv.className = 'mt-6 p-6 rounded-xl text-center border ' + resultClass;

        // Hide submit button, show "Try Again"
        document.getElementById('submit-btn').classList.add('hidden');

        // Scroll to results
        resultDiv.scrollIntoView({{ behavior: 'smooth', block: 'center' }});

        // Notify parent iframe about size change
        setTimeout(notifyParentSize, 300);
    }}
    """

    return _wrap_html(title, body, extra_js=quiz_js)


# =============================================================
# Template 2: SUMMARY
# =============================================================
#
# Clean card with numbered key points.
#
# Expected JSON format from LLM:
#   {
#     "type": "summary",
#     "title": "Document Summary",
#     "source": "AI_Chapter_2_Class_7.pdf",
#     "points": [
#       "Data science is a key part of AI...",
#       "Big Data refers to large datasets...",
#       ...
#     ]
#   }

def render_summary(data: dict) -> str:
    """Render a summary card with numbered key points.

    Args:
        data: Summary JSON with title, source, and points.

    Returns:
        Complete HTML document string.
    """

    title = data.get("title", "Summary")
    source = data.get("source", "Document")
    points = data.get("points", [])

    if not points:
        return _wrap_html(title, """
            <div class="text-center py-8">
                <p class="text-gray-400">No summary could be generated.</p>
            </div>
        """)

    # Build each point's HTML
    points_html = ""

    for i, point in enumerate(points, 1):
        points_html += f"""
            <div class="flex items-start gap-4 mb-4 p-4 rounded-lg
                        bg-surface-dark border border-surface-light">
                <!-- Number badge -->
                <div class="flex-shrink-0 w-8 h-8 rounded-full
                            bg-primary-600 flex items-center justify-center
                            text-white font-bold text-sm">
                    {i}
                </div>
                <!-- Point text -->
                <p class="text-gray-200 leading-relaxed pt-1">
                    {_escape_html(point)}
                </p>
            </div>
        """

    body = f"""
        <div class="max-w-2xl mx-auto">
            <!-- Header -->
            <div class="text-center mb-6">
                <div class="inline-block px-3 py-1 rounded-full
                            bg-primary-600/20 text-primary-500
                            text-sm font-medium mb-3">
                    Summary
                </div>
                <h1 class="text-2xl font-bold text-white mb-2">
                    {_escape_html(title)}
                </h1>
                <p class="text-sm text-gray-400">
                    Source: {_escape_html(source)} |
                    {len(points)} Key Points
                </p>
            </div>

            <!-- Points -->
            <div class="space-y-2">
                {points_html}
            </div>

            <!-- Footer -->
            <div class="mt-6 pt-4 border-t border-surface-light
                        text-center text-xs text-gray-500">
                Generated from document analysis
            </div>
        </div>
    """

    return _wrap_html(title, body)


# =============================================================
# Template 3: ARTICLE
# =============================================================
#
# Formatted article with heading, sections, and paragraphs.
#
# Expected JSON format from LLM:
#   {
#     "type": "article",
#     "title": "Understanding Data Science in AI",
#     "source": "AI_Chapter_2_Class_7.pdf",
#     "sections": [
#       {
#         "heading": "Introduction",
#         "content": "Data science is a key component..."
#       },
#       {
#         "heading": "Key Concepts",
#         "content": "Structured data is organized..."
#       },
#       ...
#     ]
#   }

def render_article(data: dict) -> str:
    """Render a formatted article with sections.

    Args:
        data: Article JSON with title, source, and sections.

    Returns:
        Complete HTML document string.
    """

    title = data.get("title", "Article")
    source = data.get("source", "Document")
    sections = data.get("sections", [])

    if not sections:
        return _wrap_html(title, """
            <div class="text-center py-8">
                <p class="text-gray-400">No article could be generated.</p>
            </div>
        """)

    # Build each section's HTML
    sections_html = ""

    for i, section in enumerate(sections):
        heading = section.get("heading", "")
        content = section.get("content", "")

        # Convert newlines to paragraphs for better formatting
        paragraphs = content.split("\n")
        content_html = ""
        for para in paragraphs:
            para = para.strip()
            if para:
                content_html += f"""
                    <p class="text-gray-200 leading-relaxed mb-3">
                        {_escape_html(para)}
                    </p>
                """

        sections_html += f"""
            <div class="mb-8">
                <h2 class="text-xl font-semibold text-white mb-3
                           pb-2 border-b border-surface-light">
                    {_escape_html(heading)}
                </h2>
                {content_html}
            </div>
        """

    body = f"""
        <div class="max-w-2xl mx-auto">
            <!-- Article Header -->
            <div class="mb-8">
                <div class="inline-block px-3 py-1 rounded-full
                            bg-primary-600/20 text-primary-500
                            text-sm font-medium mb-3">
                    Article
                </div>
                <h1 class="text-3xl font-bold text-white mb-3">
                    {_escape_html(title)}
                </h1>
                <p class="text-sm text-gray-400">
                    Source: {_escape_html(source)}
                </p>
            </div>

            <!-- Article Body -->
            <div class="prose prose-invert">
                {sections_html}
            </div>

            <!-- Footer -->
            <div class="mt-8 pt-4 border-t border-surface-light
                        text-center text-xs text-gray-500">
                Generated from document analysis
            </div>
        </div>
    """

    return _wrap_html(title, body)


# =============================================================
# Template 4: Q&A (Question & Answer)
# =============================================================
#
# Collapsible accordion — click a question to reveal its
# answer. All answers start collapsed.
#
# Expected JSON format from LLM:
#   {
#     "type": "qa",
#     "title": "Questions & Answers",
#     "source": "AI_Chapter_2_Class_7.pdf",
#     "pairs": [
#       {
#         "question": "What is Data Science?",
#         "answer": "Data science is a key part of AI..."
#       },
#       ...
#     ]
#   }

def render_qa(data: dict) -> str:
    """Render a Q&A accordion with collapsible answers.

    Args:
        data: Q&A JSON with title, source, and pairs.

    Returns:
        Complete HTML document string.
    """

    title = data.get("title", "Questions & Answers")
    source = data.get("source", "Document")
    pairs = data.get("pairs", [])

    if not pairs:
        return _wrap_html(title, """
            <div class="text-center py-8">
                <p class="text-gray-400">No Q&A could be generated.</p>
            </div>
        """)

    # Build each Q&A pair as a collapsible card
    pairs_html = ""

    for i, pair in enumerate(pairs):
        question = pair.get("question", "")
        answer = pair.get("answer", "")

        pairs_html += f"""
            <div class="mb-3 rounded-xl border border-surface-light
                        overflow-hidden">
                <!-- Question (clickable header) -->
                <button onclick="toggleAnswer({i})"
                        class="w-full text-left p-4 bg-surface-dark
                               hover:bg-surface-light/50 transition-colors
                               flex items-center justify-between gap-3">
                    <span class="font-semibold text-white flex items-center gap-3">
                        <span class="flex-shrink-0 w-7 h-7 rounded-full
                                     bg-primary-600 flex items-center
                                     justify-center text-sm text-white">
                            {i + 1}
                        </span>
                        {_escape_html(question)}
                    </span>
                    <span id="arrow-{i}"
                          class="flex-shrink-0 text-gray-400
                                 transition-transform duration-300">
                        ▼
                    </span>
                </button>

                <!-- Answer (collapsible) -->
                <div id="answer-{i}" class="qa-answer">
                    <div class="p-4 pt-2 border-t border-surface-light
                                bg-surface-darker">
                        <p class="text-gray-200 leading-relaxed">
                            {_escape_html(answer)}
                        </p>
                    </div>
                </div>
            </div>
        """

    body = f"""
        <div class="max-w-2xl mx-auto">
            <!-- Header -->
            <div class="text-center mb-6">
                <div class="inline-block px-3 py-1 rounded-full
                            bg-primary-600/20 text-primary-500
                            text-sm font-medium mb-3">
                    Q&A
                </div>
                <h1 class="text-2xl font-bold text-white mb-2">
                    {_escape_html(title)}
                </h1>
                <p class="text-sm text-gray-400">
                    Source: {_escape_html(source)} |
                    {len(pairs)} Questions
                </p>
                <p class="text-xs text-gray-500 mt-1">
                    Click a question to reveal its answer
                </p>
            </div>

            <!-- Q&A Accordion -->
            <div>
                {pairs_html}
            </div>

            <!-- Expand/Collapse All -->
            <div class="text-center mt-4">
                <button onclick="toggleAll()"
                        id="toggle-all-btn"
                        class="px-4 py-2 text-sm text-primary-500
                               hover:text-primary-400 transition-colors">
                    Expand All
                </button>
            </div>

            <!-- Footer -->
            <div class="mt-6 pt-4 border-t border-surface-light
                        text-center text-xs text-gray-500">
                Generated from document analysis
            </div>
        </div>
    """

    # Q&A accordion JavaScript
    qa_js = f"""
    // -----------------------------------------------
    // Q&A ACCORDION LOGIC
    //
    //   Each question is a collapsible card. Clicking
    //   the question header toggles its answer panel.
    //   The arrow rotates to indicate open/closed state.
    // -----------------------------------------------

    const TOTAL_PAIRS = {len(pairs)};
    let allExpanded = false;

    function toggleAnswer(index) {{
        const answer = document.getElementById('answer-' + index);
        const arrow = document.getElementById('arrow-' + index);

        if (answer.classList.contains('open')) {{
            // Close this answer
            answer.classList.remove('open');
            arrow.style.transform = 'rotate(0deg)';
        }} else {{
            // Open this answer
            answer.classList.add('open');
            arrow.style.transform = 'rotate(180deg)';
        }}

        // Notify parent about size change
        setTimeout(notifyParentSize, 350);
    }}

    function toggleAll() {{
        allExpanded = !allExpanded;

        for (let i = 0; i < TOTAL_PAIRS; i++) {{
            const answer = document.getElementById('answer-' + i);
            const arrow = document.getElementById('arrow-' + i);

            if (allExpanded) {{
                answer.classList.add('open');
                arrow.style.transform = 'rotate(180deg)';
            }} else {{
                answer.classList.remove('open');
                arrow.style.transform = 'rotate(0deg)';
            }}
        }}

        document.getElementById('toggle-all-btn').textContent =
            allExpanded ? 'Collapse All' : 'Expand All';

        setTimeout(notifyParentSize, 350);
    }}
    """

    return _wrap_html(title, body, extra_js=qa_js)


# =============================================================
# Template 5: FLASHCARDS
# =============================================================
#
# Swipeable flashcard deck — click to flip, arrows to navigate.
# Great for revision and memorization.
#
# Expected JSON format from LLM:
#   {
#     "type": "flashcards",
#     "title": "Data Science Flashcards",
#     "source": "AI_Chapter_2_Class_7.pdf",
#     "cards": [
#       {
#         "front": "What is Big Data?",
#         "back": "Large, complex datasets that require special tools to process"
#       },
#       ...
#     ]
#   }

def render_flashcards(data: dict) -> str:
    """Render an interactive flashcard deck.

    Args:
        data: Flashcard JSON with title, source, and cards.

    Returns:
        Complete HTML document string.
    """

    title = data.get("title", "Flashcards")
    source = data.get("source", "Document")
    cards = data.get("cards", [])

    if not cards:
        return _wrap_html(title, """
            <div class="text-center py-8">
                <p class="text-gray-400">No flashcards could be generated.</p>
            </div>
        """)

    # Encode cards as JSON for JavaScript
    cards_json = json.dumps(cards)

    body = f"""
        <div class="max-w-2xl mx-auto">
            <!-- Header -->
            <div class="text-center mb-6">
                <div class="inline-block px-3 py-1 rounded-full
                            bg-primary-600/20 text-primary-500
                            text-sm font-medium mb-3">
                    Flashcards
                </div>
                <h1 class="text-2xl font-bold text-white mb-2">
                    {_escape_html(title)}
                </h1>
                <p class="text-sm text-gray-400">
                    Source: {_escape_html(source)} |
                    {len(cards)} Cards
                </p>
                <p class="text-xs text-gray-500 mt-1">
                    Click the card to flip it
                </p>
            </div>

            <!-- Flashcard Container -->
            <div class="flex flex-col items-center">
                <!-- Card -->
                <div id="flashcard"
                     onclick="flipCard()"
                     class="w-full max-w-lg h-56 rounded-2xl
                            border-2 border-surface-light
                            bg-surface-dark cursor-pointer
                            flex items-center justify-center p-8
                            text-center transition-all duration-300
                            hover:border-primary-500 hover:shadow-lg
                            hover:shadow-primary-500/10
                            select-none">
                    <div>
                        <p id="card-label"
                           class="text-xs text-primary-500 font-medium
                                  mb-3 uppercase tracking-wider">
                            Question
                        </p>
                        <p id="card-text"
                           class="text-lg text-white font-medium
                                  leading-relaxed">
                        </p>
                    </div>
                </div>

                <!-- Navigation -->
                <div class="flex items-center gap-6 mt-6">
                    <button onclick="prevCard()"
                            class="px-4 py-2 rounded-lg bg-surface-dark
                                   border border-surface-light
                                   text-gray-300 hover:text-white
                                   hover:border-primary-500
                                   transition-all">
                        ← Prev
                    </button>

                    <span id="card-counter"
                          class="text-gray-400 text-sm font-mono">
                    </span>

                    <button onclick="nextCard()"
                            class="px-4 py-2 rounded-lg bg-surface-dark
                                   border border-surface-light
                                   text-gray-300 hover:text-white
                                   hover:border-primary-500
                                   transition-all">
                        Next →
                    </button>
                </div>
            </div>
        </div>
    """

    flashcard_js = f"""
    // -----------------------------------------------
    // FLASHCARD LOGIC
    //
    //   Navigate through cards with prev/next buttons.
    //   Click the card to flip between question/answer.
    //   The card counter shows current position.
    // -----------------------------------------------

    const cards = {cards_json};
    let currentIndex = 0;
    let isFlipped = false;

    function showCard() {{
        const card = cards[currentIndex];
        const textEl = document.getElementById('card-text');
        const labelEl = document.getElementById('card-label');
        const counterEl = document.getElementById('card-counter');

        if (isFlipped) {{
            textEl.textContent = card.back;
            labelEl.textContent = 'Answer';
            labelEl.className = 'text-xs text-correct font-medium mb-3 uppercase tracking-wider';
        }} else {{
            textEl.textContent = card.front;
            labelEl.textContent = 'Question';
            labelEl.className = 'text-xs text-primary-500 font-medium mb-3 uppercase tracking-wider';
        }}

        counterEl.textContent = (currentIndex + 1) + ' / ' + cards.length;
    }}

    function flipCard() {{
        isFlipped = !isFlipped;
        const flashcard = document.getElementById('flashcard');
        flashcard.style.transform = 'scale(0.95)';
        setTimeout(() => {{
            showCard();
            flashcard.style.transform = 'scale(1)';
        }}, 150);
    }}

    function nextCard() {{
        currentIndex = (currentIndex + 1) % cards.length;
        isFlipped = false;
        showCard();
    }}

    function prevCard() {{
        currentIndex = (currentIndex - 1 + cards.length) % cards.length;
        isFlipped = false;
        showCard();
    }}

    // Initialize first card
    showCard();
    """

    return _wrap_html(title, body, extra_js=flashcard_js)


# =============================================================
# Utility functions
# =============================================================

def _escape_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS.

    WHY THIS IS IMPORTANT:
      The text comes from OCR'd documents and LLM output.
      If the text contains < or > or & characters, they
      would break the HTML structure or create security
      vulnerabilities.

      This function converts:
        &  →  &amp;
        <  →  &lt;
        >  →  &gt;
        "  →  &quot;
    """

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# =============================================================
# Main entry point — render any content type
# =============================================================
#
# This is the ONLY function agent.py needs to call.
# It looks at the "type" field in the JSON data and
# routes to the correct template renderer.
#
# Usage in agent.py:
#   from content_renderer import render_content
#   html = render_content(json_data)

# Registry of content type → renderer function
RENDERERS = {
    "quiz":       render_quiz,
    "summary":    render_summary,
    "article":    render_article,
    "qa":         render_qa,
    "flashcards": render_flashcards,
}

# Content types and their trigger keywords
# (Used by agent.py to detect content requests)
CONTENT_PATTERNS = {
    "quiz": [
        "quiz", "test me", "mcq", "multiple choice",
        "create quiz", "make quiz", "generate quiz",
        "create a quiz", "make a quiz",
    ],
    "summary": [
        "summarize", "summary", "key points",
        "bullet points", "summarise", "main points",
        "give me points", "important points",
    ],
    "article": [
        "article", "write about", "essay",
        "explain in detail", "write an article",
        "detailed explanation", "write article",
    ],
    "qa": [
        "question answer", "q&a", "questions and answers",
        "give me questions", "create questions",
        "question and answer", "make questions",
    ],
    "flashcards": [
        "flashcard", "flashcards", "flash card",
        "flash cards", "revision cards", "study cards",
        "create flashcards", "make flashcards",
    ],
}


def render_content(data: dict) -> str | None:
    """Render structured content data into HTML.

    This is the main entry point. Pass the JSON data
    from the LLM and get back a complete HTML document.

    Args:
        data: Dictionary with at least a "type" field
              matching one of our templates, plus the
              template-specific fields.

    Returns:
        Complete HTML document string, or None if the
        content type is not recognized.
    """

    content_type = data.get("type", "").lower()
    renderer = RENDERERS.get(content_type)

    if renderer is None:
        logger.warning(
            "Unknown content type: '%s'. "
            "Available types: %s",
            content_type,
            list(RENDERERS.keys()),
        )
        return None

    try:
        html = renderer(data)
        logger.info(
            "Rendered '%s' template: %d chars of HTML",
            content_type,
            len(html),
        )
        return html
    except Exception as exc:
        logger.error(
            "Failed to render '%s' template: %s",
            content_type,
            exc,
        )
        return None
