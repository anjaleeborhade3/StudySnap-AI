"""StudySnap AI: turn a lecture screenshot into simple revision notes."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import pytesseract
except ImportError:
    pytesseract = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "output"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
GENERIC_TERMS = {
    "write", "using", "create", "program", "practice", "list", "make",
    "example", "examples", "following", "given", "show", "used", "use",
    "learn", "learning", "question", "answer", "note", "notes", "class",
    "chapter", "exercise", "write program", "html program",
}
TECHNICAL_TERMS = [
    "machine learning", "deep learning", "neural network", "neural networks",
    "artificial intelligence", "data science", "database management system",
    "database", "python", "html", "css", "javascript", "sql", "algorithm",
    "data structure", "data structures", "loop", "function", "variable",
    "array", "dictionary", "class", "object", "table", "query",
    "normalization", "regression", "classification", "clustering", "dataset",
    "supervised learning", "unsupervised learning", "feature", "model",
    "tag", "attributes", "attribute", "element", "web page", "markup language",
]
TOPIC_LABELS = {
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "artificial intelligence": "Artificial Intelligence",
    "data science": "Data Science",
    "database management system": "Database Management System",
    "database": "Database",
    "python": "Python",
    "html": "HTML",
    "css": "CSS",
    "javascript": "JavaScript",
    "sql": "SQL",
    "neural network": "Neural Networks",
    "neural networks": "Neural Networks",
    "data structure": "Data Structures",
    "data structures": "Data Structures",
    "regression": "Regression",
    "classification": "Classification",
    "clustering": "Clustering",
}
KEYWORD_NOUNS = {
    "algorithm", "array", "attribute", "code", "condition", "concept",
    "data", "database", "design", "element", "feature", "function",
    "input", "key", "language", "loop", "model", "network", "normalization",
    "object", "output", "process", "record", "result", "structure", "table",
    "tag", "value", "variable", "web", "page",
}
HTML_FOCUS_CONCEPTS = [
    (
        "HTML structure",
        r"<!doctype|<html\b|<head\b|<body\b|<title\b|html\s+(?:document\s+)?structure"
        r"|\btitle\b|\bbody\s+content\b",
    ),
    ("heading tags", r"<h[1-6]\b|\bheadings?\b"),
    ("tables", r"\btable(?:s)?\b|<(?:tr|td|th)\b"),
    ("ordered lists", r"ordered\s+lists?|<ol\b"),
    ("forms", r"\bforms?\b|<form\b"),
    ("paragraphs", r"<p\b|\bparagraphs?\b"),
    ("input fields", r"input\s+fields?|\b(?:password|email|dob|date of birth)\b|<input\b"),
    ("dropdowns", r"dropdowns?|\bselect(?:ion)?\b|<select\b"),
    ("radio buttons", r"radio\s+buttons?|type\s*=\s*[\"']?radio"),
    ("textarea", r"\btext\s*areas?\b|<textarea\b"),
]
HTML_INCIDENTAL_FOCUS = {
    "admission", "department", "employee", "emp", "salary", "joining date",
    "student", "marksheet", "mark sheet", "gender", "hobbies", "dob",
    "password", "email",
}
HTML_MARKUP_PATTERN = re.compile(
    r"\bhtml\b|<!doctype\b|\bweb\s?page\b|"
    r"<(?:html|head|body|title|h[1-6]|p|table|tr|td|th|form|ol|ul|li|"
    r"input|select|option|textarea)\b",
    re.IGNORECASE,
)
STOP_WORDS = sorted(
    {
        word
        for word in (
            set(TfidfVectorizer(stop_words="english").get_stop_words() or set())
            | GENERIC_TERMS
        )
        if " " not in word
    }
)


def clean_text(text: str) -> str:
    """Normalize OCR artifacts without removing meaningful numbered content."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace(r"\|", " ").replace("|", " ")
    cleaned_lines = []
    for line in normalized.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or re.fullmatch(r"\d+\s*[).]?", line):
            continue
        line = re.sub(
            r"\b(?:[A-Z]\s+){1,}[A-Z]\b",
            lambda match: re.sub(r"\s+", "", match.group()),
            line,
        )
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"([,.;:!?]){2,}", r"\1", line)
        line = re.sub(r"(?<!\w)[,;:!?](?!\w)", " ", line)
        line = re.sub(r"(?<!\w)[~`]+(?!\w)", " ", line)
        line = re.sub(r"\s+", " ", line).strip(" \t-•")
        if line and re.search(r"[A-Za-z0-9]", line):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def split_sentences(text: str) -> list[str]:
    """Split cleaned text at common sentence-ending punctuation."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        if sentence.strip()
    ]


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    """Rank useful terms and remove generic OCR instructions and duplicates."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]*\b",
        )
        scores = vectorizer.fit_transform([text]).toarray()[0]
        terms = vectorizer.get_feature_names_out()
        ranked_terms = sorted(zip(terms, scores), key=lambda item: (-item[1], item[0]))
        technical = [term for term in TECHNICAL_TERMS if term in text.lower()]
        candidates = technical + [
            term for term, _ in ranked_terms
            if _useful_keyword(term)
        ]
        keywords = []
        seen = set()
        for term in candidates:
            normalized_term = term.casefold().strip()
            if normalized_term not in seen and _useful_keyword(normalized_term):
                keywords.append(term)
                seen.add(normalized_term)
            if len(keywords) >= limit:
                break
        return keywords
    except ValueError:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", text.lower())
        return [
            word for word, _ in Counter(words).most_common(limit)
            if _useful_keyword(word)
        ][:limit]


def _useful_keyword(term: str) -> bool:
    """Keep known technical terms and words that resemble content-bearing nouns."""
    words = term.casefold().split()
    if term.casefold() in TECHNICAL_TERMS:
        return True
    if not words or any(word in GENERIC_TERMS for word in words):
        return False
    if len(words) > 1:
        return (
            words[0] in KEYWORD_NOUNS
            and words[1] in KEYWORD_NOUNS
        ) or (
            words[0].endswith(("tion", "sion", "ment", "ness", "ity", "ism", "ogy", "ics"))
            and words[1] in KEYWORD_NOUNS
        )
    word = words[0]
    return len(word) > 3 and (
        word.endswith(("tion", "sion", "ment", "ness", "ity", "ism", "ogy", "ics"))
        or word in {"data", "web", "code", "logic", "syntax", "output", "input"}
    )


def detect_topic(text: str, keywords: list[str]) -> str:
    """Prefer recognizable subject terms over arbitrary OCR sentence fragments."""
    lowered = text.casefold()
    found_topics = [
        (lowered.find(term), label)
        for term, label in TOPIC_LABELS.items()
        if term in lowered
    ]
    if found_topics:
        return min(found_topics)[1]
    meaningful = [word for word in keywords if _useful_keyword(word)]
    if meaningful:
        return " ".join(word.title() for word in meaningful[:2])
    return "Study Notes"


def summarize(text: str, sentence_limit: int = 2) -> str:
    """Select informative, distinct sentences for a concise extractive summary."""
    if _is_html_notes(text):
        concepts = [
            label for label, pattern in HTML_FOCUS_CONCEPTS
            if re.search(pattern, text, re.I)
        ]
        tasks = _html_practice_tasks(text)
        concept_summary = (
            f"These HTML notes cover {', '.join(concepts[:-1])} and {concepts[-1]}."
            if len(concepts) > 1
            else f"These HTML notes cover {concepts[0]}."
            if concepts
            else "These HTML notes cover the structure and content of web pages."
        )
        if tasks:
            task_summary = ", ".join(dict.fromkeys(task[1] for task in tasks))
            return f"{concept_summary} Practice exercises include {task_summary}."
        return f"{concept_summary} The notes focus on using these elements to build web pages."

    sentences = _content_sentences(text)
    if not sentences:
        return "No clear summary could be generated from the extracted text."
    keywords = extract_keywords(text, limit=12)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -sum(1 for keyword in keywords if keyword.casefold() in item[1].casefold()),
            abs(len(item[1].split()) - 18),
            item[0],
        ),
    )
    selected = sorted(ranked[:sentence_limit], key=lambda item: item[0])
    return " ".join(sentence for _, sentence in selected)


def make_important_questions(text: str) -> list[str]:
    """Generate five content-specific prompts, adapting to programming material."""
    sentences = _content_sentences(text) or [text.strip()]
    keywords = extract_keywords(text, limit=10)
    programming = (
        _study_type(text) in {"Programming", "Mixed"}
        and (
            _is_html_notes(text)
            or re.search(
                r"\b(?:code|program\w*|python|html|javascript|syntax|function\w*|"
                r"variable\w*|loop\w*)\b",
                text,
                re.IGNORECASE,
            )
            is not None
        )
    )
    if programming and _is_html_notes(text):
        questions = [question for _, _, question in _html_practice_tasks(text)]
        represented = set()
        for task_kind, _, _ in _html_practice_tasks(text):
            if task_kind == "webpage":
                represented.update({"HTML structure", "heading tags"})
            elif task_kind.endswith("table"):
                represented.add("tables")
            elif task_kind.endswith("form"):
                represented.update(
                    {"forms", "input fields", "dropdowns", "radio buttons", "textarea"}
                )
            elif task_kind == "ordered-list":
                represented.add("ordered lists")

        html_prompts = {
            "HTML structure": (
                "Write an HTML program with the document structure described in the notes."
            ),
            "heading tags": (
                "Write an HTML program that uses the heading tags discussed in the notes."
            ),
            "paragraphs": (
                "Write an HTML program that places the paragraph content described in the notes on a webpage."
            ),
            "tables": (
                "Write an HTML program to create a table using the table elements described in the notes."
            ),
            "ordered lists": (
                "Write an HTML program to display the listed items using an ordered list."
            ),
            "forms": (
                "Write an HTML program to create a form using the controls described in the notes."
            ),
            "input fields": (
                "Write an HTML program to add the input fields described in the notes to a form."
            ),
            "dropdowns": (
                "Write an HTML program to add the described choices to a dropdown in a form."
            ),
            "radio buttons": (
                "Write an HTML program to add the radio-button choices described in the notes."
            ),
            "textarea": (
                "Write an HTML program to add a textarea to the form described in the notes."
            ),
        }
        for concept, pattern in HTML_FOCUS_CONCEPTS:
            if (
                re.search(pattern, text, re.I)
                and concept not in represented
                and html_prompts[concept] not in questions
            ):
                questions.append(html_prompts[concept])
                represented.add(concept)
            if len(questions) == 5:
                break
        for sentence in sentences:
            if len(questions) == 5:
                break
            question = (
                "Write an HTML program that implements the requirement stated in this note: "
                f"{sentence}"
            )
            if question not in questions:
                questions.append(_trim_text(question, 240))
        fallback_index = 0
        while len(questions) < 5:
            concepts = [
                concept for concept, pattern in HTML_FOCUS_CONCEPTS
                if re.search(pattern, text, re.I)
            ]
            concept = (
                concepts[fallback_index % len(concepts)]
                if concepts else "HTML structure"
            )
            fallback_prompts = [
                f"Write an HTML program demonstrating {concept} as covered in the notes.",
                f"Build an HTML example that applies {concept} to the content described in the notes.",
                f"Create a small webpage using {concept} and state what it displays.",
            ]
            question = fallback_prompts[fallback_index % len(fallback_prompts)]
            if question in questions:
                question = f"{question[:-1]} (alternative {fallback_index + 1})."
            questions.append(question)
            fallback_index += 1
        return questions[:5]

    if programming:
        practice_sentences = [
            sentence for sentence in sentences
            if re.search(
                r"\b(?:write|create|make|build|design|implement|display|calculate)\b",
                sentence,
                re.I,
            )
        ]
        questions = [
            _trim_text(
                "Write a program that implements the requirement described in this note: "
                f"{sentence}",
                240,
            )
            for sentence in practice_sentences
        ]
        questions = list(dict.fromkeys(questions))
        excluded_terms = {"python", "html", "javascript", "program", "programming", "code", "notes"}
        for concept in keywords:
            if concept.casefold() in excluded_terms:
                continue
            question = (
                f"Write a short program using {concept} to demonstrate its role in the notes, "
                "and state the expected output."
            )
            if question not in questions:
                questions.append(question)
            if len(questions) == 5:
                break
        for sentence in sentences:
            if len(questions) == 5:
                break
            question = _trim_text(
                f"Write a small program that demonstrates the concept described here: {sentence}",
                240,
            )
            if question not in questions:
                questions.append(question)
        fallback_index = 0
        while len(questions) < 5:
            concepts = [
                concept for concept in keywords
                if concept.casefold() not in excluded_terms
            ]
            concept = (
                concepts[fallback_index % len(concepts)]
                if concepts else "the documented concept"
            )
            fallback_prompts = [
                f"Implement {concept} as a small working program, following the details in the notes.",
                f"Write another short program demonstrating {concept} from the notes and show its output.",
                f"Build a working example of {concept} using the information covered in the notes.",
            ]
            question = fallback_prompts[fallback_index % len(fallback_prompts)]
            if question in questions:
                question = f"{question[:-1]} (alternative {fallback_index + 1})."
            questions.append(question)
            fallback_index += 1
        return questions[:5]

    questions = []
    for index in range(5):
        sentence = sentences[index % len(sentences)]
        concept = next(
            (word for word in keywords if word.casefold() in sentence.casefold()),
            keywords[index % len(keywords)] if keywords else "the documented concept",
        )
        prompts = [
            f"Define {concept} using the information in this note: {sentence}",
            f"Explain the relationship described here: {sentence}",
            f"Describe one application of {concept} based on this note.",
            f"Compare {concept} with another concept mentioned in these notes.",
            f"Explain the process or property stated here: {sentence}",
        ]
        questions.append(_trim_text(prompts[index], 240))
    return questions


def _html_practice_tasks(text: str) -> list[tuple[str, str, str]]:
    """Recognize explicit HTML exercise prompts and retain their source order."""
    patterns = [
        (
            "webpage",
            r"\b(?:create|write|make|build|design)\b.{0,50}\bweb\s?page\b",
            "webpage creation",
            "Write an HTML program to create a webpage containing a title, heading, paragraph and body content.",
        ),
        (
            "employee-table",
            r"\b(?:create|write|make|build|display)\b.{0,50}\b(?:employee|emp)\b.{0,35}\btable\b"
            r"|\b(?:employee|emp)\b.{0,35}\btable\b",
            "employee tables",
            "Write an HTML program to create an employee table with Emp No., Name, Department, Salary and Joining Date.",
        ),
        (
            "student-registration-form",
            r"\bstudent\b.{0,30}\bregistration\b.{0,15}\bform\b|\bregistration\b.{0,15}\bform\b",
            "student registration forms",
            "Write an HTML program to create a Student Registration Form with Name, Password, Gender, Email, DOB, Hobbies and Submit/Reset buttons.",
        ),
        (
            "college-admission-form",
            r"\bcollege\b.{0,25}\badmission\b.{0,15}\bform\b|\badmission\b.{0,15}\bform\b",
            "college admission forms",
            "Write an HTML program to create a College Admission Form with a course dropdown, textarea, radio buttons and Submit button.",
        ),
        (
            "student-marksheet-table",
            r"\bstudent\b.{0,30}\bmarks?\s?sheet\b.{0,25}\btable\b|\bmarks?\s?sheet\b.{0,25}\btable\b",
            "student marksheet tables",
            "Write an HTML program to create a student marksheet table.",
        ),
        (
            "ordered-list",
            r"\b(?:five|5)\s+subjects?\b.{0,40}\bordered\s+list\b|\bordered\s+list\b.{0,40}\b(?:five|5)\s+subjects?\b",
            "ordered lists of five subjects",
            "Write an HTML program to display five subjects using an ordered list.",
        ),
    ]
    matches = []
    lowered = text.casefold()
    for kind, pattern, label, question in patterns:
        match = re.search(pattern, lowered, re.I)
        if match:
            matches.append((match.start(), kind, label, question))
    matches.sort(key=lambda item: item[0])
    return [(kind, label, question) for _, kind, label, question in matches]


def _is_html_notes(text: str) -> bool:
    """Recognize HTML notes by their subject name, markup, or webpage references."""
    return HTML_MARKUP_PATTERN.search(text) is not None


def make_short_answer_questions(keywords: list[str], topic: str) -> list[str]:
    """Create three short-answer prompts using extracted concepts."""
    terms = (keywords + [topic])[:3]
    while len(terms) < 3:
        terms.append(topic)
    return [
        f"Define {terms[0]} in one or two sentences.",
        f"State one use or application of {terms[1]}.",
        f"Explain the role of {terms[2]} in the topic covered.",
    ]


def make_mcqs(text: str, keywords: list[str]) -> list[dict[str, object]]:
    """Build cloze MCQs whose correct term appears in its source sentence."""
    if _is_html_notes(text):
        html_questions = _html_mcqs(text)
        if html_questions:
            return html_questions

    candidates = list(dict.fromkeys(keywords))
    for word in re.findall(r"\b[a-zA-Z][a-zA-Z'-]{3,}\b", text):
        normalized = word.casefold()
        if _useful_keyword(normalized) and normalized not in {
            item.casefold() for item in candidates
        }:
            candidates.append(word)
    candidates = candidates[:15]
    sentences = _content_sentences(text) or [text.strip()]
    matches = [
        (sentence, term)
        for sentence in sentences
        for term in sorted(candidates, key=lambda candidate: (-len(candidate), candidate.casefold()))
        if re.search(rf"\b{re.escape(term)}\b", sentence, re.IGNORECASE)
    ]
    questions = []

    for index in range(3):
        if not matches:
            source_statement = sentences[index % len(sentences)]
            options = _statement_options(sentences, source_statement, candidates)
            options = _position_answer(options, source_statement, index)
            questions.append(
                {
                    "question": "Which statement is supported by the extracted notes?",
                    "options": options,
                    "answer": options[0],
                }
            )
            continue
        sentence, answer = matches[index % len(matches)]
        distractors = [
            term for term in candidates
            if term.casefold() != answer.casefold()
        ][:2]
        options = [answer] + distractors
        if len(options) < 3:
            options = _statement_options(sentences, sentence, candidates)
            options = _position_answer(options, sentence, index)
            questions.append(
                {
                    "question": "Which statement is supported by the extracted notes?",
                    "options": options,
                    "answer": options[0],
                }
            )
            continue
        masked_sentence = re.sub(
            rf"\b{re.escape(answer)}\b", "_____", sentence, count=1, flags=re.IGNORECASE
        )
        options = _position_answer(options, answer, index)
        questions.append(
            {
                "question": f'Which term completes this statement from the notes: "{masked_sentence}"?',
                "options": options,
                "answer": answer,
            }
        )
    return questions


def _html_mcqs(text: str) -> list[dict[str, object]]:
    """Ask concept-specific HTML questions with independently verified answers."""
    concepts = [
        (
            "HTML structure",
            "Which HTML element contains the visible content of a webpage?",
            "<body>",
            ["<head>", "<title>"],
        ),
        (
            "heading tags",
            "Which HTML tag represents the largest heading?",
            "<h1>",
            ["<h6>", "<p>"],
        ),
        (
            "tables",
            "Which HTML tag defines a row in a table?",
            "<tr>",
            ["<td>", "<th>"],
        ),
        (
            "ordered lists",
            "Which HTML tag creates an ordered list?",
            "<ol>",
            ["<ul>", "<li>"],
        ),
        (
            "forms",
            "Which HTML element groups controls for submitting user input?",
            "<form>",
            ["<table>", "<ol>"],
        ),
        (
            "input fields",
            "Which HTML element is commonly used for a single-line input field?",
            "<input>",
            ["<textarea>", "<select>"],
        ),
        (
            "dropdowns",
            "Which HTML element creates a dropdown list?",
            "<select>",
            ["<textarea>", "<option>"],
        ),
        (
            "radio buttons",
            "Which input type lets a user select one option from a group?",
            'type="radio"',
            ['type="text"', 'type="password"'],
        ),
        (
            "textarea",
            "Which HTML element provides a multi-line text input?",
            "<textarea>",
            ["<input>", "<select>"],
        ),
    ]
    available = [
        item for item in concepts
        if any(item[0] == label and re.search(pattern, text, re.I)
               for label, pattern in HTML_FOCUS_CONCEPTS)
    ]
    if not available:
        return []
    return [
        {
            "question": available[index % len(available)][1],
            "options": _position_answer(
                [available[index % len(available)][2]]
                + available[index % len(available)][3],
                available[index % len(available)][2],
                index,
            ),
            "answer": available[index % len(available)][2],
        }
        for index in range(3)
    ]


def _position_answer(options: list[str], answer: str, index: int) -> list[str]:
    """Vary the correct option position while preserving the answer choices."""
    choices = [option for option in options if option.casefold() != answer.casefold()]
    choices.insert(index % (len(choices) + 1), answer)
    return choices


def _content_sentences(text: str) -> list[str]:
    """Drop OCR fragments and repeated sentences while keeping real list items."""
    candidates = []
    seen = set()
    for line in text.splitlines():
        pieces = split_sentences(line)
        for sentence in pieces:
            sentence = sentence.strip(" \t-•:;,.")
            key = sentence.casefold()
            if (
                len(sentence.split()) >= 4
                and re.search(r"[A-Za-z]{3}", sentence)
                and key not in seen
            ):
                candidates.append(sentence)
                seen.add(key)
    return candidates


def _important_points(text: str, keywords: list[str]) -> list[str]:
    """Rank useful source statements instead of copying OCR fragments in order."""
    sentences = _content_sentences(text)
    if not sentences:
        return ["No complete, meaningful points could be identified."]
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -sum(1 for keyword in keywords if keyword.casefold() in item[1].casefold()),
            abs(len(item[1].split()) - 16),
            item[0],
        ),
    )
    selected = sorted(ranked[:5], key=lambda item: item[0])
    return [_trim_text(sentence, 180) for _, sentence in selected]


def _study_type(text: str) -> str:
    """Classify notes by matching a few transparent study-type indicators."""
    lowered = text.casefold()
    programming = len(re.findall(
        r"\b(?:code|program\w*|python|html|javascript|syntax|function\w*|"
        r"variable\w*|loop\w*)\b",
        lowered,
    ))
    if _is_html_notes(text):
        programming += 1
    practical = len(re.findall(
        r"\b(?:experiment\w*|procedure\w*|step\w*|calculat\w*|practical|"
        r"implement\w*|activit\w*|measurement\w*)\b",
        lowered,
    ))
    theory = len(re.findall(
        r"\b(?:definition\w*|defin\w*|theor\w*|concept\w*|propert\w*|"
        r"advantage\w*|explain\w*|principle\w*)\b",
        lowered,
    ))
    scores = {"Programming": programming, "Practical": practical, "Theory": theory}
    ranked_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if (
        ranked_scores[1][1] >= 2
        and ranked_scores[1][1] >= ranked_scores[0][1] * 0.4
    ):
        return "Mixed"
    return ranked_scores[0][0] if ranked_scores[0][1] else "Theory"


def _exam_focus(text: str, keywords: list[str]) -> list[str]:
    """Return three to five distinct revision concepts from the screenshot."""
    if _is_html_notes(text):
        focus = [
            label for label, pattern in HTML_FOCUS_CONCEPTS
            if re.search(pattern, text, re.I)
        ]
        incidental = {term.casefold() for term in HTML_INCIDENTAL_FOCUS}
        for term in keywords:
            if (
                term.casefold() not in incidental
                and term.casefold() not in {item.casefold() for item in focus}
            ):
                focus.append(term)
            if len(focus) == 5:
                break
        return focus[:5]

    focus = list(dict.fromkeys(keywords[:5]))
    for sentence in _content_sentences(text):
        if len(focus) >= 5:
            break
        terms = [
            term for term in extract_keywords(sentence, limit=3)
            if not any(term.casefold() == item.casefold() for item in focus)
        ]
        focus.extend(terms)
    for sentence in _content_sentences(text):
        if len(focus) >= 3:
            break
        point = _trim_text(sentence, 90)
        if point.casefold() not in {item.casefold() for item in focus}:
            focus.append(point)
    if len(focus) < 3:
        for sentence in _content_sentences(text):
            words = sentence.split()
            for start in range(1, len(words)):
                point = _trim_text(" ".join(words[start:start + 4]), 60)
                if point and point.casefold() not in {
                    item.casefold() for item in focus
                }:
                    focus.append(point)
                if len(focus) >= 3:
                    break
            if len(focus) >= 3:
                break
    if len(focus) < 3:
        source_terms = keywords or [
            word for word in re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", text)
            if word.casefold() not in STOP_WORDS and word.casefold() not in GENERIC_TERMS
        ]
        if source_terms:
            term = source_terms[0]
            fallback_topics = [
                f"Definition and purpose of {term}",
                f"Uses and applications of {term}",
                f"Examples of {source_terms[1]}" if len(source_terms) > 1
                else f"How {term} works",
            ]
            for point in fallback_topics:
                if point.casefold() not in {item.casefold() for item in focus}:
                    focus.append(point)
                if len(focus) >= 3:
                    break
    return focus[:5]


def _statement_options(
    sentences: list[str], answer: str, candidates: list[str]
) -> list[str]:
    """Use one source statement and clearly false, content-based distractors."""
    options = [answer]
    content_options = [
        clause.strip()
        for sentence in sentences
        for clause in re.split(r"[,;]|\band\b", sentence, flags=re.IGNORECASE)
        if len(clause.split()) >= 3
    ]
    for sentence in sentences + content_options:
        if sentence.casefold() == answer.casefold():
            continue
        candidate = f"The notes do not mention: {_trim_text(sentence, 100)}"
        if candidate.casefold() not in {item.casefold() for item in options}:
            options.append(candidate)
        if len(options) == 3:
            break
    for term in candidates:
        if len(options) == 3:
            break
        candidate = f"The notes do not mention {term}."
        if candidate.casefold() not in {item.casefold() for item in options}:
            options.append(candidate)
    while len(options) < 3:
        candidate = (
            f"The notes do not mention: {_trim_text(answer, 100)}"
            if len(options) == 1
            else f"The notes contradict: {_trim_text(answer, 100)}"
        )
        if candidate.casefold() in {item.casefold() for item in options}:
            candidate += " (false option)"
        options.append(candidate)
    return options


def _trim_text(text: str, limit: int) -> str:
    """Trim a generated prompt at a word boundary."""
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3].rsplit(' ', 1)[0]}..."


def build_notes_text(
    topic: str,
    study_type: str,
    exam_focus: list[str],
    summary: str,
    keywords: list[str],
    important_points: list[str],
    questions: list[str],
    short_questions: list[str],
    mcqs: list[dict[str, object]],
) -> str:
    """Format generated notes as a plain-text revision sheet."""
    sections = [
        "StudySnap AI - Smart Notes",
        "=" * 30,
        f"\nDetected Topic\n{topic}",
        f"\n📚 Study Type\n{study_type}",
        "\n🎯 Exam Focus\n" + "\n".join(f"- {item}" for item in exam_focus),
        f"\nShort Summary\n{summary}",
        "\nImportant Keywords\n"
        + ("\n".join(f"- {word}" for word in keywords) or "- No keywords found"),
        "\nImportant Points\n"
        + ("\n".join(f"- {point}" for point in important_points) or "- No points found"),
        "\nQuick Revision\n- Review the important points above.\n"
        "- Cover the keywords and try to explain each one from memory.",
        "\nExam Mode - Important Questions\n"
        + "\n".join(f"{number}. {question}" for number, question in enumerate(questions, 1)),
        "\nExam Mode - Short-Answer Questions\n"
        + "\n".join(
            f"{number}. {question}"
            for number, question in enumerate(short_questions, 1)
        ),
        "\nExam Mode - MCQs\n"
        + "\n".join(
            f"{number}. {item['question']}\n"
            + "\n".join(
                f"   {letter}. {option}"
                for letter, option in zip("ABC", item["options"])
            )
            + f"\n   Answer: {item['answer']}"
            for number, item in enumerate(mcqs, 1)
        ),
    ]
    return "\n".join(sections) + "\n"


st.set_page_config(
    page_title="StudySnap AI",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #152544;
        --muted: #5d6d85;
        --accent: #5368dd;
        --accent-deep: #7549c8;
        --accent-soft: #eff1ff;
        --line: #e0e7f2;
        --surface: #ffffff;
    }
    html, body, [class*="st-"], [data-testid="stMarkdownContainer"] {
        font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(104, 124, 229, 0.09), transparent 31rem),
            linear-gradient(145deg, #f4f7fc 0%, #f8f9fd 52%, #f0f4fb 100%);
        color: var(--ink);
    }
    .main .block-container {
        max-width: 1220px;
        padding: 2.2rem 2.2rem 3rem;
    }
    h1, h2, h3, [data-testid="stHeader"] {
        color: var(--ink);
        letter-spacing: -0.025em;
    }
    h1, h2, h3 { line-height: 1.22; }
    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
        margin-bottom: 0.12rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.5rem 2.65rem;
        border-radius: 26px;
        background:
            radial-gradient(circle at 90% 10%, rgba(157, 148, 255, 0.32), transparent 16rem),
            linear-gradient(120deg, #101d3b 0%, #233d79 58%, #514d9b 100%);
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 22px 52px rgba(25, 42, 91, 0.19);
        margin-bottom: 1.6rem;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        right: -65px;
        bottom: -175px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 0 0 28px rgba(255, 255, 255, 0.035),
                    0 0 0 58px rgba(255, 255, 255, 0.025);
    }
    .hero h1 {
        color: #ffffff;
        font-size: clamp(2.2rem, 4vw, 3.05rem);
        margin: 0 0 0.55rem;
        letter-spacing: -0.04em;
    }
    .hero .hero-subtitle {
        display: inline-block;
        color: #d9ddff;
        background: linear-gradient(90deg, #f4f6ff 0%, #bfc9ff 62%, #d9caff 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(1.1rem, 2.1vw, 1.38rem);
        font-weight: 700;
        margin: 0 0 0.7rem;
    }
    .hero .hero-description {
        color: #e2e8f7;
        max-width: 720px;
        font-size: 1.02rem;
        line-height: 1.7;
        margin: 0 0 1.35rem;
    }
    .workflow-grid {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        max-width: 700px;
    }
    .workflow-step {
        color: #f6f7ff;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 12px;
        padding: 0.55rem 0.78rem;
        font-size: 0.9rem;
        font-weight: 650;
        backdrop-filter: blur(8px);
    }
    .workflow-arrow {
        color: #c8ceff;
        font-weight: 700;
        padding: 0 0.1rem;
    }
    .section-kicker {
        color: #5264c5;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .upload-panel, .st-key-upload-panel, .info-card, .question-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 9px 25px rgba(28, 45, 86, 0.06);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    .upload-panel {
        padding: 1.35rem 1.5rem 1rem;
        border: 1px solid #dfe4f1;
        border-top: 4px solid var(--accent);
        margin: 0.75rem 0 1.2rem;
    }
    .upload-subtitle {
        color: var(--muted);
        margin: 0 0 0.6rem;
    }
    .format-chip {
        display: inline-block;
        color: #4053b6;
        background: #eff1ff;
        border: 1px solid #dfe3ff;
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.8rem;
    }
    .helper-line {
        color: #6b7890;
        font-size: 0.86rem;
        margin: 0.45rem 0 0;
    }
    .info-card {
        min-height: 142px;
        padding: 1.15rem 1.15rem;
        height: 100%;
    }
    .info-card:hover, .question-card:hover {
        transform: translateY(-3px);
        border-color: #cbd3f4;
        box-shadow: 0 15px 32px rgba(33, 51, 105, 0.11);
    }
    .card-step {
        color: #707dd2;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }
    .info-card .card-title {
        color: #1b2d50;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .info-card .card-copy {
        color: var(--muted);
        font-size: 0.91rem;
        line-height: 1.5;
        margin: 0;
    }
    .result-banner {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 1rem 1.25rem;
        margin: 0.3rem 0 0.4rem;
        color: #21346b;
        background: linear-gradient(105deg, #ffffff 0%, #eff1ff 100%);
        border: 1px solid #dfe4f7;
        border-left: 4px solid var(--accent);
        border-radius: 15px;
        box-shadow: 0 7px 20px rgba(28, 45, 86, 0.05);
        font-size: 1.08rem;
        font-weight: 750;
    }
    .exam-hero {
        padding: 1.25rem 1.45rem;
        margin: 0.4rem 0 0.5rem;
        border-radius: 18px;
        color: white;
        background: linear-gradient(115deg, #17284f 0%, #344b9a 62%, #6354ad 100%);
        box-shadow: 0 14px 30px rgba(38, 53, 117, 0.18);
    }
    .exam-hero h2 {
        color: white;
        margin: 0 0 0.25rem;
    }
    .exam-hero p {
        color: #e1e6ff;
        margin: 0;
    }
    .stat-card {
        min-height: 105px;
        padding: 0.9rem 1rem;
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 15px;
        box-shadow: 0 8px 22px rgba(28, 45, 86, 0.06);
    }
    .stat-value {
        color: #4e60cd;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .stat-label {
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 650;
        margin-top: 0.2rem;
    }
    .question-label {
        display: inline-block;
        color: #4a5dc4;
        background: #eef0ff;
        border-radius: 7px;
        padding: 0.2rem 0.48rem;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    [data-testid="stMarkdownContainer"] code {
        color: #34499f;
        background: #eef1ff;
        border: 1px solid #e0e5ff;
        border-radius: 7px;
        padding: 0.22rem 0.5rem;
        margin: 0.12rem;
        font-size: 0.9em;
    }
    [data-testid="stImage"] img {
        border-radius: 11px;
    }
    [data-testid="stTabs"] [role="tab"] {
        border-radius: 9px 9px 0 0;
        font-weight: 650;
    }
    .footer {
        text-align: center;
        color: #66748b;
        border-top: 1px solid #e2e8f1;
        padding: 1.25rem 0 0.2rem;
        margin-top: 2rem;
        font-size: 0.84rem;
    }
    .footer strong { color: #34466d; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        min-height: 2.8rem;
        font-weight: 650;
        border: 1px solid #d8def0;
        transition: transform 140ms ease, box-shadow 140ms ease,
                    border-color 140ms ease, background 140ms ease;
    }
    .stButton > button[kind="primary"] {
        color: #ffffff;
        border-color: transparent;
        background: linear-gradient(105deg, #4c63d8 0%, #654fc4 100%);
        box-shadow: 0 7px 18px rgba(79, 99, 216, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        color: #ffffff;
        background: linear-gradient(105deg, #4058cb 0%, #5943b7 100%);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 9px 23px rgba(28, 45, 86, 0.15);
        border-color: #bfc9eb;
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
        outline: 3px solid rgba(83, 104, 221, 0.35);
        outline-offset: 2px;
    }
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #fafbff 0%, #f1f3ff 100%);
        border: 1px dashed #98a7e5;
        border-radius: 14px;
        padding: 0.8rem;
        transition: border-color 150ms ease, background 150ms ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #586bd4;
        background: #f2f4ff;
    }
    [data-testid="stFileUploader"] button {
        border-radius: 9px;
        font-weight: 650;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: var(--line);
        border-radius: 15px;
        box-shadow: 0 6px 18px rgba(28, 45, 86, 0.045);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #d0d8ef;
        box-shadow: 0 10px 24px rgba(28, 45, 86, 0.075);
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.58;
    }
    [data-testid="stTextArea"] textarea {
        border-color: #dce3ef;
        border-radius: 11px;
        line-height: 1.55;
    }
    @media (max-width: 700px) {
        .main .block-container { padding: 1.2rem 1rem 2.5rem; }
        .hero { padding: 1.55rem 1.3rem; border-radius: 20px; }
        .workflow-grid { align-items: flex-start; gap: 0.4rem; }
        .workflow-step { font-size: 0.83rem; }
        .workflow-arrow { display: none; }
        .info-card { min-height: 0; }
        .upload-panel { padding: 1rem; }
        .result-banner { font-size: 0.98rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>📚 StudySnap AI</h1>
      <p class="hero-subtitle">Turn Your Notes Into Smart Exam Revision</p>
      <p class="hero-description">Upload your lecture notes or screenshots and instantly transform them into organized notes, important points and exam-focused questions.</p>
      <div class="workflow-grid">
        <span class="workflow-step">📸 Upload</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">🔍 Extract</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">🧠 Understand</span><span class="workflow-arrow">→</span>
        <span class="workflow-step">🎯 Revise</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Get started</div>', unsafe_allow_html=True)
st.subheader("📸 Upload Your Notes")
st.markdown(
    '<div class="upload-subtitle">Drop your lecture screenshot here and let StudySnap AI do the rest.</div>'
    '<div class="format-chip">JPG &nbsp;•&nbsp; JPEG &nbsp;•&nbsp; PNG</div>',
    unsafe_allow_html=True,
)
with st.container(border=True, key="upload-panel"):
    uploaded_file = st.file_uploader(
        "Choose a notes screenshot",
        type=SUPPORTED_IMAGE_TYPES,
        help="Supported formats: JPG, JPEG and PNG.",
    )
st.markdown(
    '<p class="helper-line">Best results: clear screenshots with readable text. '
    'Your image is processed locally; no paid API or API key is used.</p>',
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.markdown('<div class="section-kicker">A simple study workflow</div>', unsafe_allow_html=True)
    st.subheader("What you can do")
    how_columns = st.columns(4)
    feature_cards = [
        ("📸 Smart OCR", "Extract text from screenshots."),
        ("🧠 Smart Notes", "Convert raw text into useful revision notes."),
        ("🔑 Important Points", "Identify concepts and keywords."),
        ("🎯 Exam Mode", "Generate exam-focused questions and MCQs."),
    ]
    for column, (title, copy) in zip(how_columns, feature_cards):
        with column:
            st.markdown(
                f'<div class="info-card"><div class="card-title">{title}</div>'
                f'<p class="card-copy">{copy}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="section-kicker" style="margin-top:1.7rem">From screenshot to study plan</div>',
        unsafe_allow_html=True,
    )
    st.subheader("How StudySnap AI Works")
    step_columns = st.columns(4)
    steps = [
        ("01", "📸", "Upload", "Choose a clear screenshot of your notes."),
        ("02", "🔍", "Extract", "OCR reads and cleans the text."),
        ("03", "🧠", "Organize", "Review the topic, focus, and key takeaways."),
        ("04", "🎯", "Revise", "Practice with exam-focused questions."),
    ]
    for column, (number, icon, title, copy) in zip(step_columns, steps):
        with column:
            st.markdown(
                f'<div class="info-card"><div class="card-step">{number}</div>'
                f'<div class="card-title">{icon} {title}</div>'
                f'<p class="card-copy">{copy}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="footer"><strong>StudySnap AI</strong> · Smart Learning Assistant'
        '<br>Built for smarter exam preparation.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

image_bytes = uploaded_file.getvalue()
image_fingerprint = sha256(image_bytes).hexdigest()
if st.session_state.get("analysis_image") != image_fingerprint:
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_error", None)
    st.session_state["analysis_image"] = image_fingerprint

try:
    with Image.open(BytesIO(image_bytes)) as opened_image:
        opened_image.load()
        uploaded_image = opened_image.convert("RGB")
except (UnidentifiedImageError, OSError, ValueError):
    st.error("This file is not a valid or readable image. Please upload a JPG or PNG image.")
    st.stop()

image_column, intro_column = st.columns([1, 1])
with image_column:
    st.subheader("Your screenshot")
    st.image(uploaded_image, width="stretch")
with intro_column:
    st.markdown('<div class="section-kicker">Step 1 · Review your upload</div>', unsafe_allow_html=True)
    st.subheader("Ready to study smarter?")
    st.write(
        "We’ll extract the text from this image, organize the main ideas, "
        "and prepare revision questions."
    )
    st.info("Select **Analyze screenshot** to create your revision dashboard.")

if st.button("✨ Analyze screenshot", type="primary", width="stretch"):
    st.session_state["analysis_image"] = image_fingerprint
    if pytesseract is None:
        st.session_state["analysis_error"] = (
            "The pytesseract Python package is missing. Install the project "
            "dependencies with `pip install -r requirements.txt`."
        )
        st.session_state.pop("analysis", None)
    else:
        try:
            raw_text = pytesseract.image_to_string(uploaded_image)
            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                st.session_state["analysis_error"] = (
                    "OCR did not find readable text. Try a sharper, brighter "
                    "screenshot with larger text."
                )
                st.session_state.pop("analysis", None)
            else:
                keywords = extract_keywords(cleaned_text)
                important_points = _important_points(cleaned_text, keywords)
                topic = detect_topic(cleaned_text, keywords)
                questions = make_important_questions(cleaned_text)
                short_questions = make_short_answer_questions(keywords, topic)
                mcqs = make_mcqs(cleaned_text, keywords)
                study_type = _study_type(cleaned_text)
                st.session_state["analysis"] = {
                    "raw_text": raw_text,
                    "cleaned_text": cleaned_text,
                    "topic": topic,
                    "study_type": study_type,
                    "exam_focus": _exam_focus(cleaned_text, keywords),
                    "summary": summarize(cleaned_text),
                    "keywords": keywords,
                    "important_points": important_points,
                    "questions": questions,
                    "short_questions": short_questions,
                    "mcqs": mcqs,
                }
                st.session_state.pop("analysis_error", None)
        except pytesseract.TesseractNotFoundError:
            st.session_state["analysis_error"] = (
                "The Tesseract OCR program was not found. Install the Tesseract "
                "OCR engine and make sure it is available on your system PATH."
            )
            st.session_state.pop("analysis", None)
        except pytesseract.TesseractError as error:
            st.session_state["analysis_error"] = (
                f"Tesseract could not process this image: {error}"
            )
            st.session_state.pop("analysis", None)
        except OSError as error:
            st.session_state["analysis_error"] = f"Could not read the uploaded image: {error}"
            st.session_state.pop("analysis", None)

if error_message := st.session_state.get("analysis_error"):
    st.error(error_message)

analysis = st.session_state.get("analysis")
if analysis is None:
    st.stop()

st.divider()
st.markdown(
    '<div class="result-banner">✨ Your Study Material is Ready</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-kicker">A · Source material</div>', unsafe_allow_html=True)
st.header("🔍 Extracted Text")
image_column, text_column = st.columns([0.82, 1.18], gap="large")
with image_column:
    st.markdown("### 📸 Uploaded Notes")
    with st.container(border=True):
        st.image(uploaded_image, width="stretch")
with text_column:
    st.markdown("### Extracted Text")
    raw_tab, cleaned_tab = st.tabs(["Original OCR", "Cleaned text"])
    with raw_tab:
        st.text_area("Text read from image", analysis["raw_text"], height=250)
    with cleaned_tab:
        st.text_area("Ready-to-study text", analysis["cleaned_text"], height=250)

st.divider()
st.markdown('<div class="section-kicker">B · Your revision overview</div>', unsafe_allow_html=True)
st.header("🧠 Smart Notes")
with st.container(border=True):
    st.markdown("#### Topic")
    st.metric("Detected Topic", analysis["topic"])
    setting_columns = st.columns(2, gap="large")
    with setting_columns[0]:
        st.markdown("### 📚 Study Type")
        st.info(analysis["study_type"])
    with setting_columns[1]:
        st.markdown("### 🎯 Exam Focus")
        for focus_item in analysis["exam_focus"]:
            st.markdown(f"- {focus_item}")
    st.markdown("### Short Summary")
    st.write(analysis["summary"])
    st.markdown("### 🔑 Important Keywords")
    if analysis["keywords"]:
        st.write(" · ".join(f"`{word}`" for word in analysis["keywords"]))
    else:
        st.write("No keywords could be identified.")

st.divider()
st.markdown('<div class="section-kicker">C · Key takeaways</div>', unsafe_allow_html=True)
st.header("📌 Important Points")
with st.container(border=True):
    for number, point in enumerate(analysis["important_points"], 1):
        st.markdown(f"**{number}.** {point}")
    st.markdown("### Quick Revision")
    st.markdown("- Review the important points above.")
    st.markdown("- Cover each keyword and explain it from memory.")
    st.markdown("- Try the Exam Mode questions without looking at your notes.")

st.divider()
st.markdown(
    '<div class="exam-hero"><h2>🎯 Exam Mode</h2>'
    '<p>Focus on what matters most for your exam.</p></div>',
    unsafe_allow_html=True,
)
count_columns = st.columns(3)
for column, (value, title) in zip(
    count_columns,
    (("5", "Important Questions"), ("3", "Short Answers"), ("3", "MCQs")),
):
    with column:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{value}</div>'
            f'<div class="stat-label">{title}</div></div>',
            unsafe_allow_html=True,
        )
st.markdown("Work through each question before revealing an MCQ answer.")
st.subheader("🔥 Important Questions")
for number, question in enumerate(analysis["questions"], 1):
    with st.container(border=True):
        st.markdown(f'<span class="question-label">Q{number}</span>', unsafe_allow_html=True)
        st.markdown(question)

short_column, mcq_column = st.columns([0.9, 1.1])
with short_column:
    st.subheader("⚡ Short-Answer Questions")
    for number, question in enumerate(analysis["short_questions"], 1):
        with st.container(border=True):
            st.markdown(f'<span class="question-label">Q{number}</span>', unsafe_allow_html=True)
            st.markdown(question)
with mcq_column:
    st.subheader("🧩 Quick MCQ Challenge")
    for number, item in enumerate(analysis["mcqs"], 1):
        with st.container(border=True):
            st.markdown(f'<span class="question-label">MCQ {number}</span>', unsafe_allow_html=True)
            st.markdown(item["question"])
            for letter, option in zip("ABC", item["options"]):
                st.markdown(f"**{letter}.** {option}")
            st.markdown("**D.** None of the above")
            with st.expander(f"Show answer to MCQ {number}"):
                st.write(item["answer"])

notes_text = build_notes_text(
    topic=analysis["topic"],
    study_type=analysis["study_type"],
    exam_focus=analysis["exam_focus"],
    summary=analysis["summary"],
    keywords=analysis["keywords"],
    important_points=analysis["important_points"],
    questions=analysis["questions"],
    short_questions=analysis["short_questions"],
    mcqs=analysis["mcqs"],
)

save_column, download_column = st.columns(2)
with save_column:
    if st.button("💾 Save notes to output/", width="stretch"):
        try:
            OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            notes_path = OUTPUT_DIRECTORY / "studysnap_notes.txt"
            notes_path.write_text(notes_text, encoding="utf-8")
            st.success(f"Notes saved to {notes_path.relative_to(PROJECT_ROOT)}")
        except OSError as error:
            st.error(f"Could not save the notes file: {error}")
with download_column:
    st.download_button(
        "⬇️ Download notes",
        data=notes_text,
        file_name="studysnap_notes.txt",
        mime="text/plain",
        width="stretch",
    )
st.markdown(
    '<div class="footer"><strong>StudySnap AI</strong> · Smart Learning Assistant'
    '<br>Built for smarter exam preparation.</div>',
    unsafe_allow_html=True,
)
