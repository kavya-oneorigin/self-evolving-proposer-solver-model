import argparse
import json
import logging
import sys
import os
from datetime import datetime, timezone

import fitz  # PyMuPDF
import ollama
import pytesseract
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
MANIFEST_PATH = os.path.join(CHECKPOINTS_DIR, "manifest.json")

# Max characters of source text to store in each checkpoint (for few-shot context)
SNIPPET_MAX_CHARS = 1500
# How many previous checkpoints to use as few-shot examples
MAX_FEWSHOT_EXAMPLES = 2
# Minimum solver score required to save a checkpoint
MIN_CHECKPOINT_SCORE = 0.5

SOLVER_SYSTEM_PROMPT = (
    "You are a data extraction specialist. Your sole job is to extract every factual "
    "detail from the provided document text and return it as structured JSON.\n\n"
    "Rules:\n"
    "- Extract ALL facts: names, dates, grades, scores, course codes, institutions, "
    "and any other concrete data points.\n"
    "- Use standard English for all JSON field names (e.g., \"studentName\", \"courseId\").\n"
    "- Output ONLY valid JSON. No commentary, no markdown fences, no explanations outside the JSON.\n"
    "- Preserve exact values from the source. Never paraphrase numbers, dates, or proper nouns.\n"
    "- If a field is present in the text, it must appear in your JSON. Completeness is your top priority."
)

PROPOSER_SYSTEM_PROMPT = (
    "You are a meticulous data auditor. You are given an ORIGINAL TEXT and an EXTRACTED "
    "JSON produced by another model. Your job is to find any factual information present "
    "in the original text that is missing or incorrect in the JSON.\n\n"
    "Rules:\n"
    "- Compare every fact in the original text against the JSON: names, dates, grades, "
    "scores, course codes, institutions, etc.\n"
    "- Only flag REAL missing facts. Differences in formatting or key naming are NOT issues.\n"
    "- If you find missing or incorrect data, list each item clearly with what is missing "
    "and what the correct value should be.\n"
    "- If the JSON is complete and accurate, reply with exactly: DONE\n"
    "- Never fabricate information. Only report discrepancies that exist in the original text."
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    """Load the checkpoint manifest, or return a default for iteration 0."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return {
        "current_iteration": 0,
        "solver_model": "solver",
        "proposer_model": "proposer",
    }


def save_manifest(manifest: dict):
    """Persist the manifest to disk."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(iteration: int, pdf_name: str, text_snippet: str, extracted_json: str,
                    solver_score: dict, proposer_score: dict):
    """Save a checkpoint after a successful extraction, including reward scores."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    checkpoint = {
        "iteration": iteration,
        "pdf_name": pdf_name,
        "text_snippet": text_snippet[:SNIPPET_MAX_CHARS],
        "extracted_json": extracted_json,
        "solver_score": solver_score,
        "proposer_score": proposer_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(CHECKPOINTS_DIR, f"iter_{iteration}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)
    logging.info(f"Checkpoint saved: {path}")


def load_checkpoints(max_examples: int = MAX_FEWSHOT_EXAMPLES) -> list[dict]:
    """Load the highest-scored checkpoint files for few-shot context."""
    if not os.path.isdir(CHECKPOINTS_DIR):
        return []

    # Find all iter_N.json files
    checkpoint_files = []
    for fname in os.listdir(CHECKPOINTS_DIR):
        if fname.startswith("iter_") and fname.endswith(".json"):
            try:
                int(fname.replace("iter_", "").replace(".json", ""))
                checkpoint_files.append(fname)
            except ValueError:
                continue

    # Load all checkpoints and sort by solver score (highest first)
    all_checkpoints = []
    for fname in checkpoint_files:
        path = os.path.join(CHECKPOINTS_DIR, fname)
        with open(path, 'r', encoding='utf-8') as f:
            cp = json.load(f)
            all_checkpoints.append(cp)

    all_checkpoints.sort(
        key=lambda cp: cp.get("solver_score", {}).get("total", 0.0),
        reverse=True,
    )

    # Take the best ones, then sort by iteration order for chronological context
    best = all_checkpoints[:max_examples]
    best.sort(key=lambda cp: cp.get("iteration", 0))
    return best


# ---------------------------------------------------------------------------
# Few-shot message builders
# ---------------------------------------------------------------------------

def build_fewshot_solver_messages(checkpoints: list[dict]) -> list[dict]:
    """Build few-shot example messages for the Solver from previous checkpoints."""
    messages = []
    for cp in checkpoints:
        messages.append({
            "role": "user",
            "content": f"Here is the ORIGINAL TEXT:\n\n{cp['text_snippet']}\n\nExtract all records into JSON.",
        })
        messages.append({
            "role": "assistant",
            "content": cp["extracted_json"],
        })
    return messages


def build_fewshot_proposer_messages(checkpoints: list[dict]) -> list[dict]:
    """Build few-shot example messages for the Proposer from previous checkpoints."""
    messages = []
    for cp in checkpoints:
        messages.append({
            "role": "user",
            "content": (
                f"Here is the ORIGINAL TEXT:\n\n{cp['text_snippet']}\n\n"
                f"Here is the EXTRACTED JSON:\n\n{cp['extracted_json']}\n\n"
                "Compare every fact in the original text against the extracted JSON. "
                "Only report missing info if a REAL FACT (like a grade, date, or name) is absent. "
                "If the data is present but just formatted differently, that is NOT an issue. "
                "If everything is there, reply with exactly: DONE"
            ),
        })
        messages.append({
            "role": "assistant",
            "content": "DONE",
        })
    return messages


# ---------------------------------------------------------------------------
# Reward scoring
# ---------------------------------------------------------------------------

def score_solver(extraction: str, rounds_needed: int, proposer_approved: bool) -> dict:
    """Score the Solver's extraction quality.

    Components (50/50):
      - json_valid:   1.0 if output is parseable JSON, 0.0 otherwise.
      - completeness: How quickly the Proposer approved.
                      Round 1 DONE = 1.0, Round 2 = 0.67, Round 3 = 0.33, never = 0.0.
    """
    # JSON validity
    json_valid = 0.0
    try:
        json.loads(extraction)
        json_valid = 1.0
    except (json.JSONDecodeError, TypeError):
        pass

    # Completeness
    if proposer_approved:
        completeness = round(max(0.0, 1.0 - (rounds_needed - 1) / 3), 2)
    else:
        completeness = 0.0

    total = round(0.5 * json_valid + 0.5 * completeness, 2)
    return {"json_valid": json_valid, "completeness": completeness, "total": total}


def score_proposer(audit_rounds: list[dict]) -> dict:
    """Score the Proposer's audit quality.

    Per round, two components (50/50):
      - format_valid: Did it say DONE or list specific issues?
      - impact:       If it flagged issues, did the extraction actually change?

    Returns the average across all rounds.
    """
    if not audit_rounds:
        return {"format_valid": 0.0, "impact": 0.0, "total": 0.0}

    round_scores = []
    for rd in audit_rounds:
        feedback = rd["feedback"]
        is_done = "DONE" in feedback.upper()

        # Format validity: DONE or numbered/bulleted list of issues
        has_list = any(line.strip().startswith(("-", "*")) or
                       (len(line.strip()) > 1 and line.strip()[0].isdigit() and line.strip()[1] in ".)")
                       for line in feedback.split("\n") if line.strip())
        format_valid = 1.0 if (is_done or has_list) else 0.5

        # Impact: did the Solver's output actually change after this feedback?
        if is_done:
            impact = 1.0  # Approving a good extraction is correct
        elif rd.get("extraction_after") is not None:
            before_len = len(rd["extraction_before"])
            after_len = len(rd["extraction_after"])
            change = abs(after_len - before_len)
            impact = round(min(1.0, change / max(before_len * 0.05, 1)), 2)
        else:
            impact = 0.0

        total = round(0.5 * format_valid + 0.5 * impact, 2)
        round_scores.append({"format_valid": format_valid, "impact": impact, "total": total})

    avg_total = round(sum(r["total"] for r in round_scores) / len(round_scores), 2)
    avg_format = round(sum(r["format_valid"] for r in round_scores) / len(round_scores), 2)
    avg_impact = round(sum(r["impact"] for r in round_scores) / len(round_scores), 2)
    return {"format_valid": avg_format, "impact": avg_impact, "total": avg_total, "rounds": round_scores}


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_full_text(pdf_path: str) -> str:
    """Extracts the full text content from a given PDF file."""
    logging.info(f"Opening PDF: {pdf_path}")
    text = ""

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text().strip()

            if not page_text:
                logging.info(f"Page {page_num+1} appears to be a scan. Running OCR...")
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
                for image in images:
                    page_text = pytesseract.image_to_string(image)

            text += f"\n--- Page {page_num+1} ---\n" + page_text

    return text


# ---------------------------------------------------------------------------
# Solver / Proposer interaction
# ---------------------------------------------------------------------------

def solver_extract(solver_model: str, full_text: str, solver_messages: list, options: dict) -> str:
    """Solver: initial data extraction. Operates on its own conversation state."""
    logging.info("--- Solver: Initial Extraction ---")
    solver_messages.append({
        "role": "user",
        "content": f"Here is the ORIGINAL TEXT:\n\n{full_text}\n\nExtract all academic records into JSON."
    })

    response = ollama.chat(model=solver_model, messages=solver_messages, options=options)
    assistant_output = response['message']['content']
    solver_messages.append({"role": "assistant", "content": assistant_output})
    return assistant_output


def proposer_audit(proposer_model: str, full_text: str, extracted_json: str, proposer_messages: list, options: dict) -> str:
    """Proposer: audits the Solver's extraction. Own independent conversation state."""
    proposer_messages.append({
        "role": "user",
        "content": (
            f"Here is the ORIGINAL TEXT:\n\n{full_text}\n\n"
            f"Here is the EXTRACTED JSON:\n\n{extracted_json}\n\n"
            "Compare every fact in the original text against the extracted JSON. "
            "Only report missing info if a REAL FACT (like a grade, date, or name) is absent. "
            "If the data is present but just formatted differently, that is NOT an issue. "
            "If everything is there, reply with exactly: DONE"
        )
    })

    response = ollama.chat(model=proposer_model, messages=proposer_messages, options=options)
    feedback = response['message']['content']
    proposer_messages.append({"role": "assistant", "content": feedback})
    return feedback


def solver_refine(solver_model: str, feedback: str, solver_messages: list, options: dict) -> str:
    """Solver: refines extraction based on Proposer feedback."""
    solver_messages.append({
        "role": "user",
        "content": (
            f"An auditor found the following missing or incorrect items in your JSON:\n\n"
            f"{feedback}\n\n"
            "Correct the JSON by adding those missing items. Provide the full updated JSON now."
        )
    })

    response = ollama.chat(model=solver_model, messages=solver_messages, options=options)
    assistant_output = response['message']['content']
    solver_messages.append({"role": "assistant", "content": assistant_output})
    return assistant_output


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def iterative_extraction(pdf_path: str, solver_model: str, proposer_model: str,
                         context_window: int) -> str:
    """
    Performs iterative data extraction using separate Solver and Proposer models,
    each with their own conversation state. Uses checkpoints from previous
    iterations as few-shot examples for progressive improvement.
    """
    full_text = extract_full_text(pdf_path)
    if not full_text or not full_text.strip():
        logging.error("No text could be extracted from the PDF. The document might be image-based or empty.")
        sys.exit(1)

    # Load previous checkpoints for few-shot context
    checkpoints = load_checkpoints()
    if checkpoints:
        logging.info(f"Loaded {len(checkpoints)} checkpoint(s).")
    else:
        logging.info("No previous checkpoints found.")

    options = {"num_ctx": context_window}

    # Build conversation states with few-shot examples prepended
    solver_messages = build_fewshot_solver_messages(checkpoints)
    proposer_messages = build_fewshot_proposer_messages(checkpoints)

    try:
        # --- STEP 1: Solver extracts initial JSON ---
        extraction = solver_extract(solver_model, full_text, solver_messages, options)

        # --- STEP 2: Proposer audits, Solver refines ---
        audit_rounds = []
        proposer_approved = False
        rounds_needed = 0

        for i in range(3):
            rounds_needed = i + 1
            logging.info(f"--- Proposer: Audit Round {rounds_needed} ---")
            extraction_before = extraction
            feedback = proposer_audit(proposer_model, full_text, extraction, proposer_messages, options)

            if "DONE" in feedback.upper():
                logging.info("Proposer validated extraction as complete.")
                proposer_approved = True
                audit_rounds.append({
                    "feedback": feedback,
                    "extraction_before": extraction_before,
                    "extraction_after": None,
                })
                break
            else:
                logging.warning("Proposer found missing info. Sending feedback to Solver...")
                logging.info(f"--- Solver: Refinement Round {rounds_needed} ---")
                extraction = solver_refine(solver_model, feedback, solver_messages, options)
                audit_rounds.append({
                    "feedback": feedback,
                    "extraction_before": extraction_before,
                    "extraction_after": extraction,
                })

        # --- STEP 3: Compute reward scores ---
        s_score = score_solver(extraction, rounds_needed, proposer_approved)
        p_score = score_proposer(audit_rounds)

        logging.info(f"Solver score:   {s_score['total']} "
                     f"(json_valid={s_score['json_valid']}, completeness={s_score['completeness']})")
        logging.info(f"Proposer score: {p_score['total']} "
                     f"(format={p_score['format_valid']}, impact={p_score['impact']})")

        # --- STEP 4: Save checkpoint (only if quality is sufficient) ---
        manifest = load_manifest()
        next_iter = manifest["current_iteration"] + 1
        pdf_name = os.path.basename(pdf_path)

        if s_score["total"] >= MIN_CHECKPOINT_SCORE:
            save_checkpoint(next_iter, pdf_name, full_text, extraction, s_score, p_score)
            manifest["current_iteration"] = next_iter
            save_manifest(manifest)
            logging.info(f"Iteration {next_iter} complete.")
        else:
            logging.warning(f"Solver score {s_score['total']} < {MIN_CHECKPOINT_SCORE}. "
                            f"Checkpoint NOT saved (low quality extraction).")

        return extraction

    except ollama.ResponseError as e:
        logging.error(f"Ollama API error: {e.error}")
        if "not found" in e.error:
            logging.error(f"The model '{solver_model}' or '{proposer_model}' was not found by the Ollama server.")
            logging.error("Please ensure the models are created by running: bash setup_models.sh")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during extraction: {e}")
        sys.exit(1)


def main():
    """Main function to parse command-line arguments and run the extraction process."""
    manifest = load_manifest()

    parser = argparse.ArgumentParser(description="Extracts text from a PDF using separate Solver and Proposer models.")
    parser.add_argument("pdf_path", type=str, help="The path to the PDF file.")
    parser.add_argument("--solver-model", type=str, default=manifest["solver_model"],
                        help=f"The ollama Solver model (default: {manifest['solver_model']}).")
    parser.add_argument("--proposer-model", type=str, default=manifest["proposer_model"],
                        help=f"The ollama Proposer model (default: {manifest['proposer_model']}).")
    parser.add_argument("-c", "--context", type=int, default=64000,
                        help="The context window size for the models (default: 64000).")
    parser.add_argument("-o", "--output", type=str, help="Optional path to save the output to a text file.")
    args = parser.parse_args()

    logging.info(f"Iteration: {manifest['current_iteration']} | "
                 f"Solver: {args.solver_model} | Proposer: {args.proposer_model}")

    try:
        final_extraction = iterative_extraction(
            args.pdf_path, args.solver_model, args.proposer_model, args.context,
        )

        print("\n--- FINAL EXTRACTED DATA ---\n")
        print(final_extraction)

        output_file = args.output
        if not output_file:
            output_dir = "response"
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.basename(args.pdf_path)
            filename_without_ext = os.path.splitext(base_filename)[0]
            output_file = os.path.join(output_dir, f"{filename_without_ext}_extraction.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_extraction)
        logging.info(f"Final extraction has been saved to {output_file}")

    except FileNotFoundError:
        logging.error(f"Error: The file was not found at {args.pdf_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
