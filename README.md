# Proposer-Solver Self-Evolving Agent with Llama 3.2 3B Param

This project uses a two-model, AI-driven process (Solver-Proposer) to extract comprehensive details from PDF documents. It leverages two specialized local language models via `ollama` — a **Solver** optimized for precise data extraction and a **Proposer** optimized for auditing and gap-finding — each with their own conversation state.

The system is **self-evolving**: the model checkpoint from each iteration is used as the starting point for the next. The models progressively improve at extraction the more PDFs you process — no manual intervention required.

## How It Works

1.  **Solver (Extraction)**: A specialized model (`solver:latest`) receives the full text of a PDF and extracts every factual detail into structured JSON. It operates with low temperature (0.3) for precision.
2.  **Proposer (Audit)**: A separate model (`proposer:latest`) independently receives both the original text and the Solver's JSON output. It compares every fact in the source against the extraction to find missing or incorrect data. It operates with higher temperature (0.6) for more thorough gap-finding.
3.  **Refinement Loop**: If the Proposer finds discrepancies, its feedback is sent back to the Solver to refine the JSON. This audit-refine cycle repeats up to 3 rounds, or until the Proposer replies `DONE`.
4.  **Checkpoint**: After each successful extraction, a checkpoint is created. The checkpoint from the previous iteration is used as the starting point for the next, allowing the system to progressively improve.

Both models start from the same `llama3.2:3b` base but are configured via separate Ollama Modelfiles with distinct system prompts and generation parameters, allowing them to specialize in their respective roles.

## Setup

1.  **Install System Dependencies (Tesseract and Poppler)**

    This script uses Tesseract for OCR (Optical Character Recognition) on scanned PDF pages and Poppler to handle PDF-to-image conversion. You must install them on your system.
    - **On Debian/Ubuntu:**

      ```bash
      sudo apt-get update
      sudo apt-get install tesseract-ocr poppler-utils
      ```

    - **On macOS (using Homebrew):**

      ```bash
      brew install tesseract poppler
      ```

    - **On Windows:**
      Download and run the installers for [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/). Make sure to add their installation directories (e.g., `C:\Program Files\Tesseract-OCR` and `C:\path\to\poppler\bin`) to your system's `PATH` environment variable.

2.  **Install Python Packages**: Ensure you have Python 3 installed. Create a virtual environment and install the required packages:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Ollama**: Install `ollama` and ensure it is running. See the [Ollama documentation](https://ollama.com/) for installation instructions.

4.  **Register the Solver and Proposer Models**: Run the setup script to create both specialized models from the `llama3.2:3b` base:
    ```bash
    bash setup_models.sh
    ```
    This registers `solver:latest` and `proposer:latest` in Ollama using the Modelfiles in the repo (`Modelfile.solver` and `Modelfile.proposer`).

## Usage

Run the script from your terminal, providing the path to the PDF file you want to process:

```bash
python llama_model_script.py "path/to/your document.pdf"
```

**Options:**

| Flag               | Description                 | Default                               |
| ------------------ | --------------------------- | ------------------------------------- |
| `--solver-model`   | Ollama model for extraction | `solver`                              |
| `--proposer-model` | Ollama model for auditing   | `proposer`                            |
| `-c`, `--context`  | Context window size         | `64000`                               |
| `-o`, `--output`   | Custom output file path     | `response/<filename>_extraction.json` |

**Output**: The final extracted data is printed to the terminal and saved as a JSON file inside the `response/` folder (e.g., `response/your_document_extraction.json`).

## Self-Evolution (Checkpoint-Based)

The system self-evolves automatically through checkpoint-based progressive learning:

```
Process PDF → extract data → save model checkpoint
     ↑                                  ↓
     └──── checkpoint from previous iteration used as starting point ────┘
```

Each extraction run = one iteration:

1. The model checkpoint from the previous iteration is loaded and used as the starting point for the current run
2. Both the Solver and Proposer benefit from context carried over from prior iterations
3. After a successful extraction, a new checkpoint is saved as `checkpoints/iter_N.json`
4. The next iteration automatically picks up this checkpoint, building on all prior work

**Checkpoint structure:**

```
checkpoints/
├── manifest.json       # Current iteration and model names
├── iter_1.json         # Checkpoint from 1st extraction (with scores)
├── iter_2.json         # Checkpoint from 2nd extraction (with scores)
└── ...
```

Each checkpoint file contains a snippet of the source text, the final extracted JSON, and reward scores.

## Reward System

Inspired by Dr.Zero, both models are scored after each extraction to measure quality and track improvement.

**Solver score** (two components, 50/50):

| Component | What it measures | Scoring |
|---|---|---|
| JSON validity | Is the output valid JSON? | 1.0 = yes, 0.0 = no |
| Completeness | How quickly did the Proposer approve? | Round 1 = 1.0, Round 2 = 0.67, Round 3 = 0.33, never = 0.0 |

**Proposer score** (two components, 50/50):

| Component | What it measures | Scoring |
|---|---|---|
| Format validity | Did it give clear structured feedback or a clean DONE? | 1.0 = clear, 0.5 = vague |
| Impact | When it flagged issues, did the Solver's output actually change? | Higher change = higher score |

**How scores are used:**
- Only extractions with a Solver score >= 0.5 are saved as checkpoints (keeps few-shot examples high quality)
- When loading checkpoints, the highest-scored ones are preferred over the most recent
- Scores are logged after each run and saved inside each checkpoint file for tracking improvement over time
