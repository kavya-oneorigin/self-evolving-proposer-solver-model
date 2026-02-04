# Proposer-Solver Self-Evolving Agent with Llama 3.2 3B Param

This project uses a two-step, AI-driven process (Solver-Proposer) to extract comprehensive details from PDF documents. It leverages local language models via `ollama`.

## How It Works

1.  **Solver**: The script first sends the entire text of a PDF to a language model with instructions to extract every possible detail.
2.  **Proposer (Auditor)**: A second call is made to the model, asking it to act as an auditor. It compares the initial extraction against the original text to find any missed information.
3.  **Refinement**: If the auditor finds discrepancies, a final call is made to incorporate the missing details into the final output.

## Setup

1.  **Install System Dependencies (Tesseract and Poppler)**

    This script uses Tesseract for OCR (Optical Character Recognition) on scanned PDF pages and Poppler to handle PDF-to-image conversion. You must install them on your system.

    *   **On Debian/Ubuntu:**
        ```bash
        sudo apt-get update
        sudo apt-get install tesseract-ocr poppler-utils
        ```

    *   **On macOS (using Homebrew):**
        ```bash
        brew install tesseract poppler
        ```

    *   **On Windows:**
        Download and run the installers for [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/). Make sure to add their installation directories (e.g., `C:\Program Files\Tesseract-OCR` and `C:\path\to\poppler\bin`) to your system's `PATH` environment variable.

2.  **Install Python Packages**: Ensure you have Python 3 installed. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ollama**: This script requires `ollama` to be installed and running with a suitable model (e.g., `llama3.2:3b`). Please see the Ollama documentation for installation instructions.

## Usage

Run the script from your terminal, providing the path to the PDF file you want to process as a command-line argument.

1.**Output**:The final extracted data is printed to the terminal and also saved as a JSON file inside the response folder (e.g., response/your_document_extraction.json).

```bash
python llama_model_script.py "path/to/your document.pdf"
```