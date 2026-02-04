import argparse
import logging
import sys
import os
import fitz  # PyMuPDF
import ollama
import pytesseract
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_full_text(pdf_path: str) -> str:
    """
    Extracts the full text content from a given PDF file.

    Args:
        pdf_path: The file path to the PDF.

    Returns:
        A string containing the entire text from the PDF.
    """
    logging.info(f"Opening PDF: {pdf_path}")
    text = ""
    
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text().strip()
            
            if not page_text:
                logging.info(f"Page {page_num+1} appears to be a scan. Running OCR...")
                # Convert only this specific page to an image
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
                for image in images:
                    # Run OCR on the image
                    page_text = pytesseract.image_to_string(image)
            
            text += f"\n--- Page {page_num+1} ---\n" + page_text
            
    return text


def iterative_extraction_full(pdf_path: str, model_name: str, context_window: int) -> str:
    """
    Performs an iterative data extraction process on a PDF using a Solver-Proposer model. (Note: This is the user-provided script and does not perform chunking)
    """
    full_text = extract_full_text(pdf_path)
    if not full_text or not full_text.strip():
        logging.error("No text could be extracted from the PDF. The document might be image-based or empty.")
        sys.exit(1)

    # This is the 'State' of the conversation
    messages = [
        {"role": "system", "content": "You are a data extraction expert. You have a perfect memory of the text provided."}
    ]
    
    options = {"num_ctx": context_window}

    try:
        # --- STEP 1: SOLVER (Initial Data) ---
        logging.info("--- Solver: Initial Extraction ---")
        messages.append({"role": "user", "content": f"Here is the ORIGINAL TEXT:\n\n{full_text}\n\nExtract all academic records into JSON."})
        
        response = ollama.chat(model=model_name, messages=messages, options=options)
        assistant_output = response['message']['content']
        messages.append({"role": "assistant", "content": assistant_output})

        # --- STEP 2: PROPOSER LOOP (Self-Correction) ---
        for i in range(3):  # Limit to 3 rounds of 'Self-Training'
            logging.info(f"--- Audit Attempt {i+1} ---")
            
            audit_query = (
                "Compare the ORIGINAL TEXT to the EXTRACTED JSON. "
                "Only report missing info if a REAL FACT (like a grade, date, or name) is absent. "
                "If the data is present but just formatted differently, reply 'DONE'. "
                "If everything is there, reply 'DONE'."
            )
            
            messages.append({"role": "user", "content": audit_query})
            audit_check = ollama.chat(model=model_name, messages=messages, options=options)
            feedback = audit_check['message']['content']
            messages.append({"role": "assistant", "content": feedback})

            if "DONE" in feedback.upper():
                logging.info("✔ Extraction Validated as Complete.")
                return assistant_output # Return the last good JSON
            else:
                logging.warning(f"✘ Data Loss! Model found missing info. Refining...")
                # Solver fixes its own mistake
                refine_query = "Correct the JSON by adding those missing items. Provide the full updated JSON now."
                messages.append({"role": "user", "content": refine_query})
                
                refine_res = ollama.chat(model=model_name, messages=messages, options=options)
                assistant_output = refine_res['message']['content']
                messages.append({"role": "assistant", "content": assistant_output})

        return assistant_output
    except ollama.ResponseError as e:
        logging.error(f"Ollama API error during extraction loop: {e.error}")
        logging.error(f"Is the ollama service running and the model '{model_name}' pulled?")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during the extraction process: {e}")
        sys.exit(1)


def main():
    """
    Main function to parse command-line arguments and run the extraction process.
    The final extraction is always printed to the console and saved to a file.
    """
    parser = argparse.ArgumentParser(description="Extracts text from a PDF using an iterative AI process.")
    parser.add_argument("pdf_path", type=str, help="The path to the PDF file.")
    parser.add_argument("-m", "--model", type=str, default="llama3.2:3b", help="The ollama model to use (default: llama3.2:3b).")
    parser.add_argument("-c", "--context", type=int, default=64000, help="The context window size for the model (default: 64000).")
    parser.add_argument("-o", "--output", type=str, help="Optional path to save the output to a text file.")
    args = parser.parse_args()

    try:
        final_extraction = iterative_extraction_full(args.pdf_path, args.model, args.context)

        # Always print the final output to the terminal
        print("\n--- FINAL EXTRACTED DATA ---\n")
        print(final_extraction)

        # Determine the output filename and save the result
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