import os
import io
import tempfile
from typing import List, Optional, Callable
from PIL import Image
import ollama
from langchain_core.documents import Document

try:
    import pymupdf as fitz
except ImportError:
    import fitz


class DocumentProcessor:
    """
    Processes documents (PDFs and Images) and extracts both digital text
    and handwritten notes using local Ollama vision models.
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(self, vision_model: str = "moondream", max_dimension: int = 1024):
        self.vision_model = vision_model
        self.max_dimension = max_dimension

    def _optimize_image(self, img: Image.Image) -> bytes:
        """
        Converts image to RGB, scales to an optimal bounding box for vision models,
        and returns JPEG bytes.
        """
        if img.mode in ("RGBA", "P", "LA", "CMYK"):
            img = img.convert("RGB")
        
        # Scale down if either dimension exceeds max_dimension
        if max(img.size) > self.max_dimension:
            img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
            
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    def transcribe_image_bytes(self, image_bytes: bytes, prompt: Optional[str] = None) -> str:
        """
        Calls the local Ollama vision model to transcribe handwriting and text.
        Uses a robust prompt cascade with fallback if the primary prompt produces empty output.
        """
        candidate_prompts = (
            [prompt] if prompt
            else [
                "Describe all the handwritten notes, topics, formulas, calculations, and text visible in this image.",
                "Read the text in this image.",
                "What does the text in this image say?"
            ]
        )

        for p in candidate_prompts:
            try:
                response = ollama.chat(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": p,
                            "images": [image_bytes],
                        }
                    ],
                )
                content = response["message"]["content"].strip()
                if content:
                    return content
            except Exception as e:
                err_msg = str(e)
                if "mllama" in err_msg:
                    raise RuntimeError(
                        f"Model '{self.vision_model}' requires the 'mllama' architecture which is not supported "
                        f"in your current Ollama build. Please switch the Vision Model to 'moondream' or update Ollama."
                    ) from e
                if p == candidate_prompts[-1]:
                    raise RuntimeError(f"Failed to transcribe with model '{self.vision_model}': {err_msg}") from e

        return "[No text detected in handwritten note or image]"

    def process_image(self, file_path_or_bytes, file_name: str = "image.png") -> List[Document]:
        """
        Transcribes a standalone image file (handwritten note photo, scan, etc.)
        and returns a LangChain Document.
        """
        if isinstance(file_path_or_bytes, bytes):
            img = Image.open(io.BytesIO(file_path_or_bytes))
        else:
            img = Image.open(file_path_or_bytes)

        optimized_bytes = self._optimize_image(img)
        transcribed_text = self.transcribe_image_bytes(optimized_bytes)

        if not transcribed_text:
            transcribed_text = "[No text detected in handwritten note or image]"

        doc = Document(
            page_content=transcribed_text,
            metadata={
                "source": file_name,
                "page": 1,
                "type": "handwritten_image",
                "vision_model": self.vision_model,
            },
        )
        return [doc]

    def process_pdf(
        self,
        file_path_or_bytes,
        file_name: str = "document.pdf",
        force_ocr: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Document]:
        """
        Processes a PDF document page by page.
        - If a page has substantial digital text (> 60 chars) and force_ocr is False, extracts native text.
        - If a page has little/no text (< 60 chars, e.g. scanned handwriting) or force_ocr is True,
          renders the page to an image and uses the vision model to transcribe handwritten notes.
        """
        if isinstance(file_path_or_bytes, bytes):
            doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")
        else:
            doc = fitz.open(file_path_or_bytes)

        total_pages = len(doc)
        documents: List[Document] = []

        try:
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = doc[page_idx]

                if progress_callback:
                    progress_callback(
                        page_num, total_pages, f"Processing page {page_num} of {total_pages}..."
                    )

                digital_text = page.get_text().strip()

                # Determine if page contains handwriting/scanned image or digital text
                is_scanned_or_handwritten = (len(digital_text) < 60) or force_ocr

                if is_scanned_or_handwritten:
                    if progress_callback:
                        progress_callback(
                            page_num,
                            total_pages,
                            f"Transcribing handwritten notes on page {page_num} of {total_pages}...",
                        )

                    # Render page to image (150 DPI provides good resolution for handwriting without excessive memory)
                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    optimized_bytes = self._optimize_image(img)

                    transcribed_notes = self.transcribe_image_bytes(optimized_bytes)

                    # If page had some digital text along with handwriting, combine both
                    if digital_text and not force_ocr:
                        content = f"{digital_text}\n\n[Handwritten / Visual Notes]:\n{transcribed_notes}"
                    else:
                        content = transcribed_notes or digital_text or f"[Page {page_num} is blank]"

                    doc_type = "handwritten_pdf_page"
                else:
                    content = digital_text
                    doc_type = "digital_pdf_page"

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "page": page_num,
                            "type": doc_type,
                            "vision_model": self.vision_model if is_scanned_or_handwritten else "native_pdf",
                        },
                    )
                )
        finally:
            doc.close()

        return documents

    def process_file(
        self,
        file_path_or_bytes,
        file_name: str,
        force_ocr: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Document]:
        """
        Unified entry point to process either an image or a PDF.
        """
        ext = os.path.splitext(file_name.lower())[1]

        if ext in self.IMAGE_EXTENSIONS:
            if progress_callback:
                progress_callback(1, 1, f"Transcribing handwritten notes in {file_name}...")
            return self.process_image(file_path_or_bytes, file_name=file_name)
        elif ext == ".pdf":
            return self.process_pdf(
                file_path_or_bytes,
                file_name=file_name,
                force_ocr=force_ocr,
                progress_callback=progress_callback,
            )
        else:
            raise ValueError(f"Unsupported file extension: '{ext}'. Supported: PDF, PNG, JPG, JPEG, WEBP, BMP.")
