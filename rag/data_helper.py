from pypdf import PdfReader


class PDFReader:
    def __init__(self, pdf_paths: list[str] | str):
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        self.pdf_path = pdf_paths

    def read(self) -> list[str]:
        texts = []
        for pdf_path in self.pdf_path:
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                texts.append(text)
        return texts
