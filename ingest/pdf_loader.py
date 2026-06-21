import fitz
import re


class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_and_clean(self) -> str:
        doc = fitz.open(self.file_path)
        raw_text = ""
        for page in doc: 
            raw_text += page.get_text("text") + "\n"
        
        # Drop header/footer, repeated page number
        raw_text = re.sub(r'LUAT\s+DOANHNGHIEP\s*', '', raw_text)
        raw_text = re.sub(r'\[empty\]', '', raw_text)

        # Fix OCR error
        ocr_corrections = {
            'Dièu': 'Điều', 'dièu': 'điều', 'Diéu': 'Điều',
            'Cóng ty': 'Công ty', 'cóng ty': 'công ty', 'cōng ty': 'công ty',
            'trách nhiêm': 'trách nhiệm', 'trách nhiem': 'trách nhiệm',
            'hūru': 'hữu', 'hfru': 'hữu', 'hūu': 'hữu',
            'quyét': 'quyết', 'quyét': 'quyết',
            'nghiep': 'nghiệp', 'nghip': 'nghiệp',
            'dǎng': 'đăng', 'dng': 'đăng',
            'thành lap': 'thành lập', 'thàanh': 'thành',
            'chju': 'chịu', 'chǐu': 'chịu',
            'nguòi': 'người', 'nguròi': 'người',
            'tó chúc': 'tổ chức', 't6 chúrc': 'tổ chức',
            'vón': 'vốn', 'vōn': 'vốn',
            'điều': 'Điều', 
        }

        clean_text = raw_text
        for wrong, right in ocr_corrections.items():
            clean_text = re.sub(wrong, right, clean_text)

        # Normaliza white space and outline
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
        clean_text = re.sub(r' +', ' ', clean_text)
        return clean_text.strip()
