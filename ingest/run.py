import os
import glob
from pdf_loader import PDFLoader
from parser import VietnameseLawParser


def run_ingestion():
    data_dir = "../data"
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    parser = VietnameseLawParser(law_name="Luật Doanh nghiệp số 59/2020/QH14")
    all_chunks = []
    
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file}")
        loader = PDFLoader(pdf_file)
        raw_text = loader.load_and_clean()
        
        chunks = parser.parse(raw_text)
        for chunk in chunks:
            chunk["metadata"]["source"] = os.path.basename(pdf_file)
        all_chunks.extend(chunks)
        print(f"  -> Extract {len(chunks)} Articles.")
        
    print(f"No of chunks (Aritles): {len(all_chunks)}")

    import json
    with open("../data/law_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print("Saved at ../data/law_chunks.json")

if __name__ == "__main__":
    run_ingestion()