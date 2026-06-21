import re
from typing import List, Dict, Any


class VietnameseLawParser:
    def __init__(self, law_name = "Luật Doanh Nghiệp 2020"):
        self.law_name = law_name
        self.chapter_pattern = re.compile(r'^(Chương\s+[IVXLC]+)\s*.*$', re.IGNORECASE | re.MULTILINE)
        self.article_pattern = re.compile(r'^(Điều\s+\d+)\.\s*(.*)$', re.MULTILINE)
        self.clause_pattern = re.compile(r'^(\d+)\.\s+(.*)$', re.MULTILINE)

    def parse(self, text: str) -> List[Dict[str, Any]]:
        chunks = []
        curr_chap = "Phần mở đầu"
        curr_art = ""
        curr_art_title = ""
        curr_content = []

        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line: continue

            #Check chapter
            if self.chapter_pattern.match(line):
                if curr_art: 
                    chunks.append(self._create_chunk(curr_chap, curr_art, curr_art_title, curr_content)):
                curr_chap = self.chapter_pattern.match(line).group(1)
                curr_art = ""
                curr_content = []
                continue

            # Check article
            article_match = self.article_pattern.match(line)
            if article_match:
                if curr_art:
                    chunks.append(self._create_chunk(curr_chap, curr_art, curr_art_title, curr_content))
                curr_art = article_match.group(1)
                curr_art_title = article_match.group(2)
                curr_content = [line]
                continue
            if curr_art: curr_content.append(line)

        if curr_art:
            chunks.append(self._create_chunk(curr_chap, curr_art, curr_art_title, curr_content))

        return chunks
    def _create_chunk(self, chapter, article, title, content_lines, source):
        full_text = "\n".join(content_lines)
        embedding_text = f"{self.law_name} - {chapter} - {article} {title}. {full_text}"
        
        return {
            "id": f"{article.lower().replace(' ', '_')}",
            "law_name": self.law_name,
            "chapter": chapter,
            "article": article,
            "article_title": title,
            "content": full_text,
            "embedding_text": embedding_text, 
            "metadata": {
                "source": source, 
                "type": "article",
                "chapter": chapter,
                "article": article
            }
        }
