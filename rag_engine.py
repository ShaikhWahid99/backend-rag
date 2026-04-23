import fitz       
import time
import re
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
import os
import pytesseract
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def recursive_text_split(text: str, chunk_size=400, overlap=100) -> list:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            split_idx = text.rfind('\n', start, end)
            if split_idx == -1 or split_idx <= start + (chunk_size // 2):
                split_idx = text.rfind(' ', start, end)
            if split_idx != -1 and split_idx > start:
                end = split_idx
        chunks.append(text[start:end].strip())
        start = end - overlap if end < text_len else text_len
        if start < 0: start = 0
    return [c for c in chunks if c]

TEXT_SYSTEM = """You are a precise document Q&A assistant.
STRICT RULES:
1. Answer ONLY using information explicitly present in the CONTEXT provided.
2. If the answer is not in the context say: "This information is not available in the document."
3. Never guess, infer, or add outside knowledge.
4. Quote exact numbers, names, and terms from the context.
5. Structure your answer with bullet points or numbered lists.
6. NEVER stop mid-sentence — always complete your full answer.
7. End your answer by citing which page(s) the information came from."""

VISION_PROMPT = """You are an expert at analyzing academic research paper figures and diagrams.
RULES:
1. Examine every detail visible in the image carefully.
2. Describe ALL components: boxes, arrows, labels, colors, connections.
3. Explain the data flow or process shown step by step.
4. Answer the user question directly and completely.
5. Use clear beginner-friendly language.
6. Never describe components not visible in the image.
7. Complete your full explanation — never stop mid-sentence.
User question: """

TABLE_PROMPT = """You are an expert at reading data tables from academic research papers.
RULES:
1. Read every row, column, and header visible in the table.
2. Identify what is being measured or compared.
3. Answer using specific numbers from the table.
4. Highlight the most important findings.
5. Explain what the numbers mean in plain English.
6. Never invent numbers — only report what is visible.
7. Use bullet points for clarity.
User question: """

def sep(char='─', n=62):
    print(char * n)

class MultimodalRAG:
    """
    Multimodal RAG — Gemini 2.5 Flash
    -----------------------------------
    TEXT   questions → PDF text chunks → Gemini → Answer
    IMAGE  questions → Actual image    → Gemini Vision → Explanation
    TABLE  questions → Rendered table  → Gemini Vision → Analysis
    """

    def __init__(self, llm_provider, fallback_provider=None):
        self.llm = llm_provider
        self.fallback_llm = fallback_provider

        print('🔄 Loading local embedding model...')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        self.chroma = chromadb.PersistentClient(path="chroma_db")
        self.col = self.chroma.get_or_create_collection(
            'rag', metadata={'hnsw:space': 'cosine'})

        self.images      = []  
        self.tables      = []  
        self._figure_map = {} 
        self._table_map  = {}
        self._doc_path   = None

        print(f'✅ Ready! Standard Provider Attached.')


    def _call_text(self, system: str, user: str) -> str:
        try:
            return self.llm.generate_text(system_prompt=system, user_prompt=user)
        except Exception as e:
            if self.fallback_llm:
                print(f"  ⚠️  Primary LLM failed: {e}. Falling back to LocalLLMProvider...")
                try:
                    return self.fallback_llm.generate_text(system_prompt=system, user_prompt=user)
                except Exception as fb_e:
                    print(f"  ❌ Fallback also failed: {fb_e}")
            return '❌ LLM query failed.'

    def _call_vision(self, prompt: str, img_path: str) -> str:
        try:
            return self.llm.generate_vision(prompt=prompt, image_path=img_path)
        except Exception as e:
            if self.fallback_llm:
                print(f"  ⚠️  Primary LLM vision failed: {e}. Falling back to LocalLLMProvider...")
                try:
                    return self.fallback_llm.generate_vision(prompt=prompt, image_path=img_path)
                except Exception as fb_e:
                    print(f"  ❌ Fallback vision also failed: {fb_e}")
            return '❌ Vision model query failed.'

    def _embed(self, text: str) -> list:
        return self.embedder.encode(text[:2000]).tolist()

    def _store(self, text: str, meta: dict):
        self.col.add(
            ids=[str(uuid.uuid4())],
            embeddings=[self._embed(text)],
            documents=[text],
            metadatas=[meta]
        )

    def _render_table(self, page_num: int, table_idx: int) -> str | None:
        """Render table region from PDF as high-res PNG for vision AI."""
        try:
            doc  = fitz.open(self._doc_path)
            page = doc[page_num - 1]
            tabs = page.find_tables()
            if tabs.tables and table_idx < len(tabs.tables):
                bbox = fitz.Rect(tabs.tables[table_idx].bbox) + (-20, -20, 20, 20)
                pix  = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=bbox)
            else:
                pix  = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            doc.close()
            if not os.path.exists('images'):
                os.makedirs('images')
            path = f'images/tbl_p{page_num}_{table_idx}.png'
            pix.save(path)
            return path
        except Exception as e:
            print(f'  ⚠️  Table render failed: {e}')
            return None

    def process_pdf(self, path: str, user_id: int, file_id: int):
        print(f'\n📄 Processing: {path}\n')
        self._doc_path = path
        doc = fitz.open(path)

        for pn in range(len(doc)):
            page = doc[pn]
            pnum = pn + 1
            text = page.get_text('text').strip()

            if text:
                for m in re.finditer(r'(Figure|Fig\.?)\s*(\d+)', text, re.I):
                    self._figure_map[f'{file_id}_Figure {m.group(2)}'] = pnum
                for m in re.finditer(r'Table\s*(\d+)', text, re.I):
                    key = f'Table {m.group(1)}'
                    if f"{file_id}_{key}" not in self._table_map:
                        self._table_map[f"{file_id}_{key}"] = pnum
                for chunk in recursive_text_split(text):
                    self._store(chunk, {'type': 'text', 'page': str(pnum), 'user_id': str(user_id), 'file_id': str(file_id)})
                    
            elif len(text) < 50:
                ocr_text_full = ""
                for ii, img in enumerate(page.get_images(full=True)):
                    try:
                        base_img = doc.extract_image(img[0])
                        image_bytes = base_img['image']
                        image_pil = Image.open(io.BytesIO(image_bytes))
                        ocr_text = pytesseract.image_to_string(image_pil).strip()
                        if ocr_text:
                            ocr_text_full += ocr_text + "\n"
                    except Exception as e:
                        print(f"  ⚠️ OCR failed on p{pnum} img {ii}: {e}")
                
                if ocr_text_full.strip():
                    text = ocr_text_full.strip()
                    for chunk in recursive_text_split(ocr_text_full):
                        self._store(chunk, {'type': 'ocr_text', 'page': str(pnum), 'user_id': str(user_id), 'file_id': str(file_id)})

            try:
                for ti, tab in enumerate(page.find_tables().tables):
                    rows = tab.extract()
                    if not rows:
                        continue
                    lines = [
                        ' | '.join(str(c).strip() if c else '' for c in r)
                        for r in rows
                    ]
                    ttext = f'[TABLE {ti+1} on Page {pnum}]\n' + '\n'.join(lines)
                    self._store(ttext, {
                        'type': 'table',
                        'page': str(pnum),
                        'table_idx': str(ti),
                        'user_id': str(user_id),
                        'file_id': str(file_id)
                    })
                    self.tables.append({'file_id': file_id, 'page': pnum, 'idx': ti, 'text': ttext})
                    print(f'  📊 Table {ti+1} on page {pnum}')
            except Exception:
                pass

            for ii, img in enumerate(page.get_images(full=True)):
                try:
                    base_img = doc.extract_image(img[0])
                    if not os.path.exists('images'):
                        os.makedirs('images')
                    img_path = f'images/img_p{pnum}_{ii}.png'
                    with open(img_path, 'wb') as f:
                        f.write(base_img['image'])
                    snippet = text[:200] if text else ''
                    self._store(
                        f'[FIGURE on Page {pnum}] {snippet}',
                        {'type': 'image', 'page': str(pnum), 'path': img_path, 'user_id': str(user_id), 'file_id': str(file_id)}
                    )
                    self.images.append({'file_id': file_id, 'page': pnum, 'path': img_path})
                    print(f'  🖼️  Image {ii+1} on page {pnum} → {img_path}')
                except Exception as e:
                    print(f'  ⚠️  Skip image p{pnum}: {e}')

        doc.close()
        print(f'\n✅ Indexed chunks | Tables | Images')
        print(f'   Figures detected: {list(self._figure_map.keys())}')
        print(f'   Tables  detected: {list(self._table_map.keys())}')

    def _classify(self, q: str) -> str:
        """Detect if question is about an image, table, or plain text."""
        ql = q.lower()
        if any(k in ql for k in [
            'figure', 'fig', 'diagram', 'architecture', 'block',
            'image', 'plot', 'chart', 'visual', 'show me',
            'illustration', 'drawing', 'attention map', 'structure of'
        ]):
            return 'image'
        if any(k in ql for k in [
            'table', 'result', 'score', 'bleu', 'performance',
            'metric', 'comparison', 'benchmark', 'accuracy', 'f1',
            'how does it compare', 'what are the results'
        ]):
            return 'table'
        return 'text'

    def _find_images(self, q: str, user_id: str, file_id: str) -> list:
        top_images = []
        m = re.search(r'fig(?:ure)?[\.\s]*(\d+)', q.lower())
        if m:
            page = self._figure_map.get(f'{file_id}_Figure {m.group(1)}')
            if page:
                imgs = [i for i in self.images if i['page'] == page and str(i.get('file_id')) == file_id]
                for img in imgs:
                    if img not in top_images: top_images.append(img)
                    if len(top_images) >= 3: return top_images
        
        col_count = self.col.count()
        if col_count == 0: return top_images
        res = self.col.query(
            query_embeddings=[self._embed(q)],
            n_results=min(15, col_count),
            where={"$and": [{"user_id": user_id}, {"file_id": file_id}]}
        )
        if len(res['metadatas']) > 0 and len(res['metadatas'][0]) > 0:
            for meta in res['metadatas'][0]:
                if meta['type'] == 'image':
                    img_dict = {'page': int(meta['page']), 'path': meta['path']}
                    if img_dict not in top_images:
                        top_images.append(img_dict)
                    if len(top_images) >= 3: break
        if not top_images and self.images:
            top_images.append(self.images[0])
        return top_images

    def _find_table(self, q: str, user_id: str, file_id: str) -> dict | None:
        m = re.search(r'table\s*(\d+)', q.lower())
        if m:
            page = self._table_map.get(f'{file_id}_Table {m.group(1)}')
            if page:
                tabs = [t for t in self.tables if t['page'] == page and str(t.get('file_id')) == file_id]
                if tabs:
                    return tabs[0]
                    
        col_count = self.col.count()
        if col_count == 0: return None
        res = self.col.query(
            query_embeddings=[self._embed(q)],
            n_results=min(10, col_count),
            where={"$and": [{"user_id": user_id}, {"file_id": file_id}]}
        )
        if len(res['documents']) > 0 and len(res['documents'][0]) > 0:
            for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
                if meta['type'] == 'table':
                    matches = [t for t in self.tables if str(t['page']) == str(meta['page'])]
                    if matches:
                        return matches[0]
        return self.tables[0] if self.tables else None

    def ask(self, question: str, user_id: int, file_id: int):
        sep('═')
        print(f'❓ {question}')
        sep('═')
        user_id_str = str(user_id)

        qtype = self._classify(question)
        print(f'🔎 Type: {qtype.upper()}')
        
        image_paths = []

        if qtype == 'image':
            targets = self._find_images(question, user_id_str, str(file_id))
            if not targets:
                msg = '❌ No image found in the PDF.'
                print(msg)
                return {"answer": msg, "image_paths": []}

            answer = ""
            for idx, target in enumerate(targets):
                print(f'🖼️  Image from page {target["page"]} → {target["path"]}')
                image_paths.append(target['path'])
                ans = self._call_vision(VISION_PROMPT + question, target['path'])
                answer += f"**Figure {idx+1} (Page {target['page']}):**\n{ans}\n\n"
            answer = answer.strip()

        elif qtype == 'table':
            target = self._find_table(question, user_id_str, str(file_id))
            if not target:
                msg = '❌ No table found in the PDF.'
                print(msg)
                return {"answer": msg, "image_paths": []}

            print(f'📊 Table from page {target["page"]}')
            rendered = self._render_table(target['page'], target['idx'])
            if rendered:
                print(f'   Table rendered → {rendered}')
                print('💭 Gemini reading the table image...')
                image_paths.append(rendered)
                answer = self._call_vision(TABLE_PROMPT + question, rendered)
            else:
                print('  ℹ️  Sending table as text (render failed)...')
                answer = self._call_text(
                    TABLE_PROMPT,
                    f'Table:\n{target["text"]}\n\nQuestion: {question}'
                )

        else:
            col_count = self.col.count()
            if col_count == 0:
                msg = '❌ No relevant text found in PDF.'
                return {"answer": msg, "image_paths": []}
                
            res = self.col.query(
                query_embeddings=[self._embed(question)],
                n_results=min(15, col_count),
                where={"$and": [{"user_id": user_id_str}, {"file_id": str(file_id)}]}
            )
            
            parts = []
            if len(res['documents']) > 0 and len(res['documents'][0]) > 0:
                for d, m in zip(res['documents'][0], res['metadatas'][0]):
                    if m['type'] in ('text', 'ocr_text', 'table'):
                        parts.append(f'--- Page {m["page"]} ({m["type"]}) ---\n{d}')
                        
            if not parts:
                print('  ℹ️  No strong chunks found, expanding search...')
                msg = '❌ The specific information could not be found with high confidence in this document.'
                print(msg)
                return {"answer": msg, "image_paths": []}
            print('💭 Gemini answering from PDF text...')
            answer = self._call_text(
                system=TEXT_SYSTEM,
                user=(
                    f'CONTEXT (from the PDF):\n{chr(10).join(parts)}\n\n'
                    f'QUESTION: {question}\n\n'
                    f'ANSWER (use only context, complete fully):'
                )
            )

        print()
        sep()
        print('💬 Answer:\n')
        return {"answer": answer, "image_paths": image_paths}

    def show_all_images(self):
        if not self.images:
            print('No images found in PDF.')
            return
        print(f'\n📸 {len(self.images)} image(s) extracted:\n')
        for img in self.images:
            print(f'  Page {img["page"]} → {img["path"]}')

    def show_all_tables(self):
        if not self.tables:
            print('No tables found in PDF.')
            return
        print(f'\n📊 {len(self.tables)} table(s) extracted:\n')
        for t in self.tables:
            print(f'  Page {t["page"]}:')
            print(t['text'][:300])
            print('  ...' if len(t['text']) > 300 else '')
            sep()
