import fitz       
import time
import re
import chromadb
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from PIL import Image as PILImage

MODEL = 'gemini-2.5-flash'  

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

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self._last_call  = 0
        self._call_count = 0

        print('🔄 Loading local embedding model...')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        self.chroma = chromadb.Client()
        try:
            self.chroma.delete_collection('rag')
        except Exception:
            pass
        self.col = self.chroma.create_collection(
            'rag', metadata={'hnsw:space': 'cosine'})

        self.images      = []  
        self.tables      = []  
        self._figure_map = {} 
        self._table_map  = {}
        self._idx        = 0
        self._doc_path   = None

        print(f'✅ Ready!  Model: {MODEL}')
        print(f'   Free quota: 250 req/day · 10 RPM · 1M tokens/day')


    def _throttle(self):
        """Wait so we never exceed 10 req/min (free tier limit)."""
        elapsed = time.time() - self._last_call
        if elapsed < 6:
            wait = 6 - elapsed
            print(f'  ⏱️  Throttling {wait:.1f}s...')
            time.sleep(wait)
        self._last_call = time.time()
        self._call_count += 1


    def _call_text(self, system: str, user: str, retries=3) -> str:
        for attempt in range(1, retries + 1):
            try:
                self._throttle()
                r = self.client.models.generate_content(
                    model=MODEL,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=1500,
                        temperature=0.1  
                    )
                )
                if r.text and len(r.text.strip()) > 20:
                    print(f'  ✅ {MODEL} text  (call #{self._call_count})')
                    return r.text.strip()
                print(f'  ⚠️  Empty response attempt {attempt}')
            except Exception as e:
                err = str(e)
                if '429' in err or 'quota' in err.lower() or 'RESOURCE_EXHAUSTED' in err:
                    print(f'  ⏳ Rate limited ({attempt}/{retries}) — waiting 65s...')
                    time.sleep(65)
                else:
                    print(f'  ⚠️  Attempt {attempt}: {e}')
                    time.sleep(5)
        return '❌ Rate limited. Wait 1 minute and retry.'


    def _call_vision(self, prompt: str, img_path: str, retries=3) -> str:
        pil_img = PILImage.open(img_path)
        for attempt in range(1, retries + 1):
            try:
                self._throttle()
                r = self.client.models.generate_content(
                    model=MODEL,
                    contents=[prompt, pil_img],
                    config=types.GenerateContentConfig(
                        max_output_tokens=1500,
                        temperature=0.1
                    )
                )
                if r.text and len(r.text.strip()) > 20:
                    print(f'  ✅ {MODEL} vision (call #{self._call_count})')
                    return r.text.strip()
                print(f'  ⚠️  Empty vision response attempt {attempt}')
            except Exception as e:
                err = str(e)
                if '429' in err or 'quota' in err.lower() or 'RESOURCE_EXHAUSTED' in err:
                    print(f'  ⏳ Rate limited ({attempt}/{retries}) — waiting 65s...')
                    time.sleep(65)
                else:
                    print(f'  ⚠️  Attempt {attempt}: {e}')
                    time.sleep(5)
        return '❌ Vision model rate limited. Wait 1 minute and retry.'


    def _embed(self, text: str) -> list:
        return self.embedder.encode(text[:2000]).tolist()

    def _store(self, text: str, meta: dict):
        self.col.add(
            ids=[f'c{self._idx}'],
            embeddings=[self._embed(text)],
            documents=[text],
            metadatas=[meta]
        )
        self._idx += 1

    

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
            path = f'tbl_p{page_num}_{table_idx}.png'
            pix.save(path)
            return path
        except Exception as e:
            print(f'  ⚠️  Table render failed: {e}')
            return None

   
    def process_pdf(self, path: str):
        print(f'\n📄 Processing: {path}\n')
        self._doc_path = path
        doc = fitz.open(path)

        for pn in range(len(doc)):
            page = doc[pn]
            pnum = pn + 1
            text = page.get_text('text').strip()

            
            if text:
               
                for m in re.finditer(r'(Figure|Fig\.?)\s*(\d+)', text, re.I):
                    self._figure_map[f'Figure {m.group(2)}'] = pnum
                for m in re.finditer(r'Table\s*(\d+)', text, re.I):
                    key = f'Table {m.group(1)}'
                    if key not in self._table_map:
                        self._table_map[key] = pnum
                self._store(text, {'type': 'text', 'page': str(pnum)})

          
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
                        'table_idx': str(ti)
                    })
                    self.tables.append({'page': pnum, 'idx': ti, 'text': ttext})
                    print(f'  📊 Table {ti+1} on page {pnum}')
            except Exception:
                pass

           
            for ii, img in enumerate(page.get_images(full=True)):
                try:
                    base_img = doc.extract_image(img[0])
                    img_path = f'img_p{pnum}_{ii}.png'
                    with open(img_path, 'wb') as f:
                        f.write(base_img['image'])
                    snippet = text[:150] if text else ''
                    self._store(
                        f'[FIGURE on Page {pnum}] {snippet}',
                        {'type': 'image', 'page': str(pnum), 'path': img_path}
                    )
                    self.images.append({'page': pnum, 'path': img_path})
                    print(f'  🖼️  Image {ii+1} on page {pnum} → {img_path}')
                except Exception as e:
                    print(f'  ⚠️  Skip image p{pnum}: {e}')

        doc.close()
        print(f'\n✅ Indexed {self.col.count()} chunks | '
              f'{len(self.tables)} tables | {len(self.images)} images')
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


    def _find_image(self, q: str) -> dict | None:
        m = re.search(r'fig(?:ure)?[\.\s]*(\d+)', q.lower())
        if m:
            page = self._figure_map.get(f'Figure {m.group(1)}')
            if page:
                imgs = [i for i in self.images if i['page'] == page]
                if imgs:
                    return imgs[0]
        res = self.col.query(
            query_embeddings=[self._embed(q)],
            n_results=min(10, self.col.count())
        )
        for meta in res['metadatas'][0]:
            if meta['type'] == 'image':
                return {'page': int(meta['page']), 'path': meta['path']}
        return self.images[0] if self.images else None


    def _find_table(self, q: str) -> dict | None:
        m = re.search(r'table\s*(\d+)', q.lower())
        if m:
            page = self._table_map.get(f'Table {m.group(1)}')
            if page:
                tabs = [t for t in self.tables if t['page'] == page]
                if tabs:
                    return tabs[0]
        res = self.col.query(
            query_embeddings=[self._embed(q)],
            n_results=min(10, self.col.count())
        )
        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
            if meta['type'] == 'table':
                matches = [t for t in self.tables
                           if str(t['page']) == meta['page']]
                if matches:
                    return matches[0]
        return self.tables[0] if self.tables else None


    def ask(self, question: str):
        sep('═')
        print(f'❓ {question}')
        sep('═')

        qtype = self._classify(question)
        print(f'🔎 Type: {qtype.upper()} → {MODEL}')

        if qtype == 'image':
            target = self._find_image(question)
            if not target:
                print('❌ No image found in the PDF.')
                return

            print(f'🖼️  Image from page {target["page"]} → {target["path"]}')
            print(f'   (Open {target["path"]} to view the figure)')
            print('💭 Gemini analyzing the figure...')
            answer = self._call_vision(VISION_PROMPT + question, target['path'])

        elif qtype == 'table':
            target = self._find_table(question)
            if not target:
                print('❌ No table found in the PDF.')
                return

            print(f'📊 Table from page {target["page"]}')
            rendered = self._render_table(target['page'], target['idx'])
            if rendered:
                print(f'   Table rendered → {rendered}')
                print('💭 Gemini reading the table image...')
                answer = self._call_vision(TABLE_PROMPT + question, rendered)
            else:
                print('  ℹ️  Sending table as text (render failed)...')
                answer = self._call_text(
                    TABLE_PROMPT,
                    f'Table:\n{target["text"]}\n\nQuestion: {question}'
                )

        else:
            res = self.col.query(
                query_embeddings=[self._embed(question)],
                n_results=min(8, self.col.count())
            )
            parts = [
                f'--- Page {m["page"]} ({m["type"]}) ---\n{d}'
                for d, m in zip(res['documents'][0], res['metadatas'][0])
                if m['type'] in ('text', 'table')
            ]
            if not parts:
                print('❌ No relevant text found in PDF.')
                return
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
        return answer


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
