with open(r'src/pdf_handler.py', 'a', encoding='utf-8') as f:
    f.write('''

    def get_page_thumbnails(self, pdf_path: str | Path, dpi: int = 72):
        if not PYMUPDF_AVAILABLE:
            return []
        pdf_path = Path(pdf_path)
        thumbnails = []
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=dpi)
                if pix is None:
                    continue
                if PILLOW_AVAILABLE:
                    mode = 'RGBA' if pix.alpha else 'RGB'
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    if mode == 'RGBA':
                        img = img.convert('RGB')
                else:
                    import numpy as _np
                    img = _np.frombuffer(pix.samples, dtype=_np.uint8).reshape(pix.height, pix.width, pix.n)
                    img = Image.fromarray(img, mode='RGB' if pix.n == 3 else 'RGBA').convert('RGB')
                thumbnails.append((page_num + 1, img))
            doc.close()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error('Failed to generate thumbnails: %s', e)
        return thumbnails
''')
print('done')
