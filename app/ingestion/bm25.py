"""Pencarian kata kunci BM25, untuk melengkapi pencarian vektor (hybrid search).

Kenapa: embedding bagus untuk makna, tapi kurang tajam untuk istilah persis seperti
"CRAG", "Self-RAG", "ISREL", "BM25". BM25 justru kuat di situ.

Indeks dibangun di memori dari chunk milik satu user yang sudah ada di Qdrant,
lalu disimpan di cache. Jika jumlah chunk user berubah (ada ingest baru),
indeks dibangun ulang otomatis. Untuk ratusan-ribuan chunk, ini cepat (< 1 detik).
"""
import math
import re
from collections import Counter

# Kata umum yang tidak membantu pencarian (Indonesia + Inggris)
STOPWORDS = set("""
yang dan di ke dari untuk pada dengan dalam ini itu atau adalah sebagai oleh juga tidak akan
apa bagaimana mengapa kenapa berapa serta tersebut dapat bisa lebih antara para kami kita
the a an of to in for on with by is are was were be been as at or and that this from it its
which what how why their these those using use used than into such can not we our
""".split())


def _tunggal(kata: str) -> str:
    """Penyamaan bentuk jamak bahasa Inggris yang sederhana: tokens -> token, types -> type.
    Sengaja minimal (tanpa stemmer) agar istilah teknis tidak rusak."""
    if len(kata) > 4 and kata.endswith("ies"):
        return kata[:-3] + "y"
    if len(kata) > 3 and kata.endswith("s") and not kata.endswith(("ss", "us", "is")):
        return kata[:-1]
    return kata


def tokenisasi(teks: str) -> list[str]:
    """Huruf kecil, pisah per kata. Istilah bertanda hubung disimpan utuh DAN per bagian:
    'Self-RAG' -> ['self-rag', 'self', 'rag'] supaya cocok dengan penulisan apa pun.
    Bentuk jamak disamakan: 'tokens' dan 'token' dianggap kata yang sama."""
    teks = re.sub(r"[‐-―−]", "-", teks.lower())
    token = []
    for kata in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", teks):
        if "-" in kata:
            token.append(kata)
            token.extend(_tunggal(k) for k in kata.split("-") if k and k not in STOPWORDS)
        elif kata not in STOPWORDS and len(kata) > 1:
            token.append(_tunggal(kata))
    return token


class BM25:
    def __init__(self, dokumen: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenisasi(d)) for d in dokumen]
        self.panjang = [sum(t.values()) for t in self.tf]
        self.rata2 = (sum(self.panjang) / len(self.panjang)) if self.panjang else 0.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(dokumen)
        self.idf = {kata: math.log(1 + (n - f + 0.5) / (f + 0.5)) for kata, f in df.items()}

    def skor(self, query: str) -> list[float]:
        q = [k for k in tokenisasi(query) if k in self.idf]
        hasil = []
        for tf, pj in zip(self.tf, self.panjang):
            s = 0.0
            for kata in q:
                f = tf.get(kata, 0)
                if f:
                    s += self.idf[kata] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * pj / (self.rata2 or 1)))
            hasil.append(s)
        return hasil

    def cari(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Kembalikan (indeks dokumen, skor) terbaik, hanya yang skornya > 0."""
        skor = self.skor(query)
        urut = sorted(range(len(skor)), key=lambda i: skor[i], reverse=True)
        return [(i, skor[i]) for i in urut[:limit] if skor[i] > 0]
