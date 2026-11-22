"""State yang dibawa dari node ke node di graf LangGraph.

Setiap node membaca state, lalu mengembalikan dict berisi kunci yang ia ubah saja.
"""
from typing import TypedDict


class ResearchState(TypedDict, total=False):
    # input
    question: str            # pertanyaan asli pengguna
    user_id: str             # pemilik koleksi jurnal

    # hasil router
    route: str               # "search" atau "direct"

    # hasil retrieve
    queries: list[str]       # variasi query yang dipakai mencari (EN/ID/sub-pertanyaan)
    chunks: list[dict]       # kandidat chunk dari Qdrant

    # hasil grade
    relevant: list[dict]     # chunk yang lolos penilaian relevansi (terakumulasi antar-putaran)

    # hasil evaluasi / rewrite
    putaran: int             # berapa kali sudah mencari ulang
    query_baru: list[str]    # query hasil rewrite untuk putaran berikutnya
    keputusan: str           # "generate" | "ulang" | "tidak_ada"

    # hasil generate
    answer: str
    citations: list[dict]    # sumber yang benar-benar disitasi di jawaban
    from_collection: bool    # True jika jawaban berdasar jurnal pengguna

    # jejak untuk debugging & evaluasi
    trace: list[str]
