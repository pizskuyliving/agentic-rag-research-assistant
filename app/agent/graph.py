r"""Merangkai node menjadi graf LangGraph.

    START -> router --search--> retrieve -> grade -> evaluasi --cukup--> generate -> END
                  \                ^                     |  \--kosong--> tidak_ada -> END
                   \               '-------ulang---------'
                    \--direct--> jawab_langsung -> END

Evaluasi memeriksa apakah chunk relevan sudah menjawab SEMUA aspek pertanyaan.
Jika belum, ia menulis query baru (memakai istilah dari chunk yang sudah ditemukan)
dan agen mencari ulang, maksimal MAKS_PUTARAN_ULANG kali.

Minggu berikutnya: web_search (saat koleksi tidak cukup) dan self_check setelah generate.
"""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.state import ResearchState


@lru_cache
def bangun_agen():
    g = StateGraph(ResearchState)

    g.add_node("router", nodes.router)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade", nodes.grade)
    g.add_node("evaluasi", nodes.evaluasi)
    g.add_node("generate", nodes.generate)
    g.add_node("jawab_langsung", nodes.jawab_langsung)
    g.add_node("tidak_ada", nodes.tidak_ada)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["route"],
                            {"search": "retrieve", "direct": "jawab_langsung"})
    g.add_edge("retrieve", "grade")
    g.add_edge("grade", "evaluasi")
    g.add_conditional_edges("evaluasi", nodes.setelah_evaluasi,
                            {"generate": "generate", "ulang": "retrieve", "tidak_ada": "tidak_ada"})
    g.add_edge("generate", END)
    g.add_edge("jawab_langsung", END)
    g.add_edge("tidak_ada", END)

    return g.compile()


def tanya_agen(question: str, user_id: str) -> dict:
    """Jalankan agen untuk satu pertanyaan, kembalikan state akhir."""
    return bangun_agen().invoke({"question": question, "user_id": user_id, "trace": [], "putaran": 0})
