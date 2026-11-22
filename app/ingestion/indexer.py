"""Embedding chunk lalu simpan/cari di Qdrant.

Setiap chunk menyimpan `user_id`, dan setiap pencarian WAJIB difilter per user,
supaya jurnal milik satu orang tidak pernah muncul di jawaban orang lain.
"""
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app import config
from app.ingestion.bm25 import BM25
from app.ingestion.chunker import Chunk
from app.ingestion.pdf_parser import Jurnal


@lru_cache
def _model() -> SentenceTransformer:
    # Unduhan pertama butuh internet (sekali saja), setelah itu tersimpan di cache.
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache
def _client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


def _pakai_prefix_e5() -> bool:
    # Model keluarga E5 dilatih dengan awalan "query: " / "passage: "
    return "e5" in config.EMBEDDING_MODEL.lower()


def embed_dokumen(teks: list[str]) -> list[list[float]]:
    if _pakai_prefix_e5():
        teks = [f"passage: {t}" for t in teks]
    return _model().encode(teks, normalize_embeddings=True, batch_size=32,
                           show_progress_bar=len(teks) > 64).tolist()


def embed_query(teks: str) -> list[float]:
    if _pakai_prefix_e5():
        teks = f"query: {teks}"
    return _model().encode(teks, normalize_embeddings=True).tolist()


def siapkan_koleksi() -> None:
    client = _client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        return
    model = _model()
    # nama metode berubah di sentence-transformers versi baru; dukung keduanya
    ambil_dimensi = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dimensi = ambil_dimensi()
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=dimensi, distance=models.Distance.COSINE),
    )
    # Index payload agar filter per user & per jurnal tetap cepat saat datanya banyak
    for kolom in ("user_id", "paper_id"):
        client.create_payload_index(config.QDRANT_COLLECTION, field_name=kolom,
                                    field_schema=models.PayloadSchemaType.KEYWORD)


def simpan_jurnal(jurnal: Jurnal, chunks: list[Chunk], user_id: str) -> int:
    """Simpan semua chunk satu jurnal. Upload ulang file yang sama akan menimpa, bukan menduplikasi."""
    siapkan_koleksi()
    # Hapus dulu chunk lama jurnal ini: jika cara chunking berubah (mis. tabel ditambahkan),
    # nomor urut bergeser dan chunk lama tidak boleh tertinggal.
    _client().delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="paper_id", match=models.MatchValue(value=jurnal.paper_id)),
        ])),
    )
    vektor = embed_dokumen([c.teks for c in chunks])
    points = [
        models.PointStruct(
            # ID deterministik: user + jurnal + urutan chunk
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}/{c.paper_id}/{c.urutan}")),
            vector=v,
            payload={
                "user_id": user_id,
                "paper_id": c.paper_id,
                "judul": jurnal.judul,
                "penulis": jurnal.penulis,
                "tahun": jurnal.tahun,
                "nama_file": jurnal.nama_file,
                "halaman": c.halaman,
                "urutan": c.urutan,
                "teks": c.teks,
            },
        )
        for c, v in zip(chunks, vektor)
    ]
    _client().upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    return len(points)


def _cari_satu(query: str, user_id: str, limit: int) -> list[dict]:
    hasil = _client().query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=embed_query(query),
        query_filter=models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
        ]),
        limit=limit,
        with_payload=True,
    ).points
    return [{**p.payload, "skor": round(p.score, 3)} for p in hasil]


def ambil_chunk(user_id: str, paper_id: str, urutan: list[int]) -> list[dict]:
    """Ambil chunk tertentu (berdasarkan nomor urut) dari satu jurnal milik user.
    Dipakai untuk context expansion: mengambil chunk tetangga dari chunk yang relevan."""
    if not urutan:
        return []
    points, _ = _client().scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id)),
            models.FieldCondition(key="urutan", match=models.MatchAny(any=urutan)),
        ]),
        limit=len(urutan), with_payload=True, with_vectors=False,
    )
    return [p.payload for p in points]


# ------------------------------------------------------------------ BM25 (kata kunci)
_cache_bm25: dict[str, tuple[int, BM25, list[dict]]] = {}


def _semua_chunk(user_id: str) -> list[dict]:
    """Ambil semua chunk milik user dari Qdrant (tanpa vektor) untuk indeks BM25."""
    filter_user = models.Filter(must=[
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
    hasil, offset = [], None
    while True:
        points, offset = _client().scroll(collection_name=config.QDRANT_COLLECTION,
                                          scroll_filter=filter_user, limit=256, offset=offset,
                                          with_payload=True, with_vectors=False)
        hasil.extend(p.payload for p in points)
        if offset is None:
            return hasil


def _indeks_bm25(user_id: str) -> tuple[BM25, list[dict]]:
    """Indeks BM25 per user, dibangun ulang otomatis jika jumlah chunk berubah."""
    jumlah = _client().count(
        collection_name=config.QDRANT_COLLECTION, exact=True,
        count_filter=models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]),
    ).count
    simpan = _cache_bm25.get(user_id)
    if simpan is None or simpan[0] != jumlah:
        dokumen = _semua_chunk(user_id)
        simpan = (jumlah, BM25([d["teks"] for d in dokumen]), dokumen)
        _cache_bm25[user_id] = simpan
    return simpan[1], simpan[2]


def cari_tabel(queries: list[str], user_id: str, limit: int = 3) -> list[dict]:
    """Cari KHUSUS di antara chunk tabel ("[TABEL] ...") memakai BM25.

    Dipakai untuk pertanyaan daftar ("apa saja", "jenis", "types of"): rinciannya sering
    hanya ada di tabel, tapi chunk tabel kalah bersaing dengan paragraf biasa di pencarian umum.
    """
    bm25, dokumen = _indeks_bm25(user_id)
    indeks_tabel = [i for i, d in enumerate(dokumen) if d["teks"].startswith("[TABEL]")]
    if not indeks_tabel:
        return []
    skor_terbaik: dict[int, float] = {}
    for q in queries:
        skor = bm25.skor(q)
        for i in indeks_tabel:
            if skor[i] > skor_terbaik.get(i, 0.0):
                skor_terbaik[i] = skor[i]
    urut = sorted(skor_terbaik, key=skor_terbaik.get, reverse=True)[:limit]
    return [{**dokumen[i], "skor": 0.0, "rrf": 0.0} for i in urut]


# ------------------------------------------------------------------ hybrid search
RRF_K = 60  # konstanta standar Reciprocal Rank Fusion


def cari(queries: str | list[str], user_id: str, top_k: int = config.TOP_K,
         maks_per_jurnal: int = 2) -> list[dict]:
    """Hybrid search: vektor (makna) + BM25 (kata kunci), HANYA dari jurnal milik user_id.

    - Setiap query dicari dua kali: lewat vektor dan lewat BM25.
    - Semua daftar hasil digabung dengan Reciprocal Rank Fusion (RRF):
      skor = jumlah 1 / (60 + peringkat) dari setiap daftar tempat chunk itu muncul.
      Chunk yang konsisten muncul di banyak daftar akan naik ke atas.
    - `maks_per_jurnal` mencegah satu jurnal menguasai semua hasil.
    """
    siapkan_koleksi()
    if isinstance(queries, str):
        queries = [queries]
    per_daftar = top_k * 3

    data: dict[tuple, dict] = {}
    rrf: dict[tuple, float] = {}

    def tambah(item: dict, peringkat: int) -> None:
        kunci = (item["paper_id"], item["urutan"])
        if kunci not in data:
            data[kunci] = {**item, "skor": item.get("skor", 0.0)}
        else:
            data[kunci]["skor"] = max(data[kunci]["skor"], item.get("skor", 0.0))
        rrf[kunci] = rrf.get(kunci, 0.0) + 1.0 / (RRF_K + peringkat)

    bm25, dokumen = _indeks_bm25(user_id)
    for q in queries:
        for peringkat, item in enumerate(_cari_satu(q, user_id, limit=per_daftar), start=1):
            tambah(item, peringkat)
        for peringkat, (i, _) in enumerate(bm25.cari(q, limit=per_daftar), start=1):
            tambah(dokumen[i], peringkat)

    for kunci, nilai in rrf.items():
        data[kunci]["rrf"] = round(nilai, 4)

    hasil, per_jurnal = [], {}
    for item in sorted(data.values(), key=lambda x: x["rrf"], reverse=True):
        n = per_jurnal.get(item["paper_id"], 0)
        if n >= maks_per_jurnal:
            continue
        per_jurnal[item["paper_id"]] = n + 1
        hasil.append(item)
        if len(hasil) == top_k:
            break
    return hasil
