"""Embedding chunk lalu simpan/cari di Qdrant.

Setiap chunk menyimpan `user_id`, dan setiap pencarian WAJIB difilter per user,
supaya jurnal milik satu orang tidak pernah muncul di jawaban orang lain.
"""
import re
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


def _cari_satu(query: str, user_id: str, limit: int, paper_id: str | None = None) -> list[dict]:
    syarat = [models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
    if paper_id:
        syarat.append(models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id)))
    hasil = _client().query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=embed_query(query),
        query_filter=models.Filter(must=syarat),
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


def daftar_jurnal(user_id: str) -> list[dict]:
    """Ringkasan setiap jurnal milik user (untuk UI): judul, penulis, tahun, jumlah chunk & halaman."""
    siapkan_koleksi()
    _, dokumen = _indeks_bm25(user_id)
    ringkas: dict[str, dict] = {}
    for d in dokumen:
        r = ringkas.setdefault(d["paper_id"], {
            "paper_id": d["paper_id"], "judul": d.get("judul") or "(tanpa judul)",
            "penulis": d.get("penulis") or "-", "tahun": d.get("tahun"),
            "nama_file": d.get("nama_file") or "", "chunk": 0, "halaman": 0})
        r["chunk"] += 1
        r["halaman"] = max(r["halaman"], d.get("halaman") or 0)
    return sorted(ringkas.values(), key=lambda r: (r["penulis"] or "").lower())


def jumlah_jurnal(user_id: str) -> int:
    _, dokumen = _indeks_bm25(user_id)
    return len({d["paper_id"] for d in dokumen})


def hapus_jurnal(user_id: str, paper_id: str) -> None:
    """Hapus semua chunk satu jurnal milik user dari Qdrant."""
    _client().delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id)),
        ])),
    )
    _cache_bm25.pop(user_id, None)


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


def _ringkas(teks: str) -> str:
    return re.sub(r"[^a-z0-9]", "", teks.lower())


def jurnal_untuk_istilah(istilah: str, user_id: str) -> str | None:
    """Tebak jurnal "rumah" sebuah nama metode, mis. "Self-RAG" -> paper Asai dkk.

    1. Jurnal yang JUDULNYA memuat istilah itu (paling pasti).
    2. Jika tidak ada: jurnal yang paling sering menyebut istilah itu, asalkan jelas dominan
       (mis. "CRAG" tidak ada di judul "Corrective Retrieval Augmented Generation",
       tapi disebut puluhan kali di paper itu dan hanya sesekali di paper lain).
    """
    _, dokumen = _indeks_bm25(user_id)
    kunci = _ringkas(istilah)
    if len(kunci) < 3:
        return None
    for d in dokumen:
        if kunci in _ringkas(d.get("judul", "")):
            return d["paper_id"]
    pola = re.compile(rf"(?<![\w-]){re.escape(istilah)}(?![\w-])")
    hitung: dict[str, int] = {}
    for d in dokumen:
        n = len(pola.findall(d["teks"]))
        if n:
            hitung[d["paper_id"]] = hitung.get(d["paper_id"], 0) + n
    if not hitung:
        return None
    urut = sorted(hitung.items(), key=lambda x: x[1], reverse=True)
    terbanyak, n1 = urut[0]
    n2 = urut[1][1] if len(urut) > 1 else 0
    return terbanyak if n1 >= 5 and n1 >= 2 * n2 else None


# Judul daftar pustaka: kata berhuruf kapital diikuti baris baru. Di teks PDF judul ini sering
# menempel di akhir kalimat sebelumnya ("... gifts from AI2. REFERENCES\nAkari Asai, ...").
_POLA_REFERENSI = re.compile(r"\b(References|REFERENCES|Daftar Pustaka|DAFTAR PUSTAKA|Bibliography|BIBLIOGRAPHY)\s*\n")


def struktur_paper(user_id: str, paper_id: str) -> dict:
    """Perkiraan struktur sebuah paper dari posisi chunk-nya.

    - mulai_referensi: urutan chunk tempat daftar pustaka dimulai (sesudahnya = referensi/lampiran)
    - tengah_isi: titik tengah isi utama (hal. > 2 dan sebelum referensi). Paruh pertama isi
      biasanya bagian metode; paruh kedua biasanya eksperimen dan hasil.
    """
    _, dokumen = _indeks_bm25(user_id)
    milik = [d for d in dokumen if d["paper_id"] == paper_id]
    ref = [d["urutan"] for d in milik if _POLA_REFERENSI.search(d["teks"])]
    mulai_referensi = min(ref) if ref else None
    isi = [d["urutan"] for d in milik
           if d["halaman"] > 2 and (mulai_referensi is None or d["urutan"] < mulai_referensi)]
    tengah_isi = (min(isi) + max(isi)) / 2 if isi else None
    return {"mulai_referensi": mulai_referensi, "tengah_isi": tengah_isi}


# ------------------------------------------------------------------ hybrid search
RRF_K = 60  # konstanta standar Reciprocal Rank Fusion


def cari(queries: str | list[str], user_id: str, top_k: int = config.TOP_K,
         maks_per_jurnal: int = 2, paper_id: str | None = None) -> list[dict]:
    """Hybrid search: vektor (makna) + BM25 (kata kunci), HANYA dari jurnal milik user_id.

    - Setiap query dicari dua kali: lewat vektor dan lewat BM25.
    - Semua daftar hasil digabung dengan Reciprocal Rank Fusion (RRF):
      skor = jumlah 1 / (60 + peringkat) dari setiap daftar tempat chunk itu muncul.
      Chunk yang konsisten muncul di banyak daftar akan naik ke atas.
    - `maks_per_jurnal` mencegah satu jurnal menguasai semua hasil.
    - `paper_id` (opsional) membatasi pencarian ke SATU jurnal saja.
    - Juara BM25: chunk peringkat 1 BM25 dari setiap query DIJAMIN ikut. RRF cenderung
      mengunggulkan chunk "umum" (abstrak, pendahuluan) yang muncul sedang-sedang di banyak
      daftar, dan menyingkirkan chunk yang sangat cocok di satu daftar saja. Tes: query
      "Transformer positional encoding" menempatkan bagian 3.5 di BM25 peringkat 1, tetapi
      kalah RRF dari pendahuluan hal. 1-2 dan tidak masuk 12 kandidat.
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
    juara: list[tuple] = []
    for q in queries:
        for peringkat, item in enumerate(_cari_satu(q, user_id, limit=per_daftar, paper_id=paper_id), start=1):
            tambah(item, peringkat)
        if paper_id:
            skor = bm25.skor(q)
            teratas = sorted((i for i, d in enumerate(dokumen) if d["paper_id"] == paper_id and skor[i] > 0),
                             key=lambda i: skor[i], reverse=True)[:per_daftar]
        else:
            teratas = [i for i, _ in bm25.cari(q, limit=per_daftar)]
        for peringkat, i in enumerate(teratas, start=1):
            tambah(dokumen[i], peringkat)
        if teratas:
            k = (dokumen[teratas[0]]["paper_id"], dokumen[teratas[0]]["urutan"])
            if k not in juara:
                juara.append(k)

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

    # Pastikan juara BM25 ikut: ganti kandidat RRF terlemah (yang bukan juara)
    juara = juara[:top_k]
    ada = {(h["paper_id"], h["urutan"]) for h in hasil}
    for k in juara:
        if k in ada:
            continue
        if len(hasil) >= top_k:
            for idx in range(len(hasil) - 1, -1, -1):
                if (hasil[idx]["paper_id"], hasil[idx]["urutan"]) not in juara:
                    hasil.pop(idx)
                    break
            else:
                continue
        hasil.append(data[k])
        ada.add(k)
    return hasil
