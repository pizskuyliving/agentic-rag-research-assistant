"""Node-node agen. Setiap fungsi menerima state dan mengembalikan perubahan state."""
import json
import re

from app import config
from app.agent import prompts
from app.agent.state import ResearchState
from app.ingestion.indexer import (ambil_chunk, cari, cari_tabel, jumlah_jurnal, jurnal_untuk_istilah,
                                  struktur_paper)
from app.ingestion.bm25 import tokenisasi
from app.llm import panggil_llm
from app.query import variasi_query

# Berapa kandidat chunk yang diambil sebelum disaring Grade.
# Grade terbukti andal membuang yang tidak relevan, jadi lebih aman mengambil kandidat lebih banyak.
KANDIDAT_RETRIEVE = 12
# Maksimal chunk per jurnal di daftar kandidat (3 terbukti memotong bagian metode)
MAKS_PER_JURNAL_KANDIDAT = 5
# Panjang maksimum tiap chunk saat dinilai Grade (hemat token, tapi cukup untuk melihat isinya)
POTONG_UNTUK_GRADE = 1200  # = ukuran chunk: Grade melihat chunk utuh (750 terbukti memotong bagian penting)
# Panjang cuplikan chunk saat Evaluasi memeriksa kecukupan
POTONG_UNTUK_EVALUASI = 500
# Berapa kali boleh mencari ulang (rewrite). 1 = maksimal dua kali pencarian per pertanyaan.
MAKS_PUTARAN_ULANG = 1
# Maksimal chunk relevan yang dikirim ke Generate (akumulasi semua putaran)
MAKS_SUMBER_GENERATE = 6
# Berapa chunk "jangkar" diambil dari paper asli tiap metode yang disebut di pertanyaan.
# Jangkar DIJAMIN ikut ke Generate walau ditolak Grade (lihat grade()).
JANGKAR_PER_ENTITAS = 2
# Berapa calon jangkar diambil dari paper itu sebelum dipilih ulang (lihat _kelas_jangkar)
CALON_JANGKAR = 30
# Minimal proporsi kalimat klaim yang bersitasi; di bawah ini jawaban diminta diperbaiki
MIN_RASIO_SITASI = 0.7
# Batas total karakter konteks untuk Generate setelah ditambah chunk tetangga
# (~12.000 karakter ≈ 3.000 token, aman untuk kuota 8.000 token/menit model besar)
MAKS_KARAKTER_KONTEKS = 12000


def _json(teks: str) -> dict | None:
    """Ambil objek JSON pertama yang valid dari teks (toleran terhadap teks tambahan)."""
    teks = teks or ""
    try:
        data = json.loads(teks)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    # cari setiap kemungkinan objek {...}, dari yang terpendek, sampai ada yang valid
    for m in re.finditer(r"\{", teks):
        for akhir in range(m.start() + 1, len(teks) + 1):
            if teks[akhir - 1] == "}":
                try:
                    data = json.loads(teks[m.start():akhir])
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
    return None


def _nomor_relevan(data: dict | None, mentah: str, jumlah: int) -> set[int] | None:
    """Ambil daftar nomor chunk relevan dari jawaban Grade, seberapa pun berantakan formatnya."""
    nilai = None
    if data:
        for kunci in ("relevan", "relevant", "relevan_ids", "relevant_ids", "nomor"):
            if kunci in data:
                nilai = data[kunci]
                break
    if nilai is None:
        # tanpa JSON valid: cari pola seperti "relevan": [1, 3] di teks mentah
        m = re.search(r"relevan\w*\"?\s*:\s*\[([^\]]*)\]", mentah, re.IGNORECASE)
        if not m:
            return None
        nilai = m.group(1)
    if isinstance(nilai, (int, str)):
        nilai = re.findall(r"\d+", str(nilai))
    if not isinstance(nilai, list):
        return None
    hasil = set()
    for n in nilai:
        angka = re.findall(r"\d+", str(n))
        if angka and 1 <= int(angka[0]) <= jumlah:
            hasil.add(int(angka[0]))
    return hasil


def _jejak(state: ResearchState, pesan: str) -> list[str]:
    return [*state.get("trace", []), pesan]


# ---------------------------------------------------------------- router
def router(state: ResearchState) -> dict:
    route = "search"  # default aman: kalau ragu atau error, cari di koleksi
    try:
        data = _json(panggil_llm(prompts.ROUTER + state["question"], peran="ringan",
                                 max_tokens=50, json_mode=True))
        if data and data.get("route") == "direct":
            route = "direct"
    except Exception as e:
        return {"route": route, "trace": _jejak(state, f"router error ({e}) -> search")}
    return {"route": route, "trace": _jejak(state, f"router -> {route}")}


def _kunci(c: dict) -> tuple:
    return (c["paper_id"], c["urutan"])


_POLA_DAFTAR = re.compile(
    r"\b(apa saja|sebutkan|jenis|macam|komponen|tahapan|langkah-langkah|daftar|"
    r"what are|which|types? of|kinds? of|list|components?|steps)\b", re.IGNORECASE)


# Nama metode/sistem: "Self-RAG", "Adaptive-RAG", "CRAG", "IndoBERT", "RAG-Token", "BM25"
_POLA_METODE = re.compile(r"\b(?:[A-Za-z]+-[A-Z][A-Za-z0-9]*|[A-Z]{2,}[A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b")
# Singkatan umum yang bukan nama metode spesifik (akan "menarik" paper yang salah)
_BUKAN_METODE = {"RAG", "LLM", "LLMS", "AI", "PDF", "NLP", "QA", "ID", "EN", "API", "GPU", "URL"}


def _nama_metode(pertanyaan: str) -> list[str]:
    hasil = []
    for m in _POLA_METODE.findall(pertanyaan):
        if m.upper() not in _BUKAN_METODE and m not in hasil:
            hasil.append(m)
    return hasil[:3]


def _pertanyaan_daftar(pertanyaan: str) -> bool:
    """Pertanyaan yang jawabannya berupa daftar item (sering ada di tabel)."""
    return bool(_POLA_DAFTAR.search(pertanyaan))


# ---------------------------------------------------------------- jangkar entitas
_POLA_PENUTUP = re.compile(r"\b(?:\d+\.?\s*)?(CONCLUSIONS?|Conclusions?|KESIMPULAN|Kesimpulan|PENUTUP)\b")
_POLA_KUTIPAN = re.compile(r"et al\.?,?\s*\(?\d{4}|\(\w+[^()]{0,40}?\d{4}[a-z]?\)")
MIN_KUTIPAN_RELATED_WORK = 3


def _kelas_jangkar(c: dict, struktur: dict) -> int:
    """0 = isi utama paruh pertama (biasanya bagian METODE: cara kerja dijelaskan di sini)
    1 = isi lain: abstrak/pendahuluan (hal. 1-2), kesimpulan, eksperimen (paruh kedua isi),
        atau related work (banyak kutipan "et al., 2023")
    2 = referensi/lampiran, atau tabel yang isinya sebagian besar angka

    Pelajaran dari tes pada paper asli: pencarian di dalam satu paper cenderung memilih
    kesimpulan, related work, hasil eksperimen, lampiran, atau tabel TFLOPs, karena bagian itu
    paling sering menyebut nama metode. Penjelasan cara kerjanya justru di bagian metode.
    """
    teks = c["teks"]
    ref, tengah = struktur.get("mulai_referensi"), struktur.get("tengah_isi")
    if ref is not None and c["urutan"] >= ref:
        return 2
    if teks.startswith("[TABEL]"):
        isi = re.sub(r"\s", "", teks)
        if sum(ch.isdigit() for ch in isi) / max(len(isi), 1) > 0.15:
            return 2
    if (c["halaman"] <= 2 or _POLA_PENUTUP.search(teks)
            or len(_POLA_KUTIPAN.findall(teks)) >= MIN_KUTIPAN_RELATED_WORK
            or (tengah is not None and c["urutan"] > tengah)):
        return 1
    return 0


def _query_untuk(istilah: str, queries: list[str], semua_istilah: list[str], pertanyaan: str) -> list[str]:
    """Query yang memuat istilah ini, dengan nama metode LAIN dihapus.

    "difference between Self-RAG and CRAG ..." untuk paper Self-RAG menjadi
    "difference between Self-RAG and ...": tanpa ini, bagian paper yang membandingkan kedua
    metode (biasanya related work atau hasil) selalu menang.
    """
    hasil = []
    for q in queries or [pertanyaan]:
        if istilah.lower() not in q.lower():
            continue
        for lain in semua_istilah:
            if lain != istilah:
                q = re.sub(rf"(?<![\w-]){re.escape(lain)}(?![\w-])", " ", q, flags=re.IGNORECASE)
        q = " ".join(q.split())
        if q not in hasil:
            hasil.append(q)
    return hasil or [pertanyaan]


def ambil_jangkar(pertanyaan: str, queries: list[str], user_id: str) -> list[dict]:
    """Chunk terbaik dari paper asli setiap nama metode di pertanyaan (maks. 2 per metode)."""
    jangkar: list[dict] = []
    semua = _nama_metode(pertanyaan)
    for istilah in semua:
        paper = jurnal_untuk_istilah(istilah, user_id)
        if not paper:
            continue
        calon = cari(_query_untuk(istilah, queries, semua, pertanyaan), user_id=user_id,
                     top_k=CALON_JANGKAR, maks_per_jurnal=CALON_JANGKAR, paper_id=paper)
        struktur = struktur_paper(user_id, paper)
        # urutkan ulang: kelas dulu (bagian metode di depan), lalu peringkat pencarian
        calon = sorted(enumerate(calon), key=lambda x: (_kelas_jangkar(x[1], struktur), x[0]))
        ada = {_kunci(c) for c in jangkar}
        jangkar += [{**c, "entitas": istilah} for _, c in calon if _kunci(c) not in ada][:JANGKAR_PER_ENTITAS]
    return jangkar


def _kerangka_perbandingan(jangkar: list[dict]) -> str:
    """Kerangka jawaban untuk pertanyaan yang menyebut 2+ metode (mis. "Self-RAG vs CRAG").

    Pelajaran dari tes: tanpa kerangka, model menulis 2-3 kalimat umum dan melewatkan detail
    mekanisme (nama aksi/label/token) walau detail itu ada di konteks.
    """
    entitas: dict[str, str] = {}
    for c in jangkar:
        entitas.setdefault(c["entitas"], c["penulis"])
    if len(entitas) < 2:
        return ""
    bagian = "\n".join(
        f"{i}. **{nama}**: jelaskan cara kerjanya terkait pertanyaan, terutama dari sumber {penulis.rstrip('.')}. "
        f"Sebutkan SEMUA nama token/label/aksi/kategori yang dipakai metode ini persis seperti di sumber."
        for i, (nama, penulis) in enumerate(entitas.items(), start=1))
    return (f"Susun jawaban dengan kerangka berikut:\n{bagian}\n"
            f"{len(entitas) + 1}. **Perbedaan utama**: 2-3 poin singkat.\n\n")


def _batas_per_jurnal(user_id: str) -> int:
    """Batas kandidat per jurnal, menyesuaikan ukuran koleksi.

    Batas 5 per jurnal mencegah satu jurnal menguasai hasil saat koleksinya banyak. Tapi pada
    koleksi berisi 1-2 jurnal, batas itu membuat kandidat hanya 5 dari 12 yang diminta, dan
    bagian yang dicari (mis. positional encoding di hal. 6) bisa tidak terambil sama sekali.
    """
    try:
        n = jumlah_jurnal(user_id)
    except Exception:
        return MAKS_PER_JURNAL_KANDIDAT
    return max(MAKS_PER_JURNAL_KANDIDAT, -(-KANDIDAT_RETRIEVE // max(n, 1)))


# ---------------------------------------------------------------- retrieve
def retrieve(state: ResearchState) -> dict:
    putaran = state.get("putaran", 0)
    if putaran == 0:
        queries = variasi_query(state["question"])
        chunks = cari(queries, user_id=state["user_id"], top_k=KANDIDAT_RETRIEVE,
                      maks_per_jurnal=_batas_per_jurnal(state["user_id"]))
        catatan = f"retrieve: {len(queries)} query -> {len(chunks)} kandidat"

        # Retrieval sadar entitas: untuk setiap nama metode di pertanyaan, ambil chunk terbaik
        # dari paper ASLI metode itu. Chunk ini jadi "jangkar" yang dijamin ikut ke Generate.
        # Tanpa ini, "Self-RAG vs CRAG" selalu bersumber dari related-work paper CRAG.
        jangkar = ambil_jangkar(state["question"], queries, state["user_id"])
        if jangkar:
            kunci_jangkar = {_kunci(c) for c in jangkar}
            chunks = [*jangkar, *[c for c in chunks if _kunci(c) not in kunci_jangkar]]
            halaman = ", ".join(f"{_nama_pendek(c)} hal. {c['halaman']}" for c in jangkar)
            catatan += f" + {len(jangkar)} jangkar dari paper metode ({halaman})"

        # Pertanyaan daftar: tambahkan tabel terbaik secara pasti (tidak bergantung keputusan LLM)
        if _pertanyaan_daftar(state["question"]):
            ada = {_kunci(c) for c in chunks}
            tabel = [t for t in cari_tabel(queries, user_id=state["user_id"]) if _kunci(t) not in ada]
            chunks = [*tabel, *chunks]
            catatan += f" + {len(tabel)} tabel (pertanyaan daftar)"
        return {"queries": queries, "chunks": chunks, "jangkar": jangkar,
                "trace": _jejak(state, catatan)}

    # Putaran ulang: pakai query hasil rewrite, dan lewati chunk yang sudah dinyatakan relevan
    queries = state.get("query_baru", [])
    sudah = {_kunci(c) for c in state.get("relevant", [])}
    chunks = [c for c in cari(queries, user_id=state["user_id"], top_k=KANDIDAT_RETRIEVE,
                              maks_per_jurnal=_batas_per_jurnal(state["user_id"])) if _kunci(c) not in sudah]
    return {"queries": [*state.get("queries", []), *queries], "chunks": chunks,
            "trace": _jejak(state, f"retrieve ulang #{putaran}: {len(queries)} query baru -> "
                                   f"{len(chunks)} kandidat baru")}


# ---------------------------------------------------------------- grade
def _cuplikan_terbaik(teks: str, kata: set[str], panjang: int = POTONG_UNTUK_GRADE) -> str:
    """Bagian chunk (sepanjang `panjang`) yang paling banyak memuat kata dari pertanyaan/query.

    Grade hanya melihat potongan pendek supaya hemat token. Dulu yang dikirim selalu awal chunk,
    padahal satu chunk bisa berisi sisa tabel lalu awal bagian baru (mis. "Table 1 ..." lalu
    "3.5 Positional Encoding"). Bagian yang relevan jadi terpotong dan chunk ditolak.
    """
    if len(teks) <= panjang or not kata:
        return teks[:panjang]
    terbaik, skor_terbaik = 0, -1
    for mulai in range(0, len(teks) - panjang + 150, 150):
        jendela = tokenisasi(teks[mulai:mulai + panjang])
        skor = len(kata.intersection(jendela)) * 10 + sum(t in kata for t in jendela)
        if skor > skor_terbaik:
            terbaik, skor_terbaik = mulai, skor
    potongan = teks[terbaik:terbaik + panjang]
    return ("… " if terbaik else "") + potongan + (" …" if terbaik + panjang < len(teks) else "")


def grade(state: ResearchState) -> dict:
    sebelumnya = state.get("relevant", []) if state.get("putaran", 0) > 0 else []
    chunks = state.get("chunks", [])
    if not chunks:
        return {"relevant": sebelumnya, "trace": _jejak(state, "grade: tidak ada kandidat")}

    kata = {k for q in [state["question"], *state.get("queries", [])] for k in tokenisasi(q)}
    daftar = "\n\n".join(
        f"[{i}] ({c['penulis']}, {c['tahun']}, judul: {c['judul'][:90]})\n{_cuplikan_terbaik(c['teks'], kata)}"
        for i, c in enumerate(chunks, start=1)
    )
    prompt = prompts.GRADE.format(question=state["question"], chunks=daftar)
    mentah = ""
    try:
        # SATU panggilan untuk semua chunk (bukan satu per chunk) supaya kuota hemat
        mentah = panggil_llm(prompt, peran=config.PERAN_GRADE, max_tokens=300, json_mode=True)
        dipilih = _nomor_relevan(_json(mentah), mentah, len(chunks))
        if dipilih is None:
            raise ValueError("format jawaban grade tidak valid")
        relevant = [c for i, c in enumerate(chunks, start=1) if i in dipilih]
    except Exception as e:
        # gagal menilai -> jangan buang apa pun, biar Generate yang berhati-hati.
        # Balasan mentah dicatat agar penyebabnya terlihat di jejak.
        cuplikan = " ".join(mentah.split())[:150]
        return {"relevant": [*sebelumnya, *chunks][:MAKS_SUMBER_GENERATE],
                "trace": _jejak(state, f"grade error ({e}; balasan: {cuplikan!r}) -> pakai semua")}

    # Jangkar (chunk terbaik paper asli metode yang ditanyakan) dijamin ikut walau ditolak Grade:
    # Grade sering menolaknya karena potongan 750 karakter belum menyebut "dokumen tidak relevan",
    # padahal penjelasannya ada di chunk tetangga yang nanti ditambahkan context expansion.
    dipaksa = []
    if state.get("putaran", 0) == 0:
        lolos = {_kunci(c) for c in relevant}
        dipaksa = [c for c in state.get("jangkar", []) if _kunci(c) not in lolos]

    # gabungkan: jangkar dulu, lalu hasil putaran sebelumnya, lalu yang baru lolos
    gabung = _unik([*dipaksa, *sebelumnya, *relevant])[:MAKS_SUMBER_GENERATE]
    return {"relevant": gabung,
            "trace": _jejak(state, f"grade: {len(relevant)} dari {len(chunks)} relevan"
                                   + (f" + {len(dipaksa)} jangkar dipaksa" if dipaksa else "")
                                   + (f" (total {len(gabung)})" if sebelumnya else ""))}


def _nama_pendek(c: dict) -> str:
    kata = (c.get("penulis") or "").split(",")[0].split("&")[0].split()
    return kata[-1] if kata else "?"


def _unik(chunks: list[dict]) -> list[dict]:
    hasil, ada = [], set()
    for c in chunks:
        if _kunci(c) not in ada:
            ada.add(_kunci(c))
            hasil.append(c)
    return hasil


# ---------------------------------------------------------------- evaluasi + rewrite
def evaluasi(state: ResearchState) -> dict:
    """Apakah chunk relevan sudah cukup? Jika belum, tulis query baru (rewrite) dan cari ulang.

    Query baru disusun dari istilah yang MUNCUL di chunk yang sudah ditemukan, bukan dari
    ingatan model (pelajaran dari tes: tebakan istilah dari ingatan meleset).
    """
    relevant = state.get("relevant", [])
    putaran = state.get("putaran", 0)
    if putaran >= MAKS_PUTARAN_ULANG:
        keputusan = "generate" if relevant else "tidak_ada"
        return {"keputusan": keputusan, "trace": _jejak(state, f"evaluasi: batas putaran -> {keputusan}")}

    daftar = "\n\n".join(
        f"- ({c['penulis']}, hal. {c['halaman']}) {' '.join(c['teks'].split())[:POTONG_UNTUK_EVALUASI]}"
        for c in relevant
    ) or "(belum ada potongan yang relevan)"
    try:
        data = _json(panggil_llm(prompts.EVALUASI.format(question=state["question"], chunks=daftar),
                                 peran=config.PERAN_EVALUASI, max_tokens=400, json_mode=True)) or {}
    except Exception as e:
        keputusan = "generate" if relevant else "tidak_ada"
        return {"keputusan": keputusan, "trace": _jejak(state, f"evaluasi error ({e}) -> {keputusan}")}

    query_baru = [q.strip() for q in data.get("query_baru") or [] if isinstance(q, str) and q.strip()][:3]
    if data.get("cukup") is True or not query_baru:
        keputusan = "generate" if relevant else "tidak_ada"
        return {"keputusan": keputusan, "trace": _jejak(state, f"evaluasi: cukup -> {keputusan}")}

    kurang = str(data.get("yang_kurang", ""))[:100]
    return {"keputusan": "ulang", "putaran": putaran + 1, "query_baru": query_baru,
            "trace": _jejak(state, f"evaluasi: kurang ({kurang}) -> cari ulang")}


def setelah_evaluasi(state: ResearchState) -> str:
    return state.get("keputusan", "generate")


# ---------------------------------------------------------------- context expansion
def _perluas_tetangga(relevant: list[dict], user_id: str) -> tuple[list[dict], int]:
    """Tambahkan chunk sebelum & sesudah setiap chunk relevan (context expansion).

    Paper sering menjelaskan satu ide di beberapa paragraf berurutan: chunk yang lolos Grade
    menyebut "retrieval evaluator", penjelasan aksinya ada di chunk berikutnya.
    Hasil diurutkan per jurnal lalu per urutan, supaya teks terbaca bersambung.
    """
    ada = {_kunci(c) for c in relevant}
    perlu: dict[str, set[int]] = {}
    for c in relevant:
        for u in (c["urutan"] - 1, c["urutan"] + 1):
            if u >= 0 and (c["paper_id"], u) not in ada:
                perlu.setdefault(c["paper_id"], set()).add(u)

    tetangga = []
    for paper_id, urutan in perlu.items():
        try:
            tetangga += ambil_chunk(user_id, paper_id, sorted(urutan))
        except Exception:
            continue  # gagal mengambil tetangga bukan alasan untuk gagal menjawab

    semua = {_kunci(c): c for c in [*relevant, *tetangga]}
    urut = sorted(semua.values(), key=lambda c: (c["paper_id"], c["urutan"]))

    # batasi total panjang konteks agar tetap di bawah kuota token per menit
    hasil, total = [], 0
    for c in urut:
        if total + len(c["teks"]) > MAKS_KARAKTER_KONTEKS and _kunci(c) not in ada:
            continue
        hasil.append(c)
        total += len(c["teks"])
    return hasil, len(hasil) - len(relevant)


# ---------------------------------------------------------------- generate
def generate(state: ResearchState) -> dict:
    sumber, tambahan = _perluas_tetangga(state["relevant"], state["user_id"])
    state = {**state, "trace": _jejak(state, f"perluas konteks: +{tambahan} chunk tetangga")}
    konteks = "\n\n".join(
        f"[{i}] {s['penulis']} ({s['tahun']}), hal. {s['halaman']}, judul: {s['judul']}\n{s['teks']}"
        for i, s in enumerate(sumber, start=1)
    )
    # Aturan sitasi diulang di AKHIR pesan: dengan konteks panjang, model lebih patuh
    # pada instruksi yang terakhir dibacanya.
    pesan = (f"Potongan jurnal:\n{konteks}\n\nPertanyaan: {state['question']}\n\n"
             f"{_kerangka_perbandingan(state.get('jangkar', []))}{prompts.PENGINGAT_SITASI}")
    jawaban = _rapikan_sitasi(panggil_llm(pesan, peran="berat", system=prompts.GENERATE_SYSTEM,
                                          max_tokens=1200))

    # Cek sitasi secara pasti (tanpa LLM). Jika kurang, minta perbaikan SATU kali.
    ada, total, tanpa = _cek_sitasi(jawaban)
    catatan = f"cek sitasi: {ada}/{total} kalimat bersitasi"
    if total and ada / total < MIN_RASIO_SITASI:
        daftar = "\n".join(f"- {k}" for k in tanpa[:8])
        perbaikan = prompts.PERBAIKI_SITASI.format(jawaban=jawaban, daftar=daftar)
        try:
            baru = _rapikan_sitasi(panggil_llm(f"Potongan jurnal:\n{konteks}\n\n{perbaikan}",
                                               peran="berat", system=prompts.GENERATE_SYSTEM,
                                               max_tokens=1200))
            ada2, total2, _ = _cek_sitasi(baru)
            if total2 and ada2 / total2 > ada / total:
                jawaban = baru
                catatan += f" -> diperbaiki: {ada2}/{total2}"
            else:
                catatan += " -> perbaikan tidak lebih baik, pakai versi awal"
        except Exception as e:
            catatan += f" -> perbaikan gagal ({e})"
    state = {**state, "trace": _jejak(state, catatan)}

    # Jawaban berupa pernyataan "informasi tidak ada di sumber" tidak perlu daftar sumber:
    # sitasi pada kalimat seperti itu menyesatkan (seolah-olah sumber mendukung sesuatu).
    if _jawaban_tidak_ada(jawaban):
        jawaban = re.sub(r"\s*(\[\d+\])+", "", jawaban).strip()
        return {"answer": jawaban, "citations": [], "from_collection": True,
                "trace": _jejak(state, "generate: informasi tidak ditemukan di sumber (sitasi dihapus)")}

    # Hanya tampilkan sumber yang benar-benar disitasi di jawaban
    dipakai = sorted({int(n) for n in re.findall(r"\[(\d+)\]", jawaban) if 1 <= int(n) <= len(sumber)})
    citations = [{**sumber[n - 1], "nomor": n} for n in dipakai]
    return {"answer": jawaban, "citations": citations, "from_collection": True,
            "trace": _jejak(state, f"generate: {len(citations)} sumber disitasi")}


def _rapikan_sitasi(teks: str) -> str:
    """gpt-oss kadang menulis sitasi dengan kurung lebar 【1】 / ［1］ -> samakan jadi [1].
    Juga [1, 3] / [1][3] dibiarkan, keduanya terbaca."""
    teks = re.sub(r"[【［]\s*(\d+)\s*[】］]", r"[\1]", teks)
    return re.sub(r"\[(\d+)\s*,\s*(\d+)\]", r"[\1][\2]", teks)


_POLA_TIDAK_ADA = re.compile(
    r"tidak (terdapat|ada|ditemukan|disebutkan|dibahas|memuat|menyebutkan|membahas)\b.{0,60}"
    r"(informasi|data|penjelasan|hasil|angka|nilai)|"
    r"(informasi|data)\b.{0,40}tidak (tersedia|ditemukan|disebutkan)|"
    r"(do(es)? not|no) (mention|information|data|report)", re.IGNORECASE)


def _jawaban_tidak_ada(jawaban: str) -> bool:
    """Jawaban pendek (maks. 2 kalimat) yang intinya menyatakan informasinya tidak ada di sumber."""
    tanpa_sitasi = re.sub(r"\[\d+\]", "", jawaban)
    kalimat = [k for k in re.split(r"(?<=[.!?])\s+", tanpa_sitasi.strip()) if k.strip()]
    return len(kalimat) <= 2 and len(tanpa_sitasi) < 400 and bool(_POLA_TIDAK_ADA.search(tanpa_sitasi))


def _cek_sitasi(jawaban: str) -> tuple[int, int, list[str]]:
    """Hitung kalimat berisi klaim yang punya sitasi [n].
    Kalimat pendek (judul, pembuka seperti 'Berikut perbedaannya:') tidak dihitung."""
    teks = re.sub(r"\*\*|__|`", "", jawaban)
    kalimat = [k.strip(" -•*\t") for k in re.split(r"(?<=[.!?])\s+|\n+", teks)]
    klaim = [k for k in kalimat if len(k) >= 60]
    tanpa = [k for k in klaim if not re.search(r"\[\d+\]", k)]
    return len(klaim) - len(tanpa), len(klaim), tanpa


def jawab_langsung(state: ResearchState) -> dict:
    jawaban = panggil_llm(state["question"], peran="ringan", system=prompts.DIRECT_SYSTEM, max_tokens=500)
    return {"answer": jawaban, "citations": [], "from_collection": False,
            "trace": _jejak(state, "direct: dijawab tanpa koleksi")}


def tidak_ada(state: ResearchState) -> dict:
    return {"answer": prompts.TIDAK_ADA, "citations": [], "from_collection": False,
            "trace": _jejak(state, "tidak_ada: tidak ada chunk relevan")}
