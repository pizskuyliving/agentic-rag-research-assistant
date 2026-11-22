"""UI Streamlit: unggah jurnal PDF, bertanya, lihat jawaban bersitasi sampai halaman.

Jalankan (Qdrant harus menyala: docker compose up -d):
    streamlit run streamlit_app.py

UI ini hanya "kulit": semua logika RAG tetap di app/ (agen LangGraph yang sama dengan eval).
"""
import os
import re
from pathlib import Path

import streamlit as st

from app.agent.graph import bangun_agen
from app.ingestion.indexer import daftar_jurnal, hapus_jurnal
from app.ingestion.pdf_parser import PDFTidakTerbaca
from app.ingestion.pipeline import proses_pdf
from app.llm import KuotaHarianHabis

FOLDER_PDF = Path("data/papers")
MAKS_UKURAN_MB = 20
CONTOH_PERTANYAAN = [
    "Apa perbedaan Self-RAG dan CRAG dalam menangani dokumen yang tidak relevan?",
    "Apa saja paradigma RAG menurut survei Gao dkk.?",
    "Jurnal mana saja yang memakai BM25?",
]
# Label ramah untuk setiap langkah agen (diambil dari awal baris jejak)
LABEL_LANGKAH = {
    "router": "Memahami pertanyaan",
    "retrieve": "Mencari potongan jurnal",
    "grade": "Menilai relevansi potongan",
    "evaluasi": "Memeriksa apakah sumber sudah cukup",
    "perluas": "Menambah konteks di sekitar sumber",
    "cek sitasi": "Memeriksa sitasi",
    "generate": "Menyusun jawaban",
    "tidak_ada": "Tidak menemukan sumber yang relevan",
    "jawab_langsung": "Menjawab langsung",
}

st.set_page_config(page_title="Asisten Riset & Skripsi", page_icon="📚", layout="wide")


# ------------------------------------------------------------------ helper
def _label(jejak: str) -> str:
    for awalan, label in LABEL_LANGKAH.items():
        if jejak.startswith(awalan):
            return label + "..."
    return "Memproses..."


_POLA_RUMUS = re.compile(r"\$([^$\n]{1,120}?)\$")


def _aman_markdown(teks: str) -> str:
    """Tanda $ di Streamlit berarti rumus LaTeX.

    - Rumus dari paper seperti $N = 6$ atau $\\sqrt{d_k}$ tetap ditampilkan sebagai rumus.
    - Tanda $ lain (harga "$15") di-escape supaya tidak merusak tampilan.
    """
    rumus: list[str] = []

    def simpan(m: re.Match) -> str:
        isi = m.group(1)
        if isi.strip() == isi and re.search(r"[A-Za-z\\^_={}]", isi):
            rumus.append(m.group(0))
            return f"\x00{len(rumus) - 1}\x00"
        return m.group(0)

    teks = _POLA_RUMUS.sub(simpan, teks or "").replace("$", r"\$")
    return re.sub(r"\x00(\d+)\x00", lambda m: rumus[int(m.group(1))], teks)


def _cuplikan(teks: str, maks: int = 700) -> str:
    teks = " ".join((teks or "").replace("[TABEL]", "Tabel:").split())
    return teks if len(teks) <= maks else teks[:maks] + " …"


def _tampilkan_sumber(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("**Sumber**")
    for s in citations:
        judul = f"[{s['nomor']}] {s['penulis']} ({s.get('tahun') or 't.t.'}), hal. {s['halaman']}"
        with st.expander(judul):
            st.caption(s.get("judul", ""))
            st.markdown(f"> {_aman_markdown(_cuplikan(s.get('teks', '')))}")


def _tampilkan_pesan(pesan: dict, tampil_jejak: bool) -> None:
    st.markdown(_aman_markdown(pesan["content"]))
    if pesan.get("citations"):
        _tampilkan_sumber(pesan["citations"])
    elif pesan["role"] == "assistant" and pesan.get("from_collection") is False:
        st.caption("Jawaban ini tidak berasal dari koleksi jurnalmu.")
    if tampil_jejak and pesan.get("trace"):
        with st.expander("Jejak agen"):
            for langkah in pesan["trace"]:
                st.text(f"› {langkah}")


@st.cache_data(ttl=60, show_spinner=False)
def _koleksi(user_id: str, versi: int) -> list[dict]:
    # `versi` hanya untuk membatalkan cache setelah unggah/hapus
    return daftar_jurnal(user_id)


def _jalankan_agen(pertanyaan: str, user_id: str) -> dict:
    """Jalankan agen sambil menampilkan langkah yang sedang dikerjakan."""
    awal = {"question": pertanyaan, "user_id": user_id, "trace": [], "putaran": 0}
    akhir: dict = awal
    with st.status("Memahami pertanyaan...", expanded=False) as status:
        dilihat = 0
        for nilai in bangun_agen().stream(awal, stream_mode="values"):
            akhir = nilai
            jejak = nilai.get("trace", [])
            for langkah in jejak[dilihat:]:
                status.update(label=_label(langkah))
                st.text(f"› {langkah}")
            dilihat = len(jejak)
        status.update(label="Selesai", state="complete")
    return akhir


# ------------------------------------------------------------------ state
st.session_state.setdefault("pesan", [])
st.session_state.setdefault("versi_koleksi", 0)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📚 Koleksi jurnal")
    user_id = st.text_input("Nama koleksi", value=os.getenv("DEFAULT_USER", "muhammad"),
                            help="Setiap nama punya koleksi jurnal sendiri.").strip() or "tamu"

    unggahan = st.file_uploader("Unggah jurnal (PDF)", type=["pdf"], accept_multiple_files=True)
    if unggahan and st.button("Proses PDF", type="primary", use_container_width=True):
        FOLDER_PDF.mkdir(parents=True, exist_ok=True)
        for berkas in unggahan:
            if berkas.size > MAKS_UKURAN_MB * 1024 * 1024:
                st.error(f"{berkas.name}: lebih dari {MAKS_UKURAN_MB} MB, dilewati.")
                continue
            path = FOLDER_PDF / Path(berkas.name).name
            path.write_bytes(berkas.getbuffer())
            with st.spinner(f"Memproses {berkas.name}..."):
                try:
                    hasil = proses_pdf(path, user_id)
                except PDFTidakTerbaca as e:
                    st.error(f"{berkas.name}: {e}")
                    continue
                except KuotaHarianHabis as e:
                    st.error(str(e))
                    break
            st.success(f"{hasil['judul'][:60]} — {hasil['chunk']} potongan")
            if hasil.get("perlu_dicek"):
                st.warning(f"Periksa metadata: {', '.join(hasil['perlu_dicek'])} mungkin kurang tepat.")
        st.session_state.versi_koleksi += 1

    try:
        koleksi = _koleksi(user_id, st.session_state.versi_koleksi)
    except Exception as e:  # Qdrant mati, dsb.
        koleksi = None
        st.error("Tidak bisa terhubung ke database. Jalankan `docker compose up -d` lalu muat ulang.")
        st.caption(str(e)[:200])

    if koleksi is not None:
        st.caption(f"{len(koleksi)} jurnal")
        for j in koleksi:
            with st.expander(f"{j['penulis']} ({j['tahun'] or 't.t.'})"):
                st.markdown(f"**{j['judul']}**")
                st.caption(f"{j['halaman']} halaman · {j['chunk']} potongan · {j['nama_file']}")
                if st.button("Hapus dari koleksi", key=f"hapus-{j['paper_id']}"):
                    hapus_jurnal(user_id, j["paper_id"])
                    st.session_state.versi_koleksi += 1
                    st.rerun()

    st.divider()
    tampil_jejak = st.toggle("Tampilkan jejak agen", value=False)
    if st.button("Percakapan baru", use_container_width=True):
        st.session_state.pesan = []
        st.rerun()

# ------------------------------------------------------------------ halaman utama
st.title("Asisten Riset & Skripsi")
st.caption("Jawaban hanya dari jurnal di koleksimu, dengan sitasi sampai nomor halaman.")

if koleksi == []:
    st.info("Koleksimu masih kosong. Unggah jurnal PDF lewat panel kiri untuk mulai.")

for pesan in st.session_state.pesan:
    with st.chat_message(pesan["role"]):
        _tampilkan_pesan(pesan, tampil_jejak)

pertanyaan = st.chat_input("Tanyakan sesuatu tentang jurnalmu...")
if not st.session_state.pesan and not pertanyaan and koleksi:
    st.markdown("**Contoh pertanyaan**")
    for i, contoh in enumerate(CONTOH_PERTANYAAN):
        if st.button(contoh, key=f"contoh-{i}"):
            pertanyaan = contoh

if pertanyaan:
    # Pesan pengguna baru disimpan bersama jawabannya (di akhir). Jika proses terputus di tengah
    # (mis. tombol lain diklik saat agen berjalan), pertanyaan tidak tertinggal tanpa jawaban
    # dan tidak muncul dua kali saat ditanyakan ulang.
    with st.chat_message("user"):
        st.markdown(_aman_markdown(pertanyaan))

    with st.chat_message("assistant"):
        try:
            hasil = _jalankan_agen(pertanyaan, user_id)
            pesan = {"role": "assistant", "content": hasil.get("answer", ""),
                     "citations": hasil.get("citations", []), "trace": hasil.get("trace", []),
                     "from_collection": hasil.get("from_collection")}
        except KuotaHarianHabis as e:
            pesan = {"role": "assistant", "content": f"⚠️ {e}"}
        except Exception as e:
            pesan = {"role": "assistant", "content": f"⚠️ Terjadi kesalahan: {e}"}
        _tampilkan_pesan(pesan, tampil_jejak)
    st.session_state.pesan += [{"role": "user", "content": pertanyaan}, pesan]
