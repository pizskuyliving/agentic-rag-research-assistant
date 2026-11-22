"""Satu-satunya pintu ke LLM.

Semua node agen (router, grade, generate, ...) memanggil `panggil_llm`.
Kalau nanti ingin pindah penyedia (misalnya ke Claude API), cukup ubah .env.

Peran:
- "ringan": tugas pendek & sering (router, grade, query) -> openai/gpt-oss-20b
- "berat" : menyusun jawaban & self-check            -> openai/gpt-oss-120b

Catatan model gpt-oss: ini model *reasoning*. Ia "berpikir" dulu, dan token berpikir
ikut dihitung dalam max_tokens. Karena itu:
- reasoning_effort diset "low" (cukup untuk tugas klasifikasi & sitasi, hemat token)
- max_tokens tidak boleh terlalu kecil, kalau tidak jawaban bisa kosong

Urutan model dicoba satu per satu (model utama dulu, lalu cadangan). Untuk SETIAP model:
- batas per menit (429) atau error server sementara -> TUNGGU lalu coba model yang sama lagi;
- kuota harian habis -> model itu ditandai habis (tidak dicoba lagi) lalu pindah ke berikutnya.
Dulu pindah ke cadangan terjadi seketika saat 429: kuota harian Gemini (500 permintaan)
habis tersedot hanya karena NVIDIA sesekali kena batas 40 permintaan/menit.
"""
import re
import time

import litellm
from litellm import Router

from app import config

# Jika model cadangan tidak mendukung suatu parameter (mis. reasoning_effort),
# parameter itu dibuang otomatis alih-alih membuat panggilan gagal.
litellm.drop_params = True

# Batas bawah max_tokens: ruang untuk "berpikir" + jawaban
MIN_MAX_TOKENS = 600
MAKS_COBA_ULANG = 3


def _key_untuk(model: str) -> str:
    """API key sesuai penyedia di awalan nama model (groq/..., nvidia_nim/..., gemini/...)."""
    awalan = model.split("/", 1)[0]
    return {"groq": config.GROQ_API_KEY, "nvidia_nim": config.NVIDIA_API_KEY,
            "gemini": config.GEMINI_API_KEY, "mistral": config.MISTRAL_API_KEY,
            "cloudflare": config.CLOUDFLARE_API_KEY,
            "fireworks_ai": config.FIREWORKS_AI_API_KEY}.get(awalan, "")


def _bangun_router() -> Router:
    """Model utama dari MODEL_RINGAN / MODEL_BERAT (penyedia mana pun), lalu cadangan
    berurutan: NVIDIA (model sama gpt-oss-120b), Groq, Gemini, Mistral, yang key-nya diisi.
    Cadangan yang sama persis dengan model utama dilewati."""
    model_list = [
        {"model_name": "ringan", "litellm_params": {"model": config.MODEL_RINGAN,
                                                    "api_key": _key_untuk(config.MODEL_RINGAN)}},
        {"model_name": "berat", "litellm_params": {"model": config.MODEL_BERAT,
                                                   "api_key": _key_untuk(config.MODEL_BERAT)}},
    ]
    utama = {config.MODEL_RINGAN, config.MODEL_BERAT}
    cadangan = []
    for nama, model in [("cadangan-nvidia", config.MODEL_CADANGAN_NVIDIA),
                        ("cadangan-groq", config.MODEL_CADANGAN_GROQ),
                        ("cadangan-cloudflare", config.MODEL_CADANGAN_CLOUDFLARE),
                        ("cadangan-fireworks", config.MODEL_CADANGAN_FIREWORKS),
                        ("cadangan-gemini", config.MODEL_CADANGAN_GEMINI),
                        ("cadangan-mistral", config.MODEL_CADANGAN_MISTRAL)]:
        if model and model not in utama and _key_untuk(model):
            model_list.append({"model_name": nama,
                               "litellm_params": {"model": model, "api_key": _key_untuk(model)}})
            cadangan.append(nama)

    global _cadangan
    _cadangan = cadangan
    # Tanpa fallbacks/num_retries bawaan: urutan & penantian diatur sendiri di _completion
    return Router(model_list=model_list, num_retries=0)


_cadangan: list[str] = []
_habis: set[str] = set()   # model yang kuota hariannya habis selama proses ini berjalan
_istirahat: dict[str, float] = {}   # model -> waktu (time.time) sampai kapan dilewati sementara
ISTIRAHAT_DETIK = 180              # model yang gagal terus-menerus dilewati selama 3 menit
_router = None


class KuotaHarianHabis(RuntimeError):
    """Kuota token HARIAN penyedia LLM habis. Menunggu beberapa detik tidak akan membantu."""


def _detik_tunggu(err: Exception) -> float | None:
    """Ambil saran tunggu dari pesan 429, mis. 'Please try again in 12.945s'.
    Pesan error bisa memuat beberapa error sekaligus (model utama + cadangan);
    yang dipakai adalah saran tunggu TERPENDEK."""
    teks = str(err)
    tanda = ("rate_limit", "Rate limit", "RateLimit", "429")
    if not any(t in teks for t in tanda) and "RateLimit" not in type(err).__name__:
        return None
    saran = [int(m or 0) * 60 + float(s) + 1.0
             for m, s in re.findall(r"try again in (?:(\d+)m)?([\d.]+)s", teks)]
    return min(saran) if saran else 15.0


def _kuota_harian(err: Exception) -> bool:
    teks = str(err)
    return "tokens per day" in teks or "(TPD)" in teks or "requests per day" in teks or "(RPD)" in teks


_SEMENTARA = ("Error code: 410", "Error code: 500", "Error code: 502", "Error code: 504", "503",
              "UNAVAILABLE", "high demand", "overloaded", "Timeout", "timed out")


def _rantai(peran: str) -> list[str]:
    """Urutan model yang dicoba untuk sebuah peran (tanpa duplikat model yang sama)."""
    urutan = ["ringan", "berat", *_cadangan] if peran == "ringan" else ["berat", *_cadangan, "ringan"]
    hasil, model_terpakai = [], set()
    for nama in urutan:
        model = {"ringan": config.MODEL_RINGAN, "berat": config.MODEL_BERAT}.get(nama, nama)
        if model not in model_terpakai:
            model_terpakai.add(model)
            hasil.append(nama)
    return hasil


def _completion(peran: str, **kwargs):
    global _router
    if _router is None:
        _router = _bangun_router()

    galat_terakhir, ada_harian = None, False
    for nama in _rantai(peran):
        if nama in _habis:
            ada_harian = True
            continue
        if _istirahat.get(nama, 0) > time.time():
            continue
        for percobaan in range(MAKS_COBA_ULANG + 1):
            try:
                return _router.completion(model=nama, **kwargs)
            except Exception as e:
                galat_terakhir = e
                if _kuota_harian(e):
                    _habis.add(nama)
                    ada_harian = True
                    print(f"   ({nama}: kuota harian habis, pindah ke model berikutnya)")
                    break
                tunggu = _detik_tunggu(e)
                sementara = any(t in str(e) for t in _SEMENTARA)
                if percobaan == MAKS_COBA_ULANG or (tunggu is None and not sementara):
                    # Jangan buang waktu mencoba model ini di setiap panggilan berikutnya
                    _istirahat[nama] = time.time() + ISTIRAHAT_DETIK
                    pesan = " ".join(str(e).split())[:160]
                    print(f"   ({nama} gagal, dilewati {ISTIRAHAT_DETIK // 60} menit: {pesan})")
                    break
                jeda = min(tunggu, 65) if tunggu is not None else 15
                print(f"   ({nama}: batas per menit / server sibuk, menunggu {jeda:.0f} detik...)")
                time.sleep(jeda)

    if galat_terakhir is None and _istirahat:   # semua model yang tersisa sedang "istirahat"
        _istirahat.clear()
        return _completion(peran, **kwargs)
    if ada_harian:
        raise KuotaHarianHabis(
            "Kuota HARIAN model utama dan semua cadangan sudah habis. Coba lagi besok, "
            "atau isi API key cadangan lain di .env (NVIDIA_API_KEY / GEMINI_API_KEY / GROQ_API_KEY).")
    raise galat_terakhir


def panggil_llm(prompt: str, peran: str = "berat", system: str | None = None,
                temperature: float = 0.0, max_tokens: int = 1024, json_mode: bool = False) -> str:
    """Kirim prompt ke LLM sesuai peran ("ringan" / "berat"), kembalikan teks jawabannya.

    json_mode=True memaksa model membalas JSON valid (prompt harus menyebut kata "JSON").
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    extra = {"reasoning_effort": config.REASONING_EFFORT}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}

    batas = max(max_tokens, MIN_MAX_TOKENS)
    resp = _completion(peran, messages=messages, temperature=temperature, max_tokens=batas, **extra)
    teks = resp.choices[0].message.content or ""
    if not teks.strip():
        # Balasan kosong: biasanya model reasoning menghabiskan max_tokens untuk "berpikir"
        # (penyedia seperti NVIDIA/Cloudflare mengabaikan reasoning_effort=low). Coba sekali
        # lagi dengan ruang 3x lebih besar. Tes: Grade kosong -> semua kandidat lolos tanpa disaring.
        resp = _completion(peran, messages=messages, temperature=temperature, max_tokens=batas * 3, **extra)
        teks = resp.choices[0].message.content or ""
    return teks


if __name__ == "__main__":
    # Tes cepat: python -m app.llm   (menampilkan model yang BENAR-BENAR menjawab)
    for peran, tanya in [("ringan", "Jawab satu kata: ibu kota Riau?"),
                         ("berat", "Jelaskan RAG dalam satu kalimat bahasa Indonesia.")]:
        r = _completion(peran, messages=[{"role": "user", "content": tanya}],
                        max_tokens=MIN_MAX_TOKENS, reasoning_effort=config.REASONING_EFFORT)
        print(f"{peran:6} [{r.model}]: {r.choices[0].message.content}")
