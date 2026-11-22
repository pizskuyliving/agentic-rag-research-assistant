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

Jika kena batas kuota per menit (429), panggilan menunggu sesuai saran server lalu
mencoba lagi. Jika tetap gagal, litellm pindah ke cadangan (NVIDIA/Gemini/Mistral) bila diisi.
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
            "gemini": config.GEMINI_API_KEY, "mistral": config.MISTRAL_API_KEY}.get(awalan, "")


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
                        ("cadangan-gemini", config.MODEL_CADANGAN_GEMINI),
                        ("cadangan-mistral", config.MODEL_CADANGAN_MISTRAL)]:
        if model and model not in utama and _key_untuk(model):
            model_list.append({"model_name": nama,
                               "litellm_params": {"model": model, "api_key": _key_untuk(model)}})
            cadangan.append(nama)

    # Kuota dihitung per model, jadi model utama yang lain juga bisa jadi cadangan.
    fallbacks = [
        {"ringan": ["berat", *cadangan]},
        {"berat": [*cadangan, "ringan"]},
    ]
    # num_retries=0: penantian saat 429 ditangani sendiri di panggil_llm (lebih sabar)
    return Router(model_list=model_list, fallbacks=fallbacks, num_retries=0)


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


def _completion(peran: str, **kwargs):
    global _router
    if _router is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY belum diisi di file .env")
        _router = _bangun_router()

    for percobaan in range(MAKS_COBA_ULANG + 1):
        try:
            return _router.completion(model=peran, **kwargs)
        except Exception as e:
            tunggu = _detik_tunggu(e)
            harian = _kuota_harian(e)
            # Groq habis kuota harian, TAPI cadangan hanya sibuk sementara (mis. Gemini 503
            # "high demand"): tunggu sebentar lalu coba lagi, jangan langsung menyerah.
            sementara = any(t in str(e) for t in ("503", "UNAVAILABLE", "high demand", "overloaded"))
            if harian and sementara and percobaan < MAKS_COBA_ULANG:
                print("   (kuota Groq habis dan model cadangan sedang sibuk, menunggu 20 detik...)")
                time.sleep(20)
                continue
            # kuota harian habis: menunggu sebentar tidak membantu; atau sudah dicoba berkali-kali
            if harian and tunggu is not None and (tunggu > 70 or percobaan == MAKS_COBA_ULANG):
                raise KuotaHarianHabis(
                    "Kuota token HARIAN semua model sudah habis (model utama dan cadangan). "
                    f"Coba lagi sekitar {tunggu / 60:.0f} menit lagi, atau isi NVIDIA_API_KEY / "
                    "GEMINI_API_KEY di .env sebagai cadangan.") from None
            if tunggu is None or percobaan == MAKS_COBA_ULANG:
                raise
            jeda = min(tunggu, 65)
            print(f"   (kuota per menit penuh, menunggu {jeda:.0f} detik...)")
            time.sleep(jeda)


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

    resp = _completion(peran, messages=messages, temperature=temperature,
                       max_tokens=max(max_tokens, MIN_MAX_TOKENS), **extra)
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    # Tes cepat: python -m app.llm   (menampilkan model yang BENAR-BENAR menjawab)
    for peran, tanya in [("ringan", "Jawab satu kata: ibu kota Riau?"),
                         ("berat", "Jelaskan RAG dalam satu kalimat bahasa Indonesia.")]:
        r = _completion(peran, messages=[{"role": "user", "content": tanya}],
                        max_tokens=MIN_MAX_TOKENS, reasoning_effort=config.REASONING_EFFORT)
        print(f"{peran:6} [{r.model}]: {r.choices[0].message.content}")
