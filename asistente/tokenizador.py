"""
Tokenizador propio a nivel de subpalabra (estilo WordPiece simplificado), construido desde el dataset.

1. Normalización: minúsculas, sin tildes (á->a, ñ se conserva como n), se quitan signos de puntuación.
2. Vocabulario: palabras frecuentes completas + piezas de subpalabra (prefijos de 2-4 letras y
   continuaciones "##xx") + todas las letras sueltas (así nunca falla).
3. Tokenización de una palabra: si está en el vocabulario se usa entera; si no, se parte con
   "coincidencia más larga primero" (greedy longest-match). Esto ayuda con errores de tipeo:
   "disponibilidda" (tipeo) se parte en piezas conocidas de "disponibilidad".
"""
import json
import re
import unicodedata
from collections import Counter

ESPECIALES = ["[PAD]", "[UNK]", "[CLS]"]
PAD, UNK, CLS = 0, 1, 2


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9%\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


class Tokenizador:
    def __init__(self, vocab=None, max_tokens=24):
        self.vocab = vocab or {t: i for i, t in enumerate(ESPECIALES)}
        self.inverso = {i: t for t, i in self.vocab.items()}
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    def construir(self, frases, min_freq_palabra=2, min_freq_pieza=3, max_vocab=3000):
        palabras = Counter(p for f in frases for p in normalizar(f).split())
        piezas = Counter()
        for p, n in palabras.items():
            for L in range(2, 5):
                if len(p) >= L:
                    piezas[p[:L]] += n
                for i in range(1, len(p) - L + 1):
                    piezas["##" + p[i:i + L]] += n
        letras = sorted({c for p in palabras for c in p})
        tokens = list(ESPECIALES)
        tokens += [w for w, n in palabras.most_common() if n >= min_freq_palabra]
        tokens += [pz for pz, n in piezas.most_common() if n >= min_freq_pieza and pz not in tokens]
        tokens += [c for c in letras] + ["##" + c for c in letras]
        visto, final = set(), []
        for t in tokens:
            if t not in visto:
                visto.add(t)
                final.append(t)
        final = final[:max_vocab] if len(final) > max_vocab else final
        for c in letras:          # garantizamos las letras aunque se recorte el vocabulario
            for t in (c, "##" + c):
                if t not in visto:
                    final.append(t)
        self.vocab = {t: i for i, t in enumerate(final)}
        self.inverso = {i: t for t, i in self.vocab.items()}
        return self

    def _partir_palabra(self, palabra):
        if palabra in self.vocab:
            return [palabra]
        piezas, i = [], 0
        while i < len(palabra):
            fin, encontrada = len(palabra), None
            while fin > i:
                cand = palabra[i:fin] if i == 0 else "##" + palabra[i:fin]
                if cand in self.vocab:
                    encontrada = cand
                    break
                fin -= 1
            if encontrada is None:          # carácter nunca visto: solo ese carácter es [UNK]
                piezas.append("[UNK]")
                i += 1
                continue
            piezas.append(encontrada)
            i = fin
        return piezas

    def tokenizar(self, texto):
        return [pz for p in normalizar(texto).split() for pz in self._partir_palabra(p)]

    def codificar(self, texto):
        """[CLS] + ids, recortado / rellenado con [PAD] hasta max_tokens + 1."""
        ids = [CLS] + [self.vocab.get(t, UNK) for t in self.tokenizar(texto)][: self.max_tokens]
        return ids + [PAD] * (self.max_tokens + 1 - len(ids))

    def __len__(self):
        return len(self.vocab)

    def guardar(self, ruta):
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "max_tokens": self.max_tokens}, f, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], d["max_tokens"])
