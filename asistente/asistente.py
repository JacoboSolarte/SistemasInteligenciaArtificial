"""Asistente conversacional: Transformer propio (intención) + plantillas llenadas con el estado del análisis."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asistente.respuestas import no_entendi, responder  # noqa: E402
from asistente.tokenizador import Tokenizador, normalizar  # noqa: E402
from modelos.transformer_asistente import TransformerAsistente  # noqa: E402


class Asistente:
    def __init__(self, ruta_pesos, umbral=0.5, dispositivo="cpu"):
        ck = torch.load(ruta_pesos, map_location=dispositivo, weights_only=False)
        self.intenciones = ck["intenciones"]
        self.tok = Tokenizador(ck["vocab"], ck["config"]["max_tokens"])
        c = ck["config"]
        self.modelo = TransformerAsistente(len(self.tok), len(self.intenciones), c["d_modelo"], c["cabezas"], c["capas"],
                                           c["d_ff"], 0.0, c["max_tokens"]).to(dispositivo)
        self.modelo.load_state_dict(ck["estado"])
        self.modelo.eval()
        self.umbral, self.dispositivo = umbral, dispositivo

    @torch.no_grad()
    def clasificar(self, texto):
        ids = torch.tensor([self.tok.codificar(texto)], device=self.dispositivo)
        prob = torch.softmax(self.modelo(ids), 1)[0]
        conf, i = prob.max(0)
        top = sorted(zip(self.intenciones, prob.tolist()), key=lambda t: -t[1])[:3]
        return self.intenciones[int(i)], float(conf), top

    def reconoce_alguna_palabra(self, texto):
        """Regla de cobertura: si NINGUNA palabra de la frase existe completa en el vocabulario
        (todo se partió en subpalabras o [UNK]), el texto es probablemente ruido -> no se responde a ciegas."""
        palabras = normalizar(texto).split()
        return any(p in self.tok.vocab for p in palabras)

    def responder(self, texto, estado=None):
        """Devuelve (respuesta, intención, confianza_intención)."""
        intencion, conf, _ = self.clasificar(texto)
        if not self.reconoce_alguna_palabra(texto):
            return no_entendi(), "fuera_de_tema", conf
        if conf < self.umbral:
            return no_entendi(), intencion, conf
        return responder(intencion, estado), intencion, conf
