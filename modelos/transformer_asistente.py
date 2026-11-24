"""
Modelo 3: TransformerAsistente — Transformer encoder propio para clasificar la intención de una pregunta.

Flujo:  tokens -> Embedding + codificación posicional sinusoidal -> N capas encoder -> vector [CLS] -> Linear -> intención

Cada capa encoder:
    x = LayerNorm(x + Dropout(MultiHeadSelfAttention(x)))
    x = LayerNorm(x + Dropout(FeedForward(x)))

La atención multi-cabeza está escrita a mano:  Atención(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
"""
import math

import torch
import torch.nn as nn


class CodificacionPosicional(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))."""

    def __init__(self, d_modelo: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_modelo)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_modelo, 2).float() * (-math.log(10000.0) / d_modelo))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AtencionMultiCabeza(nn.Module):
    """Multi-head self-attention implementada a mano (sin nn.MultiheadAttention)."""

    def __init__(self, d_modelo: int, cabezas: int, dropout: float = 0.1):
        super().__init__()
        assert d_modelo % cabezas == 0, "d_modelo debe ser divisible entre el número de cabezas"
        self.h, self.d_k = cabezas, d_modelo // cabezas
        self.W_q = nn.Linear(d_modelo, d_modelo)
        self.W_k = nn.Linear(d_modelo, d_modelo)
        self.W_v = nn.Linear(d_modelo, d_modelo)
        self.W_o = nn.Linear(d_modelo, d_modelo)
        self.dropout = nn.Dropout(dropout)
        self.ultimos_pesos = None   # para visualizar la atención

    def _separar(self, x):
        n, t, _ = x.shape   # [N, T, d] -> [N, h, T, d_k]
        return x.view(n, t, self.h, self.d_k).transpose(1, 2)

    def forward(self, x, mascara_relleno=None):
        Q, K, V = self._separar(self.W_q(x)), self._separar(self.W_k(x)), self._separar(self.W_v(x))
        puntajes = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)          # [N, h, T, T]
        if mascara_relleno is not None:   # True donde hay relleno <PAD>: no se debe atender ahí
            puntajes = puntajes.masked_fill(mascara_relleno[:, None, None, :], float("-inf"))
        pesos = torch.softmax(puntajes, dim=-1)
        self.ultimos_pesos = pesos.detach()
        contexto = self.dropout(pesos) @ V                                 # [N, h, T, d_k]
        n, _, t, _ = contexto.shape
        contexto = contexto.transpose(1, 2).contiguous().view(n, t, self.h * self.d_k)
        return self.W_o(contexto)


class CapaEncoder(nn.Module):
    def __init__(self, d_modelo, cabezas, d_ff, dropout):
        super().__init__()
        self.atencion = AtencionMultiCabeza(d_modelo, cabezas, dropout)
        self.ff = nn.Sequential(nn.Linear(d_modelo, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_modelo))
        self.norm1, self.norm2 = nn.LayerNorm(d_modelo), nn.LayerNorm(d_modelo)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, mascara_relleno=None):
        x = self.norm1(x + self.drop1(self.atencion(x, mascara_relleno)))   # residual + LayerNorm
        x = self.norm2(x + self.drop2(self.ff(x)))
        return x


class TransformerAsistente(nn.Module):
    def __init__(self, tam_vocab, num_intenciones, d_modelo=64, cabezas=4, capas=2, d_ff=128,
                 dropout=0.1, max_tokens=32, id_pad=0):
        super().__init__()
        self.id_pad = id_pad
        self.embedding = nn.Embedding(tam_vocab, d_modelo, padding_idx=id_pad)
        self.posicional = CodificacionPosicional(d_modelo, max_tokens + 1)
        self.escala = math.sqrt(d_modelo)
        self.capas = nn.ModuleList([CapaEncoder(d_modelo, cabezas, d_ff, dropout) for _ in range(capas)])
        self.dropout = nn.Dropout(dropout)
        self.salida = nn.Linear(d_modelo, num_intenciones)

    def forward(self, ids):
        """ids: [N, T] con el token [CLS] en la posición 0. Devuelve logits [N, num_intenciones]."""
        mascara = ids == self.id_pad
        x = self.dropout(self.posicional(self.embedding(ids) * self.escala))
        for capa in self.capas:
            x = capa(x, mascara)
        return self.salida(x[:, 0])        # pooling con el token [CLS]
