"""
Descarga y descomprime los datasets CNRPark-EXT y PKLot.

Estructura esperada después de ejecutar este script (dentro de rutas.datos):

    datos/
    └── crudos/
        ├── cnrpark_ext/
        │   ├── CNRPark+EXT.csv
        │   └── PATCHES/<SUNNY|OVERCAST|RAINY>/<fecha>/camera<N>/*.jpg
        └── pklot/
            └── PKLot/PKLot/<UFPR04|UFPR05|PUCPR>/<Sunny|Cloudy|Rainy>/<fecha>/*.jpg + *.xml

Uso:
    python scripts/descargar_datos.py                 # descarga todo
    python scripts/descargar_datos.py --solo cnrext   # solo CNRPark-EXT
    python scripts/descargar_datos.py --archivo-pklot D:/descargas/PKLot.tar.gz   # ya descargado a mano

Si un enlace oficial falla, ver la sección "Datasets" del README (espejos en Kaggle / Roboflow).
Basta con dejar las carpetas con la estructura de arriba; los scripts de preparación solo leen esas rutas.
"""
import argparse
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilidades.comun import cargar_config, ruta  # noqa: E402

URL_CNR = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/"
ARCHIVOS_CNR = ["CNRPark+EXT.csv", "CNR-EXT-Patches-150x150.zip"]
URL_PKLOT = "https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz"


def descargar(url: str, destino: Path, intentos: int = 200) -> None:
    """Descarga con reanudación (cabecera Range) y reintentos: la red puede ser inestable."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    for intento in range(1, intentos + 1):
        try:
            # tamaño total
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
                total = int(r.headers.get("Content-Length", 0))
            ya = destino.stat().st_size if destino.exists() else 0
            if total and ya >= total:
                print(f"[ok] {destino.name} ya está completo ({total/1e6:.1f} MB)")
                return
            req = urllib.request.Request(url, headers={"Range": f"bytes={ya}-"} if ya else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(destino, "ab" if ya else "wb") as f:
                print(f"Descargando {destino.name}: {ya/1e6:.0f}/{total/1e6:.0f} MB")
                while True:
                    bloque = r.read(1 << 20)
                    if not bloque:
                        break
                    f.write(bloque)
                    ya += len(bloque)
                    if ya % (200 << 20) < (1 << 20):
                        print(f"  ... {ya/1e6:.0f} MB")
            if not total or ya >= total:
                print(f"[ok] {destino.name}")
                return
        except Exception as e:  # red inestable: reintentar
            print(f"  intento {intento} falló: {e}")
            time.sleep(min(30, 2 * intento))
    raise RuntimeError(f"No se pudo descargar {url}. Descárgalo a mano (ver README).")


def extraer_zip(archivo: Path, destino: Path) -> None:
    print(f"Descomprimiendo {archivo.name} ...")
    with zipfile.ZipFile(archivo) as z:
        z.extractall(destino)


def extraer_pklot(archivo: Path, destino: Path) -> None:
    """Extrae solo las imágenes completas + XML (se omite PKLotSegmented: los recortes los
    generamos nosotros con la misma transformación de perspectiva que usa el pipeline)."""
    print(f"Descomprimiendo {archivo.name} (puede tardar varios minutos) ...")
    with tarfile.open(archivo, "r:gz") as t:
        n = 0
        for m in t:
            if "PKLotSegmented" in m.name or not m.isfile():
                continue
            t.extract(m, destino, filter="data") if sys.version_info >= (3, 12) else t.extract(m, destino)
            n += 1
            if n % 5000 == 0:
                print(f"  {n} archivos")
    print(f"[ok] {n} archivos extraídos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", choices=["cnrext", "pklot"], default=None)
    ap.add_argument("--archivo-pklot", default=None, help="ruta a PKLot.tar.gz descargado a mano")
    ap.add_argument("--archivo-cnr", default=None, help="ruta a CNR-EXT-Patches-150x150.zip descargado a mano")
    args = ap.parse_args()

    cfg = cargar_config()
    crudos = ruta(cfg, "datos") / "crudos"
    descargas = crudos / "descargas"

    if args.solo in (None, "cnrext"):
        dest = crudos / "cnrpark_ext"
        if (dest / "PATCHES").exists():
            print("[ok] CNRPark-EXT ya está descomprimido")
        else:
            for a in ARCHIVOS_CNR:
                if a.endswith(".zip") and args.archivo_cnr:
                    continue
                descargar(URL_CNR + a, descargas / a)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "CNRPark+EXT.csv").write_bytes((descargas / "CNRPark+EXT.csv").read_bytes())
            extraer_zip(Path(args.archivo_cnr) if args.archivo_cnr else descargas / ARCHIVOS_CNR[1], dest)

    if args.solo in (None, "pklot"):
        dest = crudos / "pklot"
        if list(dest.glob("**/UFPR04")):
            print("[ok] PKLot ya está descomprimido")
        else:
            arch = Path(args.archivo_pklot) if args.archivo_pklot else descargas / "PKLot.tar.gz"
            if not args.archivo_pklot:
                descargar(URL_PKLOT, arch)
            extraer_pklot(arch, dest)
    print("Listo. Siguiente paso: python scripts/preparar_cnn.py")


if __name__ == "__main__":
    main()
