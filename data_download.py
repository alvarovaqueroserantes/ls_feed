# -*- coding: utf-8 -*-
"""
Descarga CSVs de una carpeta de Google Drive y construye DataFrames por subcarpeta LS1..LS6.
- Requiere: gdown, pandas (y opcionalmente pyarrow para mayor velocidad).
- Crea variables globales df_ls1..df_ls6 si existen esas subcarpetas.
- Expone también dfs_por_ls: Dict[str, pd.DataFrame] con llaves 'ls1'..'ls6'.
"""

from __future__ import annotations
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# -------- Configuración de logging --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# -------- Motor de lectura: pyarrow si disponible --------
_ENGINE: str = "python"
try:
    import pyarrow  # noqa: F401
    _ENGINE = "pyarrow"  # pandas>=2.0 soporta motor pyarrow
except Exception:
    # caerá a "python" si no hay pyarrow; también vale "c" pero "python" permite sep=None
    _ENGINE = "python"


def descargar_carpeta_drive(folder_url: str, salida: Path) -> Path:
    """
    Descarga recursivamente una carpeta de Google Drive con gdown.
    """
    import gdown  # import local para no forzar dependencia si no se usa
    salida.mkdir(parents=True, exist_ok=True)
    logging.info("Descargando carpeta de Drive a: %s", salida.resolve())
    # remaining_ok=True permite continuar si algunos ficheros ya existen
    gdown.download_folder(
        url=folder_url,
        output=str(salida),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    logging.info("Descarga finalizada.")
    return salida


def listar_dirs_ls(raiz: Path) -> List[Path]:
    """
    Devuelve las subcarpetas inmediatas bajo 'raiz' cuyo nombre comienza por LS1..LS6.
    Acepta nombres como 'LS1', 'LS1_xxx', etc.
    """
    patron = re.compile(r"^LS([1-6])(\b|[_\-].*)?$", flags=re.IGNORECASE)
    dirs = []
    for p in raiz.iterdir():
        if p.is_dir() and patron.match(p.name):
            dirs.append(p)
    dirs_ordenados = sorted(dirs, key=lambda x: x.name.lower())
    logging.info("Subcarpetas LS detectadas: %s", [d.name for d in dirs_ordenados])
    return dirs_ordenados


def leer_csv_con_robustez(path_csv: Path) -> pd.DataFrame:
    """
    Intenta leer un CSV probando separadores comunes.
    Añade columna 'source_file' con la ruta del archivo para trazabilidad.
    """
    posibles_sep = [",", ";", "\t", "|"]
    # 1) Intento rápido con engine disponible
    for sep in posibles_sep:
        try:
            df = pd.read_csv(path_csv, sep=sep, engine=_ENGINE)
            df["source_file"] = str(path_csv)
            return df
        except Exception:
            pass
    # 2) Fallback final: intento con engine='python' por si hay casos raros
    for sep in posibles_sep:
        try:
            df = pd.read_csv(path_csv, sep=sep, engine="python")
            df["source_file"] = str(path_csv)
            return df
        except Exception:
            pass
    raise RuntimeError(f"No se pudo leer el CSV: {path_csv}")


def concat_csvs_de_directorio(dir_ls: Path) -> Optional[pd.DataFrame]:
    """
    Busca recursivamente *.csv bajo 'dir_ls' y concatena en un único DataFrame.
    Devuelve None si no hay CSVs.
    """
    csvs = sorted(dir_ls.rglob("*.csv"))
    if not csvs:
        logging.warning("No se encontraron CSVs en %s", dir_ls)
        return None

    frames: List[pd.DataFrame] = []
    for csv_path in csvs:
        try:
            frames.append(leer_csv_con_robustez(csv_path))
        except Exception as e:
            logging.error("Error leyendo %s: %s", csv_path, e)
            continue

    if not frames:
        logging.warning("No se pudo leer ningún CSV válido en %s", dir_ls)
        return None

    df = pd.concat(frames, ignore_index=True, copy=False)
    logging.info("Concatenado %d CSVs en %s -> %d filas, %d columnas",
                 len(frames), dir_ls.name, df.shape[0], df.shape[1])
    return df


def construir_dataframes_por_ls(raiz_descarga: Path) -> Dict[str, pd.DataFrame]:
    """
    Para cada subcarpeta LSx encontrada, concatena sus CSVs y
    devuelve un dict {'ls1': df, 'ls2': df, ...} con las que existan.
    """
    resultado: Dict[str, pd.DataFrame] = {}
    for dir_ls in listar_dirs_ls(raiz_descarga):
        m = re.match(r"^LS([1-6])", dir_ls.name, flags=re.IGNORECASE)
        if not m:
            continue
        idx = m.group(1)
        df = concat_csvs_de_directorio(dir_ls)
        if df is not None:
            clave = f"ls{idx}"
            resultado[clave] = df
    return resultado


def exponer_variables_globales(dfs_por_ls: Dict[str, pd.DataFrame]) -> None:
    """
    Crea variables globales df_ls1..df_ls6 según existan en dfs_por_ls.
    También imprime un resumen.
    """
    for clave, df in dfs_por_ls.items():
        var_name = f"df_{clave}"
        globals()[var_name] = df
        logging.info("Variable creada: %s  (shape=%s)", var_name, df.shape)

    if dfs_por_ls:
        resumen = {k: v.shape for k, v in dfs_por_ls.items()}
        print("\nResumen de DataFrames creados (filas, columnas):")
        for k, shape in resumen.items():
            print(f"  df_{k}: {shape[0]} filas, {shape[1]} columnas")
    else:
        print("No se crearon DataFrames (¿faltan CSVs o subcarpetas LS?).")


def main():
    # === EDITA AQUÍ CON TU URL DE DRIVE ===
    FOLDER_URL = "https://drive.google.com/drive/folders/1uUHk-56DhssZo0yGxdrfa7lTOkcv8yXA?usp=drive_link"
    SALIDA_LOCAL = Path("data")  # carpeta local donde se descargará

    # 1) Descargar la carpeta
    raiz_local = descargar_carpeta_drive(FOLDER_URL, SALIDA_LOCAL)

    # 2) Buscar subcarpetas LS* directamente bajo la raíz descargada.
    # Nota: gdown preserva estructura; a veces crea un nivel con el nombre de la carpeta principal.
    # Si detectas que la raíz real está en data_drive/<NombreCarpeta>, ajusta aquí:
    # raiz_local = next(raiz_local.iterdir()) if raiz_local.is_dir() else raiz_local

    # 3) Construir los DataFrames por LS
    dfs_por_ls = construir_dataframes_por_ls(raiz_local)

    # 4) Exponer variables globales df_ls1..df_ls6 y mostrar resumen
    exponer_variables_globales(dfs_por_ls)

    # 5) Si quieres persistir, descomenta para guardar en Parquet:
    # for clave, df in dfs_por_ls.items():
    #     outp = raiz_local / f"{clave}.parquet"
    #     df.to_parquet(outp, index=False)
    #     logging.info("Guardado %s", outp)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fallo en la ejecución: %s", e)
        sys.exit(1)
