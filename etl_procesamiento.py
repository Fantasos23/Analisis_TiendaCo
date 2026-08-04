import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Directorios de Entrada y Salida
DATA_DIR = Path('data')
OUTPUT_DIR = Path('datasets_procesados')

# Crear la carpeta de salida si no existe
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def limpiar_y_estandarizar_df(df):
    """
    Aplica reglas de limpieza estructural estricta:
    - Elimina filas vacías.
    - Limpia espacios en blanco en columnas y textos.
    - Convierte booleanos a enteros/texto limpio.
    """
    if df.empty:
        return df

    # 1. Eliminar filas completamente vacías
    df = df.dropna(how='all')

    # 2. Limpiar espacios en blanco en los nombres de las columnas
    df.columns = [str(col).strip() for col in df.columns]

    # 3. Limpiar celdas de texto (quitar espacios sobrantes al inicio y final)
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()

    # 4. Convertir columnas booleanas (True/False) a enteros (1/0) o cadenas estandarizadas
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    # 5. Reemplazar valores nulos representados como 'nan', 'null', 'none' por vacíos limpios o NaN oficial
    df = df.replace(to_replace=['nan', 'NaN', 'null', 'NULL', 'None', 'none'], value=np.nan)

    return df


def unificar_carpeta(nombre_fuente, separador_default=';', id_duplicados=None):
    """
    Busca todos los archivos en la carpeta correspondiente, los une y aplica limpieza estructural.
    """
    path_carpeta = DATA_DIR / nombre_fuente
    archivos = list(path_carpeta.glob('*.csv')) + list(path_carpeta.glob('*.xlsx'))

    print("\n" + "="*50)
    print(f"PROCESANDO Y UNIFICANDO ARCHIVOS DE: {nombre_fuente.upper()}")
    print("="*50)

    if not archivos:
        print(f"⚠️ No se encontraron archivos en: {path_carpeta}")
        return

    dfs = []
    print(f"📂 Encontrados {len(archivos)} archivos:")

    for archivo in archivos:
        print(f"  └─ Leyendo: {archivo.name}")
        try:
            if archivo.suffix == '.csv':
                try:
                    df = pd.read_csv(archivo, sep=separador_default, encoding='utf-8', low_memory=False)
                except Exception:
                    df = pd.read_csv(archivo, sep=',', encoding='utf-8-sig', low_memory=False)
            else:
                df = pd.read_excel(archivo)

            dfs.append(df)
        except Exception as e:
            print(f"  ❌ Error leyendo {archivo.name}: {e}")

    if not dfs:
        print("⚠️ No se pudieron procesar datos.")
        return

    # Unificación masiva
    df_unificado = pd.concat(dfs, ignore_index=True)
    filas_totales = len(df_unificado)

    # Limpieza estructural
    df_unificado = limpiar_y_estandarizar_df(df_unificado)

    # Deduplicación
    if id_duplicados and id_duplicados in df_unificado.columns:
        df_unificado = df_unificado.drop_duplicates(subset=[id_duplicados], keep='first')
    else:
        df_unificado = df_unificado.drop_duplicates(keep='first')

    filas_limpias = len(df_unificado)
    duplicados_eliminados = filas_totales - filas_limpias

    # Exportar archivo consolidado sin alterar la lógica de negocio
    ruta_salida = OUTPUT_DIR / f'dataset_{nombre_fuente.lower()}_unificado.csv'
    df_unificado.to_csv(ruta_salida, index=False, encoding='utf-8-sig')

    print(f"\n✅ Finalizado con éxito:")
    print(f"  • Filas totales consolidadas: {filas_totales}")
    print(f"  • Filas duplicadas omitidas: {duplicados_eliminados}")
    print(f"  • Total filas limpias guardadas: {filas_limpias}")
    print(f"📁 Guardado en: {ruta_salida}")


# ==========================================================
# EJECUCIÓN DEL PROCESO
# ==========================================================
if __name__ == '__main__':
    print("🚀 INICIANDO UNIFICACIÓN Y LIMPIEZA ESTRUCTURAL DE ARCHIVOS")

    # 1. VTEX (Ventas): Deduplica por la columna 'Order' si existe
    unificar_carpeta(nombre_fuente='VTEX', separador_default=';', id_duplicados='Order')

    # 2. META: Deduplica por filas idénticas
    unificar_carpeta(nombre_fuente='Meta', separador_default=',')

    # 3. GOOGLE: Deduplica por filas idénticas
    unificar_carpeta(nombre_fuente='Google', separador_default=',')

    print("\n" + "="*50)
    print("🎉 ¡TODOS LOS DATASETS FUERON UNIFICADOS Y LIMPIADOS EXITOSAMENTE!")
    print("="*50)