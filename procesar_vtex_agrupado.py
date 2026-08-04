from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Rutas de entrada y salida
DATASETS_DIR = Path('datasets_procesados')
ENTRADA_VTEX = DATASETS_DIR / 'dataset_vtex_unificado.csv'
SALIDA_VTEX_AGRUPADO = DATASETS_DIR / 'dataset_vtex_agrupado_ordenes.csv'


def generar_dataset_vtex_por_orden():
    print("\n" + "="*60)
    print("PROCESANDO VTEX: AGRUPACIÓN A NIVEL DE ORDEN ÚNICA")
    print("="*60)

    if not ENTRADA_VTEX.exists():
        print(f"❌ Error: No se encontró el archivo {ENTRADA_VTEX}.")
        print("   Asegúrate de haber ejecutado primero el ETL de unificación.")
        return

    print(f"📂 Leyendo: {ENTRADA_VTEX.name}...")
    df = pd.read_csv(ENTRADA_VTEX, low_memory=False)

    # 1. Mapeo de columnas requeridas (soporta variaciones comunes de nombre)
    col_mapping = {
        'Order': 'Order',
        'Creation Date': 'Creation Date',
        'Client Document': 'Client Document',
        'City': 'City',
        'Status': 'Status',
        'UtmMedium': 'UtmMedium',
        'UtmCampaign': 'UtmCampaign',
        'UtmSource': 'UtmSource',
        'Coupon': 'Coupon',
        'Payment System Name': 'Payment System Name',
        'Quantity_SKU': 'Quantity_SKU',
        'SKU Name': 'SKU Name',
        'Total Value': 'Total Value',
        'Discounts Names': 'Discounts Names'
    }

    # Renombrar columnas si existen para estandarizar
    df = df.rename(columns={c: col_mapping[c] for c in df.columns if c in col_mapping})

    # Verificar cuáles columnas de las solicitadas existen en el dataset
    columnas_deseadas = [
        'Order', 'Creation Date', 'Client Document', 'City', 'Status',
        'UtmMedium', 'UtmCampaign', 'UtmSource', 'Coupon',
        'Payment System Name', 'Quantity_SKU', 'SKU Name', 'Total Value', 'Discounts Names'
    ]

    cols_existentes = [col for col in columnas_deseadas if col in df.columns]
    
    if 'Order' not in cols_existentes:
        print("❌ Error: La columna clave 'Order' no existe en el archivo unificado.")
        return

    # Filtrar solo las columnas de interés
    df_sub = df[cols_existentes].copy()

    # 2. Asegurar tipos de datos numéricos para sumas de agregación
    if 'Quantity_SKU' in df_sub.columns:
        df_sub['Quantity_SKU'] = pd.to_numeric(df_sub['Quantity_SKU'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)

    if 'Total Value' in df_sub.columns:
        df_sub['Total Value'] = pd.to_numeric(df_sub['Total Value'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)

    # Convertir textos a string limpio para evitar errores al concatenar
    cols_texto = [c for c in cols_existentes if c not in ['Order', 'Quantity_SKU', 'Total Value']]
    for c in cols_texto:
        df_sub[c] = df_sub[c].fillna('').astype(str).str.strip()

    print(f"📊 Total de ítems/filas iniciales: {len(df_sub):,}")
    print("🔄 Agrupando por 'Order'...")

    # 3. Definir diccionario de reglas de agregación por columna
    agg_rules = {}

    for col in cols_existentes:
        if col == 'Order':
            continue
        elif col in ['Quantity_SKU', 'Total Value']:
            # Sumar cantidades y valores totales
            agg_rules[col] = 'sum'
        elif col in ['SKU Name', 'Discounts Names', 'Payment System Name']:
            # Concatenar productos/descuentos únicos separados por comas
            agg_rules[col] = lambda x: ' | '.join(unique_vals) if (unique_vals := [v for v in set(x) if v and v.lower() != 'nan']) else ''
        else:
            # Tomar el primer valor representativo para datos de la orden (Fecha, Cliente, Ciudad, UTMs, etc.)
            agg_rules[col] = 'first'

    # 4. Ejecutar la Agregación por Orden Única
    df_agrupado = df_sub.groupby('Order', as_index=False).agg(agg_rules)

    # ==========================================================
    # 4.5. NUEVA LÓGICA: CLIENTE NUEVO VS RECURRENTE POR ORDEN
    # ==========================================================
    if 'Client Document' in df_agrupado.columns and 'Creation Date' in df_agrupado.columns:
        # Asegurar tipo de dato fecha para la comparación
        df_agrupado['Creation Date_DT'] = pd.to_datetime(
            df_agrupado['Creation Date'], 
            format='mixed', 
            dayfirst=True, 
            errors='coerce'
        )
        
        # Encontrar la fecha de la primera compra de cada cliente (solo documentos válidos)
        mask_doc_valido = df_agrupado['Client Document'].astype(str).str.strip() != ''
        df_agrupado.loc[mask_doc_valido, 'First_Purchase_Date'] = df_agrupado[mask_doc_valido].groupby('Client Document')['Creation Date_DT'].transform('min')
        
        # Clasificar: Si la fecha de la orden es la primera, es Nuevo; si no, Recurrente
        df_agrupado['Tipo_Cliente'] = np.where(
            df_agrupado['Creation Date_DT'] == df_agrupado['First_Purchase_Date'], 
            'Nuevo', 
            'Recurrente'
        )
        
        # Limpiar columnas auxiliares
        df_agrupado = df_agrupado.drop(columns=['Creation Date_DT', 'First_Purchase_Date'])

    # 5. Guardar el nuevo dataset procesado
    SALIDA_VTEX_AGRUPADO.parent.mkdir(parents=True, exist_ok=True)
    df_agrupado.to_csv(SALIDA_VTEX_AGRUPADO, index=False, encoding='utf-8-sig')

    print(f"\n✅ Proceso completado exitosamente:")
    print(f"  • Filas iniciales (ítems): {len(df_sub):,}")
    print(f"  • Órdenes únicas finales: {len(df_agrupado):,}")
    print(f"📁 Guardado en: {SALIDA_VTEX_AGRUPADO}")


if __name__ == '__main__':
    generar_dataset_vtex_por_orden()