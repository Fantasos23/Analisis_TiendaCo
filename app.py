import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import warnings

# Importar funciones de procesamiento ETL
from etl_procesamiento import unificar_carpeta
from procesar_vtex_agrupado import generar_dataset_vtex_por_orden

warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(
    page_title="Dashboard Integral de Ventas & Atribución",
    page_icon="🚀",
    layout="wide"
)

# Directorios de Archivos
DATASETS_DIR = Path('datasets_procesados')
VTEX_AGRUPADO_PATH = DATASETS_DIR / 'dataset_vtex_agrupado_ordenes.csv'
META_PATH = DATASETS_DIR / 'dataset_meta_unificado.csv'
GOOGLE_PATH = DATASETS_DIR / 'dataset_google_unificado.csv'

# ==========================================================
# 1. MAPEOS Y REGLAS DE NEGOCIO (DICCIONARIOS)
# ==========================================================
DICT_CANALES = {
    'Facebook': ['facebook', 'fb', 'facebook-sitelink', 'fb-sitelink', 'facebookcpa', 'facebook-sitelink-5', 
                 'fb-sitelink-3', 'fb-sitelink-6', 'fb-sitelink-1', 'fb-sitelink-2', 'facebook_salonini', 
                 'facebook_salonin', 'trafico_andrea'],
    'Instagram': ['instagram', 'ig', 'igshopping'],
    'Google': ['google', 'google&utm_medium=cpc'],
    'YouTube': ['youtube'],
    'TikTok': ['tik tok', 'tiktok', 'tik_tok'],
    'HubSpot': ['hs_email', 'hs', 'hs_automation'],
    'Connectif': ['connectif'],
    'Icommarketing': ['icommarketing'],
    'VTEX': ['vtex', 'vtexcem'],
    'General': ['web', 'quiz'],
    'Nequi': ['nequi_app'],
    'AI': ['chatgpt.com', 'copilot.com'],
    'Directo / Sin Datos': ['nan', '', 'none', 'null']
}

DICT_ORIGENES = {
    'Publicidad (Pauta)': ['cpc', 'cpa', 'cpa+', 'cpm', 'paid', 'facebook', 'conversion', 'trafico'],
    'Orgánico / Botones Web': ['boton_tienda_superior', 'boton_tienda_inferior', 'boton_superior_tienda', 
                              'boton_inferior_tienda', 'boton_tienda', 'boton_superior_tienda_col', 'boton_inferior_tienda_col'],
    'Enlaces En Redes': ['linktree', 'social', 'content_creator'],
    'Email / Push': ['email', 'mail', 'push', 'webpush', 'abandono_carrinho', 'vtex'],
    'Alianzas': ['nequi'],
    'Directo': ['nan', '', 'none', 'null']
}

MARCAS_LISTA = [
    'SalonIn', 'Green Code', 'Luminance', 'Vitane', 'Muss', 
    'Bacterion', 'Tanga', 'CHAPSTICK', 'Deo Pies', 'Coloriss', 'Sol Eclair', 'Kleer Lac'
]

if 'custom_mappings' not in st.session_state:
    st.session_state.custom_mappings = {'Canal': {}, 'Origen': {}}


# ==========================================================
# 2. FUNCIONES DE AUXILIARES Y TRADUCCIÓN
# ==========================================================
def clasificar_valor(val, diccionario, custom_dict, default='Otros / No Asignados'):
    val_clean = str(val).lower().strip()
    if val_clean in custom_dict:
        return custom_dict[val_clean]
    for categoria, patrones in diccionario.items():
        if any(p in val_clean for p in patrones):
            return categoria
    return default

def extraer_lista_descuentos_unicos(serie_descuentos):
    """Extrae una lista limpia de todos los nombres de descuentos únicos."""
    descuentos = set()
    for item in serie_descuentos.dropna():
        partes = str(item).split('|')
        for p in partes:
            p_clean = p.strip()
            if p_clean and p_clean.lower() not in ['nan', 'none', 'null', '']:
                descuentos.add(p_clean)
    return sorted(list(descuentos))


# ==========================================================
# 3. CARGA Y PREPARACIÓN DE DATOS
# ==========================================================
@st.cache_data
def cargar_datos_vtex():
    if not VTEX_AGRUPADO_PATH.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(VTEX_AGRUPADO_PATH, low_memory=False)
    
    # Fechas
    df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce')
    df = df.dropna(subset=['Creation Date'])
    df['Año'] = df['Creation Date'].dt.year
    df['Mes_Num'] = df['Creation Date'].dt.month
    df['Día'] = df['Creation Date'].dt.date
    df['Quarter_Num'] = df['Creation Date'].dt.quarter
    df['Quarter'] = 'Q' + df['Quarter_Num'].astype(str)
    
    meses_esp = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
    df['Mes_Nombre'] = df['Mes_Num'].map(meses_esp)
    
    # Numéricos
    df['Total Value'] = pd.to_numeric(df['Total Value'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    df['Quantity_SKU'] = pd.to_numeric(df['Quantity_SKU'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    df['Discounts Names'] = df['Discounts Names'].fillna('Sin Descuento').astype(str)
    
    return df

@st.cache_data
def cargar_inversion_ads():
    df_meta, df_google = pd.DataFrame(), pd.DataFrame()

    def limpiar_numero(serie):
        """Limpia símbolos de moneda $, puntos de miles y comas decimales."""
        s = serie.astype(str).str.replace('$', '', regex=False).str.strip()
        # Si tiene puntos y comas (ej: 1.500,50), quitar puntos y reemplazar coma por punto
        s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(s, errors='coerce').fillna(0)

    # 1. Procesar Meta Ads
    if META_PATH.exists():
        df_m = pd.read_csv(META_PATH, low_memory=False)
        col_fecha = [c for c in df_m.columns if 'fecha' in c.lower() or 'date' in c.lower() or 'day' in c.lower()]
        col_costo = [c for c in df_m.columns if 'importe' in c.lower() or 'gastado' in c.lower() or 'spend' in c.lower()]
        
        if col_fecha and col_costo:
            df_m['Fecha_Clean'] = pd.to_datetime(df_m[col_fecha[0]], format='mixed', dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
            df_m['Inversion'] = limpiar_numero(df_m[col_costo[0]])
            df_meta = df_m[['Fecha_Clean', 'Inversion']].copy()

    # 2. Procesar Google Ads
    if GOOGLE_PATH.exists():
        df_g = pd.read_csv(GOOGLE_PATH, low_memory=False)
        col_fecha = [c for c in df_g.columns if 'fecha' in c.lower() or 'date' in c.lower() or 'day' in c.lower()]
        col_costo = [c for c in df_g.columns if 'coste' in c.lower() or 'cost' in c.lower()]
        
        if col_fecha and col_costo:
            df_g['Fecha_Clean'] = pd.to_datetime(df_g[col_fecha[0]], format='mixed', dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
            df_g['Inversion'] = limpiar_numero(df_g[col_costo[0]])
            df_google = df_g[['Fecha_Clean', 'Inversion']].copy()

    return df_meta, df_google


# ==========================================================
# 4. ENCABEZADO Y BOTONES DE PROCESAMIENTO
# ==========================================================
st.title("📊 Control de Mando Integral: Ventas & Atribución")

col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
with col_b1:
    st.caption("Consolidado por Órdenes Únicas, Canales, Descuentos e Inversión.")

with col_b2:
    if st.button("1. 🔄 Unificar Archivos (ETL)", type="secondary", use_container_width=True):
        with st.spinner("Unificando CSVs de origen..."):
            unificar_carpeta('VTEX', ';', 'Order')
            unificar_carpeta('Meta', ',')
            unificar_carpeta('Google', ',')
            st.cache_data.clear()
        st.success("¡Unificación completada!")

with col_b3:
    if st.button("2. 👥 Filtrar Clientes / Agrupar", type="primary", use_container_width=True):
        with st.spinner("Agrupando a nivel de Orden Única..."):
            generar_dataset_vtex_por_orden()
            st.cache_data.clear()
        st.success("¡Agregación de VTEX lista!")

st.divider()

# Carga de datasets
df_vtex = cargar_datos_vtex()
df_meta, df_google = cargar_inversion_ads()

if df_vtex.empty:
    st.info("💡 Haz clic en los botones superiores para procesar y agrupar los datos.")
    st.stop()


# ==========================================================
# 5. MODAL DE RECLASIFICACIÓN DE VALORES NUEVOS
# ==========================================================
df_vtex['Canal_Estandar'] = df_vtex['UtmSource'].apply(
    lambda x: clasificar_valor(x, DICT_CANALES, st.session_state.custom_mappings['Canal'])
)
df_vtex['Origen_Estandar'] = df_vtex['UtmMedium'].apply(
    lambda x: clasificar_valor(x, DICT_ORIGENES, st.session_state.custom_mappings['Origen'])
)

nuevos_canales = df_vtex[df_vtex['Canal_Estandar'] == 'Otros / No Asignados']['UtmSource'].dropna().unique().tolist()
nuevos_origenes = df_vtex[df_vtex['Origen_Estandar'] == 'Otros / No Asignados']['UtmMedium'].dropna().unique().tolist()

if nuevos_canales or nuevos_origenes:
    with st.expander("⚠️ ¡Atención! Se detectaron nuevos valores de Canal u Origen no asignados", expanded=False):
        if nuevos_canales:
            st.markdown("**Nuevos Canales (UtmSource):**")
            for nc in nuevos_canales[:5]:
                cat_sel = st.selectbox(f"Asignar '{nc}' a:", list(DICT_CANALES.keys()), key=f"nc_{nc}")
                if st.button(f"Guardar regla para {nc}"):
                    st.session_state.custom_mappings['Canal'][str(nc).lower().strip()] = cat_sel
                    st.rerun()


# ==========================================================
# 6. FILTROS EN CASCADA (SIDEBAR)
# ==========================================================
st.sidebar.header("🔍 Filtros de Visualización")

modo_tiempo = st.sidebar.radio("Modo Temporal:", ["Año Completo (YoY)", "Mes vs Mes Anterior (MoM)", "Quarter vs Q Año Anterior"])
anios_disp = sorted(df_vtex['Año'].unique(), reverse=True)
anio_sel = st.sidebar.selectbox("Año Principal:", anios_disp, index=0)

df_f = df_vtex[df_vtex['Año'] == anio_sel].copy()

if modo_tiempo == "Mes vs Mes Anterior (MoM)":
    meses_disp = df_f['Mes_Nombre'].unique().tolist()
    mes_sel = st.sidebar.selectbox("Selecciona el Mes:", meses_disp)
    mes_num_sel = df_f[df_f['Mes_Nombre'] == mes_sel]['Mes_Num'].iloc[0]
    df_f = df_f[df_f['Mes_Num'] == mes_num_sel]

elif modo_tiempo == "Quarter vs Q Año Anterior":
    q_disp = sorted(df_f['Quarter'].unique().tolist())
    q_sel = st.sidebar.selectbox("Selecciona el Quarter:", q_disp)
    df_f = df_f[df_f['Quarter'] == q_sel]

st.sidebar.divider()

# Tipo de Cliente
tipo_cliente_sel = st.sidebar.multiselect("Tipo de Cliente:", ["Nuevo", "Recurrente"], default=["Nuevo", "Recurrente"])
if tipo_cliente_sel:
    df_f = df_f[df_f['Tipo_Cliente'].isin(tipo_cliente_sel)]

# Canales
canales_disp = sorted(df_f['Canal_Estandar'].unique().tolist())
canales_sel = st.sidebar.multiselect("Canal (UtmSource):", canales_disp, default=canales_disp)
if canales_sel:
    df_f = df_f[df_f['Canal_Estandar'].isin(canales_sel)]

# Orígenes
origenes_disp = sorted(df_f['Origen_Estandar'].unique().tolist())
origenes_sel = st.sidebar.multiselect("Origen (UtmMedium):", origenes_disp, default=origenes_disp)
if origenes_sel:
    df_f = df_f[df_f['Origen_Estandar'].isin(origenes_sel)]

# ----------------------------------------------------------
# NUEVO: FILTRO DE DESCUENTOS Y PROMOCIONES (Discounts Names)
# ----------------------------------------------------------
lista_descuentos = extraer_lista_descuentos_unicos(df_f['Discounts Names'])
descuentos_sel = st.sidebar.multiselect("Descuentos / Promociones:", lista_descuentos, default=[])
if descuentos_sel:
    # Coincidencia si alguno de los descuentos seleccionados está en la cadena agrupada
    patron_desc = '|'.join([str(d) for d in descuentos_sel])
    df_f = df_f[df_f['Discounts Names'].str.contains(patron_desc, case=False, na=False)]

# Medios de Pago
medios_pago_disp = sorted(df_f['Payment System Name'].fillna('No Especificado').unique().tolist())
medios_sel = st.sidebar.multiselect("Medio de Pago:", medios_pago_disp, default=medios_pago_disp)
if medios_sel:
    df_f = df_f[df_f['Payment System Name'].fillna('No Especificado').isin(medios_sel)]

# Marca
marca_sel = st.sidebar.selectbox("Filtrar por Marca:", ["Todas"] + MARCAS_LISTA)
if marca_sel != "Todas":
    df_f = df_f[df_f['SKU Name'].str.lower().str.contains(marca_sel.lower(), na=False)]


# ==========================================================
# 7. COMPARATIVAS PERÍODO ANTERIOR Y MÉTRICAS
# ==========================================================
if modo_tiempo == "Año Completo (YoY)":
    df_comp = df_vtex[df_vtex['Año'] == (anio_sel - 1)]
    etiqueta_comp = f"vs Año {anio_sel - 1}"
elif modo_tiempo == "Mes vs Mes Anterior (MoM)":
    if mes_num_sel == 1:
        df_comp = df_vtex[(df_vtex['Año'] == anio_sel - 1) & (df_vtex['Mes_Num'] == 12)]
    else:
        df_comp = df_vtex[(df_vtex['Año'] == anio_sel) & (df_vtex['Mes_Num'] == mes_num_sel - 1)]
    etiqueta_comp = "vs Mes Anterior"
else:
    q_num = int(q_sel.replace('Q', ''))
    df_comp = df_vtex[(df_vtex['Año'] == anio_sel - 1) & (df_vtex['Quarter_Num'] == q_num)]
    etiqueta_comp = f"vs {q_sel} {anio_sel - 1}"

if not df_comp.empty:
    if tipo_cliente_sel: df_comp = df_comp[df_comp['Tipo_Cliente'].isin(tipo_cliente_sel)]
    if canales_sel: df_comp = df_comp[df_comp['Canal_Estandar'].isin(canales_sel)]
    if origenes_sel: df_comp = df_comp[df_comp['Origen_Estandar'].isin(origenes_sel)]
    if descuentos_sel: 
        patron_desc = '|'.join([str(d) for d in descuentos_sel])
        df_comp = df_comp[df_comp['Discounts Names'].str.contains(patron_desc, case=False, na=False)]
    if marca_sel != "Todas": df_comp = df_comp[df_comp['SKU Name'].str.lower().str.contains(marca_sel.lower(), na=False)]

ventas_actual = df_f['Total Value'].sum()
ventas_comp = df_comp['Total Value'].sum() if not df_comp.empty else 0
var_ventas = ((ventas_actual - ventas_comp) / ventas_comp * 100) if ventas_comp > 0 else 0

ordenes_actual = df_f['Order'].nunique()
ordenes_comp = df_comp['Order'].nunique() if not df_comp.empty else 0
var_ordenes = ((ordenes_actual - ordenes_comp) / ordenes_comp * 100) if ordenes_comp > 0 else 0

unidades_actual = df_f['Quantity_SKU'].sum()

# Convertir las fechas filtradas de VTEX al mismo formato de texto 'YYYY-MM-DD'
dias_actuales_str = pd.to_datetime(df_f['Creation Date']).dt.strftime('%Y-%m-%d').dropna().unique().tolist()

# Sumar la inversión que coincida en el rango de días
inv_meta_tot = df_meta[df_meta['Fecha_Clean'].isin(dias_actuales_str)]['Inversion'].sum() if not df_meta.empty else 0.0
inv_google_tot = df_google[df_google['Fecha_Clean'].isin(dias_actuales_str)]['Inversion'].sum() if not df_google.empty else 0.0

inversion_total = inv_meta_tot + inv_google_tot

# Cálculo de ROAS
roas = (ventas_actual / inversion_total) if inversion_total > 0 else 0.0


# ==========================================================
# 8. METRICAS CLAVE
# ==========================================================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ingresos Totales", f"${ventas_actual:,.0f}", f"{var_ventas:+.1f}% {etiqueta_comp}")
k2.metric("Ventas (Órdenes)", f"{ordenes_actual:,}", f"{var_ordenes:+.1f}% {etiqueta_comp}")
k3.metric("Unidades Vendidas", f"{unidades_actual:,.0f}")
k4.metric("Inversión Ads", f"${inversion_total:,.0f}")
k5.metric("ROAS General", f"{roas:.2f} x")

st.divider()


# ==========================================================
# 9. GRÁFICOS DE BARRAS INTERACTIVOS
# ==========================================================
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Ventas por Canal (UtmSource)")
    df_canal = df_f.groupby('Canal_Estandar').agg(Ingresos=('Total Value', 'sum'), Ordenes=('Order', 'nunique')).reset_index().sort_values('Ingresos', ascending=False)
    fig_canal = px.bar(
        df_canal, x='Canal_Estandar', y='Ingresos', text_auto='.2s',
        labels={'Canal_Estandar': 'Canal', 'Ingresos': 'Ingresos (COP)'},
        color='Ingresos', color_continuous_scale='Viridis'
    )
    fig_canal.update_traces(hovertemplate="<b>Canal:</b> %{x}<br><b>Ingresos:</b> $%{y:,.0f}<extra></extra>")
    st.plotly_chart(fig_canal, use_container_width=True)

with col_g2:
    st.subheader("Adquisición: Clientes Nuevos vs Recurrentes")
    df_tipo = df_f.groupby(['Mes_Nombre', 'Tipo_Cliente']).agg(Ordenes=('Order', 'nunique'), Ingresos=('Total Value', 'sum')).reset_index()
    fig_tipo = px.bar(
        df_tipo, x='Mes_Nombre', y='Ordenes', color='Tipo_Cliente', barmode='group', text_auto=True
    )
    fig_tipo.update_traces(hovertemplate="<b>Mes:</b> %{x}<br><b>Órdenes:</b> %{y:,d}<extra></extra>")
    st.plotly_chart(fig_tipo, use_container_width=True)

col_g3, col_g4 = st.columns(2)

with col_g3:
    # NUEVO: GRÁFICO DE USO DE DESCUENTOS Y PROMOCIONES
    st.subheader("Descuentos / Promociones Más Utilizados")
    
    # Explotar la columna Discounts Names separada por '|' para contar individualmente
    df_desc_exp = df_f.assign(Descuento_Unico=df_f['Discounts Names'].str.split('|')).explode('Descuento_Unico')
    df_desc_exp['Descuento_Unico'] = df_desc_exp['Descuento_Unico'].str.strip()
    df_desc_exp = df_desc_exp[~df_desc_exp['Descuento_Unico'].isin(['Sin Descuento', '', 'nan', 'none'])]
    
    if not df_desc_exp.empty:
        df_desc_agg = df_desc_exp.groupby('Descuento_Unico').agg(
            Ordenes=('Order', 'nunique'),
            Ingresos=('Total Value', 'sum')
        ).reset_index().sort_values('Ordenes', ascending=False).head(8)
        
        fig_desc = px.bar(
            df_desc_agg, x='Ordenes', y='Descuento_Unico', orientation='h',
            text_auto=True, color='Ingresos', color_continuous_scale='Purples'
        )
        fig_desc.update_traces(hovertemplate="<b>Descuento:</b> %{y}<br><b>Órdenes:</b> %{x:,d}<extra></extra>")
        st.plotly_chart(fig_desc, use_container_width=True)
    else:
        st.info("No hay cupones/descuentos aplicados en el rango filtrado.")

with col_g4:
    st.subheader("Preferencias por Medio de Pago")
    df_pago = df_f.groupby('Payment System Name').agg(Ordenes=('Order', 'nunique')).reset_index().sort_values('Ordenes', ascending=False).head(8)
    fig_pago = px.bar(
        df_pago, x='Ordenes', y='Payment System Name', orientation='h',
        text_auto=True, color='Ordenes', color_continuous_scale='Blues'
    )
    fig_pago.update_traces(hovertemplate="<b>Medio de Pago:</b> %{y}<br><b>Órdenes:</b> %{x:,d}<extra></extra>")
    st.plotly_chart(fig_pago, use_container_width=True)


# ==========================================================
# 10. RESUMEN EN TABLA EJECUTIVA
# ==========================================================
st.subheader("📋 Tabla de Atribución Detallada")
tabla_resumen = df_f.groupby(['Canal_Estandar', 'Origen_Estandar', 'Discounts Names']).agg(
    Órdenes=('Order', 'nunique'),
    Unidades=('Quantity_SKU', 'sum'),
    Ingresos_Totales=('Total Value', 'sum')
).reset_index().sort_values('Ingresos_Totales', ascending=False)

st.dataframe(
    tabla_resumen.style.format({'Órdenes': '{:,}', 'Unidades': '{:,.0f}', 'Ingresos_Totales': '${:,.0f}'}),
    use_container_width=True
)