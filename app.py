import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Inicializar session_state si no existe
if 'custom_mappings' not in st.session_state:
    st.session_state.custom_mappings = {'Canal': {}, 'Origen': {}}

st.markdown("""
<style>
    /* 1. Cambiar el color de los 'chips' seleccionados en multiselect a un gris/azul neutro */
    span[data-baseweb="tag"] {
        background-color: #334155 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    
    /* 2. Ocultar la barra de color (Colorbar) redundante en los gráficos de Plotly */
    .coloraxis {
        display: none !important;
    }
    
    /* 3. Ajuste de contenedor de filtros */
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

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

DICT_MARCAS_VARIANTES = {
    'SalonIn': ['salonin', 'salon in', 'vegan keratin collagen', 'VEGAN KERATIN'],
    'Sol Eclair': ['sol eclair', 'soleclair', 'sol-eclair'],
    'Green Code': ['green code', 'greencode'],
    'Luminance': ['luminance'],
    'Vitane': ['vitane'],
    'Muss': ['muss'],
    'Bacterion': ['bacterion'],
    'Tanga': ['tanga'],
    'CHAPSTICK': ['chapstick', 'chap stick'],
    'Deo Pies': ['deo pies', 'deopies'],
    'Coloriss': ['coloriss'],
    'Kleer Lac': ['kleer lac', 'kleerlac']
}

MARCAS_LISTA = list(DICT_MARCAS_VARIANTES.keys())

# ==========================================================
# 2. FUNCIONES AUXILIARES
# ==========================================================
def clasificar_valor(val, diccionario, custom_dict, default='Otros / No Asignados'):
    val_clean = str(val).lower().strip()
    if val_clean in custom_dict:
        return custom_dict[val_clean]
    for categoria, patrones in diccionario.items():
        if any(p in val_clean for p in patrones):
            return categoria
    return default

def formatear_cifra_corta(valor, es_moneda=True):
    simbolo = "$" if es_moneda else ""
    cifra_exacta = f"{simbolo}{valor:,.0f}"
    abs_val = abs(valor)
    
    if abs_val >= 1_000_000_000:
        corta = f"{simbolo}{valor / 1_000_000_000:.2f} Bill"
    elif abs_val >= 1_000_000:
        corta = f"{simbolo}{valor / 1_000_000:.2f} Mill"
    elif abs_val >= 100_000:
        corta = f"{simbolo}{valor / 1_000:.1f} Mil"
    else:
        corta = cifra_exacta
        
    return corta, cifra_exacta

def obtener_top_productos(df_entrada, marca_filtro="Todas", top_n=10):
    if df_entrada.empty or 'SKU Name' not in df_entrada.columns:
        return pd.DataFrame()

    df_exp = df_entrada.assign(
        SKU_Individual=df_entrada['SKU Name'].astype(str).str.split('|')
    ).explode('SKU_Individual')

    df_exp['SKU_Individual'] = df_exp['SKU_Individual'].str.strip()
    df_exp = df_exp[~df_exp['SKU_Individual'].str.lower().isin(['nan', 'none', '', 'sin sku'])]

    if df_exp.empty:
        return pd.DataFrame()

    def asignar_marca(sku):
        sku_clean = str(sku).lower()
        for marca_oficial, variantes in DICT_MARCAS_VARIANTES.items():
            if any(variante in sku_clean for variante in variantes):
                return marca_oficial
        return 'Otros / Sin Marca'

    df_exp['Marca_Detectada'] = df_exp['SKU_Individual'].apply(asignar_marca)

    if marca_filtro == "Otros":
        df_exp = df_exp[df_exp['Marca_Detectada'] == 'Otros / Sin Marca']
    elif marca_filtro != "Todas":
        df_exp = df_exp[df_exp['Marca_Detectada'] == marca_filtro]

    df_ranking = df_exp.groupby(['SKU_Individual', 'Marca_Detectada']).agg(
        Ordenes=('Order', 'nunique'),
        Unidades=('Quantity_SKU', 'sum')
    ).reset_index()

    return df_ranking.sort_values(by=['Unidades', 'Ordenes'], ascending=[False, False]).head(top_n)

def limpiar_cadena_descuentos(cadena_raw):
    if not cadena_raw or str(cadena_raw).lower() in ['nan', 'none', 'sin descuento', '']:
        return "Sin Descuento"
    elementos = [e.strip() for e in str(cadena_raw).replace('|', ',').split(',')]
    elementos_unicos = list(dict.fromkeys([e for e in elementos if e and e.lower() != 'nan']))
    return " + ".join(elementos_unicos) if elementos_unicos else "Sin Descuento"

def generar_grafico_pareto(df_entrada):
    """Genera la Curva de Pareto enfocada en el rango real de acumulación."""
    if df_entrada.empty or 'SKU Name' not in df_entrada.columns:
        return None

    df_exp = df_entrada.assign(
        SKU_Individual=df_entrada['SKU Name'].astype(str).str.split('|')
    ).explode('SKU_Individual')
    df_exp['SKU_Individual'] = df_exp['SKU_Individual'].str.strip()
    df_exp = df_exp[~df_exp['SKU_Individual'].str.lower().isin(['nan', 'none', '', 'sin sku'])]

    if df_exp.empty:
        return None

    df_pareto = df_exp.groupby('SKU_Individual')['Total Value'].sum().reset_index()
    df_pareto = df_pareto.sort_values(by='Total Value', ascending=False).reset_index(drop=True)
    
    total_ventas = df_pareto['Total Value'].sum()
    if total_ventas == 0:
        return None
        
    df_pareto['Ingreso_Acumulado'] = df_pareto['Total Value'].cumsum()
    df_pareto['Pct_Acumulado'] = (df_pareto['Ingreso_Acumulado'] / total_ventas) * 100
    
    # Tomar Top 15
    df_top = df_pareto.head(15).copy()
    df_top['SKU_Corto'] = df_top['SKU_Individual'].apply(lambda x: x[:22] + '...' if len(x) > 25 else x)

    pct_alcanzado = df_top['Pct_Acumulado'].max()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=df_top['SKU_Corto'], 
            y=df_top['Total Value'],
            name="Ventas ($)",
            marker_color='#2563eb',
            customdata=df_top['SKU_Individual'],
            hovertemplate="<b>Producto:</b> %{customdata}<br><b>Ventas:</b> $%{y:,.0f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_top['SKU_Corto'], 
            y=df_top['Pct_Acumulado'],
            name="% Acumulado",
            line=dict(color='#f59e0b', width=3),
            mode='lines+markers',
            hovertemplate="<b>% Acumulado:</b> %{y:.1f}%<extra></extra>"
        ),
        secondary_y=True
    )

    # Ajustar límite Y secundario según el acumulado del Top 15 para evitar compresión
    max_y_2 = min(100, max(60, int(pct_alcanzado + 15)))

    # Línea del 80% solo si el Top 15 lo alcanza, si no, línea guía del acumulado
    if pct_alcanzado >= 80:
        fig.add_shape(
            type="line", x0=-0.5, x1=len(df_top)-0.5, y0=80, y1=80,
            yref="y2", line=dict(color="red", width=2, dash="dash")
        )

    fig.update_layout(
        hovermode="x unified",
        height=420,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=100),
        xaxis=dict(tickangle=-45)
    )
    fig.update_yaxes(title_text="Ingresos ($ COP)", secondary_y=False)
    fig.update_yaxes(title_text="% Acumulado", secondary_y=True, range=[0, max_y_2])
    
    return fig, pct_alcanzado

def generar_matriz_cohortes(df_completo_vtex):
    """Calcula la matriz de retención en porcentaje."""
    if df_completo_vtex.empty or 'Client Document' not in df_completo_vtex.columns:
        return pd.DataFrame()

    df_c = df_completo_vtex.dropna(subset=['Client Document', 'Creation Date']).copy()
    df_c['Client Document'] = df_c['Client Document'].astype(str)
    
    df_c['Order_Month'] = df_c['Creation Date'].dt.to_period('M').astype(str)
    df_c['Cohort_Month'] = df_c.groupby('Client Document')['Creation Date'].transform('min').dt.to_period('M').astype(str)

    df_cohort_data = df_c.groupby(['Cohort_Month', 'Order_Month']).agg(Clientes=('Client Document', 'nunique')).reset_index()
    
    df_cohort_data['Periodo_Mes'] = (
        pd.to_datetime(df_cohort_data['Order_Month']).dt.to_period('M') - 
        pd.to_datetime(df_cohort_data['Cohort_Month']).dt.to_period('M')
    ).apply(lambda x: x.n)

    cohort_pivot = df_cohort_data.pivot(index='Cohort_Month', columns='Periodo_Mes', values='Clientes')
    
    if cohort_pivot.empty:
        return pd.DataFrame()

    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100

    return retention_matrix

def generar_grafico_tiempo_entre_compras(df_completo_vtex):
    """Calcula el promedio de días que pasan entre la 1ª, 2ª, 3ª y 4ª compra."""
    if df_completo_vtex.empty or 'Client Document' not in df_completo_vtex.columns:
        return None

    df_frec = df_completo_vtex.dropna(subset=['Client Document', 'Creation Date']).copy()
    df_frec['Client Document'] = df_frec['Client Document'].astype(str)
    
    # Ordenar por cliente y fecha
    df_frec = df_frec.sort_values(['Client Document', 'Creation Date'])
    
    # Asignar número de orden consecutiva por cliente (1, 2, 3...)
    df_frec['Num_Orden_Cliente'] = df_frec.groupby('Client Document').cumcount() + 1
    
    # Calcular la diferencia de días respecto a la orden anterior
    df_frec['Fecha_Previo'] = df_frec.groupby('Client Document')['Creation Date'].shift(1)
    df_frec['Dias_Entre_Compras'] = (df_frec['Creation Date'] - df_frec['Fecha_Previo']).dt.days

    # Filtrar solo recompras (Órdenes >= 2)
    df_recompras = df_frec[df_frec['Num_Orden_Cliente'].isin([2, 3, 4, 5])].copy()

    if df_recompras.empty:
        return None

    df_promedios = df_recompras.groupby('Num_Orden_Cliente')['Dias_Entre_Compras'].mean().reset_index()
    
    etiquetas_map = {
        2: "1ª ➔ 2ª Compra",
        3: "2ª ➔ 3ª Compra",
        4: "3ª ➔ 4ª Compra",
        5: "4ª ➔ 5ª Compra"
    }
    df_promedios['Etiqueta'] = df_promedios['Num_Orden_Cliente'].map(etiquetas_map)

    # Gráfico de Barras Verticales
    fig = px.bar(
        df_promedios,
        x='Etiqueta',
        y='Dias_Entre_Compras',
        text_auto='.0f',
        labels={'Etiqueta': 'Salto de Recompra', 'Dias_Entre_Compras': 'Días Promedio'},
        color_discrete_sequence=['#3b82f6']
    )
    fig.update_traces(
        textposition='outside',
        hovertemplate="<b>Transición:</b> %{x}<br><b>Tiempo Promedio:</b> %{y:.1f} días<extra></extra>"
    )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Días Promedio"
    )
    return fig

# ==========================================================
# 3. CARGA Y PREPARACIÓN DE DATOS (PARSER ROBUSATO)
# ==========================================================
@st.cache_data
def cargar_datos_vtex():
    if not VTEX_AGRUPADO_PATH.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(VTEX_AGRUPADO_PATH, low_memory=False)
    
    # Convertir a datetime y TRUNCAR LAS HORAS de inmediato (.dt.floor('D'))
    df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce').dt.floor('D')
    df = df.dropna(subset=['Creation Date'])
    
    # Formato de fecha estricto YYYY-MM-DD sin hora
    df['Fecha_Clean'] = df['Creation Date'].dt.strftime('%Y-%m-%d')
    
    # Variables de calendario
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

    def limpiar_monto(serie):
        s = serie.astype(str).str.replace('$', '', regex=False).str.strip()
        s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(s, errors='coerce').fillna(0.0)

    # 1. Google Ads
    if GOOGLE_PATH.exists():
        df_g = pd.read_csv(GOOGLE_PATH, low_memory=False)
        col_fecha_g = next((c for c in ['Día', 'Dia', 'Day', 'Fecha'] if c in df_g.columns), None)
        col_costo_g = next((c for c in ['Coste', 'Cost', 'Cost / Inversión', 'Inversión'] if c in df_g.columns), None)

        if col_fecha_g and col_costo_g:
            # Parseo directo YYYY-MM-DD
            fechas_parsed = pd.to_datetime(df_g[col_fecha_g], errors='coerce').dt.floor('D')
            df_g['Fecha_Clean'] = fechas_parsed.dt.strftime('%Y-%m-%d')
            df_g['Inversion'] = limpiar_monto(df_g[col_costo_g])
            df_google = df_g[['Fecha_Clean', 'Inversion']].dropna(subset=['Fecha_Clean'])

    # 2. Meta Ads
    if META_PATH.exists():
        df_m = pd.read_csv(META_PATH, low_memory=False)
        col_fecha_m = next((c for c in ['Inicio del informe', 'Reporting starts', 'Day', 'Día', 'Fecha'] if c in df_m.columns), None)
        col_costo_m = next((c for c in ['Importe gastado (COP)', 'Amount spent (COP)', 'Importe gastado', 'Coste', 'Spend'] if c in df_m.columns), None)

        if col_fecha_m and col_costo_m:
            # Parseo directo YYYY-MM-DD
            fechas_parsed = pd.to_datetime(df_m[col_fecha_m], errors='coerce').dt.floor('D')
            df_m['Fecha_Clean'] = fechas_parsed.dt.strftime('%Y-%m-%d')
            df_m['Inversion'] = limpiar_monto(df_m[col_costo_m])
            df_meta = df_m[['Fecha_Clean', 'Inversion']].dropna(subset=['Fecha_Clean'])

    return df_meta, df_google

# ==========================================================
# 4. ENCABEZADO Y PROCESAMIENTO
# ==========================================================
st.title("Data Tienda Colombia")

col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
with col_b1:
    st.caption("Consolidado por Órdenes Únicas, Canales, Descuentos e Inversión.")

with col_b2:
    if st.button("1.Unificar Archivos (ETL)", type="secondary", use_container_width=True):
        with st.spinner("Unificando CSVs..."):
            unificar_carpeta('VTEX', ';', 'Order')
            unificar_carpeta('Meta', ',')
            unificar_carpeta('Google', ',')
            st.cache_data.clear()
        st.success("¡Completado!")

with col_b3:
    if st.button("2.Filtrar Clientes / Agrupar", type="primary", use_container_width=True):
        with st.spinner("Agrupando a nivel de Orden Única..."):
            generar_dataset_vtex_por_orden()
            st.cache_data.clear()
        st.success("¡Agregación lista!")

st.divider()

df_vtex = cargar_datos_vtex()
df_meta, df_google = cargar_inversion_ads()

if df_vtex.empty:
    st.info("Haz clic en los botones superiores para procesar y agrupar los datos.")
    st.stop()

# ==========================================================
# 5. RECLASIFICACIÓN DE CANALES/ORÍGENES
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
    with st.expander("⚠️ Atributos no asignados detectados", expanded=False):
        if nuevos_canales:
            st.markdown("**Nuevos Canales:**")
            for nc in nuevos_canales[:5]:
                cat_sel = st.selectbox(f"Asignar '{nc}' a:", list(DICT_CANALES.keys()), key=f"nc_{nc}")
                if st.button(f"Guardar regla para {nc}"):
                    st.session_state.custom_mappings['Canal'][str(nc).lower().strip()] = cat_sel
                    st.rerun()

# ==========================================================
# 6A. BARRA SUPERIOR DE FILTRO POR FECHAS
# ==========================================================
anios_disp = sorted(df_vtex['Año'].unique(), reverse=True)
col_año, col_segmento, col_mes, col_dias = st.columns([1.2, 3.5, 1.8, 1.8])

with col_año:
    anio_sel = st.selectbox("Año", anios_disp, index=0, label_visibility="collapsed")

df_f = df_vtex[df_vtex['Año'] == anio_sel].copy()

with col_segmento:
    opciones_periodo = ["Año", "Q1", "Q2", "Q3", "Q4", "Mes"]
    try:
        periodo_sel = st.segmented_control("Periodo", options=opciones_periodo, default="Año", label_visibility="collapsed")
    except AttributeError:
        periodo_sel = st.radio("Periodo", options=opciones_periodo, horizontal=True, label_visibility="collapsed")

MAP_MESES = {
    'Ene': 'Enero', 'Feb': 'Febrero', 'Mar': 'Marzo', 'Abr': 'Abril',
    'May': 'Mayo', 'Jun': 'Junio', 'Jul': 'Julio', 'Ago': 'Agosto',
    'Sep': 'Septiembre', 'Oct': 'Octubre', 'Nov': 'Noviembre', 'Dic': 'Diciembre'
}
MAP_MESES_REV = {v: k for k, v in MAP_MESES.items()}

mes_sel_nombre = None
opcion_dias_sel = "Todo el mes"

if periodo_sel == "Mes":
    meses_cortos_presentes = df_f['Mes_Nombre'].unique()
    meses_largos_presentes = [MAP_MESES[m] for m in meses_cortos_presentes if m in MAP_MESES]
    if not meses_largos_presentes:
        meses_largos_presentes = list(MAP_MESES.values())

    with col_mes:
        mes_sel_nombre = st.selectbox("Mes", meses_largos_presentes, index=len(meses_largos_presentes)-1, label_visibility="collapsed")
    with col_dias:
        opcion_dias_sel = st.selectbox("Días", ["Todo el mes", "Seleccionar Rango"], label_visibility="collapsed")

# Lógica del rango temporal
if periodo_sel == "Año":
    modo_tiempo = "Año Completo (YoY)"

elif periodo_sel in ["Q1", "Q2", "Q3", "Q4"]:
    modo_tiempo = "Quarter vs Q Año Anterior"
    q_num = int(periodo_sel.replace("Q", ""))
    df_f = df_f[df_f['Quarter_Num'] == q_num]

elif periodo_sel == "Mes":
    modo_tiempo = "Mes vs Mes Anterior (MoM)"
    if mes_sel_nombre:
        mes_corto_sel = MAP_MESES_REV.get(mes_sel_nombre, mes_sel_nombre)
        df_f = df_f[df_f['Mes_Nombre'] == mes_corto_sel]
        mes_num_sel = df_f['Mes_Num'].iloc[0] if not df_f.empty else 1

    if opcion_dias_sel == "Seleccionar Rango":
        min_date = df_f['Día'].min()
        max_date = df_f['Día'].max()
        if pd.notnull(min_date) and pd.notnull(max_date):
            rango = st.date_input("Rango", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(rango, tuple) and len(rango) == 2:
                df_f = df_f[(df_f['Día'] >= rango[0]) & (df_f['Día'] <= rango[1])]

# ==========================================================
# 6B. FILTROS SECUNDARIOS (SIDEBAR)
# ==========================================================
st.sidebar.header("Filtros de Segmentación")

tipo_cliente_sel = st.sidebar.multiselect("Tipo de Cliente:", ["Nuevo", "Recurrente"], default=[], placeholder="Todos los clientes")
if tipo_cliente_sel:
    df_f = df_f[df_f['Tipo_Cliente'].isin(tipo_cliente_sel)]

canales_disp = sorted(df_f['Canal_Estandar'].unique().tolist())
canales_sel = st.sidebar.multiselect("Canal (UtmSource):", canales_disp, default=[], placeholder="Todos los canales")
if canales_sel:
    df_f = df_f[df_f['Canal_Estandar'].isin(canales_sel)]

origenes_disp = sorted(df_f['Origen_Estandar'].unique().tolist())
origenes_sel = st.sidebar.multiselect("Origen (UtmMedium):", origenes_disp, default=[], placeholder="Todos los orígenes")
if origenes_sel:
    df_f = df_f[df_f['Origen_Estandar'].isin(origenes_sel)]

df_f['Discounts_Clean'] = df_f['Discounts Names'].apply(limpiar_cadena_descuentos)
descuentos_sel = []

medios_pago_disp = sorted(df_f['Payment System Name'].fillna('No Especificado').unique().tolist())
medios_sel = st.sidebar.multiselect("Medio de Pago:", medios_pago_disp, default=[], placeholder="Todos los medios de pago")
if medios_sel:
    df_f = df_f[df_f['Payment System Name'].fillna('No Especificado').isin(medios_sel)]

opciones_marca = ["Todas"] + MARCAS_LISTA + ["Otros"]
marca_sel = st.sidebar.selectbox("Filtrar por Marca:", opciones_marca)

if marca_sel == "Otros":
    todas_variantes = [v for lista in DICT_MARCAS_VARIANTES.values() for v in lista]
    patron_todas = '|'.join(todas_variantes)
    df_f = df_f[~df_f['SKU Name'].str.lower().str.contains(patron_todas, na=False)]
elif marca_sel != "Todas":
    variantes_marca = DICT_MARCAS_VARIANTES.get(marca_sel, [marca_sel.lower()])
    patron_marca = '|'.join(variantes_marca)
    df_f = df_f[df_f['SKU Name'].str.lower().str.contains(patron_marca, na=False)]

# ==========================================================
# 7. COMPARATIVAS Y CÁLCULO DIRECTO DE INVERSIÓN
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
    q_num = int(periodo_sel.replace('Q', ''))
    df_comp = df_vtex[(df_vtex['Año'] == anio_sel - 1) & (df_vtex['Quarter_Num'] == q_num)]
    etiqueta_comp = f"vs {periodo_sel} {anio_sel - 1}"

if not df_comp.empty:
    if tipo_cliente_sel: df_comp = df_comp[df_comp['Tipo_Cliente'].isin(tipo_cliente_sel)]
    if canales_sel: df_comp = df_comp[df_comp['Canal_Estandar'].isin(canales_sel)]
    if origenes_sel: df_comp = df_comp[df_comp['Origen_Estandar'].isin(origenes_sel)]
    if marca_sel != "Todas": df_comp = df_comp[df_comp['SKU Name'].str.lower().str.contains(marca_sel.lower(), na=False)]

ventas_actual = df_f['Total Value'].sum()
ventas_comp = df_comp['Total Value'].sum() if not df_comp.empty else 0
var_ventas = ((ventas_actual - ventas_comp) / ventas_comp * 100) if ventas_comp > 0 else 0

ordenes_actual = df_f['Order'].nunique()
ordenes_comp = df_comp['Order'].nunique() if not df_comp.empty else 0
var_ordenes = ((ordenes_actual - ordenes_comp) / ordenes_comp * 100) if ordenes_comp > 0 else 0

unidades_actual = df_f['Quantity_SKU'].sum()

# --- CÁLCULO DE INVERSIÓN SIN INTERFERENCIA DE HORA ---
# Obtener todas las fechas YYYY-MM-DD únicas presentes en la vista actual
dias_filtrados_str = df_f['Fecha_Clean'].dropna().unique().tolist()

# Sumar la inversión que coincida con esos días
inv_meta_tot = df_meta[df_meta['Fecha_Clean'].isin(dias_filtrados_str)]['Inversion'].sum() if not df_meta.empty else 0.0
inv_google_tot = df_google[df_google['Fecha_Clean'].isin(dias_filtrados_str)]['Inversion'].sum() if not df_google.empty else 0.0

inversion_total = inv_meta_tot + inv_google_tot
roas = (ventas_actual / inversion_total) if inversion_total > 0 else 0.0

# ==========================================================
# 8. MÉTRICAS CLAVE (KPIs)
# ==========================================================
df_nuevos = df_f[df_f['Tipo_Cliente'] == 'Nuevo']
df_recurrentes = df_f[df_f['Tipo_Cliente'] == 'Recurrente']

nuevos_clientes_cant = df_nuevos['Client Document'].nunique() if 'Client Document' in df_nuevos.columns else 0
recurrentes_cant = df_recurrentes['Client Document'].nunique() if 'Client Document' in df_recurrentes.columns else 0

cac = (inversion_total / nuevos_clientes_cant) if nuevos_clientes_cant > 0 else 0.0
ingresos_recurrentes = df_recurrentes['Total Value'].sum()
ltv_recurrente = (ingresos_recurrentes / recurrentes_cant) if recurrentes_cant > 0 else 0.0
ratio_ltv_cac = (ltv_recurrente / cac) if cac > 0 else 0.0

ingresos_corta, ingresos_exacta = formatear_cifra_corta(ventas_actual, es_moneda=True)
inv_corta, inv_exacta = formatear_cifra_corta(inversion_total, es_moneda=True)
cac_corta, cac_exacta = formatear_cifra_corta(cac, es_moneda=True)
ltv_corta, ltv_exacta = formatear_cifra_corta(ltv_recurrente, es_moneda=True)
unidades_corta, unidades_exacta = formatear_cifra_corta(unidades_actual, es_moneda=False)

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

k1.metric("Ingresos Totales", ingresos_corta, f"{var_ventas:+.1f}% {etiqueta_comp}", help=f"Valor exacto: {ingresos_exacta}")
k2.metric("Ventas (Órdenes)", f"{ordenes_actual:,}", f"{var_ordenes:+.1f}% {etiqueta_comp}", help=f"Total órdenes: {ordenes_actual:,}")
k3.metric("Unidades Vendidas", unidades_corta, help=f"Unidades exactas: {unidades_exacta}")
k4.metric("Inversión Ads", inv_corta, help=f"Inversión acumulada Meta + Google: {inv_exacta}")
k5.metric("ROAS General", f"{roas:.2f} x", help="Retorno de Inversión: Ventas Totales / Inversión Ads")
k6.metric("CAC (Nuevos)", cac_corta if cac > 0 else "N/A", help=f"Costo de Adquisición: {cac_exacta}" if cac > 0 else "Sin datos")
k7.metric("LTV Recurrente", ltv_corta if ltv_recurrente > 0 else "N/A", f"Ratio: {ratio_ltv_cac:.1f}x", help=f"LTV Promedio: {ltv_exacta}" if ltv_recurrente > 0 else "Sin datos")

st.divider()

# ==========================================================
# 9. SECCIÓN DE ADQUISICIÓN Y CANALES
# ==========================================================
st.markdown("### Comportamiento de Clientes")

total_ord_sub = df_f['Order'].nunique()
ord_nuevos = df_f[df_f['Tipo_Cliente'] == 'Nuevo']['Order'].nunique()
ord_recurrentes = df_f[df_f['Tipo_Cliente'] == 'Recurrente']['Order'].nunique()

pct_nuevos = (ord_nuevos / total_ord_sub * 100) if total_ord_sub > 0 else 0
pct_recurrentes = (ord_recurrentes / total_ord_sub * 100) if total_ord_sub > 0 else 0

col_kpi_n, col_kpi_r = st.columns(2)
col_kpi_n.metric("👤 Clientes Nuevos", f"{ord_nuevos:,} órd.", f"{pct_nuevos:.1f}% del Total", delta_color="normal")
col_kpi_r.metric("🔄 Clientes Recurrentes", f"{ord_recurrentes:,} órd.", f"{pct_recurrentes:.1f}% del Total", delta_color="off")

st.divider()

st.subheader("Ventas por Canal (UtmSource)")

df_canal_agg = df_f.groupby('Canal_Estandar').agg(Ingresos=('Total Value', 'sum'), Ordenes=('Order', 'nunique')).reset_index()
row_directo = df_canal_agg[df_canal_agg['Canal_Estandar'] == 'Directo / Sin Datos']
ingresos_directo = row_directo['Ingresos'].sum() if not row_directo.empty else 0
pct_directo = (ingresos_directo / df_f['Total Value'].sum() * 100) if df_f['Total Value'].sum() > 0 else 0

st.info(f"📍 **Ventas en Directo / Sin Atribución:** ${ingresos_directo:,.0f} ({pct_directo:.1f}% del total)")

df_canal_bars = df_canal_agg[df_canal_agg['Canal_Estandar'] != 'Directo / Sin Datos'].sort_values('Ingresos', ascending=True)

fig_canal = px.bar(
    df_canal_bars, y='Canal_Estandar', x='Ingresos', orientation='h', text_auto='.2s',
    labels={'Canal_Estandar': '', 'Ingresos': 'Ingresos (COP)'}
)
fig_canal.update_traces(marker_color='#2563eb', hovertemplate="<b>Canal:</b> %{y}<br><b>Ingresos:</b> $%{x:,.0f}<extra></extra>")
fig_canal.update_layout(coloraxis_showscale=False, showlegend=False)
st.plotly_chart(fig_canal, use_container_width=True)

# ==========================================================
# 10. ANALÍTICA PROFUNDA & PESTAÑAS
# ==========================================================
st.markdown("### Analítica Profunda & Rendimiento")

tab_tendencias, tab_pauta, tab_carrito, tab_retencion = st.tabs([
    "Tendencia Temporal & Adquisición", 
    "Eficiencia de Pauta (ROAS)", 
    "Comportamiento de Compra (AOV & Pagos)",
    "Retención & Pareto (80/20)"
])

# ----------------------------------------------------------
# PESTAÑA 1: TENDENCIA TEMPORAL Y ADQUISICIÓN
# ----------------------------------------------------------
with tab_tendencias:
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("Evolución Diaria: Ventas vs Inversión Ads")
        
        # Agrupar Ventas por Día (utilizando la fecha limpia de VTEX)
        df_ventas_dia = df_f.groupby('Fecha_Clean').agg(
            Ingresos=('Total Value', 'sum'),
            Ordenes=('Order', 'nunique')
        ).reset_index()

        if not df_ventas_dia.empty:
            # 1. Obtener la fecha mínima y máxima del período seleccionado
            fecha_min_str = df_ventas_dia['Fecha_Clean'].min()
            fecha_max_str = df_ventas_dia['Fecha_Clean'].max()
            
            # 2. Generar calendario continuo en formato texto YYYY-MM-DD
            rango_fechas = pd.date_range(start=fecha_min_str, end=fecha_max_str, freq='D').strftime('%Y-%m-%d')
            df_timeline = pd.DataFrame({'Fecha_Clean': rango_fechas})
            
            # 3. Filtrar Meta y Google por el rango del período completo
            df_meta_f = df_meta[(df_meta['Fecha_Clean'] >= fecha_min_str) & (df_meta['Fecha_Clean'] <= fecha_max_str)] if not df_meta.empty else pd.DataFrame()
            df_goog_f = df_google[(df_google['Fecha_Clean'] >= fecha_min_str) & (df_google['Fecha_Clean'] <= fecha_max_str)] if not df_google.empty else pd.DataFrame()
            
            # Unir Inversión de Meta + Google
            df_inv_combined = pd.concat([df_meta_f, df_goog_f], ignore_index=True) if (not df_meta_f.empty or not df_goog_f.empty) else pd.DataFrame(columns=['Fecha_Clean', 'Inversion'])
            df_inv_dia = df_inv_combined.groupby('Fecha_Clean')['Inversion'].sum().reset_index() if not df_inv_combined.empty else pd.DataFrame(columns=['Fecha_Clean', 'Inversion'])

            # 4. Merge sobre la línea de tiempo maestra continua
            df_tendencia = pd.merge(df_timeline, df_ventas_dia, on='Fecha_Clean', how='left')
            df_tendencia = pd.merge(df_tendencia, df_inv_dia, on='Fecha_Clean', how='left').fillna(0)
            df_tendencia = df_tendencia.sort_values('Fecha_Clean')

            # 5. Renderizado del Gráfico Limpio
            fig_tendencia = go.Figure()

            # Barras tenues de Inversión Ads
            fig_tendencia.add_trace(
                go.Bar(
                    x=df_tendencia['Fecha_Clean'], 
                    y=df_tendencia['Inversion'], 
                    name="Inversión Ads ($)", 
                    marker_color='rgba(245, 158, 11, 0.45)', # Naranja translúcido
                    hovertemplate="<b>Fecha:</b> %{x}<br><b>Inversión:</b> $%{y:,.0f}<extra></extra>"
                )
            )

            # Línea prominente de Ventas
            fig_tendencia.add_trace(
                go.Scatter(
                    x=df_tendencia['Fecha_Clean'], 
                    y=df_tendencia['Ingresos'], 
                    name="Ventas ($ COP)", 
                    line=dict(color='#2563eb', width=2.5), # Azul sólido
                    hovertemplate="<b>Ventas:</b> $%{y:,.0f}<extra></extra>"
                )
            )

            fig_tendencia.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=30, b=10),
                height=380,
                xaxis_title="",
                yaxis_title="Monto ($ COP)"
            )

            st.plotly_chart(fig_tendencia, use_container_width=True)
        else:
            st.info("No hay datos de ventas en el período seleccionado.")

    # 2. Gráfico de Barras Apiladas: Adquisición Temporal (Nuevos vs Recurrentes)
    with col_t2:
        st.subheader("Adquisición Temporal: Nuevos vs Recurrentes")
        
        df_acq_dia = df_f.groupby(['Fecha_Clean', 'Tipo_Cliente']).agg(Ordenes=('Order', 'nunique')).reset_index()
        
        fig_apiladas = px.bar(
            df_acq_dia,
            x='Fecha_Clean',
            y='Ordenes',
            color='Tipo_Cliente',
            color_discrete_map={'Nuevo': '#2563eb', 'Recurrente': '#10b981'},
            labels={'Ordenes': 'Órdenes', 'Fecha_Clean': ''}
        )
        fig_apiladas.update_layout(
            barmode='stack', 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        fig_apiladas.update_traces(hovertemplate="<b>Fecha:</b> %{x}<br><b>Órdenes:</b> %{y:,d}<extra></extra>")
        st.plotly_chart(fig_apiladas, use_container_width=True)

# ----------------------------------------------------------
# PESTAÑA 2: RENDIMIENTO FINANCIERO Y EFICIENCIA DE PAUTA
# ----------------------------------------------------------
with tab_pauta:
    st.subheader("Eficiencia de Pauta: Inversión vs ROAS por Canal")
    
    ventas_meta = df_f[df_f['Canal_Estandar'].isin(['Facebook', 'Instagram'])]['Total Value'].sum()
    ventas_google = df_f[df_f['Canal_Estandar'] == 'Google']['Total Value'].sum()
    
    data_pauta = [
        {'Canal': 'Meta (FB / IG)', 'Inversion': inv_meta_tot, 'Ventas': ventas_meta, 'ROAS': (ventas_meta / inv_meta_tot) if inv_meta_tot > 0 else 0},
        {'Canal': 'Google Ads', 'Inversion': inv_google_tot, 'Ventas': ventas_google, 'ROAS': (ventas_google / inv_google_tot) if inv_google_tot > 0 else 0}
    ]
    df_roas_canal = pd.DataFrame(data_pauta)
    
    fig_roas = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Barras: Inversión ($) con texto INSIDE en color blanco
    fig_roas.add_trace(
        go.Bar(
            x=df_roas_canal['Canal'], 
            y=df_roas_canal['Inversion'], 
            name="Inversión ($)", 
            marker_color='#2563eb', 
            text=df_roas_canal['Inversion'].apply(lambda x: f"${x:,.0f}"), 
            textposition='inside', # Texto dentro de la barra
            textfont=dict(color='white', size=13, family='Arial Black')
        ),
        secondary_y=False
    )
    
    # Línea: ROAS (x) flotando arriba
    fig_roas.add_trace(
        go.Scatter(
            x=df_roas_canal['Canal'], 
            y=df_roas_canal['ROAS'], 
            name="ROAS (x)", 
            mode='lines+markers+text', 
            text=df_roas_canal['ROAS'].apply(lambda x: f"{x:.2f}x"), 
            textposition='top center', 
            line=dict(color='#ef4444', width=3), 
            marker=dict(size=12, color='#ef4444')
        ),
        secondary_y=True
    )
    
    fig_roas.update_layout(
        hovermode="x unified", 
        showlegend=True, 
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_roas.update_yaxes(title_text="Inversión ($ COP)", secondary_y=False)
    fig_roas.update_yaxes(title_text="ROAS (Múltiplo)", secondary_y=True, range=[0, max(df_roas_canal['ROAS'].max()*1.3, 2)])
    
    st.plotly_chart(fig_roas, use_container_width=True)
# ----------------------------------------------------------
# PESTAÑA 3: COMPORTAMIENTO DE COMPRA (AOV & PAGOS LIMPIOS)
# ----------------------------------------------------------
with tab_carrito:
    col_aov, col_dona = st.columns(2)
    
    # 1. Ticket Medio (AOV) por Medio de Pago
    with col_aov:
        st.subheader("Ticket Medio (AOV) por Medio de Pago")
        
        # Copia local limpia
        df_pago_clean = df_f.copy()
        df_pago_clean['Payment System Name'] = df_pago_clean['Payment System Name'].fillna('No Especificado').astype(str)
        
        df_aov = df_pago_clean.groupby('Payment System Name').agg(
            Ingresos=('Total Value', 'sum'),
            Ordenes=('Order', 'nunique')
        ).reset_index()
        
        df_aov['AOV'] = (df_aov['Ingresos'] / df_aov['Ordenes']).fillna(0)
        df_aov = df_aov.sort_values('AOV', ascending=True).tail(8) # Top 8
        
        fig_aov = px.bar(
            df_aov,
            y='Payment System Name',
            x='AOV',
            orientation='h',
            text_auto='.2s',
            labels={'Payment System Name': '', 'AOV': 'Valor Promedio Orden ($)'},
            color_discrete_sequence=['#10b981'] # Verde financiero
        )
        fig_aov.update_traces(
            textposition='inside',
            hovertemplate="<b>Medio:</b> %{y}<br><b>AOV:</b> $%{x:,.0f}<extra></extra>"
        )
        fig_aov.update_layout(height=380)
        st.plotly_chart(fig_aov, use_container_width=True)

    # 2. Gráfico de Dona: Participación por Medio de Pago (Agrupando <2% en Otros)
    with col_dona:
        st.subheader("Participación por Medio de Pago")
        
        df_pago_donuts = df_pago_clean.groupby('Payment System Name').agg(
            Ordenes=('Order', 'nunique')
        ).reset_index().sort_values('Ordenes', ascending=False)
        
        total_ordenes_pagos = df_pago_donuts['Ordenes'].sum()
        
        if total_ordenes_pagos > 0:
            df_pago_donuts['Pct'] = (df_pago_donuts['Ordenes'] / total_ordenes_pagos) * 100
            
            # Agrupar pagos < 2% en 'Otros'
            df_principales = df_pago_donuts[df_pago_donuts['Pct'] >= 2.0].copy()
            df_menores = df_pago_donuts[df_pago_donuts['Pct'] < 2.0]
            
            if not df_menores.empty:
                otros_row = pd.DataFrame([{
                    'Payment System Name': 'Otros',
                    'Ordenes': df_menores['Ordenes'].sum(),
                    'Pct': df_menores['Pct'].sum()
                }])
                df_pago_donuts = pd.concat([df_principales, otros_row], ignore_index=True)
            else:
                df_pago_donuts = df_principales

        fig_dona = px.pie(
            df_pago_donuts,
            names='Payment System Name',
            values='Ordenes',
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_dona.update_traces(
            textinfo='percent+label',
            insidetextorientation='radial',
            hovertemplate="<b>Medio:</b> %{label}<br><b>Órdenes:</b> %{value:,d}<extra></extra>"
        )
        fig_dona.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_dona, use_container_width=True)

# ----------------------------------------------------------
# PESTAÑA 4: RETENCIÓN, PARETO & FRECUENCIA
# ----------------------------------------------------------
with tab_retencion:
    col_p1, col_p2 = st.columns(2)
    
    # 1. PARTE SUPERIOR IZQUIERDA: PARETO REAJUSTADO
    with col_p1:
        st.subheader("⚖️ Concentración de Ventas (Pareto)")
        
        res_pareto = generar_grafico_pareto(df_f)
        if res_pareto:
            fig_p, pct_alc = res_pareto
            st.caption(f"El Top 15 de productos concentra el **{pct_alc:.1f}%** del total de ingresos del período.")
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Sin datos suficientes para calcular Pareto.")

    # 2. PARTE SUPERIOR DERECHA: COHORTES (SIN MES 0 Y CON MAPA DE CALOR INTENSO)
    with col_p2:
        st.subheader("🔄 Cohortes de Retención (% Recompra)")
        st.caption("Porcentaje de clientes que vuelven a comprar a partir del **Mes 1** (excluyendo la 1ª compra).")
        
        matrix_ret = generar_matriz_cohortes(df_vtex)
        
        if not matrix_ret.empty and matrix_ret.shape[1] > 1:
            # OCULTAR MES 0: Tomamos de la columna 1 en adelante (Mes 1, Mes 2...)
            matrix_show = matrix_ret.tail(8).iloc[:, 1:7]
            
            y_labels = [str(idx) for idx in matrix_show.index]
            x_labels = [f"Mes {i}" for i in matrix_show.columns]

            # Escala de calor intensa para destacar valores pequeños (ej. 1% al 6%)
            fig_cohort = px.imshow(
                matrix_show.values,
                labels=dict(x="Meses Después de 1ª Compra", y="Cohorte (1ª Compra)", color="% Recompra"),
                x=x_labels,
                y=y_labels,
                text_auto=".1f",
                color_continuous_scale="Blues", # Mapa de azul pálido a azul marino intenso
                aspect="auto"
            )
            
            fig_cohort.update_layout(height=420, coloraxis_showscale=False)
            fig_cohort.update_traces(hovertemplate="<b>Cohorte:</b> %{y}<br><b>%{x}:</b> %{z:.1f}% de recompra<extra></extra>")
            st.plotly_chart(fig_cohort, use_container_width=True)
        else:
            st.info("No hay datos de recompras suficientes para generar la matriz de cohortes.")

    st.divider()

    # 3. PARTE INFERIOR COMPLETA: TIEMPO ENTRE COMPRAS
    st.subheader("🕒 Frecuencia de Compra (Tiempo Promedio Entre Órdenes)")
    st.caption("Días promedio que transcurren para que un cliente realice su siguiente transacción.")
    
    fig_frecuencia = generar_grafico_tiempo_entre_compras(df_vtex)
    if fig_frecuencia:
        st.plotly_chart(fig_frecuencia, use_container_width=True)
    else:
        st.info("No se registraron recompras en la base para medir el tiempo entre transacciones.")

# ==========================================================
# 11. TOP PRODUCTOS & TABLA
# ==========================================================
st.subheader(f"Top Productos — {marca_sel}")
df_top_prod = obtener_top_productos(df_f, marca_filtro=marca_sel, top_n=10)

if not df_top_prod.empty:
    max_uds = df_top_prod['Unidades'].max() if df_top_prod['Unidades'].max() > 0 else 1
    with st.container():
        for idx, row in df_top_prod.reset_index(drop=True).iterrows():
            posicion = idx + 1
            nombre_prod = row['SKU_Individual']
            uds = int(row['Unidades'])
            ordenes = int(row['Ordenes'])
            porcentaje_barra = int((uds / max_uds) * 100)
            color_num = "#f59e0b" if posicion <= 3 else "#9ca3af"

            st.markdown(f"""
            <div style="margin-bottom: 12px; padding: 4px 8px; border-bottom: 1px solid #f3f4f6;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-weight: bold; color: {color_num}; margin-right: 10px; font-size: 1.1em;">{posicion}</span>
                    <span style="flex-grow: 1; font-weight: 500; font-size: 0.95em; color: #374151;">{nombre_prod}</span>
                    <span style="font-weight: bold; color: #1f2937; font-size: 0.9em; margin-left: 10px;">
                        {uds:,} uds <span style="font-weight: normal; color: #6b7280; font-size: 0.85em;">({ordenes:,} órd.)</span>
                    </span>
                </div>
                <div style="width: 100%; background-color: #f3f4f6; height: 6px; border-radius: 3px;">
                    <div style="width: {porcentaje_barra}%; background-color: #f59e0b; height: 6px; border-radius: 3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info(f"No se encontraron productos para la marca '{marca_sel}'.")

st.subheader("📋 Tabla de Atribución Detallada")
tabla_resumen = df_f.groupby(['Canal_Estandar', 'Origen_Estandar', 'Tipo_Cliente']).agg(
    Órdenes=('Order', 'nunique'),
    Unidades=('Quantity_SKU', 'sum'),
    Ingresos_Totales=('Total Value', 'sum')
).reset_index().sort_values('Ingresos_Totales', ascending=False)

st.dataframe(
    tabla_resumen.style.format({'Órdenes': '{:,}', 'Unidades': '{:,.0f}', 'Ingresos_Totales': '${:,.0f}'}),
    use_container_width=True
)