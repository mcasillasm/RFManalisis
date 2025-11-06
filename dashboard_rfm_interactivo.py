import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# === CONFIGURACIÓN ===
st.set_page_config(
    page_title="RFM Customer Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
st.markdown("""
# RFM Customer Segmentation Dashboard
### Segmenta, analiza y actúa sobre tus clientes en tiempo real
""")

# === SIDEBAR - FILTROS ===
st.sidebar.header("Filtros de Segmentación")
segmento = st.sidebar.multiselect(
    "Segmento RFM",
    options=['Campeones', 'Clientes Leales', 'Nuevos / Potenciales', 'En Riesgo', 'Perdidos', 'Hibernando'],
    default=['Campeones', 'Clientes Leales', 'En Riesgo']
)

recency_range = st.sidebar.slider(
    "Recencia (días sin compra)",
    min_value=0, max_value=365, value=(0, 120)
)

monetary_range = st.sidebar.slider(
    "Valor Monetario (MXN)",
    min_value=0, max_value=50000, value=(500, 20000)
)

# === GENERAR / CARGAR DATOS ===
@st.cache_data
def generar_rfm():
    np.random.seed(42)
    hoy = datetime(2025, 11, 6)
    fecha_ref = hoy + timedelta(days=1)
    n_clientes = 100
    data = []

    for cliente in range(1, n_clientes + 1):
        n_compras = np.random.poisson(5)
        if n_compras == 0: n_compras = 1
        monto_promedio = np.random.lognormal(8, 1)
        
        for _ in range(n_compras):
            dias = np.random.randint(1, 365)
            fecha = hoy - timedelta(days=dias)
            monto = max(100, np.random.normal(monto_promedio, monto_promedio * 0.5))
            data.append({
                'Cliente_ID': f'C{cliente:03d}',
                'Fecha_Compra': fecha,
                'Monto_MXN': round(monto, 2)
            })
    
    df = pd.DataFrame(data)
    df['Fecha_Compra'] = pd.to_datetime(df['Fecha_Compra'])

    rfm = df.groupby('Cliente_ID').agg({
        'Fecha_Compra': lambda x: (fecha_ref - x.max()).days,
        'Cliente_ID': 'count',
        'Monto_MXN': 'sum'
    }).rename(columns={
        'Fecha_Compra': 'Recency',
        'Cliente_ID': 'Frequency',
        'Monto_MXN': 'Monetary'
    }).reset_index()

    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm[['R_Score', 'F_Score', 'M_Score']] = rfm[['R_Score', 'F_Score', 'M_Score']].astype(int)
    rfm['RFM_Score'] = rfm['R_Score']*100 + rfm['F_Score']*10 + rfm['M_Score']

    def segmento(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4: return 'Campeones'
        elif r >= 3 and f >= 3: return 'Clientes Leales'
        elif r >= 4 and f <= 2: return 'Nuevos / Potenciales'
        elif r <= 2 and f >= 3: return 'En Riesgo'
        elif r <= 2 and f <= 2: return 'Perdidos'
        else: return 'Hibernando'
    
    rfm['Segmento'] = rfm.apply(segmento, axis=1)
    return rfm

rfm = generar_rfm()

# Aplicar filtros
df_f = rfm[
    rfm['Segmento'].isin(segmento) &
    rfm['Recency'].between(recency_range[0], recency_range[1]) &
    rfm['Monetary'].between(monetary_range[0], monetary_range[1])
]

# === KPIs ===
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Clientes", len(df_f))
col2.metric("Ingresos Totales", f"${df_f['Monetary'].sum():,.0f}")
col3.metric("Compra Promedio", f"${df_f['Monetary'].mean():,.0f}")
col4.metric("Recencia Promedio", f"{df_f['Recency'].mean():.0f} días")

# === GRÁFICO 1: Distribución de Segmentos ===
st.markdown("---")
st.subheader("Distribución de Segmentos RFM")
seg_count = df_f['Segmento'].value_counts().reset_index()
seg_count.columns = ['Segmento', 'Clientes']

fig_pie = px.pie(
    seg_count, values='Clientes', names='Segmento',
    color_discrete_map={
        'Campeones': '#10b981', 'Clientes Leales': '#3b82f6',
        'Nuevos / Potenciales': '#8b5cf6', 'En Riesgo': '#f59e0b',
        'Perdidos': '#ef4444', 'Hibernando': '#6b7280'
    },
    hole=0.4
)
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_pie, use_container_width=True)

# === GRÁFICO 2: Matriz RF ===
st.subheader("Matriz Recency vs Frequency")
rf_matrix = df_f.pivot_table(
    index='R_Score', columns='F_Score',
    values='Cliente_ID', aggfunc='count', fill_value=0
)
fig_heatmap = px.imshow(
    rf_matrix.values,
    x=rf_matrix.columns, y=rf_matrix.index,
    text_auto=True, aspect="auto",
    color_continuous_scale='Blues',
    labels=dict(x="Frequency Score", y="Recency Score", color="Clientes")
)
fig_heatmap.update_layout(height=500)
st.plotly_chart(fig_heatmap, use_container_width=True)

# === GRÁFICO 3: Top 10 Clientes ===
st.subheader("Top 10 Clientes por Valor Monetario")
top10 = df_f.nlargest(10, 'Monetary')[['Cliente_ID', 'Monetary', 'Frequency', 'Recency', 'Segmento']]
top10['Monetary'] = top10['Monetary'].apply(lambda x: f"${x:,.0f}")
st.dataframe(top10, use_container_width=True)

# === TABLA COMPLETA ===
st.subheader("Datos Completos (Filtrados)")
st.dataframe(df_f.style.format({
    'Monetary': '${:,.0f}',
    'Recency': '{:.0f} días',
    'Frequency': '{:.0f}'
}), use_container_width=True)

# === BOTÓN EXPORTAR ===
csv = df_f.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar Datos Filtrados (CSV)",
    data=csv,
    file_name='rfm_clientes_filtrados.csv',
    mime='text/csv'
)

# === FOOTER ===
st.markdown("---")
st.caption("""
**RFM Dashboard** • Desarrollado con Streamlit | 
[Ver código en GitHub](https://github.com/tu-usuario/rfm-customer-segmentation) | 
[Contactar para análisis personalizado](mailto:mcasillasm@outlook.com)
""")