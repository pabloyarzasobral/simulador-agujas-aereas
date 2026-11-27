import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import streamlit as st

# ==========================================
# 1. CONFIGURACIÓN GENERAL DE LA APLICACIÓN
# ==========================================
st.set_page_config(page_title="Simulador Aguja Aérea Ferroviaria", layout="wide")
st.title("🚆 Simulador Integral de Replanteo y Dinámica de Agujas Aéreas")
st.markdown(
    "Herramienta de ingeniería ferroviaria para análisis geométrico multivano, "
    "interacción pantógrafo-catenaria y gálibo de mesilla."
)

# ==========================================
# 2. BARRA LATERAL: PARÁMETROS Y PLANTILLAS
# ==========================================
st.sidebar.header("1. Parámetros Operativos")
v_kmh = st.sidebar.slider("Velocidad del tren (km/h)", min_value=40, max_value=300, value=140, step=10)
T_hilo = st.sidebar.slider("Tensión mecánica del hilo (kN)", min_value=10.0, max_value=25.0, value=15.0, step=0.5) * 1000.0
F_estatica = st.sidebar.slider("Fuerza estática pantógrafo (N)", min_value=60, max_value=140, value=90, step=5)
itinerario = st.sidebar.radio("Itinerario del tren a simular", ["Vía Directa", "Vía Desviada"])

st.sidebar.header("2. Plantillas de Desvío (Presets)")
preset = st.sidebar.selectbox(
    "Cargar configuración tipo",
    ["Desvío P50 (Tangente 0.09 - Convencional)", "Desvío P90 (Tangente 0.0625 - Alta Velocidad)"],
)


def obtener_tabla_preset(tipo):
    if "P50" in tipo:
        return pd.DataFrame([
            {"Poste": "P1", "Pos_X (m)": 0.0,   "Tipo": "Normal",    "Desc_Directa (m)": -0.20, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)": np.nan, "Alt_Desviada (m)": np.nan},
            {"Poste": "P2", "Pos_X (m)": 45.0,  "Tipo": "Aguja",     "Desc_Directa (m)":  0.20, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)":  0.65,  "Alt_Desviada (m)": 5.34},
            {"Poste": "P3", "Pos_X (m)": 85.0,  "Tipo": "Cruce",     "Desc_Directa (m)": -0.15, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)": -0.15,  "Alt_Desviada (m)": 5.30},
            {"Poste": "P4", "Pos_X (m)": 130.0, "Tipo": "Elevación", "Desc_Directa (m)":  0.20, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)": -0.85,  "Alt_Desviada (m)": 5.65},
            {"Poste": "P5", "Pos_X (m)": 175.0, "Tipo": "Normal",    "Desc_Directa (m)": -0.20, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)": np.nan, "Alt_Desviada (m)": np.nan},
        ])
    return pd.DataFrame([
        {"Poste": "P1", "Pos_X (m)": 0.0,   "Tipo": "Normal",    "Desc_Directa (m)": -0.20, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)": np.nan, "Alt_Desviada (m)": np.nan},
        {"Poste": "P2", "Pos_X (m)": 55.0,  "Tipo": "Aguja",     "Desc_Directa (m)":  0.20, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)":  0.55,  "Alt_Desviada (m)": 5.33},
        {"Poste": "P3", "Pos_X (m)": 110.0, "Tipo": "Cruce",     "Desc_Directa (m)": -0.10, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)": -0.10,  "Alt_Desviada (m)": 5.30},
        {"Poste": "P4", "Pos_X (m)": 165.0, "Tipo": "Elevación", "Desc_Directa (m)":  0.20, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)": -0.75,  "Alt_Desviada (m)": 5.60},
        {"Poste": "P5", "Pos_X (m)": 220.0, "Tipo": "Normal",    "Desc_Directa (m)": -0.20, "Alt_Directa (m)": 5.30, "Desc_Desviada (m)": np.nan, "Alt_Desviada (m)": np.nan},
    ])


if "ultimo_preset" not in st.session_state or st.session_state.ultimo_preset != preset:
    st.session_state.df_postes = obtener_tabla_preset(preset)
    st.session_state.ultimo_preset = preset

# ==========================================
# 3. TABLA DE REPLANTEO EDITABLE Y EXPORTACIÓN
# ==========================================
col_t1, col_t2 = st.columns([4, 1])
with col_t1:
    st.subheader("📋 Tabla de Replanteo por Poste (Editable)")
    st.caption(
        "Modifica distancias entre postes (Pos_X), tipos de poste, descentramientos o alturas. "
        "Deja vacío (NaN) donde no exista hilo desviado."
    )

df_editado = st.data_editor(
    st.session_state.df_postes,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Tipo": st.column_config.SelectboxColumn(
            "Tipo de Poste", options=["Normal", "Aguja", "Cruce", "Elevación"], required=True
        ),
        "Pos_X (m)": st.column_config.NumberColumn("Posición X (m)", format="%.1f", step=5.0),
        "Desc_Directa (m)": st.column_config.NumberColumn("Descentr. Directa (m)", format="%.2f", step=0.05),
        "Alt_Directa (m)": st.column_config.NumberColumn("Altura Directa (m)", format="%.2f", step=0.02),
        "Desc_Desviada (m)": st.column_config.NumberColumn("Descentr. Desviada (m)", format="%.2f", step=0.05),
        "Alt_Desviada (m)": st.column_config.NumberColumn("Altura Desviada (m)", format="%.2f", step=0.02),
    },
)

with col_t2:
    st.write("")
    st.write("")
    csv_data = df_editado.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Exportar Replanteo CSV",
        data=csv_data,
        file_name="replanteo_aguja_aerea.csv",
        mime="text/csv",
    )

df_clean = df_editado.dropna(subset=["Pos_X (m)", "Desc_Directa (m)", "Alt_Directa (m)"]).sort_values("Pos_X (m)")

if len(df_clean) < 2:
    st.error("Debes definir al menos 2 postes válidos para ejecutar la simulación.")
    st.stop()

# ==========================================
# 4. INTERPOLACIÓN GEOMÉTRICA MULTIVANO
# ==========================================
x_postes = df_clean["Pos_X (m)"].values
x_min, x_max = float(x_postes[0]), float(x_postes[-1])
if x_max <= x_min:
    st.error("La posición X del último poste debe ser mayor que la del primer poste.")
    st.stop()

x_sim = np.linspace(x_min, x_max, 1000)

f_y_dir = interp1d(x_postes, df_clean["Desc_Directa (m)"].values, kind="linear", fill_value="extrapolate")
f_z_dir = interp1d(x_postes, df_clean["Alt_Directa (m)"].values, kind="linear", fill_value="extrapolate")
y_directo = f_y_dir(x_sim)
z_directo = f_z_dir(x_sim)

df_desv = df_clean.dropna(subset=["Desc_Desviada (m)", "Alt_Desviada (m)"])
tiene_desviada = len(df_desv) >= 2

if tiene_desviada:
    x_desv_postes = df_desv["Pos_X (m)"].values
    x_desv_min, x_desv_max = float(x_desv_postes[0]), float(x_desv_postes[-1])
    f_y_desv = interp1d(x_desv_postes, df_desv["Desc_Desviada (m)"].values, kind="linear")
    f_z_desv = interp1d(x_desv_postes, df_desv["Alt_Desviada (m)"].values, kind="linear")

    mask_desv = (x_sim >= x_desv_min) & (x_sim <= x_desv_max)
    x_sim_desv = x_sim[mask_desv]
    y_desviado = f_y_desv(x_sim_desv)
    z_desviado = f_z_desv(x_sim_desv)
else:
    x_desv_min, x_desv_max = np.nan, np.nan
    x_sim_desv, y_desviado, z_desviado = np.array([]), np.array([]), np.array([])

gradiente_dir = np.gradient(z_directo, x_sim) * 1000.0
gradiente_desv = np.gradient(z_desviado, x_sim_desv) * 1000.0 if tiene_desviada and len(x_sim_desv) > 1 else np.array([])

# ==========================================
# 5. MOTOR FÍSICO: DINÁMICA PANTÓGRAFO-CATENARIA
# ==========================================
if itinerario == "Vía Desviada" and tiene_desviada:
    z_objetivo_arr = np.copy(z_directo)
    for idx, x_val in enumerate(x_sim):
        if x_desv_min <= x_val <= x_desv_max:
            z_d_val = float(f_z_desv(x_val))
            if z_d_val <= z_directo[idx] + 0.02:
                z_objetivo_arr[idx] = z_d_val
else:
    z_objetivo_arr = np.copy(z_directo)

z_rel_hilo = z_objetivo_arr - 5.30
f_z_rel = interp1d(x_sim, z_rel_hilo, kind="linear", fill_value="extrapolate")


def calcular_rigidez_catenaria(pos_x, T_mec):
    idx_vano = np.searchsorted(x_postes, pos_x) - 1
    idx_vano = np.clip(idx_vano, 0, len(x_postes) - 2)

    x_izq = x_postes[idx_vano]
    x_der = x_postes[idx_vano + 1]
    L_local = max(10.0, x_der - x_izq)

    k_base = (T_mec / 1000.0) * (60.0 / L_local) * 1000.0
    factor_vano = 1.0 + 0.40 * np.cos(2 * np.pi * (pos_x - x_izq) / L_local)

    factor_especial = 0.0
    for _, row in df_clean.iterrows():
        if row["Tipo"] in ["Aguja", "Cruce", "Elevación"]:
            dist = abs(pos_x - row["Pos_X (m)"])
            peso = 0.45 if row["Tipo"] == "Cruce" else 0.25
            factor_especial += peso * np.exp(-((dist / 2.5) ** 2))

    return k_base * (factor_vano + factor_especial)


def modelo_dinamico_multivano(t, estado, v_tren, T_mec, F_est):
    z1, vz1, z2, vz2 = estado
    pos_x = float(np.clip(x_min + v_tren * t, x_min, x_max))

    m1, m2 = 7.2, 15.0        # Masas mesilla y bastidor (kg)
    c1, c2 = 12.0, 50.0       # Amortiguamiento (Ns/m)
    k1, k2 = 4200.0, 50.0     # Rigidez (N/m)

    k_cat = calcular_rigidez_catenaria(pos_x, T_mec)
    z_hilo_actual = float(f_z_rel(pos_x))

    penetracion = z1 - z_hilo_actual
    F_contacto = max(0.0, k_cat * penetracion)

    az1 = (-F_contacto - c1 * (vz1 - vz2) - k1 * (z1 - z2)) / m1
    az2 = (F_est + c1 * (vz1 - vz2) + k1 * (z1 - z2) - c2 * vz2 - k2 * z2) / m2

    return [vz1, az1, vz2, az2]


v_ms = v_kmh / 3.6
t_total = (x_max - x_min) / v_ms
t_eval = np.linspace(0, t_total, 1000)

k_ini = calcular_rigidez_catenaria(x_min, T_hilo)
z_ini = float(f_z_rel(x_min)) + (F_estatica / k_ini)

sol = solve_ivp(
    modelo_dinamico_multivano,
    [0, t_total],
    [z_ini, 0.0, z_ini, 0.0],
    args=(v_ms, T_hilo, F_estatica),
    t_eval=t_eval,
    method="RK45",
)

F_contacto_arr = []
for i, pos_x in enumerate(x_sim):
    k_c = calcular_rigidez_catenaria(pos_x, T_hilo)
    pen = sol.y[0][i] - float(f_z_rel(pos_x))
    F_contacto_arr.append(max(0.0, k_c * pen))
F_contacto_arr = np.array(F_contacto_arr)

# ==========================================
# 6. INDICADORES KPI Y VALIDACIÓN NORMATIVA
# ==========================================
Fm = float(np.mean(F_contacto_arr))
sigma = float(np.std(F_contacto_arr))
F_min = float(np.min(F_contacto_arr))
F_max = float(np.max(F_contacto_arr))
grad_max = max(
    float(np.max(np.abs(gradiente_dir))),
    float(np.max(np.abs(gradiente_desv))) if len(gradiente_desv) > 0 else 0.0,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fuerza Media (Fm)", f"{Fm:.1f} N")
col2.metric("Desviación Típica (σ)", f"{sigma:.1f} N", delta=f"Límite EN 50367: {0.3 * Fm:.1f} N", delta_color="inverse")
col3.metric("Fuerza Mín / Máx", f"{F_min:.0f} N / {F_max:.0f} N")
col4.metric("Pendiente Máx. Hilo", f"{grad_max:.2f} ‰", delta="Recomendado < 2.0 ‰", delta_color="inverse")

if F_min <= 1.0:
    st.error(
        "⚠️ PÉRDIDA DE CONTACTO (ARCO ELÉCTRICO): El pantógrafo despega del hilo. "
        "Revisa pendientes en postes de elevación o reduce velocidad."
    )
elif sigma > 0.3 * Fm:
    st.warning("⚠️ INESTABILIDAD DINÁMICA (EN 50367): σ supera el 30% de Fm. Revisa longitud de vanos o rigidez en el cruce.")
elif grad_max > 2.5:
    st.warning("⚠️ PENDIENTE EXCESIVA: El cambio de altura entre postes supera los 2.5 mm/m.")
else:
    st.success("✅ REPLANTEO Y DINÁMICA CORRECTOS: Cumple criterios operativos y norma EN 50367.")

# ==========================================
# 7. PESTAÑAS GRÁFICAS INTERACTIVAS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🚊 Sección Transversal YZ (Mesilla Pantógrafo)",
    "📈 Fuerza de Contacto",
    "📐 Planta y Alzado",
    "🌐 Vista Espacial 3D",
])

# ------------------------------------------
# PESTAÑA 1: SECCIÓN TRANSVERSAL YZ + ANIMACIÓN
# ------------------------------------------
with tab1:
    st.subheader("Vista Frontal del Tren (Plano YZ): Interacción Mesilla - Hilos")
    col_yz1, col_yz2 = st.columns([2.2, 1])

    with col_yz2:
        perfil_panto = st.selectbox(
            "Perfil de mesilla",
            ["UIC 1600 mm (Alta Velocidad)", "Ibérico / 1950 mm (Convencional)"],
        )
        default_pk = float(np.clip(float(x_postes[len(x_postes) // 2]), x_min, x_max))
        if "slider_pk_widget" not in st.session_state:
            st.session_state.slider_pk_widget = default_pk

        col_btn1, col_btn2 = st.columns(2)
        iniciar_anim = col_btn1.button("▶️ Simular paso tren", use_container_width=True)
        reset_anim = col_btn2.button("⏮️ Inicio (PK 0)", use_container_width=True)

        if reset_anim:
            st.session_state.slider_pk_widget = float(np.clip(x_min, x_min, x_max))

        st.session_state.slider_pk_widget = float(np.clip(st.session_state.slider_pk_widget, x_min, x_max))

        pk_actual = st.slider(
            "Posición del tren PK (m)",
            min_value=x_min,
            max_value=x_max,
            step=1.0,
            key="slider_pk_widget",
        )

    if "1600" in perfil_panto:
        ancho_carbon, ancho_total, caida_cuerno = 0.45, 0.80, 0.12
    else:
        ancho_carbon, ancho_total, caida_cuerno = 0.55, 0.975, 0.14

    def generar_figura_yz(pk_val):
        idx_pk = int(np.argmin(np.abs(x_sim - pk_val)))
        z_mesilla_din = 5.30 + float(sol.y[0][idx_pk])
        fc_val = float(F_contacto_arr[idx_pk])

        y_dir_pt = float(f_y_dir(pk_val))
        z_dir_pt = max(float(f_z_dir(pk_val)), z_mesilla_din)

        existe_desv = bool(tiene_desviada and (x_desv_min <= pk_val <= x_desv_max))
        y_desv_pt, z_desv_pt = None, None
        alerta_eng = False

        if existe_desv:
            y_desv_pt = float(f_y_desv(pk_val))
            z_desv_raw = float(f_z_desv(pk_val))
            if abs(y_desv_pt) <= ancho_carbon and z_desv_raw <= z_mesilla_din:
                z_desv_pt = z_mesilla_din
            else:
                z_desv_pt = z_desv_raw

            if ancho_carbon < abs(y_desv_pt) <= ancho_total:
                pendiente_cuerno = caida_cuerno / (ancho_total - ancho_carbon)
                z_cuerno_en_y = z_mesilla_din - pendiente_cuerno * (abs(y_desv_pt) - ancho_carbon)
                if z_desv_pt < z_cuerno_en_y:
                    alerta_eng = True

        fig = go.Figure()
        # Bastidor
        fig.add_trace(
            go.Scatter(
                x=[-0.35, 0.0, 0.35],
                y=[z_mesilla_din - 0.35, z_mesilla_din - 0.15, z_mesilla_din - 0.35],
                mode="lines",
                name="Bastidor",
                line=dict(color="gray", width=4),
            )
        )
        # Cuernos
        fig.add_trace(
            go.Scatter(
                x=[-ancho_total, -ancho_carbon],
                y=[z_mesilla_din - caida_cuerno, z_mesilla_din],
                mode="lines",
                name="Cuernos",
                line=dict(color="#ff7f0e", width=7),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[ancho_carbon, ancho_total],
                y=[z_mesilla_din, z_mesilla_din - caida_cuerno],
                mode="lines",
                showlegend=False,
                line=dict(color="#ff7f0e", width=7),
            )
        )
        # Banda de carbón
        fig.add_trace(
            go.Scatter(
                x=[-ancho_carbon, ancho_carbon],
                y=[z_mesilla_din, z_mesilla_din],
                mode="lines",
                name="Banda de Carbón",
                line=dict(color="#111111", width=9),
            )
        )
        # Hilo Directo
        fig.add_trace(
            go.Scatter(
                x=[y_dir_pt],
                y=[z_dir_pt + 0.006],
                mode="markers+text",
                name="Hilo Directo",
                text=["Directa"],
                textposition="top center",
                marker=dict(size=14, color="#1f77b4", line=dict(width=2, color="white")),
            )
        )
        # Hilo Desviado
        if existe_desv:
            fig.add_trace(
                go.Scatter(
                    x=[y_desv_pt],
                    y=[z_desv_pt + 0.006],
                    mode="markers+text",
                    name="Hilo Desviado",
                    text=["Desviada"],
                    textposition="top center",
                    marker=dict(size=14, color="#d62728", line=dict(width=2, color="white")),
                )
            )

        fig.add_vline(x=0, line_dash="dash", line_color="lightgray", annotation_text="Eje Vía")
        fig.update_layout(
            title=f"Sección Transversal YZ en PK = {pk_val:.1f} m | Fuerza Contacto: {fc_val:.1f} N",
            xaxis_title="Desplazamiento lateral Y respecto al eje de vía (m)",
            yaxis_title="Altura sobre plano de rodadura Z (m)",
            xaxis=dict(range=[-1.3, 1.3], zeroline=True),
            yaxis=dict(range=[4.95, 5.80]),
            height=480,
        )
        return fig, z_mesilla_din, y_dir_pt, z_dir_pt, existe_desv, y_desv_pt, z_desv_pt, alerta_eng

    placeholder_graf = col_yz1.empty()
    placeholder_info = col_yz2.empty()

    def mostrar_estado_yz(pk_v):
        fig, z_mes, yd, zd, ex_desv, ydes, zdes, enganche = generar_figura_yz(pk_v)
        placeholder_graf.plotly_chart(fig, use_container_width=True)
        with placeholder_info.container():
            st.metric("Altura Mesilla", f"{z_mes:.3f} m")
            st.metric("Y / Z Hilo Directo", f"{yd:+.2f} m / {zd:.2f} m")
            if ex_desv:
                st.metric("Y / Z Hilo Desviado", f"{ydes:+.2f} m / {zdes:.2f} m")
            if enganche:
                st.error("🚨 ¡PELIGRO DE ENGANCHE! El hilo desviado pasa por debajo del cuerno.")
            elif ex_desv and abs(ydes) <= ancho_carbon and abs(zdes - z_mes) < 0.015:
                st.info("⚡ DOBLE CAPTACIÓN: Ambos hilos apoyan sobre la banda de carbón.")
            else:
                st.success("✅ GÁLIBO SEGURO: Sin riesgo de enganche en este punto.")

    if iniciar_anim:
        pasos = np.linspace(x_min, x_max, 45)
        for p in pasos:
            mostrar_estado_yz(float(p))
            time.sleep(0.06)
        mostrar_estado_yz(x_max)
    else:
        mostrar_estado_yz(pk_actual)

# ------------------------------------------
# PESTAÑA 2: FUERZA DE CONTACTO
# ------------------------------------------
with tab2:
    fig_din = go.Figure()
    fig_din.add_trace(
        go.Scatter(
            x=x_sim,
            y=F_contacto_arr,
            mode="lines",
            name="Fuerza de Contacto (N)",
            line=dict(color="#2ca02c", width=2.5),
        )
    )
    for _, row in df_clean.iterrows():
        color_p = "red" if row["Tipo"] in ["Aguja", "Cruce", "Elevación"] else "gray"
        fig_din.add_vline(
            x=row["Pos_X (m)"],
            line_dash="dash",
            line_color=color_p,
            annotation_text=f"{row['Poste']} ({row['Tipo']})",
        )
    fig_din.update_layout(
        title=f"Fuerza de Contacto a {v_kmh} km/h - Itinerario: {itinerario}",
        xaxis_title="PK (m)",
        yaxis_title="Fuerza (N)",
        height=450,
    )
    st.plotly_chart(fig_din, use_container_width=True)

# ------------------------------------------
# PESTAÑA 3: PLANTA Y ALZADO
# ------------------------------------------
with tab3:
    fig_planta = go.Figure()
    fig_planta.add_trace(
        go.Scatter(x=x_sim, y=y_directo, mode="lines", name="Hilo Directo (Y)", line=dict(color="#1f77b4", width=3))
    )
    if tiene_desviada:
        fig_planta.add_trace(
            go.Scatter(
                x=x_sim_desv,
                y=y_desviado,
                mode="lines",
                name="Hilo Desviado (Y)",
                line=dict(color="#d62728", width=3),
            )
        )
    fig_planta.add_hline(y=0.40, line_dash="dot", line_color="orange", annotation_text="Límite carbón (+0.40 m)")
    fig_planta.add_hline(y=-0.40, line_dash="dot", line_color="orange", annotation_text="Límite carbón (-0.40 m)")
    fig_planta.update_layout(title="Planta: Descentramientos (Y)", xaxis_title="PK (m)", yaxis_title="Y (m)", height=350)
    st.plotly_chart(fig_planta, use_container_width=True)

    fig_alzado = go.Figure()
    fig_alzado.add_trace(
        go.Scatter(x=x_sim, y=z_directo, mode="lines", name="Hilo Directo (Z)", line=dict(color="#1f77b4", width=3))
    )
    if tiene_desviada:
        fig_alzado.add_trace(
            go.Scatter(
                x=x_sim_desv,
                y=z_desviado,
                mode="lines",
                name="Hilo Desviado (Z)",
                line=dict(color="#d62728", width=3),
            )
        )
    fig_alzado.update_layout(title="Alzado: Alturas y Elevación (Z)", xaxis_title="PK (m)", yaxis_title="Z (m)", height=350)
    st.plotly_chart(fig_alzado, use_container_width=True)

# ------------------------------------------
# PESTAÑA 4: VISTA ESPACIAL 3D
# ------------------------------------------
with tab4:
    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Scatter3d(
            x=x_sim,
            y=y_directo,
            z=z_directo,
            mode="lines",
            name="Hilo Directo",
            line=dict(color="#1f77b4", width=6),
        )
    )
    if tiene_desviada:
        fig_3d.add_trace(
            go.Scatter3d(
                x=x_sim_desv,
                y=y_desviado,
                z=z_desviado,
                mode="lines",
                name="Hilo Desviado",
                line=dict(color="#d62728", width=6),
            )
        )
    for _, row in df_clean.iterrows():
        px, py, pz = row["Pos_X (m)"], row["Desc_Directa (m)"], row["Alt_Directa (m)"]
        fig_3d.add_trace(
            go.Scatter3d(
                x=[px, px],
                y=[py, py],
                z=[4.8, pz + 0.3],
                mode="lines+text",
                text=["", row["Poste"]],
                showlegend=False,
                line=dict(color="black", width=4),
            )
        )
    fig_3d.update_layout(
        title="Modelo 3D Multivano",
        scene=dict(
            xaxis_title="PK (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectratio=dict(x=3, y=1, z=0.8),
        ),
        height=550,
    )
    st.plotly_chart(fig_3d, use_container_width=True)