"""
╔══════════════════════════════════════════════════════════╗
║         ML MODELS — App interactiva para clase           ║
║  Instalar: pip install streamlit plotly scikit-learn     ║
║  Correr:   streamlit run ml_modelos_app.py               ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Modelos — Clase",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
.stApp                       { background: #0d0f14; color: #e8e6df; }
h1, h2, h3                  { font-family: 'Syne', sans-serif !important; color: #f0ede4 !important; }
p, li, .stMarkdown p        { color: #9a9890; line-height: 1.7; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #151820; border-radius: 12px;
    padding: 5px; gap: 4px; border: 1px solid #1e2230;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 1rem; color: #4a4860; border-radius: 8px;
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    background: #1e2230 !important; color: #f0ede4 !important;
}

/* ── Font sizes ── */
html, body, [class*="css"] { font-size: 16px !important; }
p, li, .stMarkdown p { font-size: 1.05rem !important; line-height: 1.75 !important; }
.stSlider label, .stRadio label, .stSelectbox label, .stToggle label { font-size: 1rem !important; }
.stMetric label { font-size: 0.95rem !important; }
.stMetric [data-testid="metric-value"] { font-size: 2rem !important; }
.stAlert p { font-size: 1rem !important; }
code, .mono { font-size: 0.97rem !important; }
.stCaption { font-size: 0.9rem !important; }
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.65rem !important; }
h3 { font-size: 1.3rem !important; }

/* ── Info table ── */
.info-table {
    width: 100%; border-collapse: collapse;
    font-family: 'DM Sans', sans-serif; font-size: 0.88rem;
    margin-bottom: 1.2rem;
}
.info-table th {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.09em; text-transform: uppercase;
    color: #4a4860; padding: 6px 14px; text-align: left;
    border-bottom: 1px solid #1e2230; background: #0d0f14;
}
.info-table td {
    padding: 10px 14px; color: #c8c4b8;
    border-bottom: 1px solid #151820; vertical-align: top;
}
.info-table tr:last-child td { border-bottom: none; }
.info-table tr:hover td { background: #151820; }
.tag-yes {
    display: inline-block; padding: 2px 10px; border-radius: 99px;
    font-size: 0.75rem; font-weight: 500;
    background: #0d2218; color: #34D399; border: 1px solid #34D39933;
}
.tag-no {
    display: inline-block; padding: 2px 10px; border-radius: 99px;
    font-size: 0.75rem; background: #1a0a0a; color: #555; border: 1px solid #1e2230;
}
.mono { font-family: 'DM Mono', monospace; font-size: 0.82rem; color: #a0c4ff; }
.section-label {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.1em; color: #4a4860; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.divider { height: 1px; background: #1e2230; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────
BG, GRID = "#0d0f14", "#1e2230"

def pstyle(fig, h=370, title=""):
    kw = dict(text=title, font=dict(family="Syne", size=14, color="#e8e6df"), x=0, xanchor="left") if title else {}
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, height=h,
        margin=dict(l=10, r=10, t=38 if title else 20, b=10),
        font=dict(family="DM Sans", color="#9a9890", size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        title=kw,
    )
    return fig

MPL_BG = "#0d0f14"

# ─────────────────────────────────────────────────────────────
#  DATOS (fijos por reproducibilidad)
# ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# Regresión lineal
X_lr  = np.sort(rng.uniform(0, 10, 50))
y_lr  = 2.3 * X_lr + 4.0 + rng.normal(0, 5.5, 50)

# Clasificación binaria 1D para logística (bien separada y con solapamiento)
rng2  = np.random.default_rng(7)
n_log = 80
x_neg = rng2.normal(-2.0, 1.2, n_log // 2)
x_pos = rng2.normal( 2.0, 1.2, n_log // 2)
X_log = np.concatenate([x_neg, x_pos])
y_log = np.concatenate([np.zeros(n_log // 2), np.ones(n_log // 2)]).astype(int)
sh    = rng2.permutation(n_log)
X_log, y_log = X_log[sh], y_log[sh]

# Moons para árbol / RF
X_cls, y_cls = make_moons(n_samples=120, noise=0.25, random_state=42)


# ═════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════
st.markdown('<h1 style="margin-bottom:0.1rem">🧠 4 Modelos de ML</h1>', unsafe_allow_html=True)
st.caption("Visualización interactiva · Mueve los sliders para ver cómo aprende cada modelo")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Regresión Lineal",
    "🔀  Regresión Logística",
    "🌳  Árbol de Decisión",
    "🌲  Random Forest",
])


# ═════════════════════════════════════════════════════════════
#  TAB 1 — REGRESIÓN LINEAL
# ═════════════════════════════════════════════════════════════
with tab1:

    st.markdown("### Regresión Lineal")
    st.write(
        "Modela la relación entre una variable de entrada **x** y una salida continua **y** "
        "encontrando la línea recta que **minimiza el error total** de las predicciones. "
        "Es el modelo más simple y es la base conceptual de casi todo en ML supervisado."
    )

    st.markdown("""
    <table class="info-table">
      <tr><th>Campo</th><th>Detalle</th></tr>
      <tr>
        <td>Modelo</td>
        <td><span class="mono">ŷ = m·x + b</span></td>
      </tr>
      <tr>
        <td>Función de costo</td>
        <td><b>MSE — Error Cuadrático Medio</b><br>
            <span class="mono">J(m,b) = (1/n) Σ (yᵢ − ŷᵢ)²</span></td>
      </tr>
      <tr>
        <td>Optimización</td>
        <td>Gradiente descendente <i>o</i> solución cerrada <span class="mono">β = (XᵀX)⁻¹Xᵀy</span></td>
      </tr>
      <tr>
        <td>¿Clasificación?</td>
        <td><span class="tag-no">✗ No</span></td>
      </tr>
      <tr>
        <td>¿Regresión?</td>
        <td><span class="tag-yes">✓ Sí — predice valores continuos</span></td>
      </tr>
      <tr>
        <td>Cuándo usarlo</td>
        <td>Relación aproximadamente lineal · Interpretabilidad importante · Línea base rápida</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    lr_model = LinearRegression().fit(X_lr.reshape(-1, 1), y_lr)
    m_opt    = float(lr_model.coef_[0])
    b_opt    = float(lr_model.intercept_)
    mse_opt  = mean_squared_error(y_lr, lr_model.predict(X_lr.reshape(-1, 1)))

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown('<div class="section-label">Controles</div>', unsafe_allow_html=True)
        m_user   = st.slider("Pendiente  m", -1.0, 6.0, 0.0, 0.05, key="lr_m")
        show_res = st.toggle("Mostrar residuos", True)

        y_hat    = m_user * X_lr + b_opt
        mse_user = mean_squared_error(y_lr, y_hat)
        delta    = (mse_user - mse_opt) / mse_opt * 100

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.metric("MSE actual",         f"{mse_user:.1f}", delta=f"{delta:+.0f}% vs óptimo", delta_color="inverse")
        st.metric("MSE mínimo posible", f"{mse_opt:.1f}")
        st.metric("Pendiente óptima",   f"m = {m_opt:.2f}")

        if delta < 2:
            st.success("🎯 ¡Encontraste el mínimo!")
        elif delta < 25:
            st.info(f"📉 Cerca — {delta:.0f}% sobre el mínimo")
        else:
            st.warning(f"📈 {delta:.0f}% sobre el mínimo")

    with col2:
        x_line = np.linspace(-0.3, 10.3, 300)

        fig = go.Figure()
        if show_res:
            for xi, yi, yhi in zip(X_lr, y_lr, y_hat):
                fig.add_trace(go.Scatter(x=[xi, xi], y=[yi, yhi], mode="lines",
                    line=dict(color="#FF6B6B", width=1, dash="dot"),
                    showlegend=False, opacity=0.4))

        fig.add_trace(go.Scatter(x=X_lr, y=y_lr, mode="markers", name="Datos",
            marker=dict(color="#4A9EFF", size=7, opacity=0.85, line=dict(color="white", width=0.4))))
        fig.add_trace(go.Scatter(x=x_line, y=m_opt * x_line + b_opt, mode="lines",
            line=dict(color="#4A9EFF", width=1.5, dash="dot"),
            name=f"Óptimo m={m_opt:.2f}", opacity=0.5))
        fig.add_trace(go.Scatter(x=x_line, y=m_user * x_line + b_opt, mode="lines",
            line=dict(color="#FF6B6B", width=3), name=f"Tu línea m={m_user:.2f}"))

        pstyle(fig, 300, "Ajusta la pendiente y observa los residuos")
        fig.update_layout(xaxis_title="x", yaxis_title="y", legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

        m_sweep = np.linspace(-1.0, 6.0, 400)
        jm      = [mean_squared_error(y_lr, mi * X_lr + b_opt) for mi in m_sweep]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=m_sweep, y=jm, mode="lines",
            line=dict(color="#4A9EFF", width=2), fill="tozeroy",
            fillcolor="rgba(74,158,255,0.07)", name="J(m)"))
        fig2.add_trace(go.Scatter(x=[m_opt], y=[mse_opt], mode="markers",
            marker=dict(color="#4AFF9E", size=14, symbol="star", line=dict(color="white", width=1.5)),
            name=f"Mínimo m={m_opt:.2f}"))
        fig2.add_trace(go.Scatter(x=[m_user], y=[mse_user], mode="markers",
            marker=dict(color="#FF6B6B", size=14, line=dict(color="white", width=2)),
            name=f"Tu m={m_user:.2f}"))
        fig2.add_shape(type="line", x0=m_user, x1=m_user, y0=0, y1=mse_user,
                       line=dict(color="#FF6B6B", width=1, dash="dot"))

        pstyle(fig2, 240, "Función de costo J(m) — parábola convexa (único mínimo global)")
        fig2.update_layout(xaxis_title="pendiente m", yaxis_title="MSE", legend=dict(x=0.6, y=0.99))
        st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  TAB 2 — REGRESIÓN LOGÍSTICA
# ═════════════════════════════════════════════════════════════
with tab2:

    st.markdown("### Regresión Logística")
    st.write(
        "A pesar del nombre, es un **clasificador**. La idea central: tomamos una combinación "
        "lineal de los datos y la pasamos por la **función sigmoide**, que convierte cualquier "
        "número real en una probabilidad entre 0 y 1. Esa probabilidad decide la clase."
    )

    st.markdown("""
    <table class="info-table">
      <tr><th>Campo</th><th>Detalle</th></tr>
      <tr>
        <td>Modelo</td>
        <td><span class="mono">z = θ₀ + θ₁·x</span> &nbsp;→&nbsp;
            <span class="mono">p = σ(z) = 1/(1+e⁻ᶻ)</span> &nbsp;→&nbsp;
            <span class="mono">ŷ = 1 si p ≥ umbral</span></td>
      </tr>
      <tr>
        <td>Función de costo</td>
        <td><b>Log Loss (Binary Cross-Entropy)</b><br>
            <span class="mono">J(θ) = −(1/n) Σ [ yᵢ·log(p̂ᵢ) + (1−yᵢ)·log(1−p̂ᵢ) ]</span></td>
      </tr>
      <tr>
        <td>Optimización</td>
        <td>Gradiente descendente (no hay solución cerrada)</td>
      </tr>
      <tr>
        <td>¿Clasificación?</td>
        <td><span class="tag-yes">✓ Sí — binaria (o multiclase con softmax)</span></td>
      </tr>
      <tr>
        <td>¿Regresión?</td>
        <td><span class="tag-no">✗ No — aunque la salida p es continua, no modela variables continuas</span></td>
      </tr>
      <tr>
        <td>Cuándo usarlo</td>
        <td>Clasificación interpretable · Probabilidades de salida necesarias · Línea base rápida</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Controles ────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        k_user  = st.slider("Pendiente k  (empinada de la S)", 0.3, 6.0, 1.5, 0.1, key="lg_k",
                            help="k grande → frontera más brusca y modelo más 'seguro'.")
    with ctrl2:
        x0_user = st.slider("Punto de inflexión x₀  (donde p = 0.5)", -3.0, 3.0, 0.0, 0.1, key="lg_x0")
    with ctrl3:
        umbral  = st.slider("Umbral de decisión", 0.1, 0.9, 0.5, 0.05, key="lg_thr")
        show_proj = st.toggle("Mostrar proyecciones", True, key="lg_proj")

    # Métricas rápidas
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    p_hat     = sigmoid(k_user * (X_log - x0_user))
    y_pred_lg = (p_hat >= umbral).astype(int)
    acc_lg    = accuracy_score(y_log, y_pred_lg)
    loss_lg   = log_loss(y_log, np.clip(p_hat, 1e-7, 1 - 1e-7))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",             f"{acc_lg:.1%}")
    m2.metric("Log Loss",             f"{loss_lg:.3f}")
    m3.metric("Frontera en x =",      f"{x0_user + np.log(umbral/(1-umbral))/k_user:.2f}")
    m4.metric("σ(x₀) =",             "0.5  (siempre)")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 3 gráficas en columnas ────────────────────────────────
    g1, g2, g3 = st.columns(3, gap="medium")

    x_range = np.linspace(X_log.min() - 1, X_log.max() + 1, 400)
    sig_y   = sigmoid(k_user * (x_range - x0_user))
    x_front = x0_user + np.log(umbral / (1 - umbral)) / k_user

    # ─ Gráfica 1: datos en espacio original + sigmoide ───────
    with g1:
        st.markdown('<div class="section-label">① Datos + sigmoide ajustada</div>', unsafe_allow_html=True)
        fig1 = go.Figure()

        # Proyecciones verticales
        if show_proj:
            for xi, yi in zip(X_log, y_log):
                p_i = sigmoid(k_user * (xi - x0_user))
                fig1.add_trace(go.Scatter(
                    x=[xi, xi], y=[float(yi), p_i], mode="lines",
                    line=dict(color="#FBBF24", width=0.8, dash="dot"),
                    showlegend=False, opacity=0.3,
                ))

        # Sigmoide
        fig1.add_trace(go.Scatter(
            x=x_range, y=sig_y, mode="lines",
            line=dict(color="#34D399", width=2.5),
            name=f"σ  k={k_user:.1f}  x₀={x0_user:.1f}",
            fill="tozeroy", fillcolor="rgba(52,211,153,0.06)",
        ))
        # Umbral horizontal
        fig1.add_hline(y=umbral, line_dash="dash", line_color="#FBBF24", line_width=1,
                       annotation_text=f" umbral={umbral:.2f}",
                       annotation_font_color="#FBBF24", annotation_font_size=10)
        # Frontera vertical
        fig1.add_vline(x=x_front, line_dash="dash", line_color="#FBBF24", line_width=1,
                       annotation_text=f" x={x_front:.2f}",
                       annotation_font_color="#FBBF24", annotation_font_size=10,
                       annotation_position="bottom right")

        # Puntos clase 0 (en y=0)
        mask0 = y_log == 0
        fig1.add_trace(go.Scatter(
            x=X_log[mask0], y=np.zeros(mask0.sum()) - 0.04, mode="markers",
            marker=dict(color="#FF6B6B", size=8, opacity=0.8,
                       symbol="circle", line=dict(color="white", width=0.5)),
            name="Clase 0",
        ))
        # Puntos clase 1 (en y=1)
        mask1 = y_log == 1
        fig1.add_trace(go.Scatter(
            x=X_log[mask1], y=np.ones(mask1.sum()) + 0.04, mode="markers",
            marker=dict(color="#4A9EFF", size=8, opacity=0.8,
                       symbol="circle", line=dict(color="white", width=0.5)),
            name="Clase 1",
        ))

        pstyle(fig1, 400, "Sigmoide sobre los datos")
        fig1.update_layout(
            xaxis_title="x  (característica)",
            yaxis_title="p = P(clase = 1)",
            yaxis=dict(range=[-0.15, 1.2]),
            legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ─ Gráfica 2: espacio z (lineal) → curva sigmoide ────────
    with g2:
        st.markdown('<div class="section-label">② Proyección al espacio lineal z</div>', unsafe_allow_html=True)

        z_range = np.linspace(-7, 7, 400)
        sig_std = sigmoid(z_range)                    # sigmoide estándar en espacio z
        z_data  = k_user * (X_log - x0_user)          # cada punto proyectado a z
        p_data  = sigmoid(z_data)

        fig2 = go.Figure()

        # Zonas de fondo
        fig2.add_vrect(x0=-7, x1=0, fillcolor="rgba(255,107,107,0.04)", line_width=0)
        fig2.add_vrect(x0=0, x1=7,  fillcolor="rgba(74,158,255,0.04)",  line_width=0)

        # Sigmoide estándar
        fig2.add_trace(go.Scatter(
            x=z_range, y=sig_std, mode="lines",
            line=dict(color="#34D399", width=2.5),
            name="σ(z)", fill="tozeroy", fillcolor="rgba(52,211,153,0.05)",
        ))

        # Proyecciones verticales de puntos a la curva
        if show_proj:
            for zi, pi, yi in zip(z_data, p_data, y_log):
                color = "#4A9EFF" if yi == 1 else "#FF6B6B"
                fig2.add_trace(go.Scatter(
                    x=[zi, zi], y=[0, pi], mode="lines",
                    line=dict(color=color, width=0.7, dash="dot"),
                    showlegend=False, opacity=0.35,
                ))

        # Puntos sobre la curva
        fig2.add_trace(go.Scatter(
            x=z_data[y_log==0], y=p_data[y_log==0], mode="markers",
            marker=dict(color="#FF6B6B", size=7, opacity=0.85, line=dict(color="white", width=0.5)),
            name="Clase 0 en σ",
        ))
        fig2.add_trace(go.Scatter(
            x=z_data[y_log==1], y=p_data[y_log==1], mode="markers",
            marker=dict(color="#4A9EFF", size=7, opacity=0.85, line=dict(color="white", width=0.5)),
            name="Clase 1 en σ",
        ))

        # Umbral y línea z=0
        fig2.add_hline(y=umbral, line_dash="dash", line_color="#FBBF24", line_width=1,
                       annotation_text=f" p={umbral:.2f}", annotation_font_color="#FBBF24",
                       annotation_font_size=10)
        fig2.add_vline(x=0, line_dash="dot", line_color="#333", line_width=1.5)

        # Etiquetas asíntotas
        fig2.add_annotation(x=-6, y=0.07, text="z→−∞  p→0", font=dict(color="#FF6B6B", size=10), showarrow=False)
        fig2.add_annotation(x= 6, y=0.93, text="z→+∞  p→1", font=dict(color="#4A9EFF", size=10), showarrow=False, xanchor="right")

        pstyle(fig2, 400, "z = k·(x − x₀) en la curva σ(z)")
        fig2.update_layout(
            xaxis_title="z = k·(x − x₀)  [espacio lineal]",
            yaxis_title="σ(z)  [probabilidad]",
            yaxis=dict(range=[-0.08, 1.1]),
            legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─ Gráfica 3: Log Loss vs k ───────────────────────────────
    with g3:
        st.markdown('<div class="section-label">③ Función de costo — Log Loss</div>', unsafe_allow_html=True)

        k_sweep    = np.linspace(0.2, 6.0, 150)
        loss_sweep = []
        for ki in k_sweep:
            pi = np.clip(sigmoid(ki * (X_log - x0_user)), 1e-7, 1 - 1e-7)
            loss_sweep.append(log_loss(y_log, pi))

        k_opt_idx = int(np.argmin(loss_sweep))
        k_opt_val = float(k_sweep[k_opt_idx])
        loss_min  = float(loss_sweep[k_opt_idx])

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=k_sweep, y=loss_sweep, mode="lines",
            line=dict(color="#34D399", width=2.5),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.06)",
            name="Log Loss(k)",
        ))
        # Mínimo
        fig3.add_trace(go.Scatter(
            x=[k_opt_val], y=[loss_min], mode="markers",
            marker=dict(color="#4AFF9E", size=14, symbol="star", line=dict(color="white", width=1.5)),
            name=f"Mínimo k={k_opt_val:.2f}",
        ))
        # Posición actual
        fig3.add_trace(go.Scatter(
            x=[k_user], y=[loss_lg], mode="markers",
            marker=dict(color="#FBBF24", size=14, line=dict(color="white", width=2)),
            name=f"Actual k={k_user:.1f}",
        ))
        fig3.add_shape(type="line", x0=k_user, x1=k_user, y0=0, y1=loss_lg,
                       line=dict(color="#FBBF24", width=1, dash="dot"))

        pstyle(fig3, 400, "Log Loss vs pendiente k")
        fig3.update_layout(
            xaxis_title="pendiente k",
            yaxis_title="J(k)  [Log Loss]",
            legend=dict(x=0.45, y=0.98, font=dict(size=10)),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Nota pedagógica ───────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.info(
        "**¿Por qué Log Loss y no MSE?**  Con MSE la función de costo resulta no-convexa — "
        "el gradiente descendente puede quedar atascado. El **Log Loss** es convexa "
        "(una sola \"valle\"), garantizando convergencia al mínimo global. "
        "Además, penaliza muy fuerte cuando el modelo predice con alta confianza y se equivoca."
    )


# ═════════════════════════════════════════════════════════════
#  TAB 3 — ÁRBOL DE DECISIÓN
# ═════════════════════════════════════════════════════════════
with tab3:

    st.markdown("### Árbol de Decisión")
    st.write(
        "El árbol hace **preguntas de sí/no** sobre las características, dividiendo los datos "
        "en grupos cada vez más puros. Cada división elige el corte que **más reduce la impureza** "
        "(Gini o entropía). Es el modelo más **interpretable** — cualquiera puede seguir sus decisiones paso a paso."
    )

    st.markdown("""
    <table class="info-table">
      <tr><th>Campo</th><th>Detalle</th></tr>
      <tr>
        <td>Criterio de división</td>
        <td><b>Índice de Gini</b>: <span class="mono">G = 1 − Σ pᵢ²</span><br>
            <b>Entropía</b>: <span class="mono">H = −Σ pᵢ log₂(pᵢ)</span></td>
      </tr>
      <tr>
        <td>Función de costo</td>
        <td>Impureza ponderada de los nodos hijo:<br>
            <span class="mono">J = (n_izq/n)·G_izq + (n_der/n)·G_der</span></td>
      </tr>
      <tr>
        <td>Optimización</td>
        <td>Greedy — en cada nodo busca el mejor corte (no garantiza árbol global óptimo)</td>
      </tr>
      <tr>
        <td>¿Clasificación?</td>
        <td><span class="tag-yes">✓ Sí — clase mayoritaria del nodo hoja</span></td>
      </tr>
      <tr>
        <td>¿Regresión?</td>
        <td><span class="tag-yes">✓ Sí — media del nodo hoja (criterio MSE)</span></td>
      </tr>
      <tr>
        <td>Cuándo usarlo</td>
        <td>Interpretabilidad crítica · Relaciones no lineales · Punto de partida para ensembles</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        depth    = st.slider("Profundidad máxima", 1, 6, 3, key="dt_d")
        criterio = st.selectbox("Criterio", ["gini", "entropy"], key="dt_c")

    dt_model = DecisionTreeClassifier(max_depth=depth, criterion=criterio, random_state=42).fit(X_cls, y_cls)
    acc_dt   = accuracy_score(y_cls, dt_model.predict(X_cls))

    with col2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",      f"{acc_dt:.1%}")
        c2.metric("Nodos totales",  dt_model.tree_.node_count)
        c3.metric("Hojas",          dt_model.get_n_leaves())

    st.divider()
    st.markdown("#### El árbol construido")
    st.caption("Cada nodo muestra la pregunta de corte, el Gini del nodo y cuántas muestras llegaron.")

    fig_tree, ax = plt.subplots(figsize=(max(9, depth * 4), max(4, depth * 2.2)))
    fig_tree.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_BG)
    plot_tree(dt_model, ax=ax,
              feature_names=["x₁", "x₂"],
              class_names=["Clase 0", "Clase 1"],
              filled=True, rounded=True, impurity=True,
              proportion=False, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_tree, use_container_width=True)
    plt.close()

    st.divider()

    gcol1, gcol2 = st.columns(2)

    with gcol1:
        st.markdown("#### Frontera de decisión")
        st.caption("El árbol divide el espacio en rectángulos — una pregunta por eje a la vez.")
        xx, yy = np.meshgrid(
            np.linspace(X_cls[:,0].min()-0.5, X_cls[:,0].max()+0.5, 200),
            np.linspace(X_cls[:,1].min()-0.5, X_cls[:,1].max()+0.5, 200))
        Z = dt_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig_f = go.Figure()
        fig_f.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z.astype(float),
            colorscale=[[0,"#1a0808"],[1,"#081a10"]], showscale=False, opacity=0.5, line_width=0))
        fig_f.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z.astype(float),
            colorscale=[[0,"#FBBF24"],[1,"#FBBF24"]],
            showscale=False, contours=dict(start=0.5,end=0.5,size=0,coloring="lines"),
            line=dict(color="#FBBF24", width=2)))
        for cls, col, name in [(0,"#FF6B6B","Clase 0"),(1,"#4AFF9E","Clase 1")]:
            mask = y_cls == cls
            fig_f.add_trace(go.Scatter(x=X_cls[mask,0], y=X_cls[mask,1], mode="markers",
                marker=dict(color=col, size=7, opacity=0.85, line=dict(color="white",width=0.4)), name=name))
        pstyle(fig_f, 360, f"Frontera · profundidad={depth} · Accuracy={acc_dt:.1%}")
        fig_f.update_layout(xaxis_title="x₁", yaxis_title="x₂")
        st.plotly_chart(fig_f, use_container_width=True)

    with gcol2:
        st.markdown("#### Impureza Gini vs proporción de clase")
        st.caption("Un nodo puro (una sola clase) tiene Gini=0. Mitad/mitad → Gini máximo=0.5.")
        props = np.linspace(0.001, 0.999, 300)
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=props, y=2*props*(1-props), mode="lines",
            line=dict(color="#FBBF24", width=2.5), fill="tozeroy",
            fillcolor="rgba(251,191,36,0.07)", name="Gini"))
        fig_g.add_trace(go.Scatter(
            x=props, y=-(props*np.log2(props)+(1-props)*np.log2(1-props))/2,
            mode="lines", line=dict(color="#A78BFA", width=2, dash="dash"), name="Entropía (÷2)"))
        fig_g.add_vline(x=0.5, line_dash="dot", line_color="#5a5860",
                        annotation_text="  máxima impureza", annotation_font_color="#5a5860")
        pstyle(fig_g, 360, "Gini y Entropía según proporción de clase")
        fig_g.update_layout(xaxis_title="p (proporción clase mayoritaria)",
                            yaxis_title="impureza", legend=dict(x=0.5, y=0.98))
        st.plotly_chart(fig_g, use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  TAB 4 — RANDOM FOREST
# ═════════════════════════════════════════════════════════════
with tab4:

    st.markdown("### Random Forest")
    st.write(
        "Idea simple y poderosa: en vez de confiar en **un solo árbol**, entrena "
        "**muchos árboles diferentes** — cada uno con una muestra aleatoria de datos "
        "y características — y deja que **voten**. La mayoría gana. "
        "Más árboles = más estabilidad, menos varianza."
    )

    st.markdown("""
    <table class="info-table">
      <tr><th>Campo</th><th>Detalle</th></tr>
      <tr>
        <td>Algoritmo</td>
        <td>Ensemble de árboles entrenados con <b>Bagging</b> (Bootstrap Aggregating)</td>
      </tr>
      <tr>
        <td>Función de costo</td>
        <td>Cada árbol minimiza Gini/Entropía internamente. El bosque se evalúa con <b>OOB Error</b>:<br>
            <span class="mono">OOB = fracción de predicciones incorrectas en datos "out-of-bag"</span></td>
      </tr>
      <tr>
        <td>Optimización</td>
        <td>Paralela — cada árbol se entrena independientemente sobre su muestra bootstrap</td>
      </tr>
      <tr>
        <td>¿Clasificación?</td>
        <td><span class="tag-yes">✓ Sí — voto mayoritario de los árboles</span></td>
      </tr>
      <tr>
        <td>¿Regresión?</td>
        <td><span class="tag-yes">✓ Sí — promedio de las predicciones de los árboles</span></td>
      </tr>
      <tr>
        <td>Cuándo usarlo</td>
        <td>Alta precisión necesaria · Resistente a overfitting · No requiere mucho tuning</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        n_trees = st.slider("Número de árboles", 1, 50, 5, key="rf_n")

    rf_model = RandomForestClassifier(
        n_estimators=n_trees, max_depth=3, random_state=42,
        oob_score=(n_trees > 1)
    ).fit(X_cls, y_cls)
    acc_rf = accuracy_score(y_cls, rf_model.predict(X_cls))

    with col2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy (train)", f"{acc_rf:.1%}")
        if n_trees > 1:
            c2.metric("OOB Score", f"{rf_model.oob_score_:.1%}",
                      help="Estimación del error en datos no vistos por cada árbol")
        c3.metric("Árboles entrenados", n_trees)

    st.divider()
    st.markdown("#### Los árboles individuales del bosque")
    st.caption("Cada árbol vio **datos distintos** (muestreo bootstrap) → aprendió algo diferente. Juntos se compensan.")

    n_show    = min(n_trees, 4)
    cols_show = st.columns(n_show)
    for i in range(n_show):
        tree_i      = rf_model.estimators_[i]
        fig_t, ax_t = plt.subplots(figsize=(4, 3))
        fig_t.patch.set_facecolor(MPL_BG)
        ax_t.set_facecolor(MPL_BG)
        plot_tree(tree_i, ax=ax_t,
                  feature_names=["x₁","x₂"],
                  class_names=["C0","C1"],
                  filled=True, rounded=True,
                  impurity=False, proportion=False, fontsize=7)
        ax_t.set_title(f"Árbol {i+1}", color="#a0a098", fontsize=10)
        plt.tight_layout()
        cols_show[i].pyplot(fig_t, use_container_width=True)
        plt.close()

    st.divider()

    vc1, vc2 = st.columns(2)

    with vc1:
        st.markdown("#### Votación del bosque en un punto de ejemplo")
        sample_idx    = 10
        x_sample      = X_cls[sample_idx:sample_idx+1]
        y_true_sample = y_cls[sample_idx]
        votes         = np.array([t.predict(x_sample)[0] for t in rf_model.estimators_])
        vote_counts   = [int((votes==0).sum()), int((votes==1).sum())]
        winner        = int(np.argmax(vote_counts))

        st.caption(f"Clase real: **{y_true_sample}** · Predicción del bosque: **Clase {winner}**")
        fig_v = go.Figure(go.Bar(
            x=["Clase 0","Clase 1"], y=vote_counts,
            marker_color=["#FF6B6B","#4AFF9E"],
            text=[f"{v} votos" for v in vote_counts],
            textposition="outside",
        ))
        fig_v.update_traces(textfont_color="#a0a098")
        pstyle(fig_v, 300, f"Votación de {n_trees} árbol{'es' if n_trees>1 else ''} → Clase {winner}")
        fig_v.update_layout(yaxis=dict(range=[0, n_trees+3]), showlegend=False)
        st.plotly_chart(fig_v, use_container_width=True)

    with vc2:
        st.markdown("#### Accuracy al agregar más árboles")
        st.caption("El error disminuye y se estabiliza — más árboles más allá de cierto punto no mejoran.")
        ns   = list(range(1, min(51, max(n_trees*3, 20)+1)))
        accs = []
        for k in ns:
            rf_k = RandomForestClassifier(k, max_depth=3, random_state=42).fit(X_cls, y_cls)
            accs.append(accuracy_score(y_cls, rf_k.predict(X_cls)))

        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=ns, y=accs, mode="lines",
            line=dict(color="#A78BFA", width=2.5), fill="tozeroy",
            fillcolor="rgba(167,139,250,0.08)", name="Accuracy"))
        fig_err.add_vline(x=n_trees, line_dash="dash", line_color="#FBBF24",
                          annotation_text=f" n={n_trees}", annotation_font_color="#FBBF24")
        pstyle(fig_err, 300, "Accuracy vs número de árboles")
        fig_err.update_layout(xaxis_title="número de árboles", yaxis_title="Accuracy")
        st.plotly_chart(fig_err, use_container_width=True)

    st.divider()
    st.markdown("#### Frontera del bosque completo")
    st.caption("Más suave y curva que un árbol solo — el ensemble generaliza mejor.")

    xx, yy = np.meshgrid(
        np.linspace(X_cls[:,0].min()-0.5, X_cls[:,0].max()+0.5, 200),
        np.linspace(X_cls[:,1].min()-0.5, X_cls[:,1].max()+0.5, 200))
    probs_rf = rf_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)

    fig5 = go.Figure()
    fig5.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=probs_rf,
        colorscale=[[0,"#1a0808"],[0.5,"#0d0f14"],[1,"#0d0818"]],
        showscale=True, opacity=0.7, line_width=0,
        colorbar=dict(title="P(Clase 1)", thickness=12, tickfont=dict(size=10))))
    for cls, col, name in [(0,"#FF6B6B","Clase 0"),(1,"#4AFF9E","Clase 1")]:
        mask = y_cls == cls
        fig5.add_trace(go.Scatter(x=X_cls[mask,0], y=X_cls[mask,1], mode="markers",
            marker=dict(color=col, size=7, opacity=0.85, line=dict(color="white",width=0.4)), name=name))
    pstyle(fig5, 400, f"Frontera Random Forest · {n_trees} árboles · Accuracy={acc_rf:.1%}")
    fig5.update_layout(xaxis_title="x₁", yaxis_title="x₂")
    st.plotly_chart(fig5, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption("ML Modelos · Clase interactiva · Streamlit + Plotly + scikit-learn")