import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from io import StringIO

st.set_page_config(
    page_title="서울 기온 상승 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 상승 분석")
st.markdown("**1980년 이전 vs 이후** 기온 상승 속도 비교 분석")

# ── 데이터 로드 ──────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))
    else:
        st.error("CSV 파일을 업로드해 주세요.")
        return None

    df.columns = df.columns.str.strip()
    df["날짜"] = df["날짜"].astype(str).str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온(℃)"])
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["시기"] = df["연도"].apply(lambda y: "1980년 이전" if y < 1980 else "1980년 이후")
    return df

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    uploaded = st.file_uploader("기상 데이터 CSV 업로드", type="csv")
    cutoff = st.slider("기준 연도", 1950, 2010, 1980, step=5)
    rolling_years = st.slider("이동평균 (년)", 5, 20, 10)
    temp_col = st.selectbox("분석 기온 항목", ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"])

df = load_data(uploaded)
if df is None:
    st.info("👈 왼쪽에서 CSV 파일을 업로드하면 분석이 시작됩니다.")
    st.stop()

# cutoff 재적용
df["시기"] = df["연도"].apply(lambda y: f"{cutoff}년 이전" if y < cutoff else f"{cutoff}년 이후")

# ── 연간 평균 ────────────────────────────────────────────────
annual = df.groupby("연도")[temp_col].mean().reset_index()
annual.columns = ["연도", "기온"]
annual["이동평균"] = annual["기온"].rolling(rolling_years, center=True).mean()

before = annual[annual["연도"] < cutoff]
after  = annual[annual["연도"] >= cutoff]

def linear_trend(sub):
    slope, intercept, r, p, se = stats.linregress(sub["연도"], sub["기온"])
    return slope, intercept, r**2, p

s_b, i_b, r2_b, p_b = linear_trend(before)
s_a, i_a, r2_a, p_a = linear_trend(after)

# ── 요약 카드 ────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{cutoff}년 이전 상승 속도", f"{s_b*10:.3f}°C/10년",
          help=f"R²={r2_b:.3f}, p={p_b:.4f}")
c2.metric(f"{cutoff}년 이후 상승 속도", f"{s_a*10:.3f}°C/10년",
          help=f"R²={r2_a:.3f}, p={p_a:.4f}",
          delta=f"{(s_a-s_b)*10:+.3f}°C/10년 차이")
avg_b = before["기온"].mean()
avg_a = after["기온"].mean()
c3.metric(f"{cutoff}년 이전 평균", f"{avg_b:.2f}°C")
c4.metric(f"{cutoff}년 이후 평균", f"{avg_a:.2f}°C",
          delta=f"{avg_a-avg_b:+.2f}°C")

st.markdown("---")

# ── 탭 ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 추세 분석", "📊 분포 비교", "🗓️ 월별 변화", "📋 통계 요약"
])

# ── Tab 1: 추세 ──────────────────────────────────────────────
with tab1:
    st.subheader("연간 기온 추세 및 회귀선")

    fig = go.Figure()

    # 원시 데이터
    fig.add_trace(go.Scatter(
        x=annual["연도"], y=annual["기온"],
        mode="markers", name="연간 평균",
        marker=dict(size=4, color="lightgray"), opacity=0.6
    ))

    # 이동평균
    fig.add_trace(go.Scatter(
        x=annual["연도"], y=annual["이동평균"],
        mode="lines", name=f"{rolling_years}년 이동평균",
        line=dict(color="steelblue", width=2)
    ))

    # before 회귀
    x_b = np.array([before["연도"].min(), before["연도"].max()])
    y_b = i_b + s_b * x_b
    fig.add_trace(go.Scatter(
        x=x_b, y=y_b, mode="lines",
        name=f"{cutoff}년 이전 추세 ({s_b*10:+.3f}°C/10년)",
        line=dict(color="royalblue", width=3, dash="dash")
    ))

    # after 회귀
    x_a = np.array([after["연도"].min(), after["연도"].max()])
    y_a = i_a + s_a * x_a
    fig.add_trace(go.Scatter(
        x=x_a, y=y_a, mode="lines",
        name=f"{cutoff}년 이후 추세 ({s_a*10:+.3f}°C/10년)",
        line=dict(color="crimson", width=3, dash="dash")
    ))

    # 기준선 표시
    fig.add_vline(x=cutoff, line_dash="dot", line_color="orange",
                  annotation_text=f"{cutoff}년", annotation_position="top right")

    fig.update_layout(
        xaxis_title="연도", yaxis_title=temp_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 기울기 배율
    if s_b != 0:
        ratio = s_a / s_b
        st.info(f"💡 {cutoff}년 이후 기온 상승 속도는 이전 대비 **{ratio:.1f}배** 빠릅니다.")

# ── Tab 2: 분포 ──────────────────────────────────────────────
with tab2:
    st.subheader("기온 분포 비교")

    col1, col2 = st.columns(2)

    with col1:
        # 박스플롯
        fig_box = go.Figure()
        for label, color in [(f"{cutoff}년 이전", "royalblue"), (f"{cutoff}년 이후", "crimson")]:
            sub = annual[annual["연도"].apply(
                lambda y: y < cutoff if "이전" in label else y >= cutoff)]
            fig_box.add_trace(go.Box(
                y=sub["기온"], name=label,
                marker_color=color, boxmean="sd"
            ))
        fig_box.update_layout(title="연간 평균기온 박스플롯",
                              yaxis_title=temp_col, height=380, template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        # 히스토그램
        fig_hist = go.Figure()
        for label, color in [(f"{cutoff}년 이전", "royalblue"), (f"{cutoff}년 이후", "crimson")]:
            sub = annual[annual["연도"].apply(
                lambda y: y < cutoff if "이전" in label else y >= cutoff)]
            fig_hist.add_trace(go.Histogram(
                x=sub["기온"], name=label,
                marker_color=color, opacity=0.6,
                nbinsx=20
            ))
        fig_hist.update_layout(
            barmode="overlay", title="연간 평균기온 분포",
            xaxis_title=temp_col, yaxis_title="빈도",
            height=380, template="plotly_white"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # T-검정
    t_stat, t_p = stats.ttest_ind(
        before["기온"].dropna(), after["기온"].dropna()
    )
    st.markdown(f"**독립표본 t-검정:** t = {t_stat:.3f}, **p = {t_p:.2e}**")
    if t_p < 0.05:
        st.success("✅ p < 0.05 → 두 시기의 평균 기온 차이가 **통계적으로 유의**합니다.")
    else:
        st.warning("⚠️ p ≥ 0.05 → 통계적으로 유의한 차이가 없습니다.")

# ── Tab 3: 월별 변화 ─────────────────────────────────────────
with tab3:
    st.subheader("월별 평균 기온 변화")

    monthly_before = df[df["연도"] < cutoff].groupby("월")[temp_col].mean()
    monthly_after  = df[df["연도"] >= cutoff].groupby("월")[temp_col].mean()
    months = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"]

    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(
        x=months, y=monthly_before.values,
        mode="lines+markers", name=f"{cutoff}년 이전",
        line=dict(color="royalblue", width=2.5),
        marker=dict(size=8)
    ))
    fig_m.add_trace(go.Scatter(
        x=months, y=monthly_after.values,
        mode="lines+markers", name=f"{cutoff}년 이후",
        line=dict(color="crimson", width=2.5),
        marker=dict(size=8)
    ))
    fig_m.update_layout(
        xaxis_title="월", yaxis_title=temp_col,
        height=420, template="plotly_white"
    )
    st.plotly_chart(fig_m, use_container_width=True)

    # 월별 차이 바 차트
    diff = monthly_after.values - monthly_before.values
    colors = ["crimson" if d > 0 else "royalblue" for d in diff]
    fig_diff = go.Figure(go.Bar(
        x=months, y=diff,
        marker_color=colors,
        text=[f"{d:+.2f}°C" for d in diff],
        textposition="outside"
    ))
    fig_diff.update_layout(
        title="월별 기온 변화량 (이후 − 이전)",
        yaxis_title="기온 변화 (°C)", height=320, template="plotly_white"
    )
    st.plotly_chart(fig_diff, use_container_width=True)

# ── Tab 4: 통계 요약 ─────────────────────────────────────────
with tab4:
    st.subheader("통계 요약 테이블")

    summary = pd.DataFrame({
        "항목": [
            "분석 기간",
            "데이터 수 (연도)",
            "평균 기온 (°C)",
            "기온 상승 속도 (°C/10년)",
            "회귀 R²",
            "회귀 p-value",
            "최저 연평균 (°C)",
            "최고 연평균 (°C)",
            "표준편차 (°C)"
        ],
        f"{cutoff}년 이전": [
            f"{before['연도'].min()}–{before['연도'].max()}",
            len(before),
            f"{before['기온'].mean():.2f}",
            f"{s_b*10:.4f}",
            f"{r2_b:.4f}",
            f"{p_b:.4e}",
            f"{before['기온'].min():.2f}",
            f"{before['기온'].max():.2f}",
            f"{before['기온'].std():.4f}"
        ],
        f"{cutoff}년 이후": [
            f"{after['연도'].min()}–{after['연도'].max()}",
            len(after),
            f"{after['기온'].mean():.2f}",
            f"{s_a*10:.4f}",
            f"{r2_a:.4f}",
            f"{p_a:.4e}",
            f"{after['기온'].min():.2f}",
            f"{after['기온'].max():.2f}",
            f"{after['기온'].std():.4f}"
        ]
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**가설 검증 결과 요약**")
    ratio_str = f"{s_a/s_b:.1f}배" if s_b != 0 else "N/A"
    conclusion = f"""
- {cutoff}년 **이전** 상승 속도: **{s_b*10:.3f}°C/10년** (R²={r2_b:.3f})
- {cutoff}년 **이후** 상승 속도: **{s_a*10:.3f}°C/10년** (R²={r2_a:.3f})
- 평균 기온 상승: **{avg_a - avg_b:+.2f}°C** (이후 − 이전)
- 상승 속도 배율: **{ratio_str}** 더 빠름
- t-검정 p-value: **{t_p:.2e}** → {"통계적으로 유의함 ✅" if t_p < 0.05 else "유의하지 않음 ⚠️"}
"""
    st.markdown(conclusion)

st.caption("데이터 출처: 기상청 기상자료개방포털 | 서울(108) 지점")
