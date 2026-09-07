# Voice Analysis UI (목BTI) Mobile Design & Technical Specification (`design.md`)

## 1. Executive Summary & Design Goals
This specification defines the responsive UI layout and component architecture for the **Voice Type Analysis Result Dashboard ("목BTI")**. 

### Primary Objectives
* **Mobile-First Ratio Compliance**: Ensure optimized display across mobile devices (390px - 430px base viewports, scalable up to 100%).
* **Overlapping Character Art Layering**: Position transparent PNG assets over card boundaries using CSS absolute coordinates and z-indexing.
* **Dynamic Type Mapping**: Support 5 distinct voice types with mapped character image assets and dynamic color themes.

---

## 2. Dynamic Asset & Type Mapping Strategy

| Voice Type | Character Asset (`.png`) | Accent Color Code | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **하이톤형** | `assets/dolphin.png` | `#00E5FF` (Cyan) | High Tone Sparkling / 청량함 / 고음 |
| **중저음형** | `assets/bear.png` | `#10B981` (Teal/Green) | Warm & Deep / 안정감 / 중저음 |
| **울림형** | `assets/owl.png` | `#8B5CF6` (Purple) | Resonant / 풍부한 울림 / 울림톤 |
| **차분형** | `assets/cat.png` | `#64748B` (Slate/Gray) | Calm & Soft / 편안함 / 차분함 |
| **비성형** | `assets/fox.png` | `#F59E0B` (Amber) | Nasal Charm / 개성있는 톤 / 비성 |

---

## 3. UI Layout Architecture & Overlapping Image CSS

### 3.1 Structural Component Hierarchy
```
+-------------------------------------------------------+
|  [ Header Title: 당신의 목소리 유형은? ]                 |
+-------------------------------------------------------+
|  [ Card Wrapper (.card-wrapper: relative) ]           |
|                                                       |
|   (Overlapping Character PNG: absolute, z-index: 10)  |
|      / \                                              |
|     ( 🐬 ) --- Overlaps Top-Left Card Boundary        |
|      \ /                                              |
|   +-----------------------------------------------+   |
|   | [ Result Card (.result-card) ]                |   |
|   | - Left Accent Border (border-left: 8px)       |   |
|   | - Badge: 대표 분류 유형 Top 1 (77.6%)           |   |
|   | - Type Name: 하이톤형                          |   |
|   | - Subtitle: (High-Tone Sparkling)             |   |
|   | - Summary & Descriptive Paragraphs            |   |
|   +-----------------------------------------------+   |
+-------------------------------------------------------+
|  [ Section 2: Metric Stat Grid (4 Metrics) ]          |
+-------------------------------------------------------+
|  [ Section 3: 5대 목BTI Donut Chart (Plotly) ]         |
|  - Center Annotation: BEST TYPE 하이톤형 77.6%         |
+-------------------------------------------------------+
|  [ Section 4: Detailed Profile Breakdown Card ]       |
+-------------------------------------------------------+
|  [ Section 5: Audio & Spectrogram Player Comparison ] |
|  - Raw Audio Player + Waveform Visualizer             |
|  - Clean Audio Player + Spectrogram                   |
+-------------------------------------------------------+
```

### 3.2 Key CSS Specifications (`styles.css`)

```css
/* Mobile Container Constraint */
.main-container {
    max-width: 480px;
    margin: 0 auto;
    padding: 16px;
    background-color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Outer Card Wrapper to allow negative margin overlapping */
.card-wrapper {
    position: relative;
    margin-top: 50px;
    margin-bottom: 24px;
}

/* Main Result Card Body */
.result-card {
    position: relative;
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-left: 8px solid #00e5ff; /* Dynamic per type */
    border-radius: 20px;
    padding: 28px 20px 20px 80px; /* Padding left allocates space for overlapping PNG */
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
    text-align: center;
}

/* Overlapping Floating Character PNG Layer */
.character-img {
    position: absolute;
    top: -40px;          /* Overhang above top border */
    left: -30px;         /* Overhang beyond left border */
    width: 140px;        /* Optimized asset width for 390px+ viewports */
    height: auto;
    z-index: 10;         /* Layered above the card container */
    filter: drop-shadow(0px 8px 12px rgba(0, 0, 0, 0.15));
}

/* Responsive Audio Player */
audio {
    width: 100%;
    height: 40px;
    border-radius: 20px;
}
```

---

## 4. Full Reference Implementation (`app.py`)

```python
import streamlit as st
import plotly.graph_objects as go

# 1. Page Configuration for Mobile Viewports
st.set_page_config(
    page_title="목BTI 결과 리포트",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Type Configuration Mapping
TYPE_CONFIG = {
    "하이톤형": {
        "image": "assets/dolphin.png",
        "color": "#00e5ff",
        "badge": "대표 분류 유형: Top 1 (77.6%)",
        "english": "(High-Tone Sparkling)",
        "summary": "청량함 100%! 분위기를 단숨에 반전시키는 청아한 고음",
        "desc_1": "밝고 통통 튀며 귀에 쏙쏙 박히는 에너제틱한 고음 보이스입니다.",
        "desc_2": "대화방의 분위기 메이커 역할을 톡톡히 하며 청중의 집중을 즉각 이끌어냅니다."
    }
}

current_type = "하이톤형"
data = TYPE_CONFIG[current_type]

# 3. Custom Mobile CSS Injection
st.markdown(f"""
    <style>
    .block-container {{
        max-width: 480px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}
    
    .card-wrapper {{
        position: relative;
        margin-top: 50px;
        margin-bottom: 24px;
    }}

    .result-card {{
        position: relative;
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-left: 8px solid {data['color']};
        border-radius: 20px;
        padding: 28px 20px 20px 80px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center;
    }}

    .character-img {{
        position: absolute;
        top: -40px;
        left: -30px;
        width: 140px;
        height: auto;
        z-index: 10;
        filter: drop-shadow(0px 8px 12px rgba(0,0,0,0.15));
    }}

    .type-badge {{
        background: {data['color']};
        color: #ffffff;
        font-size: 11px;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 10px;
        display: inline-block;
        margin-bottom: 8px;
    }}

    .main-title {{
        text-align: center;
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 12px;
    }}
    </style>
""", unsafe_allow_html=True)

# 4. Header Title
st.markdown('<div class="main-title">당신의 목소리 유형은?</div>', unsafe_allow_html=True)

# 5. Overlapping Character & Result Card
st.markdown(f"""
<div class="card-wrapper">
    <img src="{data['image']}" class="character-img" alt="Voice Type Character">
    <div class="result-card">
        <div class="type-badge">{data['badge']}</div>
        <h1 style="margin:0; font-size: 28px; color: #0f172a;">{current_type}</h1>
        <div style="font-size: 13px; color: #475569; font-weight: 600; margin-bottom: 12px;">{data['english']}</div>
        <p style="color: #0284c7; font-weight: bold; font-size: 14px; margin-bottom: 8px;">{data['summary']}</p>
        <p style="font-size: 12px; color: #334155; margin: 4px 0;">{data['desc_1']}</p>
        <p style="font-size: 12px; color: #64748b; margin: 4px 0;">{data['desc_2']}</p>
    </div>
</div>
""", unsafe_allow_html=True)

# 6. Donut Chart Visualization
st.markdown("### 5대 목BTI 유형별 점유율 및 원 그래프 분석")

labels = ['하이톤형', '중저음형', '울림형', '차분형', '비성형']
values = [77.6, 22.4, 0.0, 0.0, 0.0]
colors = ['#00e5ff', '#10b981', '#cbd5e1', '#e2e8f0', '#f1f5f9']

fig = go.Figure(data=[go.Pie(
    labels=labels, 
    values=values, 
    hole=.6,
    marker=dict(colors=colors),
    textinfo='percent',
    hoverinfo='label+percent'
)])

fig.update_layout(
    showlegend=True,
    margin=dict(t=10, b=10, l=10, r=10),
    height=280,
    annotations=[dict(text=f'<b>BEST TYPE</b><br>{current_type}<br>77.6%', x=0.5, y=0.5, font_size=14, showarrow=False)]
)

st.plotly_chart(fig, use_container_width=True)

# 7. Audio Comparison Player
st.markdown("### 노이즈 캔슬링 전/후 비교 청취")
st.caption("원본 오디오 (Raw Input)")
st.audio("https://www.w3schools.com/html/horse.ogg", format="audio/ogg")

st.caption("정제 오디오 (FFT Filtered)")
st.audio("https://www.w3schools.com/html/horse.ogg", format="audio/ogg")
```

---

## 5. Verification Checklist
- [ ] Ensure all 5 PNG character assets are tightly cropped to remove transparent boundary margins.
- [ ] Confirm `position: relative` on `.card-wrapper` and `position: absolute` on `.character-img`.
- [ ] Verify `z-index: 10` forces image rendering on top of `.result-card` border.
- [ ] Validate responsive scaling on viewport widths down to `360px`.
