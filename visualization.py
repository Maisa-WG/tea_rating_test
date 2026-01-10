import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import streamlit as st

# ==========================================
# 1. 核心计算逻辑
# ==========================================

def calculate_section_scores(scores):
    """
    将六因子得分为：前调(Top)、中调(Mid)、尾调(Base)
    """
    # 辅助函数：安全获取分数，默认为 0
    def get(key):
        return float(scores.get(key, 0))

    top = (get('优雅性') + get('辨识度')) / 2
    mid = (get('协调性') + get('饱和度')) / 2
    base = (get('持久性') + get('苦涩度')) / 2
    
    return top, mid, base

# ==========================================
# 2. 六因子雷达图 (Plotly)
# ==========================================

def plot_radar_chart(scores_data):
    """
    绘制单次评测的六因子雷达图
    """
    categories = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
    values = [float(scores_data.get(c, 0)) for c in categories]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='当前风味',
        line_color='#1E88E5',
        fillcolor='rgba(30, 136, 229, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=10, color="gray"),
                linecolor="rgba(0,0,0,0.1)"
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="black"),
                rotation=90,
                direction="clockwise"
            )
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ==========================================
# 3. 三段风味形态图 (Matplotlib)
# ==========================================

def plot_flavor_shape(scores_data):
    """
    绘制基于 '前中后' 三调的茶汤形态图
    """
    top, mid, base = calculate_section_scores(scores_data)
    
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    y = np.array([1, 2, 3]) 
    x = np.array([base, mid, top])
    
    y_new = np.linspace(1, 3, 300)
    try:
        spl = make_interp_spline(y, x, k=2)
        x_smooth = spl(y_new)
    except:
        x_smooth = np.interp(y_new, y, x)
    
    x_smooth = np.maximum(x_smooth, 0.1)

    colors = {'base': '#8B4513', 'mid': '#D2691E', 'top': '#FFD700'}
    
    mask_base = (y_new >= 1.0) & (y_new <= 1.6)
    ax.fill_betweenx(y_new[mask_base], -x_smooth[mask_base], x_smooth[mask_base], 
                     color=colors['base'], alpha=0.9, edgecolor=None)
    
    mask_mid = (y_new > 1.6) & (y_new <= 2.4)
    ax.fill_betweenx(y_new[mask_mid], -x_smooth[mask_mid], x_smooth[mask_mid], 
                     color=colors['mid'], alpha=0.85, edgecolor=None)
    
    mask_top = (y_new > 2.4) & (y_new <= 3.0)
    ax.fill_betweenx(y_new[mask_top], -x_smooth[mask_top], x_smooth[mask_top], 
                     color=colors['top'], alpha=0.8, edgecolor=None)

    ax.plot(x_smooth, y_new, color='black', linewidth=1, alpha=0.2)
    ax.plot(-x_smooth, y_new, color='black', linewidth=1, alpha=0.2)
    
    ax.axhline(y=1.6, color='white', linestyle=':', alpha=0.5)
    ax.axhline(y=2.4, color='white', linestyle=':', alpha=0.5)
    
    font_style = {'ha': 'center', 'va': 'center', 'color': 'white', 'fontweight': 'bold', 'fontsize': 12}
    ax.text(0, 2.7, f"Top\n{top:.1f}", **font_style)
    ax.text(0, 2.0, f"Mid\n{mid:.1f}", **font_style)
    ax.text(0, 1.3, f"Base\n{base:.1f}", **font_style)
    
    ax.axis('off')
    ax.set_xlim(-10, 10)
    ax.set_ylim(0.8, 3.2)
        
    return fig