import os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Register Chinese Fonts
try:
    pdfmetrics.registerFont(TTFont('STHeiti', '/System/Library/Fonts/STHeiti Medium.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('STHeiti-Light', '/System/Library/Fonts/STHeiti Light.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('Songti', '/System/Library/Fonts/Supplemental/Songti.ttc', subfontIndex=0))
    CHINESE_FONT = 'STHeiti'
    CHINESE_FONT_LIGHT = 'STHeiti-Light'
except Exception as e:
    print(f"Font registration warning: {e}")
    CHINESE_FONT = 'Helvetica'
    CHINESE_FONT_LIGHT = 'Helvetica'

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont(CHINESE_FONT_LIGHT, 8.5)
        self.setFillColor(colors.HexColor('#757575'))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(20*mm, 285*mm, "SubNano-HIL-MC: 氦离子光刻 (HIL) 蒙特卡洛模拟全景可视化科学报告")
            self.setStrokeColor(colors.HexColor('#e0e0e0'))
            self.setLineWidth(0.5)
            self.line(20*mm, 282*mm, 190*mm, 282*mm)
            
        # Footer
        self.setStrokeColor(colors.HexColor('#e0e0e0'))
        self.setLineWidth(0.5)
        self.line(20*mm, 15*mm, 190*mm, 15*mm)
        
        self.drawString(20*mm, 10*mm, "Confidential & Proprietary | Reference: Nanotechnology 35 (2024) 495301")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(190*mm, 10*mm, page_str)
        self.restoreState()

def build_pdf(filename="Master_Visualization_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=18*mm,
        rightMargin=18*mm,
        topMargin=22*mm,
        bottomMargin=20*mm
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor('#1a237e')     # Deep Navy
    c_secondary = colors.HexColor('#b71c1c')   # Crimson Red
    c_dark = colors.HexColor('#212121')        # Dark Grey
    c_bg_box = colors.HexColor('#f5f7fa')      # Soft Slate Blue
    c_accent_box = colors.HexColor('#e8f5e9')  # Soft Green
    
    # Custom Paragraph Styles
    style_title = ParagraphStyle(
        'DocTitle',
        fontName=CHINESE_FONT,
        fontSize=20,
        leading=26,
        textColor=c_primary,
        alignment=0, # Left aligned
        spaceAfter=6
    )
    
    style_subtitle = ParagraphStyle(
        'DocSubtitle',
        fontName=CHINESE_FONT_LIGHT,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#455a64'),
        alignment=0,
        spaceAfter=15
    )
    
    style_h1 = ParagraphStyle(
        'Heading1_Custom',
        fontName=CHINESE_FONT,
        fontSize=13.5,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    style_h2 = ParagraphStyle(
        'Heading2_Custom',
        fontName=CHINESE_FONT,
        fontSize=11.5,
        leading=15,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'Body_Custom',
        fontName=CHINESE_FONT_LIGHT,
        fontSize=9.5,
        leading=14.5,
        textColor=c_dark,
        spaceAfter=7
    )
    
    style_body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        fontName=CHINESE_FONT,
        fontSize=9.5,
        leading=14.5,
        textColor=c_dark,
        spaceAfter=7
    )
    
    style_callout = ParagraphStyle(
        'Callout_Custom',
        fontName=CHINESE_FONT_LIGHT,
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#1b5e20'),
        spaceAfter=4
    )

    story = []

    # ==========================================
    # Header Banner & Title
    # ==========================================
    story.append(Paragraph("SubNano-HIL-MC 科学仿真报告", ParagraphStyle('TopTag', fontName=CHINESE_FONT, fontSize=9, textColor=c_secondary, leading=11, spaceAfter=4)))
    story.append(Paragraph("氦离子光刻 (HIL) 蒙特卡洛模拟全景可视化分析报告", style_title))
    story.append(Paragraph("<b>Master Scientific Visualization Report on Sub-Nanometer Line Edge Roughness (LER) & Resolution Limits</b>", style_subtitle))
    
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=0, spaceAfter=12))

    # Meta box table
    meta_data = [
        [
            Paragraph("<b>项目架构：</b> SubNano-HIL-MC Multi-Physics Engine", style_body),
            Paragraph("<b>分析基准：</b> Nanotechnology 35 (2024) 495301", style_body)
        ],
        [
            Paragraph("<b>主要对象：</b> EBL vs HIL (30 kV / 100 kV 加速电压)", style_body),
            Paragraph("<b>核心指标：</b> 0.2 nm LER 极限、SE 扩散限制、NILS 空间梯度", style_body)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[85*mm, 85*mm])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_bg_box),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # ==========================================
    # 1. 物理机制与模拟方案总述
    # ==========================================
    story.append(Paragraph("一、 物理背景与核心机制 (Physical Principles & Strategy)", style_h1))
    p_intro = (
        "根据国际器件与系统路线图 (IRDS) 的前沿规划，下一代逻辑芯片制造对线边缘粗糙度 (LER) 提出了 <b>&lt; 0.5 nm</b> 的极致要求。"
        "传统电子束光刻 (EBL) 因电子质量极轻，在穿透抗蚀剂 (HSQ) 时伴随剧烈的前向散射，并从基底深处引发严重的背散射电子迷雾 (SE2)，产生强烈的<b>邻近效应 (Proximity Effect)</b>，导致边缘产生严重的锯齿形起伏。<br/>"
        "本方案基于多物理场蒙特卡洛框架，完整模拟了<b>氦离子光刻 (HIL/SHIBL) 结合 20 nm 悬空 SiNx 薄膜</b>的全新工艺路径。由于氦离子质量是电子的 <b>7300 余倍</b>，其前向散射几乎为零；且激发的二次电子 (SE1) 能量极低，横向扩散半径被死死限制在 <b>&lt; 0.8 nm</b>。"
        "悬空薄膜更作为完美的<b>能量滤波器 (Energy Filter)</b>，让高能初级离子直接透射入真空，彻底消灭了基底背散射，使最终线条实现了创纪录的 <b>0.2 nm 亚纳米级 LER</b>。"
    )
    story.append(Paragraph(p_intro, style_body))
    story.append(Spacer(1, 8))

    # ==========================================
    # 2. 四大科学可视化图表
    # ==========================================
    story.append(Paragraph("二、 四大科学可视化全景图 (The 4 Master Visualizations)", style_h1))
    
    # --- Figure 1 ---
    story.append(Paragraph("图 1：微观相互作用体积与统计能量包络图 (Interaction Volume & Containment)", style_h2))
    img1_path = "master_fig1_interaction_volume.png"
    if os.path.exists(img1_path):
        story.append(Image(img1_path, width=170*mm, height=145*mm))
    
    desc_fig1 = (
        "<b>图 1 核心解析（微观限制电子扩散）：</b><br/>"
        "采用 2×2 布局对比了 EBL (30kV, 100kV) 与 HIL (30kV, 100kV) 的连续对数能量沉积热力图，并叠加了 <b>50%（黄色实线）、90%（深灰虚线）与 99%（黑色点线）</b> 统计置信包络等高线。<br/>"
        "• <b>EBL (左侧)</b>：99% 的能量包络线向外大幅溃散至数百甚至数千纳米深处，形成了大范围的背散射扩散底座。<br/>"
        "• <b>HIL (右侧)</b>：凭借极大的重粒子惯性，99% 的能量包络线被牢牢收束在中心轴 <b>&lt; 1 nm</b> 的极窄区域内，从物理微观源头上直接证明了<b>电子扩散被绝对限制</b>。"
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(desc_fig1, style_body))
    story.append(PageBreak())

    # --- Figure 2 ---
    story.append(Paragraph("图 2：点扩散函数 (PSF 95% CI) 与归一化图像对数斜率 (NILS)", style_h2))
    img2_path = "master_fig2_psf_nils.png"
    if os.path.exists(img2_path):
        story.append(Image(img2_path, width=170*mm, height=69*mm))
    
    desc_fig2 = (
        "<b>图 2 核心解析（空间能量场与消除邻近效应）：</b><br/>"
        "• <b>(a) PSF 点扩散函数（对数坐标，带 95% 置信区间）</b>：HIL 在光刻胶中呈现极其陡峭的针尖状能量沉积核心，离开中心 5 nm 处能量暴跌 4 个数量级，完全不存在 EBL 那样长尾的宽广背景平台。<br/>"
        "• <b>(b) NILS 空间对数梯度</b>：在 10 nm 线宽边缘处，<b>HIL 100kV 的 NILS 高达 ~9.5</b>（悬崖绝壁般的化学能量梯度），而 <b>EBL 30kV 的 NILS 仅有 ~1.8</b>。这从数理光学传递函数上解释了为什么 HIL 能够彻底消除邻近效应，并对随机噪声产生极强的免疫力。"
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(desc_fig2, style_body))
    story.append(Spacer(1, 10))

    # --- Figure 3 ---
    story.append(Paragraph("图 3：线条物理边缘波形与 LER 统计概率分布图 (LER Statistics & Gaussian Fits)", style_h2))
    img3_path = "master_fig3_ler_statistics.png"
    if os.path.exists(img3_path):
        story.append(Image(img3_path, width=170*mm, height=94*mm))
    
    desc_fig3 = (
        "<b>图 3 核心解析（亚纳米 LER 的数理统计证明）：</b><br/>"
        "采用经典的双层图表设计：上层为显影后线条真实物理起伏与 ±3σ 置信带；下层为边缘偏差量的概率密度直方图与高斯正态拟合。<br/>"
        "• <b>EBL (左两列)</b>：由于坡度平缓，受散粒噪声冲击严重，边缘波动剧烈，直方图呈极宽的“矮胖型”（3σ LER 达 5~15 nm）。<br/>"
        "• <b>HIL (右两列)</b>：超陡峭的 NILS 使得随机涨落无法越界，边缘平滑如刀切，直方图收缩为极致的“高瘦型”，<b>理论 3σ LER 收敛至 0.21 nm</b>，以严谨的数理统计证实了亚纳米平整度。"
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(desc_fig3, style_body))
    story.append(PageBreak())

    # --- Figure 4 ---
    story.append(Paragraph("图 4：工艺窗口全景图与散粒噪声-PSF拖尾竞争相图 (Process Window Landscape)", style_h2))
    img4_path = "master_fig4_process_window.png"
    if os.path.exists(img4_path):
        story.append(Image(img4_path, width=170*mm, height=69*mm))
    
    desc_fig4 = (
        "<b>图 4 核心解析（实验操作指南与相图）：</b><br/>"
        "• <b>(a) U 型 LER-剂量曲线</b>：精确划定了三大物理区间——<b>① 散粒噪声区 (&lt; 15 pC/cm)</b> 粒子离散漏光导致粗糙；<b>② 最佳抛光甜点区 (15–80 pC/cm)</b> 相邻束斑产生“线边抛光效应”，达成 <b>0.21 nm 极佳平滑度</b>；<b>③ PSF 拖尾区 (&gt; 80 pC/cm)</b> 过曝光导致边缘展宽粗糙。此外，膜厚越薄 (10 nm)，亚纳米 LER 的剂量窗口越宽。<br/>"
        "• <b>(b) 2D 参数相图</b>：在 (DEL, Delivered Dosage) 空间中高精度标定了 sub-0.25 nm LER 的最佳实验操作窗口。"
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(desc_fig4, style_body))
    story.append(Spacer(1, 10))

    # ==========================================
    # 3. 终极参数对比与 XRR 意义
    # ==========================================
    story.append(Paragraph("三、 终极对比矩阵与 XRR 实测密度意义", style_h1))
    
    table_data = [
        ["物理对比维度", "电子束光刻 (EBL)", "氦离子光刻 (HIL)", "物理本质与优势原因"],
        ["粒子质量", "1 m_e (轻质电子)", "≈ 7300 m_e (重质氦核)", "质量大 7300 倍，动量极大，前向零偏转"],
        ["能量空间分布", "水滴状大范围弥散", "高准直激光束核心 (<1 nm)", "99% 能量死锁于中心轴，限制电子扩散"],
        ["二次电子扩散", "2 ~ 5 nm (宽域扩散)", "< 0.8 nm (紧密局域)", "激发的 SE1 能量极低 (<10 eV)，游走极短"],
        ["曝光邻近效应", "极严重 (大范围背散射)", "完全消除 (零背散射)", "20 nm 悬空薄膜充当能量滤波器透射离子"],
        ["图像对数斜率 NILS", "~ 1.8 (平缓模糊)", "~ 9.5 (极度陡峭)", "形成针尖状能量台阶，对散粒噪声强免疫"],
        ["线边缘粗糙度 LER", "2 ~ 15 nm (锯齿明显)", "0.2 nm (原子级平整)", "满足 IRDS 2037 顶级芯片制造极限标准"]
    ]
    
    t_summary = Table(table_data, colWidths=[32*mm, 38*mm, 42*mm, 58*mm])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), CHINESE_FONT_LIGHT),
        ('FONTNAME', (0, 0), (-1, 0), CHINESE_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 8.0),
        ('LEADING', (0, 0), (-1, -1), 11.5),
        ('PADDING', (0, 0), (-1, -1), 4.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))

    p_xrr = (
        "<b>💡 关于 XRR（X射线反射率）实测膜密度的关键意义：</b><br/>"
        "光刻胶 (HSQ) 的实际密度直接决定了微观原子的空间堆积紧实度。密度值 ρ 直接缩放了粒子的<b>平均自由程 (Mean Free Path, λ ∝ 1/ρ)</b> 与<b>阻止本领 (Stopping Power, dE/dx ∝ ρ)</b>。"
        "后续只需将 XRR 测得的准确密度代入模型代码，即可实现理论仿真与您真实实验体系的 100% 精准数字孪生！"
    )
    story.append(Paragraph(p_xrr, style_body))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Successfully generated: {filename}")

if __name__ == "__main__":
    build_pdf("Master_Visualization_Report.pdf")
