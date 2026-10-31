import collections
import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_presentation():
    prs = Presentation()
    # 16:9 ratio
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    DARK_BLUE = RGBColor(0, 51, 102)
    GRAY = RGBColor(100, 100, 100)
    BLACK = RGBColor(0, 0, 0)
    
    # Standard Layouts
    TITLE_SLIDE_LAYOUT = prs.slide_layouts[0]
    TITLE_AND_CONTENT_LAYOUT = prs.slide_layouts[1]
    TITLE_ONLY_LAYOUT = prs.slide_layouts[5]

    def add_footer_and_number(slide, slide_num):
        # Footer
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(7.1), Inches(12), Inches(0.3))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = "IIIT Allahabad | Mid-Semester Project Evaluation"
        p.font.size = Pt(10)
        p.font.color.rgb = GRAY
        
        # Slide Number
        txBox2 = slide.shapes.add_textbox(Inches(12.5), Inches(7.1), Inches(0.5), Inches(0.3))
        tf2 = txBox2.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = str(slide_num)
        p2.font.size = Pt(10)
        p2.font.color.rgb = GRAY
        p2.alignment = PP_ALIGN.RIGHT

    def set_title(slide, text):
        title = slide.shapes.title
        title.text = text
        for p in title.text_frame.paragraphs:
            p.font.color.rgb = DARK_BLUE
            p.font.bold = True
            p.font.name = "Arial"

    def add_bullet(tf, text, level=0):
        p = tf.add_paragraph() if len(tf.paragraphs) > 0 and tf.paragraphs[0].text != "" else tf.paragraphs[0]
        p.text = text
        p.level = level
        p.font.size = Pt(18 - level*2)
        p.font.name = "Arial"

    # ==========================
    # SLIDE 1 - TITLE
    # ==========================
    slide = prs.slides.add_slide(TITLE_SLIDE_LAYOUT)
    title = slide.shapes.title
    title.text = "Selective Model-Impact and Temporal State-Change Gating\nfor Clustered Federated Learning under Data Drift"
    for p in title.text_frame.paragraphs:
        p.font.color.rgb = DARK_BLUE
        p.font.size = Pt(36)
        p.font.bold = True
        
    subtitle = slide.placeholders[1]
    subtitle.text = "Mid-Semester Project Evaluation\n\nDevansh Rai (IIB2024006)\nPrince Raj (IIT2024232)\nRishit Balaji (IIT2024242)\n\nDepartment of Information Technology\nIndian Institute of Information Technology, Allahabad"
    for p in subtitle.text_frame.paragraphs:
        p.font.size = Pt(20)
    add_footer_and_number(slide, 1)

    # ==========================
    # SLIDE 2 - MOTIVATION & PROBLEM
    # ==========================
    slide = prs.slides.add_slide(TITLE_AND_CONTENT_LAYOUT)
    set_title(slide, "Motivation & Problem Context")
    tf = slide.placeholders[1].text_frame
    tf.clear()
    
    add_bullet(tf, "Federated Learning (FL):")
    add_bullet(tf, "Clients collaboratively train models without sharing raw data (non-IID and heterogeneous).", 1)
    
    add_bullet(tf, "Clustered Federated Learning (CFL):")
    add_bullet(tf, "Clients grouped into similar clusters; each gets a specialized model.", 1)
    
    add_bullet(tf, "The Problem (Streaming Data Drift):")
    add_bullet(tf, "Client data distributions change over time, gradually destroying cluster homogeneity.", 1)
    add_bullet(tf, "FIELDING responds to detected drift by unconditionally re-clustering.", 1)
    
    add_bullet(tf, "Key Motivation:", 0)
    add_bullet(tf, "\"Distribution drift does not necessarily mean model-performance degradation.\"", 1)
    
    # Add visual chain box
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(5.8), Inches(12.3), Inches(1))
    tf2 = txBox.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Heterogeneous Clients  →  Clustered FL  →  Data Drift  →  Homogeneity Degrades  →  Adaptive Re-Clustering"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    p.alignment = PP_ALIGN.CENTER
    
    add_footer_and_number(slide, 2)

    # ==========================
    # SLIDE 3 - LITERATURE REVIEW
    # ==========================
    slide = prs.slides.add_slide(TITLE_ONLY_LAYOUT)
    set_title(slide, "Literature Review")
    
    rows = 7
    cols = 5
    table_shape = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.5), Inches(12.3), Inches(4.5))
    table = table_shape.table
    
    headers = ["System", "Year", "Methodology", "Dataset", "Key Features / Evaluation Focus"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h
        table.cell(0, i).text_frame.paragraphs[0].font.bold = True
        table.cell(0, i).text_frame.paragraphs[0].font.size = Pt(14)
        table.cell(0, i).fill.solid()
        table.cell(0, i).fill.fore_color.rgb = DARK_BLUE
        table.cell(0, i).text_frame.paragraphs[0].font.color.rgb = RGBColor(255,255,255)
        
    data = [
        ("FedAvg (McMahan et al.)", "2017", "Standard FL", "-", "Baseline distributed optimization"),
        ("IFCA (Ghosh et al.)", "2020", "Fixed K CFL", "-", "Loss-based static clustering"),
        ("FedProx (Li et al.)", "2020", "Proximal FL", "-", "Stability under heterogeneity"),
        ("FlexCFL (Duan et al.)", "2021", "Fixed K CFL", "-", "Parameter-based dynamic grouping"),
        ("FedDrift (Jothimurugesan et al.)", "2023", "Global re-cluster", "-", "High communication cost local-loss eval"),
        ("FIELDING (Li et al.) [BASE PAPER]", "2024", "Hybrid CFL", "FMoW", "L1 label drift, adaptive AIMD threshold")
    ]
    
    for r in range(len(data)):
        for c in range(len(data[r])):
            table.cell(r+1, c).text = data[r][c]
            table.cell(r+1, c).text_frame.paragraphs[0].font.size = Pt(14)
            if r == 5:
                table.cell(r+1, c).text_frame.paragraphs[0].font.bold = True
                
    add_footer_and_number(slide, 3)

    # ==========================
    # SLIDE 4 - RESEARCH GAPS
    # ==========================
    slide = prs.slides.add_slide(TITLE_ONLY_LAYOUT)
    set_title(slide, "Research Gaps & Base Paper Logic")
    
    # LEFT
    tx_left = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(5.5), Inches(4))
    tf_left = tx_left.text_frame
    p = tf_left.paragraphs[0]
    p.text = "Base Paper: FIELDING Logic"
    p.font.bold = True
    p.font.size = Pt(22)
    p.font.color.rgb = DARK_BLUE
    
    add_bullet(tf_left, "Data Drift Detected\n          ↓", 1)
    add_bullet(tf_left, "Re-Clustering Triggered\n          ↓", 1)
    add_bullet(tf_left, "Cluster Models Re-Initialized via Averaging", 1)

    # RIGHT
    tx_right = slide.shapes.add_textbox(Inches(6.5), Inches(1.5), Inches(6.5), Inches(4))
    tf_right = tx_right.text_frame
    p = tf_right.paragraphs[0]
    p.text = "Research Gaps Identified"
    p.font.bold = True
    p.font.size = Pt(22)
    p.font.color.rgb = DARK_BLUE
    
    add_bullet(tf_right, "1. Benign-Drift Over-Reaction", 0)
    add_bullet(tf_right, "A distribution shift may not reduce actual model accuracy.", 1)
    
    add_bullet(tf_right, "2. Single-Round Evaluation Noise", 0)
    add_bullet(tf_right, "One-round accuracy drops are often transient.", 1)
    
    add_bullet(tf_right, "3. State Blindness", 0)
    add_bullet(tf_right, "No comparison against a client's own historical drift behavior.", 1)
    
    # BOTTOM
    tx_bottom = slide.shapes.add_textbox(Inches(0.5), Inches(6), Inches(12.3), Inches(1))
    p = tx_bottom.text_frame.paragraphs[0]
    p.text = "Research Question: Can re-clustering be triggered selectively using evidence of genuine and sustained model impact?"
    p.font.bold = True
    p.font.italic = True
    p.font.size = Pt(22)
    p.alignment = PP_ALIGN.CENTER
    
    add_footer_and_number(slide, 4)

    # ==========================
    # SLIDE 5 - PROBLEM & OBJECTIVES
    # ==========================
    slide = prs.slides.add_slide(TITLE_AND_CONTENT_LAYOUT)
    set_title(slide, "Problem Statement & Objectives")
    tf = slide.placeholders[1].text_frame
    tf.clear()
    
    add_bullet(tf, "Problem Statement:")
    p = tf.paragraphs[0]
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    add_bullet(tf, "FIELDING treats every detected drift event as actionable, which causes excessive re-clustering and severe 'cluster churn' under shorter, dynamic training horizons.", 1)
    
    add_bullet(tf, "\nProject Objectives:")
    p = tf.paragraphs[-1]
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    add_bullet(tf, "1. Reproduce FIELDING: Run the baseline on FMoW (FedScale) and verify the published accuracy range.", 1)
    add_bullet(tf, "2. Design Selective Gating: Implement a three-stage mechanism (Impact + Smoothing + Temporal State).", 1)
    add_bullet(tf, "3. Counterfactual Evaluation: Evaluate avoided re-clustering telemetry over multiple simulation phases.", 1)
    
    add_footer_and_number(slide, 5)

    # ==========================
    # SLIDE 6 - METHODOLOGY
    # ==========================
    slide = prs.slides.add_slide(TITLE_ONLY_LAYOUT)
    set_title(slide, "Proposed Methodology: The Selective Pipeline")
    
    tx_top = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(12.3), Inches(1))
    tf_top = tx_top.text_frame
    p = tf_top.paragraphs[0]
    p.text = "FIELDING Baseline:\nData Drift → Re-cluster"
    p.font.size = Pt(16)
    p.font.color.rgb = GRAY
    
    p2 = tf_top.add_paragraph()
    p2.text = "PROPOSED:\nData Drift → Model Impact → Rolling Smoothing → Temporal State → Conditional Re-cluster"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = DARK_BLUE
    
    # 3 Boxes
    box_w = Inches(3.8)
    # Stage 1
    s1 = slide.shapes.add_textbox(Inches(0.5), Inches(3), box_w, Inches(2.5))
    s1.text_frame.text = "Stage 1:\nModel-Impact Gate\n\n• Measure Top-1 accuracy degradation.\n• Threshold τ_impact = 0.05 (5% drop required)."
    s1.text_frame.paragraphs[0].font.bold = True
    for p in s1.text_frame.paragraphs: p.font.size = Pt(16)
    
    # Stage 2
    s2 = slide.shapes.add_textbox(Inches(4.7), Inches(3), box_w, Inches(2.5))
    s2.text_frame.text = "Stage 2:\nRolling-Window Smoothing\n\n• Window W = 3.\n• Averages contiguous history to filter out transient one-round dips."
    s2.text_frame.paragraphs[0].font.bold = True
    for p in s2.text_frame.paragraphs: p.font.size = Pt(16)
    
    # Stage 3
    s3 = slide.shapes.add_textbox(Inches(8.9), Inches(3), box_w, Inches(2.5))
    s3.text_frame.text = "Stage 3:\nTemporal ε-Gate\n\n• Builds client state vector [D_L, D_X, D_C, M, V, P].\n• Adaptive ε threshold based on 95th-percentile state history."
    s3.text_frame.paragraphs[0].font.bold = True
    for p in s3.text_frame.paragraphs: p.font.size = Pt(16)
    
    tx_bot = slide.shapes.add_textbox(Inches(0.5), Inches(6), Inches(12.3), Inches(1))
    p = tx_bot.text_frame.paragraphs[0]
    p.text = "Final Rule: Re-cluster ⇔ (Smoothed Impact ≥ 5%) AND (Temporal State Change > ε)"
    p.font.size = Pt(22)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    
    add_footer_and_number(slide, 6)

    # ==========================
    # SLIDE 7 - BASELINE
    # ==========================
    slide = prs.slides.add_slide(TITLE_ONLY_LAYOUT)
    set_title(slide, "Baseline Replication & 500-Round Stress Test")
    
    # LEFT
    tx_left = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(5.5), Inches(4.5))
    tf_left = tx_left.text_frame
    p = tf_left.paragraphs[0]
    p.text = "Section A: 2000-Round Replication"
    p.font.bold = True
    p.font.size = Pt(20)
    p.font.color.rgb = DARK_BLUE
    
    add_bullet(tf_left, "Dataset: FMoW", 1)
    add_bullet(tf_left, "Framework: FedScale (ResNet-18, FedProx)", 1)
    add_bullet(tf_left, "Full-scale run: 2000 rounds", 1)
    add_bullet(tf_left, "Final recorded Top-1 accuracy: 59.38%", 1)
    add_bullet(tf_left, "Result was fully consistent with the published FIELDING accuracy levels.", 1)

    # RIGHT
    tx_right = slide.shapes.add_textbox(Inches(6.5), Inches(1.5), Inches(6.5), Inches(4.5))
    tf_right = tx_right.text_frame
    p = tf_right.paragraphs[0]
    p.text = "Section B: 500-Round Stress Test"
    p.font.bold = True
    p.font.size = Pt(20)
    p.font.color.rgb = DARK_BLUE
    
    add_bullet(tf_right, "Global / unclustered model climbed to ≈ 38%.", 1)
    add_bullet(tf_right, "Specialized clusters collapsed to 2–15%.", 1)
    add_bullet(tf_right, "Reason: Shorter dynamic horizon triggered constant unconditional cluster churn.", 1)
    
    p_alert = tf_right.add_paragraph()
    p_alert.text = "\nCore failure exposed: frequent unconditional re-clustering prevents cluster models from converging."
    p_alert.font.bold = True
    p_alert.font.size = Pt(18)
    p_alert.font.color.rgb = RGBColor(180, 0, 0)
    
    add_footer_and_number(slide, 7)

    # ==========================
    # SLIDE 8 - RESULTS
    # ==========================
    slide = prs.slides.add_slide(TITLE_ONLY_LAYOUT)
    set_title(slide, "Experimental Results & Counterfactuals")
    
    rows = 5
    cols = 3
    table_shape = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.5), Inches(12.3), Inches(4.5))
    table = table_shape.table
    
    table.columns[0].width = Inches(4.0)
    table.columns[1].width = Inches(4.15)
    table.columns[2].width = Inches(4.15)
    
    headers = ["Experimental Phase", "Specialized Cluster Accuracy", "Re-Clustering Frequency"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h
        table.cell(0, i).text_frame.paragraphs[0].font.bold = True
        table.cell(0, i).fill.solid()
        table.cell(0, i).fill.fore_color.rgb = DARK_BLUE
        table.cell(0, i).text_frame.paragraphs[0].font.color.rgb = RGBColor(255,255,255)
        
    data = [
        ("Unmodified FIELDING (Stress Test)", "2 – 15% (Collapse)", "249 / 249 (100%)"),
        ("Phase II: Model-Impact Gate", "45 – 69%", "52 / 249 (20.8%)"),
        ("Phase III: Impact + Smoothing (W=3)", "43 – 61% (Most Stable)", "13 / 249 (94.78% reduction)"),
        ("Phase IV: Full Pipeline (+ ε-Gate)", "40 – 55%", "114 / 249 (45.7%)")
    ]
    
    for r in range(len(data)):
        for c in range(len(data[r])):
            table.cell(r+1, c).text = data[r][c]
            table.cell(r+1, c).text_frame.paragraphs[0].font.size = Pt(16)
            if r == 2:  # Highlight Phase III
                table.cell(r+1, c).text_frame.paragraphs[0].font.bold = True
                table.cell(r+1, c).fill.solid()
                table.cell(r+1, c).fill.fore_color.rgb = RGBColor(230, 240, 250)

    # Note text
    tx_note = slide.shapes.add_textbox(Inches(0.5), Inches(6.2), Inches(12.3), Inches(0.8))
    p = tx_note.text_frame.paragraphs[0]
    p.text = "*Note: Phase III and Phase IV represent separate 500-round simulated telemetry runs. Because decisions change the accuracy trajectory directly, they are not a strict seed-controlled paired ablation."
    p.font.size = Pt(12)
    p.font.italic = True
    
    add_footer_and_number(slide, 8)

    # ==========================
    # SLIDE 9 - FINDINGS, LIMITATIONS
    # ==========================
    slide = prs.slides.add_slide(TITLE_ONLY_LAYOUT)
    set_title(slide, "Key Findings, Limitations & Future Work")
    
    box_w = Inches(4.0)
    
    # Left
    s1 = slide.shapes.add_textbox(Inches(0.3), Inches(1.5), box_w, Inches(5.5))
    s1.text_frame.text = "Key Findings"
    s1.text_frame.paragraphs[0].font.bold = True
    s1.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    s1.text_frame.paragraphs[0].font.size = Pt(20)
    add_bullet(s1.text_frame, "Model-impact gating substantially rescues specialized-cluster performance.", 1)
    add_bullet(s1.text_frame, "Rolling-window smoothing drastically reduces false triggers (249 → 13 events).", 1)
    add_bullet(s1.text_frame, "Temporal ε-gate avoided 0 additional re-clusterings in the Phase IV run.", 1)
    
    # Mid
    s2 = slide.shapes.add_textbox(Inches(4.6), Inches(1.5), box_w, Inches(5.5))
    s2.text_frame.text = "Limitations"
    s2.text_frame.paragraphs[0].font.bold = True
    s2.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    s2.text_frame.paragraphs[0].font.size = Pt(20)
    add_bullet(s2.text_frame, "Feature (D_X) and Context (D_C) drifts are currently zero-valued placeholders.", 1)
    add_bullet(s2.text_frame, "Phase III/IV are separate runs, not paired ablations.", 1)
    add_bullet(s2.text_frame, "Most ε-gate evaluations were still trapped in the 50-round warm-up.", 1)
    
    # Right
    s3 = slide.shapes.add_textbox(Inches(8.9), Inches(1.5), box_w, Inches(5.5))
    s3.text_frame.text = "Future Work"
    s3.text_frame.paragraphs[0].font.bold = True
    s3.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE
    s3.text_frame.paragraphs[0].font.size = Pt(20)
    add_bullet(s3.text_frame, "Activate D_X and D_C vector components.", 1)
    add_bullet(s3.text_frame, "Trace why ε-gate offered zero incremental filtering.", 1)
    add_bullet(s3.text_frame, "Run seed-controlled ablations over longer horizons.", 1)
    add_bullet(s3.text_frame, "Evaluate on Cityscapes or Waymo Open Dataset.", 1)
    
    for shape in [s1, s2, s3]:
        for p in shape.text_frame.paragraphs[1:]:
            p.font.size = Pt(16)
            
    add_footer_and_number(slide, 9)

    # ==========================
    # SLIDE 10 - CONCLUSION & REF
    # ==========================
    slide = prs.slides.add_slide(TITLE_ONLY_LAYOUT)
    set_title(slide, "Conclusion & Key References")
    
    # Top
    tx_top = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(12.3), Inches(2.5))
    tf_top = tx_top.text_frame
    p = tf_top.paragraphs[0]
    p.text = "Conclusion"
    p.font.bold = True
    p.font.size = Pt(20)
    p.font.color.rgb = DARK_BLUE
    
    add_bullet(tf_top, "1. FIELDING successfully reproduced; short-horizon testing exposed severe cluster churn.", 1)
    add_bullet(tf_top, "2. Selective model-impact gating and temporal smoothing substantially reduced unnecessary re-clustering and structurally improved cluster accuracy.", 1)
    add_bullet(tf_top, "3. The incremental value of the Temporal ε-Gate remains an open research question requiring controlled experiments.", 1)
    
    for p in tf_top.paragraphs[1:]: p.font.size = Pt(16)

    # Bottom
    tx_bot = slide.shapes.add_textbox(Inches(0.5), Inches(4.0), Inches(12.3), Inches(3.0))
    tf_bot = tx_bot.text_frame
    p = tf_bot.paragraphs[0]
    p.text = "Key References"
    p.font.bold = True
    p.font.size = Pt(20)
    p.font.color.rgb = DARK_BLUE
    
    refs = [
        "[1] B. McMahan et al., \"Communication-Efficient Learning of Deep Networks from Decentralized Data,\" AISTATS, 2017.",
        "[2] A. Ghosh et al., \"An Efficient Framework for Clustered Federated Learning,\" NeurIPS, 2020.",
        "[3] T. Li et al., \"Federated Optimization in Heterogeneous Networks,\" MLSys, 2020.",
        "[4] M. Duan et al., \"Flexible Clustered Federated Learning for Client-Level Data Distribution Shift,\" IEEE TPDS, 2021.",
        "[5] F. Lai et al., \"FedScale: Benchmarking Model and System Performance of Federated Learning at Scale,\" ICML, 2022.",
        "[6] J. Liu et al., \"Auxo: Efficient Federated Learning via Scalable Client Clustering,\" SoCC, 2023.",
        "[7] E. Jothimurugesan et al., \"Federated Learning under Distributed Concept Drift,\" AISTATS, 2023.",
        "[8] Y. Zhang et al., \"FedAC: An Adaptive Clustered Federated Learning Framework...\" arXiv:2403.16460, 2024.",
        "[9] M. Li et al., \"FIELDING: Clustered Federated Learning with Data Drift,\" arXiv:2411.01580, 2024. [Base Paper]"
    ]
    
    for r in refs:
        p = tf_bot.add_paragraph()
        p.text = r
        p.font.size = Pt(11)
        p.font.name = "Arial"
        
    add_footer_and_number(slide, 10)

    # Save
    prs.save("IIIT_Allahabad_MidSem_Evaluation.pptx")

create_presentation()
