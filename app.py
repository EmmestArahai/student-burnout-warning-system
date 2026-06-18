import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Burnout Warning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; }
    
    .main { background-color: #0e1117; }
    
    .section-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 8px;
        border: 1px solid #2d3250;
    }
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.05em;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 8px 14px;
        border-radius: 8px;
        margin-bottom: 14px;
        display: inline-block;
    }
    .header-academic {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
    }
    .header-lifestyle {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white;
    }
    .header-personal {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        color: white;
    }
    .insight-box { background: #1a2332; border-left: 4px solid #1f77b4; padding: 15px; border-radius: 8px; margin: 8px 0; font-family: 'Inter', sans-serif; }
    .good-box    { background: #0d2818; border-left: 4px solid #2ca02c; padding: 15px; border-radius: 8px; margin: 8px 0; color: #a8e6b8; }
    .warn-box    { background: #2a1f0a; border-left: 4px solid #ff9800; padding: 15px; border-radius: 8px; margin: 8px 0; color: #ffd59e; }
    .danger-box  { background: #2a0d0d; border-left: 4px solid #d62728; padding: 15px; border-radius: 8px; margin: 8px 0; color: #ffb3b3; }
    
    .stMetric > div { background: #1e2130; border-radius: 12px; padding: 12px; border: 1px solid #2d3250; }
    .stDownloadButton > button { 
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important; font-weight: 600 !important;
        border-radius: 10px !important; border: none !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #11998e, #38ef7d) !important;
        color: #0e1117 !important; font-weight: 700 !important;
        border-radius: 10px !important; border: none !important;
        font-size: 1.1em !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load('models/burnout_model.pkl')
    features = joblib.load('models/feature_columns.pkl')
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv('data/featured_data.csv')

model, FEATURES = load_model()
df_data = load_data()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def build_features(inputs: dict) -> pd.DataFrame:
    d = inputs.copy()
    d['Sleep_Debt']         = max(0, 6 - d['Sleep_Hours_Per_Night'])
    d['Digital_Overload']   = d['Screen_Time_Hours'] + d['Social_Media_Hours']
    d['Academic_Pressure']  = d['Study_Hours_Per_Day'] * (1 - d['Assignments_Completed'] / 100)
    d['Lifestyle_Balance']  = d['Physical_Activity_Hours_Per_Week'] / 7 + d['Sleep_Hours_Per_Night']
    d['Burnout_Risk_Score'] = (
        d['Sleep_Debt'] * 2.0
        + d['Digital_Overload'] * 0.5
        + d['Academic_Pressure'] * 1.5
        + d['Study_Hours_Per_Day'] * 0.8
        - d['Lifestyle_Balance'] * 1.0
        - d['Motivation_Level'] * 0.5
    )
    return pd.DataFrame([d])[FEATURES]

def get_risk_level(prob):
    if prob >= 0.70:
        return "HIGH", "🚨", "#d62728", "danger-box"
    elif prob >= 0.45:
        return "MODERATE", "⚠️", "#ff9800", "warn-box"
    else:
        return "LOW", "✅", "#2ca02c", "good-box"

def score_to_stars(value, max_val, reverse=False):
    """Convert numeric value to visual bar"""
    pct = value / max_val
    if reverse:
        pct = 1 - pct
    filled = int(pct * 10)
    return "█" * filled + "░" * (10 - filled)

# ─────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────
def generate_pdf(inputs, prob, risk_level, tips_bad, tips_good):
    buffer = BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                 fontSize=20, textColor=colors.HexColor('#1a1a2e'),
                                 spaceAfter=6)
    story.append(Paragraph("🧠 Student Burnout Assessment Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.3*cm))

    # Risk Result
    risk_colors = {"LOW": colors.green, "MODERATE": colors.orange, "HIGH": colors.red}
    result_style = ParagraphStyle('Result', parent=styles['Heading1'],
                                  fontSize=16, textColor=risk_colors[risk_level])
    story.append(Paragraph(f"Burnout Risk Level: {risk_level}", result_style))
    story.append(Paragraph(f"Risk Probability: <b>{prob:.1%}</b>", styles['Normal']))
    story.append(Spacer(1, 0.3*cm))

    # What this means
    story.append(Paragraph("📖 What does this mean?", styles['Heading2']))
    meanings = {
        "LOW":      "Your current lifestyle shows low signs of burnout. You are managing your academic and personal life well. Keep maintaining these healthy habits!",
        "MODERATE": "Some warning signs of burnout have been detected. This means you may be feeling more tired, stressed, or unmotivated than usual. It's a good time to review your daily habits before things get worse.",
        "HIGH":     "Strong signs of burnout detected. Burnout can seriously affect your mental health, academic performance, and overall wellbeing. We strongly recommend speaking with a school counselor or trusted adult."
    }
    story.append(Paragraph(meanings[risk_level], styles['Normal']))
    story.append(Spacer(1, 0.4*cm))

    # Input Summary
    story.append(Paragraph("📋 Your Input Summary", styles['Heading2']))
    table_data = [["Factor", "Your Value", "Healthy Range"]]
    reference = [
        ("Sleep Hours/Night",       f"{inputs['Sleep_Hours_Per_Night']}h",      "7–9 hours"),
        ("Study Hours/Day",         f"{inputs['Study_Hours_Per_Day']}h",        "4–6 hours"),
        ("Screen Time/Day",         f"{inputs['Screen_Time_Hours']}h",          "< 6 hours"),
        ("Social Media/Day",        f"{inputs['Social_Media_Hours']}h",         "< 2 hours"),
        ("Physical Activity/Week",  f"{inputs['Physical_Activity_Hours_Per_Week']}h", "> 3 hours"),
        ("Attendance",              f"{inputs['Attendance_Percentage']}%",      "> 80%"),
        ("Assignments Completed",   f"{inputs['Assignments_Completed']}%",      "> 70%"),
        ("Motivation Level",        f"{inputs['Motivation_Level']}/10",         "> 6/10"),
    ]
    for row in reference:
        table_data.append(list(row))

    t = Table(table_data, colWidths=[6*cm, 4*cm, 5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('PADDING',    (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # Warnings
    if tips_bad:
        story.append(Paragraph("⚠️ Areas Needing Attention", styles['Heading2']))
        for tip in tips_bad:
            clean = tip.replace("**", "")
            story.append(Paragraph(f"• {clean}", styles['Normal']))
        story.append(Spacer(1, 0.3*cm))

    # Positives
    if tips_good:
        story.append(Paragraph("✅ What You're Doing Well", styles['Heading2']))
        for tip in tips_good:
            clean = tip.replace("**", "")
            story.append(Paragraph(f"• {clean}", styles['Normal']))
        story.append(Spacer(1, 0.3*cm))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    disc_style = ParagraphStyle('Disc', parent=styles['Normal'],
                                fontSize=8, textColor=colors.grey)
    story.append(Paragraph(
        "Disclaimer: This report is generated by an AI model for educational purposes only. "
        "It is not a medical or psychological diagnosis. If you are struggling, please reach out "
        "to a qualified mental health professional.", disc_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("🧠 Burnout Warning System")
    st.markdown("---")
    st.markdown("""
    **How to use this app:**
    1. Go to **Assessment** tab
    2. Fill in your daily habits
    3. Click **Analyze** to get your results
    4. Download your PDF report
    
    ---
    **What is Burnout?**
    
    Burnout is a state of chronic stress that leads to physical and emotional exhaustion, 
    cynicism, and feelings of ineffectiveness.
    
    ---
    """)
    st.caption("Built with ❤️ using Python & Streamlit")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏠 Burnout Assessment", "📊 EDA Dashboard", "ℹ️ Guide & About"])

# ═══════════════════════════════════════════════════════════
# TAB 1: ASSESSMENT
# ═══════════════════════════════════════════════════════════
with tab1:
    st.header("📝 Tell us about your daily routine")
    st.markdown("*Answer honestly — this tool is for your benefit. All data stays on your device.*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-card"><span class="section-header header-academic">📚 Academic Habits</span></div>', unsafe_allow_html=True)
        study_hours   = st.slider("📖 How many hours do you study per day?", 0.0, 12.0, 4.0, 0.1,
                                   help="Include all self-study, homework, and revision time")
        attendance    = st.slider("🏫 What is your class attendance rate?", 40.0, 100.0, 80.0, 0.5,
                                   help="Percentage of classes you attend")
        assignments   = st.slider("✅ What % of assignments do you complete?", 0.0, 100.0, 75.0, 1.0,
                                   help="Percentage of assigned tasks you finish on time")
        participation = st.slider("🙋 Class participation score (0–10)", 0.0, 10.0, 6.0, 0.1,
                                   help="How actively you engage in class discussions")
        prev_gpa      = st.slider("🎓 Previous semester GPA (0–10)", 0.0, 10.0, 6.5, 0.1,
                                   help="Your GPA from last semester")

    with col2:
        st.markdown('<div class="section-card"><span class="section-header header-lifestyle">🌙 Lifestyle & Health</span></div>', unsafe_allow_html=True)
        sleep_hours   = st.slider("😴 How many hours do you sleep per night?", 3.0, 10.0, 7.0, 0.1,
                                   help="Average sleep duration. Healthy range: 7–9 hours")
        screen_time   = st.slider("💻 Total screen time per day (hours)", 1.0, 15.0, 6.0, 0.1,
                                   help="All devices: phone, laptop, TV combined")
        social_media  = st.slider("📱 Social media usage per day (hours)", 0.0, 10.0, 3.0, 0.1,
                                   help="TikTok, Instagram, Facebook, etc.")
        physical      = st.slider("🏃 Physical activity per week (hours)", 0.0, 15.0, 3.0, 0.5,
                                   help="Exercise, sports, gym, walking — anything active")
        ai_usage      = st.slider("🤖 AI tool usage per week (hours)", 0.0, 25.0, 5.0, 0.5,
                                   help="ChatGPT, Claude, Copilot, etc.")

    with col3:
        st.markdown('<div class="section-card"><span class="section-header header-personal">🧩 Personal Context</span></div>', unsafe_allow_html=True)
        age           = st.slider("🎂 Your age", 16, 25, 20,
                                   help="Your current age")
        motivation    = st.slider("💪 Motivation level (0–10)", 1.0, 10.0, 6.0, 0.1,
                                   help="0 = completely unmotivated, 10 = extremely motivated")
        extracurr     = st.slider("🎭 Extracurricular involvement (0–10)", 0.0, 10.0, 4.0, 0.1,
                                   help="Clubs, societies, volunteering, hobbies outside class")
        gender        = st.selectbox("👤 Gender", ["Male", "Female", "Other"])
        internet      = st.selectbox("🌐 Internet quality at home", ["Good", "Average", "Poor"],
                                      help="This affects your ability to study and access resources")
        part_time     = st.selectbox("💼 Do you have a part-time job?", ["No", "Yes"],
                                      help="Working while studying adds significant stress load")

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze My Burnout Risk", use_container_width=True, type="primary")

    if analyze_btn:
        gender_map   = {"Female": 0, "Male": 1, "Other": 2}
        internet_map = {"Poor": 0, "Average": 1, "Good": 2}
        part_time_map= {"No": 0, "Yes": 1}

        inputs = {
            "Age":                              age,
            "Gender":                           gender_map[gender],
            "Study_Hours_Per_Day":              study_hours,
            "Sleep_Hours_Per_Night":            sleep_hours,
            "Screen_Time_Hours":                screen_time,
            "Social_Media_Hours":               social_media,
            "Attendance_Percentage":            attendance,
            "Assignments_Completed":            assignments,
            "Class_Participation_Score":        participation,
            "Physical_Activity_Hours_Per_Week": physical,
            "Internet_Quality":                 internet_map[internet],
            "Part_Time_Job":                    part_time_map[part_time],
            "Extracurricular_Involvement":      extracurr,
            "AI_Tool_Usage_Hours_Per_Week":     ai_usage,
            "Previous_Semester_GPA":            prev_gpa,
            "Motivation_Level":                 motivation,
        }

        X            = build_features(inputs)
        prob         = model.predict_proba(X)[0][1]
        risk_level, icon, color, box_class = get_risk_level(prob)

        # ── RESULT HEADER ──
        st.markdown("## 📊 Your Burnout Risk Assessment Results")

        # Risk gauge using plotly
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prob * 100,
            title = {'text': "Burnout Risk Score", 'font': {'size': 20}},
            number= {'suffix': "%", 'font': {'size': 36}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar':  {'color': color},
                'steps': [
                    {'range': [0,  45], 'color': '#d4edda'},
                    {'range': [45, 70], 'color': '#fff3cd'},
                    {'range': [70,100], 'color': '#f8d7da'},
                ],
                'threshold': {
                    'line':  {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))

        g1, g2 = st.columns([1, 1])
        with g1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with g2:
            st.markdown(f"### {icon} Risk Level: **{risk_level}**")
            risk_explain = {
                "LOW":      ("You're doing well! Your habits suggest a **healthy balance** between study and personal life. "
                             "The model found fewer than 45% probability of burnout signs based on your lifestyle."),
                "MODERATE": ("Some patterns in your lifestyle **resemble early burnout signs**. "
                             "This doesn't mean you're burned out — but it's a signal worth paying attention to. "
                             "Small changes now can prevent bigger problems later."),
                "HIGH":     ("Your lifestyle patterns strongly match those associated with **burnout and high stress**. "
                             "This is serious and worth addressing. Please consider talking to someone you trust "
                             "— a counselor, teacher, or family member.")
            }
            st.markdown(risk_explain[risk_level])

        st.markdown("---")

        # ── BREAKDOWN METRICS ──
        st.markdown("### 🔬 What's Affecting Your Score?")
        st.markdown("*Here's how each area of your life is contributing to your burnout risk:*")

        sleep_debt       = max(0, 6 - sleep_hours)
        digital_overload = screen_time + social_media
        acad_pressure    = study_hours * (1 - assignments / 100)
        lifestyle_bal    = physical / 7 + sleep_hours

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            status = "🔴 Concerning" if sleep_hours < 6 else ("🟡 Borderline" if sleep_hours < 7 else "🟢 Healthy")
            st.metric("😴 Sleep Quality", f"{sleep_hours}h/night", status)
            st.caption("**Why it matters:** Sleep is the #1 recovery mechanism for your brain. "
                       "Less than 7h consistently leads to poor memory, mood swings, and burnout.")

        with m2:
            status = "🔴 Overloaded" if digital_overload > 10 else ("🟡 Watch out" if digital_overload > 7 else "🟢 Healthy")
            st.metric("📱 Digital Load", f"{digital_overload:.1f}h/day", status)
            st.caption("**Why it matters:** Excessive screen time overstimulates your brain, "
                       "reduces focus, and disrupts sleep — all major burnout contributors.")

        with m3:
            status = "🔴 High" if acad_pressure > 3 else ("🟡 Moderate" if acad_pressure > 1.5 else "🟢 Manageable")
            st.metric("📚 Academic Pressure", f"{acad_pressure:.1f}", status)
            st.caption("**Why it matters:** This score combines how much you study vs how much you actually complete. "
                       "High study hours with low completion = high unproductive stress.")

        with m4:
            status = "🟢 Balanced" if lifestyle_bal > 8 else ("🟡 Could improve" if lifestyle_bal > 6 else "🔴 Imbalanced")
            st.metric("⚖️ Lifestyle Balance", f"{lifestyle_bal:.1f}", status)
            st.caption("**Why it matters:** This combines your sleep and physical activity. "
                       "A balanced lifestyle builds resilience against stress and burnout.")

        st.markdown("---")

        # ── PERSONALIZED TIPS ──
        st.markdown("### 💡 Your Personalized Action Plan")
        st.markdown("*Based on your inputs, here's what needs attention and what you're already doing right:*")

        tips_bad  = []
        tips_good = []

        # Sleep
        if sleep_hours < 6:
            tips_bad.append(("😴 Critical: Sleep Deprivation",
                             f"You sleep only {sleep_hours}h/night. Scientists recommend 7–9 hours. "
                             "Sleep deprivation impairs memory consolidation, emotional regulation, and decision-making. "
                             "**Action:** Try going to bed 30 minutes earlier each week until you reach 7+ hours."))
        elif sleep_hours < 7:
            tips_bad.append(("😴 Warning: Borderline Sleep",
                             f"At {sleep_hours}h/night, you're slightly below the recommended 7–9 hours. "
                             "**Action:** Aim to add just 30 more minutes of sleep per night."))
        else:
            tips_good.append(("😴 Great Sleep Habits",
                              f"Sleeping {sleep_hours}h/night is within the healthy range. Keep it up!"))

        # Digital
        if digital_overload > 10:
            tips_bad.append(("📱 Warning: Digital Overload",
                             f"Your total screen + social media time is {digital_overload:.1f}h/day. "
                             "This is very high. Excessive screen time is linked to anxiety, poor sleep, and reduced attention span. "
                             "**Action:** Use app timers. Try a 1-hour screen-free period before bed."))
        elif digital_overload > 7:
            tips_bad.append(("📱 Caution: High Screen Time",
                             f"{digital_overload:.1f}h of daily screen time is above average. "
                             "**Action:** Try reducing social media to under 1.5h/day."))
        else:
            tips_good.append(("📱 Healthy Digital Habits",
                              f"Your screen time of {digital_overload:.1f}h/day is manageable. Well done!"))

        # Study
        if study_hours > 8:
            tips_bad.append(("📚 Warning: Over-Studying",
                             f"Studying {study_hours}h/day without proper breaks leads to diminishing returns. "
                             "Research shows that 4–6 focused hours beats 8+ unfocused hours. "
                             "**Action:** Use the Pomodoro technique: 25 min study, 5 min break."))
        elif study_hours < 2:
            tips_bad.append(("📚 Warning: Under-Studying",
                             f"Only {study_hours}h/day of study may not be enough to keep up with coursework. "
                             "**Action:** Try to gradually increase to at least 3–4 focused hours per day."))
        else:
            tips_good.append(("📚 Reasonable Study Hours",
                              f"{study_hours}h/day is a healthy study duration. Focus on quality over quantity!"))

        # Physical activity
        if physical < 1.5:
            tips_bad.append(("🏃 Warning: Sedentary Lifestyle",
                             f"Only {physical}h of exercise per week is very low. "
                             "Physical activity is one of the most powerful stress-busters known to science — "
                             "it releases endorphins, improves sleep, and boosts mood. "
                             "**Action:** Start with just 20-minute walks 3x/week. Any movement counts!"))
        elif physical < 3:
            tips_bad.append(("🏃 Caution: Low Activity",
                             f"{physical}h/week of activity is below the recommended 3–5 hours. "
                             "**Action:** Add one more activity session per week."))
        else:
            tips_good.append(("🏃 Active Lifestyle",
                              f"{physical}h/week of physical activity is great! Exercise is one of the best burnout preventions."))

        # Motivation
        if motivation < 4:
            tips_bad.append(("💪 Warning: Low Motivation",
                             f"A motivation score of {motivation}/10 is concerning. Low motivation is often an early symptom of burnout, "
                             "depression, or excessive stress — not laziness. "
                             "**Action:** Talk to a counselor. Break goals into tiny, achievable steps. Celebrate small wins."))
        elif motivation >= 7:
            tips_good.append(("💪 High Motivation",
                              f"A motivation score of {motivation}/10 is excellent! This is a strong protective factor against burnout."))

        # Attendance
        if attendance < 70:
            tips_bad.append(("🏫 Warning: Low Attendance",
                             f"{attendance}% attendance means you're missing significant learning time. "
                             "This creates catch-up stress later. **Action:** Identify what's keeping you from class and address it."))
        else:
            tips_good.append(("🏫 Good Attendance",
                              f"{attendance}% attendance shows commitment to your education. Keep showing up!"))

        # Part time
        if part_time == "Yes" and study_hours > 6:
            tips_bad.append(("💼 Caution: Work + Heavy Study Load",
                             "Having a part-time job while studying 6+ hours/day is a significant stressor. "
                             "**Action:** Make sure you're protecting your sleep and leisure time carefully."))

        # Display tips
        ta, tb = st.columns(2)
        with ta:
            st.markdown("#### ⚠️ Areas Needing Attention")
            if tips_bad:
                for title, body in tips_bad:
                    st.markdown(f'<div class="warn-box"><b>{title}</b><br>{body}</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="good-box">No major red flags! You\'re doing great across all areas.</div>',
                            unsafe_allow_html=True)

        with tb:
            st.markdown("#### ✅ What You're Doing Well")
            if tips_good:
                for title, body in tips_good:
                    st.markdown(f'<div class="good-box"><b>{title}</b><br>{body}</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">Keep working on the areas above — improvements will show!</div>',
                            unsafe_allow_html=True)

        st.markdown("---")

        # ── COMPARISON WITH DATASET ──
        st.markdown("### 📈 How Do You Compare to Other Students?")
        st.markdown("*Your values vs. the average student in our dataset of 10,000 students:*")

        compare_metrics = {
            "Study Hours/Day":    (study_hours,  df_data['Study_Hours_Per_Day'].mean(),  "hours"),
            "Sleep Hours/Night":  (sleep_hours,  df_data['Sleep_Hours_Per_Night'].mean(),"hours"),
            "Screen Time/Day":    (screen_time,  df_data['Screen_Time_Hours'].mean(),    "hours"),
            "Physical Activity":  (physical,     df_data['Physical_Activity_Hours_Per_Week'].mean(), "h/week"),
            "Motivation":         (motivation,   df_data['Motivation_Level'].mean(),     "/10"),
        }

        comp_df = pd.DataFrame([
            {"Metric": k, "You": v[0], "Average Student": round(v[1], 2), "Unit": v[2]}
            for k, v in compare_metrics.items()
        ])

        fig_compare = px.bar(
            comp_df.melt(id_vars=["Metric", "Unit"], value_vars=["You", "Average Student"]),
            x="Metric", y="value", color="variable", barmode="group",
            color_discrete_map={"You": "#1f77b4", "Average Student": "#aec7e8"},
            labels={"value": "Value", "variable": ""},
            title="You vs. Average Student"
        )
        fig_compare.update_layout(height=350, margin=dict(t=40, b=0))
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("---")

        # ── PDF EXPORT ──
        st.markdown("### 📄 Download Your Report")
        pdf_buffer = generate_pdf(inputs, prob, risk_level,
                                  [f"{t[0]}: {t[1]}" for t in tips_bad],
                                  [f"{t[0]}: {t[1]}" for t in tips_good])
        st.download_button(
            label     = "⬇️ Download PDF Report",
            data      = pdf_buffer,
            file_name = f"burnout_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime      = "application/pdf",
            use_container_width=True
        )
        st.caption("*The PDF contains your full assessment including all insights and recommendations.*")

# ═══════════════════════════════════════════════════════════
# TAB 2: EDA DASHBOARD
# ═══════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Dataset Explorer — Understanding Student Burnout Patterns")
    st.markdown("*Explore patterns from our dataset of 10,000 students. "
                "These charts show what the data says about student burnout.*")
    st.markdown("---")

    # Chart 1: Burnout distribution
    c1, c2 = st.columns(2)
    with c1:
        burnout_counts = df_data['Burnout_Label_v2'].value_counts().reset_index()
        burnout_counts.columns = ['Status', 'Count']
        burnout_counts['Status'] = burnout_counts['Status'].map({0: '✅ Safe', 1: '🚨 At Risk'})
        fig1 = px.pie(burnout_counts, values='Count', names='Status',
                      title='Burnout Risk Distribution in Dataset',
                      color_discrete_sequence=['#2ca02c', '#d62728'])
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("**What this shows:** 56% of students in the dataset show burnout risk indicators. "
                   "This highlights how common burnout is among students.")

    with c2:
        fig2 = px.histogram(df_data, x='Stress_Level', nbins=30,
                            title='How Stressed Are Students? (Distribution)',
                            color_discrete_sequence=['#ff7f0e'],
                            labels={'Stress_Level': 'Stress Level (1=Low, 10=High)'})
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("**What this shows:** Stress levels follow a roughly normal distribution, "
                   "with most students feeling moderate stress (4–7 out of 10).")

    st.markdown("---")

    # Chart 2: Sleep vs Productivity
    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.scatter(df_data.sample(500), x='Sleep_Hours_Per_Night', y='Productivity_Score',
                          color='Burnout_Label_v2',
                          color_discrete_map={0: '#2ca02c', 1: '#d62728'},
                          title='Sleep Hours vs Productivity Score',
                          labels={'Sleep_Hours_Per_Night': 'Sleep Hours/Night',
                                  'Productivity_Score': 'Productivity Score',
                                  'Burnout_Label_v2': 'Burnout Risk'},
                          opacity=0.6)
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("**Key insight:** Students who sleep more tend to have higher productivity. "
                   "Red dots (at-risk) are clustered at lower sleep and productivity levels.")

    with c4:
        fig4 = px.scatter(df_data.sample(500), x='Screen_Time_Hours', y='Productivity_Score',
                          color='Burnout_Label_v2',
                          color_discrete_map={0: '#2ca02c', 1: '#d62728'},
                          title='Screen Time vs Productivity Score',
                          labels={'Screen_Time_Hours': 'Screen Time (hours/day)',
                                  'Productivity_Score': 'Productivity Score',
                                  'Burnout_Label_v2': 'Burnout Risk'},
                          opacity=0.6)
        fig4.update_layout(height=380)
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("**Key insight:** Higher screen time is associated with lower productivity. "
                   "Students with burnout risk tend to have more screen exposure.")

    st.markdown("---")

    # Chart 3: Average metrics by burnout status
    st.subheader("📋 Average Habits: Safe Students vs At-Risk Students")
    st.markdown("*This table compares the average lifestyle of students with and without burnout risk.*")

    compare_cols = ['Sleep_Hours_Per_Night', 'Study_Hours_Per_Day', 'Screen_Time_Hours',
                    'Social_Media_Hours', 'Physical_Activity_Hours_Per_Week',
                    'Motivation_Level', 'Attendance_Percentage', 'Assignments_Completed']

    avg_by_burnout = df_data.groupby('Burnout_Label_v2')[compare_cols].mean().round(2)
    avg_by_burnout.index = ['✅ Safe (0)', '🚨 At Risk (1)']
    avg_by_burnout.columns = ['Sleep (h)', 'Study (h)', 'Screen (h)',
                               'Social Media (h)', 'Exercise (h/wk)',
                               'Motivation', 'Attendance %', 'Assignments %']

    # Fix: sau melt(), cột group tên là 'variable' không phải 'Burnout_Label_v2'
    melted = avg_by_burnout.T.reset_index()
    melted.columns.name = None
    melted = melted.rename(columns={'index': 'Metric'})
    melted = melted.melt(id_vars='Metric', var_name='Student Group', value_name='Average Value')

    fig5 = px.bar(
                    melted,
                    x='Metric', y='Average Value', color='Student Group', barmode='group',
                    title='Average Lifestyle Metrics: Safe vs At-Risk Students',
                    color_discrete_map={'✅ Safe (0)': '#2ca02c', '🚨 At Risk (1)': '#d62728'},
                )
    fig5.update_layout(height=420, xaxis_tickangle=-30,
                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                   font=dict(family='Inter', color='white'))
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("**What to notice:** Safe students sleep more, exercise more, and are more motivated. "
               "At-risk students have higher screen time and lower assignment completion rates.")

    st.markdown("---")

    # Chart 4: Correlation heatmap
    st.subheader("🌡️ What Factors Most Affect Productivity?")
    corr_cols = ['Sleep_Hours_Per_Night', 'Screen_Time_Hours', 'Study_Hours_Per_Day',
                 'Physical_Activity_Hours_Per_Week', 'Motivation_Level',
                 'Stress_Level', 'Productivity_Score', 'Burnout_Risk_Score']
    corr_matrix = df_data[corr_cols].corr()
    fig6 = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                     color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                     title='Correlation Matrix — Key Variables')
    fig6.update_layout(height=450)
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("**How to read this:** Values close to **+1.0** mean two factors move together (positive relationship). "
               "Values close to **-1.0** mean they move in opposite directions. "
               "Sleep and Motivation show the strongest positive links to Productivity.")

# ═══════════════════════════════════════════════════════════
# TAB 3: GUIDE & ABOUT
# ═══════════════════════════════════════════════════════════
with tab3:
    st.header("ℹ️ Guide: Understanding Your Results")

    st.markdown("""
    ## 🤔 What is Burnout?
    Burnout is not just feeling tired after a long day. It's a **chronic state of physical and emotional exhaustion**
    caused by prolonged stress — most often from academic, work, or personal pressures.
    
    **Signs of burnout include:**
    - Constant fatigue even after sleeping
    - Loss of motivation and interest in things you used to enjoy
    - Feeling detached or cynical about your studies
    - Difficulty concentrating
    - Physical symptoms: headaches, getting sick often
    
    ---
    
    ## 📊 Understanding the Risk Score
    
    Our AI model analyzes **20 different factors** from your lifestyle and academic habits,
    then calculates a probability that you show signs of burnout.
    
    | Risk Level | Score Range | What it means |
    |------------|------------|---------------|
    | ✅ LOW | 0–44% | Healthy balance detected. Keep it up! |
    | ⚠️ MODERATE | 45–69% | Early warning signs. Time to make small changes. |
    | 🚨 HIGH | 70–100% | Strong burnout indicators. Please seek support. |
    
    ---
    
    ## 🔬 Understanding the 4 Key Metrics
    
    ### 😴 Sleep Quality
    Sleep is your brain's reset button. During sleep, your brain consolidates memories,
    repairs stress damage, and regulates emotions. **Less than 7 hours consistently** impairs all of these.
    
    ### 📱 Digital Load  
    This combines your screen time and social media hours. High digital exposure
    **overstimulates your brain's reward system**, making it harder to focus, sleep, and feel satisfied
    with real life. It's also strongly linked to anxiety and depression in students.
    
    ### 📚 Academic Pressure Index
    This isn't just about how much you study — it measures **productive vs unproductive effort**.
    If you study 8 hours but only complete 40% of assignments, that's high pressure with low output.
    The goal is focused, efficient studying.
    
    ### ⚖️ Lifestyle Balance
    This combines sleep and physical activity — your two most powerful natural stress reducers.
    Exercise releases endorphins (natural mood boosters) and improves sleep quality.
    Even 20 minutes of walking daily makes a measurable difference.
    
    ---
    
    ## 🆘 When to Seek Help
    
    This app is a **self-awareness tool**, not a medical diagnosis. But if you consistently score HIGH,
    or feel overwhelmed, please reach out to:
    
    - 🏫 **Your school/university counseling center**
    - 👨‍👩‍👧 **A trusted family member or friend**
    - 🩺 **A doctor or mental health professional**
    - 📞 **Mental health hotlines** available in your country
    
    *Remember: Seeking help is a sign of strength, not weakness.*
    
    ---
    
    ## 🤖 About the AI Model
    
    | Detail | Value |
    |--------|-------|
    | Algorithm | XGBoost Classifier |
    | Training Data | 10,000 student records |
    | Features Used | 20 lifestyle & academic factors |
    | Model Accuracy | ~60% |
    | ROC-AUC Score | 0.656 |
    
    **What is ROC-AUC?** It measures how well the model distinguishes between at-risk and safe students.
    0.5 = random guessing, 1.0 = perfect. Our score of 0.656 means the model performs
    meaningfully better than chance.
    
    **Disclaimer:** This tool is for educational and self-reflection purposes only.
    It is not a clinical diagnosis tool.
    """)