import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import easyocr
import re
import tempfile

# ------------------ Initialize OCR Reader ------------------
reader = easyocr.Reader(['en'])

# ------------------ Reference Ranges ------------------
reference_ranges = {
    "Hemoglobin": (13, 17),
    "WBC": (4800, 10800),
    "Platelets": (1.5, 4.1),
    "RBC": (4.5, 5.5),
    "MCV": (83, 101),
    "MCHC": (31.5, 34.5),
    "Hematocrit": (40, 50),
    "Lymphocyte": (20, 40),
    "Monocytes": (2, 10),
    "Neutrophils": (40, 80),
    "Eosinophils": (1, 6),
    "Basophils": (0, 2)
}

# ------------------ Doctor & Diet Recommendations ------------------
doctor_map = {
    "Hemoglobin": "Hematologist",
    "WBC": "Immunologist",
    "Platelets": "Hematologist",
    "RBC": "General Physician",
    "MCV": "General Physician",
    "MCHC": "General Physician",
    "Hematocrit": "Hematologist",
    "Lymphocyte": "Immunologist",
    "Monocytes": "Infectious Disease Specialist",
    "Neutrophils": "Immunologist",
    "Eosinophils": "Allergist",
    "Basophils": "Allergist"
}

diet_map = {
    "Hemoglobin": "Iron-rich foods: spinach, red meat, lentils",
    "WBC": "Vitamin C foods: oranges, strawberries, bell peppers",
    "Platelets": "Vitamin K foods: kale, broccoli, green leafy vegetables",
    "RBC": "Iron and B12: eggs, meat, dairy",
    "MCV": "B12 and folate: leafy greens, legumes",
    "MCHC": "Balanced iron and B12 diet",
    "Hematocrit": "Hydration and iron-rich foods",
    "Lymphocyte": "Immune boosters: citrus fruits, garlic, turmeric",
    "Monocytes": "Anti-inflammatory foods: berries, salmon, green tea",
    "Neutrophils": "Zinc and probiotics: yogurt, nuts",
    "Eosinophils": "Avoid allergens, include anti-inflammatory foods",
    "Basophils": "Low-histamine foods, anti-inflammatory diet"
}

# ------------------ Functions ------------------


def extract_text_from_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = reader.readtext(image_rgb, detail=0)
    return '\n'.join(results)


def parse_values(text):
    lines = text.split('\n')
    values = {}
    for i, line in enumerate(lines):
        clean_line = line.replace(',', '').strip()
        for param in reference_ranges:
            if param.lower() in clean_line.lower():
                match = re.search(r'(\d+\.?\d*)', clean_line)
                if match:
                    values[param] = float(match.group(1))
                elif i + 1 < len(lines):  # check next line if no number found
                    next_line = lines[i + 1].replace(',', '').strip()
                    next_match = re.search(r'(\d+\.?\d*)', next_line)
                    if next_match:
                        values[param] = float(next_match.group(1))
    return values


def evaluate_status(val, ref):
    if val < ref[0]:
        return "Low"
    if val > ref[1]:
        return "High"
    return "Normal"


def generate_report(values):
    report = {"Parameter": [], "Value": [], "Normal Range": [],
              "Status": [], "Doctor": [], "Diet": []}
    for param, val in values.items():
        ref = reference_ranges[param]
        status = evaluate_status(val, ref)
        report["Parameter"].append(param)
        report["Value"].append(val)
        report["Normal Range"].append(f"{ref[0]} - {ref[1]}")
        report["Status"].append(status)
        report["Doctor"].append(
            doctor_map[param] if status != "Normal" else "-")
        report["Diet"].append(diet_map[param] if status != "Normal" else "-")
    return pd.DataFrame(report)


def plot_bar_chart(df):
    st.subheader("📈 Parameter Values vs Normal Range")
    plt.figure(figsize=(12, 5))
    sns.barplot(x="Parameter", y="Value", hue="Status",
                data=df, palette="coolwarm")
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())


# ------------------ Streamlit App ------------------
st.set_page_config(page_title="MediScan AI", layout="wide")
st.title("🩸 MediScan AI: Blood Report Analyzer")

mode = st.radio("Select Input Mode", ["Upload Image", "Enter Manually"])
values = {}

if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload Blood Report Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Report",
                 use_column_width=True)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        img = cv2.imread(tmp_path)
        ocr_text = extract_text_from_image(img)
        values = parse_values(ocr_text)

elif mode == "Enter Manually":
    st.subheader("📝 Manual Input")
    for param, (low, high) in reference_ranges.items():
        values[param] = st.number_input(
            f"{param} ({low}-{high})", min_value=0.0, step=0.1)

# ------------------ Output ------------------
if values:
    df = generate_report(values)
    st.subheader("📊 Blood Test Summary")
    st.dataframe(df)

    st.subheader("🚨 Abnormal Findings")
    abnormal_df = df[df['Status'] != 'Normal']
    if not abnormal_df.empty:
        for _, row in abnormal_df.iterrows():
            st.markdown(
                f"**{row['Parameter']}** is **{row['Status']}** → Consult a **{row['Doctor']}**. Suggested diet: *{row['Diet']}*.")
    else:
        st.success("All values are within normal range.")

    plot_bar_chart(df)
