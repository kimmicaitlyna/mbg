import cobacoba.streamlit as st
import pandas as pd
import pickle
import os
from prepo import clean_text, prepro_sentimen, lexicon_handle

model_path = {
    "SVM_sentimen": 'SVM_sentimen.pkl',
    "SVM_gaya": 'SVM_gaya.pkl'
}

models = {}

for model_name, path in model_path.items():
    with open(path, 'rb') as f:
        models[model_name] = pickle.load(f)

st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>
        📊 Sistem Analisis Sentimen & Klasifikasi Gaya Tweet mengenai Makan Bergizi Gratis
    </h1>
    <p style='text-align: center; color: gray;'>
        Model menggunakan Support Vector Machine (SVM) sebagai model terbaik
    </p>
""", unsafe_allow_html=True)

st.divider()
st.markdown("### ✍️ Input Tweet (Multi Input)")
text_input = st.text_area(
    "Masukkan beberapa tweet (1 baris = 1 tweet)",
    placeholder="contoh:\nProgram MBG sangat bagus\nSaya kurang setuju dengan program ini\nMakan gratis ini membantu masyarakat",
    height=180
)
st.divider()


if st.button("🚀 Analisis Sekarang"):
    if text_input.strip() == "" or text_input.strip() == "-":
        st.warning("⚠️ Input tidak boleh kosong.")
    else:
        
        def sentimen(pred):
            if pred == "positif":
                return "🟢 Positif"
            elif pred == "netral":
                return "🟡 Netral"
            else:
                return "🔴 Negatif"

        def gaya(pred):
            if pred == "formal":
                return "🔵 Formal"
            elif pred == "informal":
                return "🟢 Informal"
            else:
                return "🟠 Gaul"

        texts = [t for t in text_input.split("\n") if t.strip() != ""]

        model_sentimen = models["SVM_sentimen"]
        model_gaya = models["SVM_gaya"]

        results = []

        for t in texts:
            cleaned_gaya = clean_text(t)
            cleaned_sentimen = prepro_sentimen(cleaned_gaya)
            
            pred_sentimen = model_sentimen.predict([cleaned_sentimen])[0]
            pred_gaya = model_gaya.predict([cleaned_gaya])[0]
            
            tokens = cleaned_sentimen.split()
            score = lexicon_handle(tokens)

            # negation
            has_negation = any("_" in token for token in tokens)
            if has_negation and score != 0:
                if score > 0:
                    pred_sentimen = "positif"

                elif score < 0:
                    pred_sentimen = "negatif"
            
            # pred + lexicon
            if score < 0 and pred_sentimen != "negatif":
                pred_sentimen = "negatif"
            elif score > 0 and pred_sentimen != "positif":
                pred_sentimen = "positif"

            results.append({
                "Tweet": t,
                "Sentimen": sentimen(pred_sentimen),
                "Gaya Bahasa": gaya(pred_gaya)
            })

        hasil = pd.DataFrame(results)
        
        st.markdown("## 📌 Hasil Analisis")

        st.dataframe(
            hasil,
            use_container_width=True,
            hide_index=True
        )
        
