import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
import os

# --- Data Loading and Session Management ---
def load_dataset():
    """Load the uploaded Excel file into a DataFrame and manage session state."""
    uploaded = st.file_uploader("Upload dataset musik (XLSX)", type="xlsx")
    if not uploaded:
        return None
    # reset on new file
    if 'previous_file' not in st.session_state or st.session_state.previous_file != uploaded.name:
        st.session_state.clear()
        st.session_state.previous_file = uploaded.name
    return pd.read_excel(uploaded)

# --- Column Selection UI ---
def select_feature_columns(df):
    """Let user choose the relevant columns for track, artist, valence, and energy."""
    cols = df.columns.tolist()
    default = lambda name: cols.index(name) if name in cols else 0
    st.subheader("Konfirmasi Kolom Dataset")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        track_col = st.selectbox("Nama Track", cols, index=default('track_name'))
    with c2:
        artist_col = st.selectbox("Nama Artis", cols, index=default('artist_name'))
    with c3:
        valence_col = st.selectbox("Valence", cols, index=default('valence'))
    with c4:
        energy_col = st.selectbox("Energy", cols, index=default('energy'))
    return track_col, artist_col, valence_col, energy_col

# --- Preprocessing ---
def preprocess_data(df, valence_col, energy_col):
    """Drop NA rows for the chosen numeric features and store cleaned data in session."""
    clean = df.dropna(subset=[valence_col, energy_col])
    if clean.empty:
        st.error("Tidak ada data valid setelah menghapus nilai kosong!")
        return False
    st.session_state.df_clean = clean
    st.session_state.data = clean[[valence_col, energy_col]].values
    st.session_state.valence_col = valence_col
    st.session_state.energy_col = energy_col
    return True

# --- Clustering and Metrics ---
def evaluate_clusters(data):
    """Perform KMeans for k=2..10, compute evaluation metrics, and save results."""
    metrics, results = [], {}
    progress = st.progress(0)
    status = st.empty()
    for i, k in enumerate(range(2, 11)):
        status.text(f"Memproses k = {k}...")
        progress.progress((i+1)/9)
        model = KMeans(n_clusters=k, init='random', random_state=42, tol=0, max_iter=100)
        labels = model.fit_predict(data)
        centroids = model.cluster_centers_
        try:
            sil = silhouette_score(data, labels)
            cal = calinski_harabasz_score(data, labels)
            dbi = davies_bouldin_score(data, labels)
        except Exception as e:
            st.error(f"Error menghitung metrik untuk k={k}: {e}")
            continue
        metrics.append({'k': k, 'Silhouette Score': sil, 'Calinski-Harabasz': cal, 'Davies-Bouldin': dbi})
        results[k] = {'labels': labels, 'centroids': centroids}
    st.session_state.metrics_df = pd.DataFrame(metrics).set_index('k')
    st.session_state.results_dict = results

# --- Display and Highlight Metrics ---
def highlight_metrics(s):
    styles = []
    for val in s:
        if s.name == 'Silhouette Score':
            best = st.session_state.metrics_df['Silhouette Score'].max()
            is_best = val == best
        elif s.name == 'Calinski-Harabasz':
            best = st.session_state.metrics_df['Calinski-Harabasz'].max()
            is_best = val == best
        elif s.name == 'Davies-Bouldin':
            best = st.session_state.metrics_df['Davies-Bouldin'].min()
            is_best = val == best
        else:
            is_best = False
        styles.append('background-color: yellow' if is_best else '')
    return styles

def show_metrics():
    """Render the evaluation metrics table with highlights."""
    st.subheader("📊 Hasil Evaluasi Klasterisasi")
    styled = st.session_state.metrics_df.style.apply(highlight_metrics).format({
        'Silhouette Score':'{:.2f}','Calinski-Harabasz':'{:.2f}','Davies-Bouldin':'{:.2f}'
    })
    st.dataframe(styled)

# --- Recommend Optimal k ---
def recommend_k():
    """Determine recommended cluster counts based on metrics and display."""
    st.subheader("💡 Rekomendasi Jumlah Klaster Optimal")
    try:
        best_sil = st.session_state.metrics_df['Silhouette Score'].idxmax()
        best_cal = st.session_state.metrics_df['Calinski-Harabasz'].idxmax()
        best_dbi = st.session_state.metrics_df['Davies-Bouldin'].idxmin()
        st.write(f"- **Silhouette Score Terbaik**: k={best_sil}")
        st.write(f"- **Calinski-Harabasz Terbaik**: k={best_cal}")
        st.write(f"- **Davies-Bouldin Terbaik**: k={best_dbi}")
        votes = pd.Series([best_sil, best_cal, best_dbi]).value_counts()
        st.success(f"**Rekomendasi k Optimal**: {votes.idxmax()}")
    except:
        st.warning("Tidak dapat menentukan rekomendasi k optimal")

# --- Visualization ---
def plot_clusters():
    """Allow selection of k and plot cluster scatter with centroids and centroid table."""
    st.subheader("👁️ Visualisasi Klaster")
    ks = list(st.session_state.results_dict.keys())
    k = st.selectbox("Pilih jumlah klaster:", options=ks)
    labels = st.session_state.results_dict[k]['labels']
    centroids = st.session_state.results_dict[k]['centroids']
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(st.session_state.data[:,0], st.session_state.data[:,1], c=labels, cmap='tab10', alpha=0.6, edgecolor='w', s=50)
    ax.scatter(centroids[:,0], centroids[:,1], marker='X', s=200, c='red', label='Centroid')
    ax.set_xlabel('Valence')
    ax.set_ylabel('Energy')
    ax.set_title(f'Visualisasi Klaster (k={k})')
    plt.colorbar(ax.collections[0], label='Klaster')
    plt.legend()
    st.pyplot(fig)
    # centroid table
    st.subheader("📌 Nilai Centroid Klaster")
    dfc = pd.DataFrame(centroids, columns=['Valence','Energy'], index=[f'Klaster {i+1}' for i in range(k)]).round(2)
    st.caption("Nilai koordinat centroid untuk masing-masing klaster:")
    st.dataframe(dfc.style.format("{:.2f}"), use_container_width=True)

# --- Saving Results ---
def save_results():
    """UI for saving the chosen k clustering results to Excel and Pickle, plus download buttons."""
    st.subheader("💾 Simpan Hasil Klasterisasi")
    os.makedirs('saved_models', exist_ok=True)
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Nama File (tanpa ekstensi):")
    with col2:
        k = st.selectbox("Simpan untuk k =", options=list(st.session_state.results_dict.keys()))
    if st.button("💿 Simpan Hasil"):
        if not name:
            st.error("Harap beri nama file!")
            return
        try:
            # prepare data
            df_save = st.session_state.df_clean.copy()
            df_save['Cluster'] = st.session_state.results_dict[k]['labels']
            model_data = {
                'centroids': st.session_state.results_dict[k]['centroids'],
                'k': k,
                'features': [st.session_state.valence_col, st.session_state.energy_col],
                'data': st.session_state.data
            }
            # file paths
            excel_path = os.path.join('saved_models', f"{name}.xlsx")
            pkl_path = os.path.join('saved_models', f"{name}_k{k}.pkl")
            # write files
            df_save.to_excel(excel_path, index=False)
            with open(pkl_path, 'wb') as f:
                pickle.dump(model_data, f)
            st.success(f"Berhasil disimpan di: {excel_path} dan {pkl_path}")
            # download buttons
            with open(excel_path, 'rb') as f_excel:
                st.download_button(
                    label="📥 Unduh Excel Hasil",
                    data=f_excel,
                    file_name=f"{name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with open(pkl_path, 'rb') as f_pkl:
                st.download_button(
                    label="📥 Unduh Pickle Model",
                    data=f_pkl,
                    file_name=f"{name}_k{k}.pkl",
                    mime="application/octet-stream"
                )
            st.balloons()
        except Exception as e:
            st.error(f"Gagal menyimpan: {e}")

# --- Main App ---
def main():
    st.title('🎵 Music Mood Clustering Analysis')
    df = load_dataset()
    if df is None:
        return
    track_col, artist_col, valence_col, energy_col = select_feature_columns(df)
    if st.button("🚀 Jalankan Analisis Klasterisasi"):
        if not preprocess_data(df, valence_col, energy_col):
            return
        evaluate_clusters(st.session_state.data)
    if 'metrics_df' in st.session_state:
        show_metrics()
        recommend_k()
        plot_clusters()
        save_results()

if __name__ == '__main__':
    main()
