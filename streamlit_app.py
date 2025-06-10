import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
import os

def main():
    st.title('🎵 Music Mood Clustering Analysis')
    
    # Upload file
    uploaded_file = st.file_uploader("Upload dataset musik (XLSX)", type="xlsx")
    
    if uploaded_file:
        # Reset session state jika upload file baru
        if 'previous_file' not in st.session_state or st.session_state.previous_file != uploaded_file.name:
            st.session_state.clear()
            st.session_state.previous_file = uploaded_file.name
            
        df = pd.read_excel(uploaded_file)
        columns = df.columns.tolist()
        
        # Konfirmasi kolom
        st.subheader("Konfirmasi Kolom Dataset")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            track_col = st.selectbox("Nama Track", columns, index=columns.index('track_name') if 'track_name' in columns else 0)
        with col2:
            artist_col = st.selectbox("Nama Artis", columns, index=columns.index('artist_name') if 'artist_name' in columns else 0)
        with col3:
            valence_col = st.selectbox("Valence", columns, index=columns.index('valence') if 'valence' in columns else 0)
        with col4:
            energy_col = st.selectbox("Energy", columns, index=columns.index('energy') if 'energy' in columns else 0)
        
        if st.button("🚀 Jalankan Analisis Klasterisasi"):
            # Preprocessing data
            df_clean = df.dropna(subset=[valence_col, energy_col])
            if df_clean.empty:
                st.error("Tidak ada data valid setelah menghapus nilai kosong!")
                return
            
            # Simpan data ke session state
            st.session_state.df_clean = df_clean
            st.session_state.data = df_clean[[valence_col, energy_col]].values
            st.session_state.valence_col = valence_col
            st.session_state.energy_col = energy_col
            
            # Inisialisasi penyimpanan hasil
            metrics_list = []
            results_dict = {}
            
            # Proses klasterisasi untuk k 2-10
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, k in enumerate(range(2, 11)):
                status_text.text(f"Memproses k = {k}...")
                progress_bar.progress((i+1)/9)
                
                # Klasterisasi K-Means
                kmeans = KMeans(
                    n_clusters=k,
                    init='random',  # Gunakan inisialisasi acak seperti implementasi manual
                    random_state=42,
                    tol=0,          # Nonaktifkan toleransi, berhenti hanya jika centroid tidak berubah
                    max_iter=100
                    )
                labels = kmeans.fit_predict(st.session_state.data)
                centroids = kmeans.cluster_centers_
                
                # Hitung metrik evaluasi
                try:
                    silhouette = silhouette_score(st.session_state.data, labels)
                    calinski = calinski_harabasz_score(st.session_state.data, labels)
                    dbi = davies_bouldin_score(st.session_state.data, labels)
                except Exception as e:
                    st.error(f"Error menghitung metrik untuk k={k}: {str(e)}")
                    continue

                # Simpan hasil
                metrics_list.append({
                    'k': k,
                    'Silhouette Score': silhouette,
                    'Calinski-Harabasz': calinski,
                    'Davies-Bouldin': dbi
                })

                results_dict[k] = {
                    'labels': labels,
                    'centroids': centroids
                }
            
            # Simpan ke session state
            st.session_state.results_dict = results_dict
            st.session_state.metrics_df = pd.DataFrame(metrics_list).set_index('k')
            
        # Tampilkan hasil jika ada di session state
        if 'metrics_df' in st.session_state:
            # Tampilkan metrik evaluasi
            st.subheader("📊 Hasil Evaluasi Klasterisasi")
            
            # Fungsi highlight
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
            
            # Tampilkan tabel dengan highlight
            styled_df = st.session_state.metrics_df.style.apply(highlight_metrics).format({
                'Silhouette Score': '{:.2f}',
                'Calinski-Harabasz': '{:.2f}',
                'Davies-Bouldin': '{:.2f}'
            })
            
            st.dataframe(styled_df)
            
            # Rekomendasi k optimal
            st.subheader("💡 Rekomendasi Jumlah Klaster Optimal")
            try:
                best_silhouette = st.session_state.metrics_df['Silhouette Score'].idxmax()
                best_calinski = st.session_state.metrics_df['Calinski-Harabasz'].idxmax()
                best_dbi = st.session_state.metrics_df['Davies-Bouldin'].idxmin()
                
                recommendations = {
                    'Silhouette Score': best_silhouette,
                    'Calinski-Harabasz': best_calinski,
                    'Davies-Bouldin': best_dbi
                }
                
                st.write(f"- **Silhouette Score Terbaik**: k={best_silhouette}")
                st.write(f"- **Calinski-Harabasz Terbaik**: k={best_calinski}")
                st.write(f"- **Davies-Bouldin Terbaik**: k={best_dbi}")
                
                # Mencari k optimal
                vote_counts = pd.Series(list(recommendations.values())).value_counts()
                optimal_k = vote_counts.idxmax()
                st.success(f"**Rekomendasi k Optimal**: {optimal_k}")
            except:
                st.warning("Tidak dapat menentukan rekomendasi k optimal")
            
            
            # Visualisasi klaster
            st.subheader("👁️ Visualisasi Klaster")
            if 'results_dict' in st.session_state:
                selected_k = st.selectbox("Pilih jumlah klaster:", options=list(st.session_state.results_dict.keys()))
                
                labels = st.session_state.results_dict[selected_k]['labels']
                centroids = st.session_state.results_dict[selected_k]['centroids']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(st.session_state.data[:,0], 
                                   st.session_state.data[:,1], 
                                   c=labels, 
                                   cmap='tab10',
                                   alpha=0.6,
                                   edgecolor='w',
                                   s=50)
                ax.scatter(centroids[:,0], centroids[:,1],
                        marker='X',
                        s=200,
                        c='red',
                        label='Centroid')
                ax.set_xlabel('Valence')
                ax.set_ylabel('Energy')
                ax.set_title(f'Visualisasi Klaster (k={selected_k})')
                plt.colorbar(scatter, label='Klaster')
                plt.legend()
                st.pyplot(fig)

                # Tampilkan tabel centroid
                st.subheader("📌 Nilai Centroid Klaster")
                centroid_df = pd.DataFrame(
                    centroids,
                    columns=['Valence', 'Energy'],
                    index=[f'Klaster {i+1}' for i in range(selected_k)]
                ).round(2)

                st.caption("Nilai koordinat centroid untuk masing-masing klaster:")
                st.dataframe(
                    centroid_df.style.format("{:.2f}"),
                    use_container_width=True
                )

                # Simpan Hasil Klasterisasi
                st.divider()
                st.subheader("💾 Simpan Hasil Klasterisasi")

                # Buat direktori penyimpanan jika belum ada
                save_dir = "saved_models"
                os.makedirs(save_dir, exist_ok=True)

                col1, col2 = st.columns(2)
                with col1:
                    save_name = st.text_input(
                        "Nama File (tanpa ekstensi):",
                        help="Contoh: 'hasil_klaster_spotify'"
                    )
    
                with col2:
                    save_k = st.selectbox(
                        "Simpan untuk k =",
                        options=list(st.session_state.results_dict.keys()),
                        help="Pilih jumlah klaster yang ingin disimpan"
                    )

                # Simpan dalam 2 format
                if st.button("💿 Simpan Hasil"):
                    if not save_name:
                        st.error("Harap beri nama file!")
                        return
                    try:
                        # 1. Simpan data + label dalam Excel
                        df_save = st.session_state.df_clean.copy()
                        df_save['Cluster'] = st.session_state.results_dict[save_k]['labels']
        
                        # 2. Simpan model dalam Pickle
                        model_data = {
                            'centroids': st.session_state.results_dict[save_k]['centroids'],
                            'k': save_k,
                            'features': [st.session_state.valence_col, st.session_state.energy_col],
                            'data': st.session_state.data
                        }
        
                        # Export ke file
                        excel_path = os.path.join(save_dir, f"{save_name}.xlsx")
                        pkl_path = os.path.join(save_dir, f"{save_name}_k{save_k}.pkl")
        
                        df_save.to_excel(excel_path, index=False)
                        with open(pkl_path, 'wb') as f:
                            pickle.dump(model_data, f)
            
                        st.success(f"Berhasil disimpan di: {excel_path} dan {pkl_path}")
                        st.balloons()
        
                    except Exception as e:
                        st.error(f"Gagal menyimpan: {str(e)}")
            else:
                st.warning("Silakan jalankan analisis klasterisasi terlebih dahulu")

if __name__ == '__main__':
    main()
