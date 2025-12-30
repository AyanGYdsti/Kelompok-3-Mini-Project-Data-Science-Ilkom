
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Dashboard Analisis Produksi Padi Sumatra",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #2E7D32;
        padding-bottom: 1rem;
    }
    h2 {
        color: #388E3C;
        padding-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #81C784;
    }
    h3 {
        color: #43A047;
        padding-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_and_preprocess_data():
    try:
        # Load data using exact filenames from the notebook
        tanaman = pd.read_csv('dataset/Data_Tanaman_Padi_Sumatera.csv')
        iklim = pd.read_csv('dataset/data_iklim_sumatera.csv')
        produktivitas = pd.read_csv('dataset/data_produktivitas_sumatera.csv')

        # Standardize province names
        for df_data in [tanaman, iklim, produktivitas]:
            df_data['Provinsi'] = df_data['Provinsi'].astype(str).str.title()

        # Convert 'Produktivitas (ku/ha)' to float
        produktivitas['Produktivitas (ku/ha)'] = (
            produktivitas['Produktivitas (ku/ha)']
            .str.replace(',', '.', regex=False)
            .astype(float)
        )

        # ==================== COLUMN DROPPING (as per notebook) ====================
        # Drop 'Curah hujan' from 'Tanaman' to avoid duplication during merge
        if 'Curah hujan' in tanaman.columns:
            tanaman = tanaman.drop(columns=['Curah hujan'])

        # Drop 'Suhu Rata-rata (°C)' from 'Iklim' to keep 'Suhu rata-rata' from 'Tanaman'
        cols_to_drop_iklim = [c for c in iklim.columns if 'Suhu Rata-rata' in c]
        if cols_to_drop_iklim:
            iklim = iklim.drop(columns=cols_to_drop_iklim)

        # ==================== MERGE DATA (as per notebook) ====================
        df_merged = pd.merge(tanaman, iklim, on=['Provinsi', 'Tahun'], how='inner')
        df_final = pd.merge(df_merged, produktivitas, on=['Provinsi', 'Tahun'], how='inner')

        # Ensure consistent column names after merge for later use
        # The notebook's final df columns are: 'Suhu rata-rata' (from Tanaman), 'Curah Hujan (mm)' (from Iklim)
        # No specific rename map is needed if previous drops are handled correctly and original names are preserved.

        # ==================== OUTLIER CAPPING (IQR) ====================
        def cap_outliers(df, cols):
            for col in cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            return df

        df_final = cap_outliers(df_final, [
            'Curah Hujan (mm)',
            'Suhu rata-rata', # Using 'Suhu rata-rata' as per merged notebook df
            'Kelembapan'
        ])
        
        # Normalisasi nama kolom (anti typo & beda format) - keeping this for robustness
        df_final.columns = (
            df_final.columns
            .str.strip()
            .str.replace('  ', ' ')
        )

        return df_final

    except FileNotFoundError as e:
        st.error(f"Gagal memuat data. Pastikan file CSV tersedia di folder: {e}")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        return None

# ==================== HEADER ====================
st.title('🌾 Analisis Produksi Padi dan Indikator Iklim di Sumatra')

with st.expander("ℹ️ Tentang Dashboard", expanded=False):
    st.markdown("""
    Dashboard ini menyajikan analisis mendalam mengenai hubungan antara faktor-faktor iklim,
    luas panen, dan produktivitas terhadap total **Produksi** tanaman padi di Sumatra periode 2018-2024.

    **Fitur Utama:**
    - 📊 Overview Data & Statistik Deskriptif
    - 🔍 Analisis Korelasi Multi-Variabel
    - 📈 Tren Temporal Produksi
    - 🗺️ Analisis Regional per Provinsi
    - 🎯 Clustering & Zonasi Iklim (Dinamika Perubahan & PCA Biplot)
    - 🤖 Pemodelan Prediktif Machine Learning (Target: Produktivitas ku/ha)
    """)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    # Load data
    with st.spinner('Memuat data...'):
        df_final = load_and_preprocess_data()

    if df_final is None:
        st.stop()

    st.success(f"✅ Data berhasil dimuat: {len(df_final)} baris")

    # Filters
    st.subheader("🔍 Filter Data")

    selected_provinces = st.multiselect(
        "Pilih Provinsi",
        options=sorted(df_final['Provinsi'].unique()),
        default=sorted(df_final['Provinsi'].unique())
    )

    year_range = st.slider(
        "Rentang Tahun",
        min_value=int(df_final['Tahun'].min()),
        max_value=int(df_final['Tahun'].max()),
        value=(int(df_final['Tahun'].min()), int(df_final['Tahun'].max()))
    )

    # Apply filters
    df_filtered = df_final[
        (df_final['Provinsi'].isin(selected_provinces)) &
        (df_final['Tahun'] >= year_range[0]) &
        (df_final['Tahun'] <= year_range[1])
    ]

    st.info(f"📋 Data terfilter: {len(df_filtered)} baris")

    # Analysis selection
    st.subheader("📑 Navigasi Analisis")
    analysis_options = [
        "📊 Overview Data",
        "🔗 Analisis Korelasi",
        "📈 Tren Temporal",
        "🗺️ Analisis Regional",
        "🎯 Clustering Iklim",
        "🤖 Pemodelan Prediktif"
    ]
    selected_analysis = st.radio("Pilih Analisis", analysis_options, index=0)

# ==================== MAIN CONTENT ====================

# 📊 OVERVIEW DATA
if selected_analysis == "📊 Overview Data":
    st.header("📊 Overview Data")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data", f"{len(df_filtered):,}")
    with col2:
        st.metric("Jumlah Provinsi", len(df_filtered['Provinsi'].unique()))
    with col3:
        st.metric("Rentang Tahun", f"{df_filtered['Tahun'].min()}-{df_filtered['Tahun'].max()}")
    with col4:
        st.metric("Total Produksi (Ton)", f"{df_filtered['Produksi'].sum():,.0f}")

    st.subheader("📋 Data Sample")
    st.dataframe(df_filtered.head(10), use_container_width=True)

    st.subheader("📊 Statistik Deskriptif")
    st.dataframe(df_filtered.describe().T, use_container_width=True)

# 🔗 ANALISIS KORELASI
elif selected_analysis == "🔗 Analisis Korelasi":
    st.header("🔗 Analisis Korelasi")

    st.subheader("🔥 Heatmap Korelasi")

    # Using columns as present in the notebook's df_final and correlation section
    corr_cols = [
        'Produksi',
        'Produktivitas (ku/ha)',
        'Luas Panen',
        'Curah Hujan (mm)',
        'Kelembapan',
        'Suhu rata-rata', # Changed to match notebook's merged df
        'Lama Penyinaran Matahari (%)'
    ]


    df_corr = df_filtered[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                linewidths=1, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Heatmap Korelasi Antar Variabel', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("📊 Korelasi dengan Produktivitas (ku/ha)") # Changed title to reflect notebook analysis

    relevant_columns = [
        'Produksi',
        'Produktivitas (ku/ha)',
        'Luas Panen',
        'Curah Hujan (mm)',
        'Kelembapan',
        'Suhu rata-rata', # Changed to match notebook's merged df
        'Lama Penyinaran Matahari (%)'
    ]

    df_correlation = df_filtered[relevant_columns].corr()
    corr_target = df_correlation['Produktivitas (ku/ha)'].drop('Produktivitas (ku/ha)').sort_values(ascending=True); # Changed target to Produktivitas (ku/ha)

    correlation_threshold = st.slider("Threshold Korelasi", 0.0, 1.0, 0.1, 0.05)
    selected_features = corr_target[abs(corr_target) >= correlation_threshold]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_features)))
    selected_features.plot(kind='barh', color=colors, ax=ax)
    ax.axvline(x=correlation_threshold, color='red', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axvline(x=-correlation_threshold, color='red', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(f'Korelasi dengan Produktivitas (threshold ≥ {correlation_threshold})', fontweight='bold') # Changed title
    ax.set_xlabel('Nilai Korelasi')
    ax.set_ylabel('Variabel')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# 📈 TREN TEMPORAL
elif selected_analysis == "📈 Tren Temporal":
    st.header("📈 Tren Temporal")

    df_trend = df_filtered.groupby('Tahun').mean(numeric_only=True).reset_index()

    # These columns are directly used in notebook's trend analysis
    trend_cols = ['Produksi', 'Produktivitas (ku/ha)', 'Kelembapan', 'Suhu rata-rata', 'Lama Penyinaran Matahari (%)']

    selected_trend = st.selectbox("Pilih Variabel untuk Analisis Tren", trend_cols)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Sum for total production, mean for other climate/productivity factors
        if selected_trend == 'Produksi':
            df_trend_sum = df_filtered.groupby('Tahun')['Produksi'].sum().reset_index()
            sns.lineplot(data=df_trend_sum, x='Tahun', y=selected_trend, marker='o', linewidth=2.5, ax=ax)
            ax.set_title(f'Total Tren {selected_trend} (2018-2024)', fontsize=14, fontweight='bold')
            stats_df = df_trend_sum
        else:
            sns.lineplot(data=df_trend, x='Tahun', y=selected_trend, marker='o', linewidth=2.5, ax=ax)
            ax.set_title(f'Rata-rata Tren {selected_trend} (2018-2024)', fontsize=14, fontweight='bold')
            stats_df = df_trend

        ax.set_xlabel('Tahun', fontsize=12)
        ax.set_ylabel(selected_trend, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Statistik Tren")
        st.metric("Nilai Tertinggi", f"{stats_df[selected_trend].max():,.2f}")
        st.metric("Nilai Terendah", f"{stats_df[selected_trend].min():,.2f}")
        st.metric("Rata-rata", f"{stats_df[selected_trend].mean():,.2f}")
        st.metric("Std Deviasi", f"{stats_df[selected_trend].std():,.2f}")

    st.subheader("📊 Distribusi Data")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_filtered, x=selected_trend, kde=True, bins=30, ax=ax)
    ax.set_title(f'Distribusi {selected_trend}', fontsize=14, fontweight='bold')
    ax.set_xlabel(selected_trend, fontsize=12)
    ax.set_ylabel('Frekuensi', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

# 🗺️ ANALISIS REGIONAL
elif selected_analysis == "🗺️ Analisis Regional":
    st.header("🗺️ Analisis Regional per Provinsi")

    selected_province = st.selectbox("Pilih Provinsi untuk Analisis Detail",
                                     sorted(df_filtered['Provinsi'].unique()))

    df_province = df_filtered[df_filtered['Provinsi'] == selected_province]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Produksi", f"{df_province['Produksi'].sum():,.0f} Ton")
    with col2:
        st.metric("Rata-rata Produktivitas", f"{df_province['Produktivitas (ku/ha)'].mean():.2f} ku/ha")
    with col3:
        st.metric("Total Luas Panen", f"{df_province['Luas Panen'].sum():,.0f} ha")

    st.subheader(f"📈 Tren Produksi - {selected_province}")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_province, x='Tahun', y='Produksi', marker='o', linewidth=2.5, ax=ax)
    ax.set_title(f'Tren Produksi {selected_province}', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🔍 Korelasi Variabel Iklim dengan Produktivitas (ku/ha)") # Changed title

    climate_vars = ['Kelembapan', 'Lama Penyinaran Matahari (%)', 'Curah Hujan (mm)', 'Suhu rata-rata'] # Changed column name
    corr_data = []

    for var in climate_vars:
        if var in df_province.columns and df_province[var].nunique() > 1 and df_province['Produktivitas (ku/ha)'].nunique() > 1:
            corr = df_province['Produktivitas (ku/ha)'].corr(df_province[var]) # Target is Produktivitas
            corr_data.append({'Variabel': var, 'Korelasi': corr})

    if corr_data:
        df_corr_province = pd.DataFrame(corr_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_corr_province, x='Korelasi', y='Variabel', palette='viridis', ax=ax)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'Korelasi Variabel Iklim thd Produktivitas - {selected_province}', fontsize=14, fontweight='bold') # Changed title
        ax.set_xlabel('Nilai Korelasi')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Data tidak cukup untuk menampilkan korelasi variabel iklim.")

# 🎯 CLUSTERING IKLIM
elif selected_analysis == "🎯 Clustering Iklim":
    st.header("🎯 Analisis Clustering Berdasarkan Karakteristik Iklim")

    st.info("💡 Clustering membantu mengidentifikasi zonasi iklim. Kita akan melihat karakteristik Produksi di setiap cluster.")

    # Fitur untuk clustering
    clustering_features = [
        'Curah Hujan (mm)',
        'Lama Penyinaran Matahari (%)',
        'Suhu rata-rata',
        'Kelembapan'
    ]

    # Bersihkan Data
    df_clustering = df_filtered[clustering_features].dropna()

    if df_clustering.empty or len(df_clustering) < 2:
        st.warning("⚠️ Data tidak cukup untuk melakukan clustering. Silakan pilih lebih dari satu tahun atau provinsi.")
    else:
        # Standardisasi Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clustering)

        # ---------------------------------------------------------
        # MULAI PERBAIKAN: METODE ELBOW GEOMETRIS
        # ---------------------------------------------------------
        st.subheader("📊 Metode Elbow untuk Menentukan Jumlah Cluster Optimal")

        # Tentukan Range K (Maksimal 10 atau sejumlah data jika < 10)
        max_k = min(10, len(df_clustering))
        range_values = range(1, max_k + 1)
        inertia = []

        # Hitung Inertia
        with st.spinner("Menghitung optimal cluster..."):
            for i in range_values:
                kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)

        # Tentukan Optimal K Secara Otomatis (Metode Jarak Titik ke Garis)
        optimal_k = 1
        if len(range_values) > 1:
            p1 = np.array([range_values[0], inertia[0]])
            p2 = np.array([range_values[-1], inertia[-1]])

            distances = []
            for i in range(len(range_values)):
                p = np.array([range_values[i], inertia[i]])
                # Rumus jarak titik ke garis lurus
                dist = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
                distances.append(dist)
            
            # Ambil K dengan jarak terbesar
            optimal_k = range_values[np.argmax(distances)]

        st.success(f"K Optimal yang ditemukan secara otomatis: **{optimal_k}**")

        # Plotting dengan Garis Merah (Sesuai Referensi)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range_values, inertia, marker='o', label='Inertia')

        # >>> GARIS MERAH <<<
        ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K = {optimal_k}')

        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal K')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # ---------------------------------------------------------
        # SELESAI PERBAIKAN ELBOW
        # ---------------------------------------------------------

        # Slider untuk memilih K (Default ke Optimal K yang ditemukan)
        optimal_k_slider = st.slider("Pilih Jumlah Cluster", 1, max_k, int(optimal_k))

        if optimal_k_slider >= 1:
            # Jalankan KMeans Final dengan K pilihan user
            kmeans_model = KMeans(
                n_clusters=optimal_k_slider,
                random_state=42,
                n_init='auto'
            )

            df_filtered_copy = df_filtered.copy()
            cluster_labels = np.full(len(df_filtered_copy), -1)

            # Masking untuk sinkronisasi indeks
            valid_mask = df_filtered[clustering_features].notna().all(axis=1)

            # Fit & Predict
            cluster_labels[valid_mask] = kmeans_model.fit_predict(X_scaled)
            df_filtered_copy['Cluster'] = cluster_labels

            st.subheader(f"📍 Hasil Clustering (Visualisasi PCA Biplot)")

            # --- LOGIKA PCA & BIPLOT ---
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)

            df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
            df_pca['Cluster'] = cluster_labels[valid_mask]

            var_ratio = pca.explained_variance_ratio_
            pc1_info = var_ratio[0] * 100
            pc2_info = var_ratio[1] * 100
            total_info = pc1_info + pc2_info

            fig, ax = plt.subplots(figsize=(12, 8))

            # Scatter plot
            sns.scatterplot(
                x='PC1', y='PC2', hue='Cluster', data=df_pca,
                palette='viridis', s=120, alpha=0.8, edgecolor='w', ax=ax
            )

            # Vektor Biplot
            scale_arrow = 3 
            features = clustering_features

            for i, feature in enumerate(features):
                ax.arrow(0, 0,
                         pca.components_[0, i] * scale_arrow,
                         pca.components_[1, i] * scale_arrow,
                         color='red', alpha=0.5, head_width=0.1)
                ax.text(pca.components_[0, i] * scale_arrow * 1.15,
                        pca.components_[1, i] * scale_arrow * 1.15,
                        feature, color='darkred', ha='center', va='center',
                        fontsize=10, weight='bold')

            ax.set_title(f'Visualisasi Cluster K-Means dengan PCA\n(Total Informasi: {total_info:.2f}%)', fontsize=15)
            ax.set_xlabel(f'Principal Component 1 ({pc1_info:.2f}%)', fontsize=12)
            ax.set_ylabel(f'Principal Component 2 ({pc2_info:.2f}%)', fontsize=12)
            ax.legend(title='Cluster', loc='best')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

            st.pyplot(fig)

            # Statistik Cluster
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Distribusi Cluster:**")
                st.dataframe(df_filtered_copy['Cluster'].value_counts().sort_index())
            with col2:
                st.write("**Statistik Produksi (Ton) per Cluster:**")
                cluster_stats = (
                    df_filtered_copy
                    .groupby('Cluster')['Produksi']
                    .agg(['mean', 'sum', 'std', 'count'])
                )
                st.dataframe(cluster_stats, use_container_width=True)

            # --- Heatmap Dinamika Cluster ---
            st.markdown("---")
            st.subheader("🗓️ Dinamika Perubahan Cluster Agroklimat (2018-2024)")
            
            if df_filtered_copy['Tahun'].nunique() > 1:
                pivot_cluster = df_filtered_copy.pivot(index='Provinsi', columns='Tahun', values='Cluster')
                fig_height = max(6, len(pivot_cluster) * 0.5)

                fig, ax = plt.subplots(figsize=(12, fig_height))
                sns.heatmap(pivot_cluster, annot=True, cmap='viridis', cbar=False,
                           linewidths=1, linecolor='white', ax=ax)

                ax.set_title('Peta Perubahan Cluster per Provinsi', fontsize=14, fontweight='bold', pad=15)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("ℹ️ Pilih rentang tahun lebih dari 1 untuk melihat dinamika perubahan cluster.")


# 🤖 PEMODELAN PREDIKTIF
elif selected_analysis == "🤖 Pemodelan Prediktif":
    st.header("🤖 Pemodelan Prediktif Produktivitas (ku/ha)") # Changed title

    st.info("🎯 Membandingkan performa berbagai model machine learning untuk prediksi **Produktivitas (ku/ha)** padi.") # Changed target in info

    # Feature columns as used in notebook's predictive modeling section (Hipotesis 3)
    feature_cols = [
        'Produksi',
        'Luas Panen',
        'Curah Hujan (mm)',
        'Lama Penyinaran Matahari (%)',
        'Kelembapan',
        'Suhu rata-rata' # Changed to match notebook's merged df
    ]

    # Target variable as used in notebook's predictive modeling section (Hipotesis 3)
    target_col = 'Produktivitas (ku/ha)'

    df_model = df_filtered[feature_cols + [target_col]].dropna()

    if df_model.empty:
        st.warning("⚠️ Data tidak cukup untuk Pemodelan Prediktif. Silakan sesuaikan filter data.")
    else:
        X = df_model[feature_cols]
        y = df_model[target_col]

        test_size = st.slider("Ukuran Data Test (%)", 10, 40, 20, 5) / 100

        # Ensure enough samples for both train and test sets after split
        if len(X) * (1 - test_size) < 1 or len(X) * test_size < 1:
            st.warning("⚠️ Ukuran data training atau testing terlalu kecil. Sesuaikan filter data atau ukuran data test.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            st.write(f"📊 Data Training: {len(X_train)} | Data Testing: {len(X_test)}")

            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
                'SVR': SVR()
            }

            results = []

            with st.spinner('🔄 Training models...'):
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test) # Corrected typo here

                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    results.append({
                        'Model': name,
                        'MAE': mae,
                        'RMSE': rmse,
                        'R²': r2
                    })

            df_results = pd.DataFrame(results)

            st.subheader("📊 Perbandingan Performa Model")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.dataframe(df_results.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                            .highlight_max(subset=['R²'], color='lightgreen'), use_container_width=True)

            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                df_results.plot(x='Model', y='R²', kind='bar', ax=ax, color='skyblue', legend=False)
                ax.set_title('Perbandingan R² Score', fontsize=14, fontweight='bold')
                ax.set_ylabel('R² Score')
                ax.set_xlabel('')
                ax.set_xticklabels(df_results['Model'], rotation=45, ha='right')
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

            best_model = df_results.loc[df_results['R²'].idxmax(), 'Model']
            st.success(f"🏆 Model terbaik: **{best_model}** dengan R² = {df_results['R²'].max():.4f}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Dashboard Analisis Produksi Padi Sumatra 2018-2024</p>
        <p>Data Processing & Visualization | Machine Learning Analytics</p>
    </div>
    """, unsafe_allow_html=True)
