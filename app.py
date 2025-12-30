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
from sklearn.decomposition import PCA  # <--- NEW: Import PCA
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
        # Load data
        tanaman = pd.read_csv('dataset/Data_Tanaman_Padi_Sumatera.csv')
        iklim = pd.read_csv('dataset/data_iklim_sumatera.csv')
        produktivitas = pd.read_csv('dataset/data_produktivitas_sumatera.csv')

        # Standarisasi nama provinsi
        for df in [tanaman, iklim, produktivitas]:
            df['Provinsi'] = df['Provinsi'].astype(str).str.title()

        # Konversi produktivitas
        produktivitas['Produktivitas (ku/ha)'] = (
            produktivitas['Produktivitas (ku/ha)']
            .str.replace(',', '.', regex=False)
            .astype(float)
        )

        # ==================== HAPUS DUPLIKAT IKLIM DI TANAMAN ====================
        kolom_iklim = ['Curah hujan', 'Suhu rata-rata']
        tanaman = tanaman.drop(columns=[c for c in kolom_iklim if c in tanaman.columns])

        # ==================== MERGE DATA ====================
        df = pd.merge(tanaman, iklim, on=['Provinsi', 'Tahun'], how='inner')
        df = pd.merge(df, produktivitas, on=['Provinsi', 'Tahun'], how='inner')

        # ==================== RENAME KOLUMN IKLIM ====================
        rename_map = {}

        for col in df.columns:
            if 'suhu' in col.lower():
                rename_map[col] = 'Suhu rata-rata (°C)'
            if 'curah' in col.lower():
                rename_map[col] = 'Curah Hujan (mm)'

        df = df.rename(columns=rename_map)


        # ==================== OUTLIER CAPPING (IQR) ====================
        def cap_outliers(df, cols):
            for col in cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            return df

        df = cap_outliers(df, [
            'Curah Hujan (mm)',
            'Suhu rata-rata (°C)',
            'Kelembapan'
        ])
        # Normalisasi nama kolom (anti typo & beda format)
        df.columns = (
            df.columns
            .str.strip()
            .str.replace('  ', ' ')
        )


        return df

    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
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
    - 🤖 Pemodelan Prediktif Machine Learning (Target: Produksi)
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

    corr_cols = [
        'Produksi',
        'Produktivitas (ku/ha)',
        'Luas Panen',
        'Curah Hujan (mm)',
        'Kelembapan',
        'Suhu rata-rata (°C)',
        'Lama Penyinaran Matahari (%)'
    ]


    df_corr = df_filtered[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                linewidths=1, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Heatmap Korelasi Antar Variabel', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("📊 Korelasi dengan Produksi")

    relevant_columns = [
        'Produksi',
        'Produktivitas (ku/ha)',
        'Luas Panen',
        'Curah Hujan (mm)',
        'Kelembapan',
        'Suhu rata-rata (°C)',
        'Lama Penyinaran Matahari (%)'
    ]

    df_correlation = df_filtered[relevant_columns].corr()
    corr_target = df_correlation['Produksi'].drop('Produksi').sort_values(ascending=True)

    correlation_threshold = st.slider("Threshold Korelasi", 0.0, 1.0, 0.1, 0.05)
    selected_features = corr_target[abs(corr_target) >= correlation_threshold]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_features)))
    selected_features.plot(kind='barh', color=colors, ax=ax)
    ax.axvline(x=correlation_threshold, color='red', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axvline(x=-correlation_threshold, color='red', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(f'Korelasi dengan Produksi (threshold ≥ {correlation_threshold})', fontweight='bold')
    ax.set_xlabel('Nilai Korelasi')
    ax.set_ylabel('Variabel')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# 📈 TREN TEMPORAL
elif selected_analysis == "📈 Tren Temporal":
    st.header("📈 Tren Temporal")

    df_trend = df_filtered.groupby('Tahun').mean(numeric_only=True).reset_index()

    trend_cols = ['Produksi', 'Produktivitas (ku/ha)', 'Kelembapan', 'Lama Penyinaran Matahari (%)']

    selected_trend = st.selectbox("Pilih Variabel untuk Analisis Tren", trend_cols)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
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

    st.subheader("🔍 Korelasi Variabel Iklim dengan Produksi")

    climate_vars = ['Kelembapan', 'Lama Penyinaran Matahari (%)', 'Curah Hujan (mm)', 'Suhu rata-rata (°C)']
    corr_data = []

    for var in climate_vars:
        if var in df_province.columns and df_province[var].nunique() > 1:
            corr = df_province['Produksi'].corr(df_province[var])
            corr_data.append({'Variabel': var, 'Korelasi': corr})

    if corr_data:
        df_corr_province = pd.DataFrame(corr_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_corr_province, x='Korelasi', y='Variabel', palette='viridis', ax=ax)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'Korelasi Variabel Iklim thd Produksi - {selected_province}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Nilai Korelasi')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Data tidak cukup untuk menampilkan korelasi variabel iklim.")

# 🎯 CLUSTERING IKLIM
elif selected_analysis == "🎯 Clustering Iklim":
    st.header("🎯 Analisis Clustering Berdasarkan Karakteristik Iklim")

    st.info("💡 Clustering membantu mengidentifikasi zonasi iklim. Kita akan melihat karakteristik Produksi di setiap cluster.")

    clustering_features = [
        'Lama Penyinaran Matahari (%)',
        'Kelembapan',
        'Suhu rata-rata (°C)'
    ]

    # Drop NA untuk keperluan clustering
    df_clustering = df_filtered[clustering_features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clustering)

    # Elbow method
    st.subheader("📊 Metode Elbow untuk Menentukan Jumlah Cluster Optimal")

    inertia = []
    K_range = range(1, min(11, len(df_clustering)))

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    inertia_diff = np.diff(inertia)
    optimal_k = K_range[np.argmin(inertia_diff) + 2]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(K_range, inertia, marker='o', linewidth=2, markersize=8)
    ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K = {optimal_k}')
    ax.set_xlabel('Jumlah Cluster (K)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    optimal_k = st.slider("Pilih Jumlah Cluster", 2, 5, 3)

    # KMeans
    kmeans_model = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    df_filtered_copy = df_filtered.copy()
    cluster_labels = np.full(len(df_filtered_copy), -1)
    
    # Mask untuk data valid agar indeks sinkron
    valid_mask = df_filtered[clustering_features].notna().all(axis=1)
    
    # Fit & Predict
    cluster_labels[valid_mask] = kmeans_model.fit_predict(X_scaled)
    df_filtered_copy['Cluster'] = cluster_labels

    st.subheader(f"📍 Hasil Clustering (Visualisasi PCA Biplot)")

    # ==========================================
    # LOGIKA PCA & BIPLOT (REPLACED PLOTLY)
    # ==========================================
    
    # 1. Jalankan PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled) # X_scaled derived from valid_mask rows

    # 2. DataFrame untuk Plotting
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = cluster_labels[valid_mask] # Label cluster pada baris valid

    # 3. Hitung Variance Explained
    var_ratio = pca.explained_variance_ratio_
    pc1_info = var_ratio[0] * 100
    pc2_info = var_ratio[1] * 100
    total_info = pc1_info + pc2_info

    # 4. Plotting (Matplotlib/Seaborn)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot titik data
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue='Cluster', 
        data=df_pca, 
        palette='viridis', 
        s=120, 
        alpha=0.8, 
        edgecolor='w',
        ax=ax
    )

    # Panah Vektor (Biplot)
    scale_arrow = 3 # Scaling factor agar panah terlihat jelas
    features = clustering_features # Gunakan nama fitur yang dipakai clustering

    for i, feature in enumerate(features):
        # Gambar panah
        ax.arrow(0, 0, 
                 pca.components_[0, i] * scale_arrow, 
                 pca.components_[1, i] * scale_arrow,
                 color='red', alpha=0.5, head_width=0.1)
        # Tulis label fitur
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

    # ==========================================
    # METRIK EVALUASI PCA
    # ==========================================
    with st.expander("ℹ️ Detail Evaluasi PCA (Klik untuk melihat)", expanded=True):
        col_eval1, col_eval2 = st.columns(2)
        
        with col_eval1:
            st.markdown("#### Explained Variance")
            st.write(f"Seberapa banyak informasi data asli yang tersimpan di grafik 2D ini?")
            st.metric("PC1 Explained", f"{pc1_info:.2f}%")
            st.metric("PC2 Explained", f"{pc2_info:.2f}%")
            st.metric("TOTAL Informasi", f"{total_info:.2f}%")
            if total_info > 70:
                st.success("✅ Grafik ini sangat akurat merepresentasikan data asli.")
            else:
                st.warning("⚠️ Grafik ini mungkin kehilangan beberapa detail data asli.")

        with col_eval2:
            st.markdown("#### Loading Scores (Pengaruh Variabel)")
            st.write("Variabel mana yang paling mempengaruhi pembentukan sumbu X (PC1) dan Y (PC2)?")
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2'],
                index=features
            )
            st.dataframe(loadings.style.background_gradient(cmap='coolwarm'))


    # Stats Cluster
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

    # ========================================================
    # 🆕 FITUR TAMBAHAN: HEATMAP DINAMIKA CLUSTER PER TAHUN
    # ========================================================
    st.markdown("---")
    st.subheader("🗓️ Dinamika Perubahan Cluster Agroklimat (2018-2024)")
    st.write("Visualisasi ini menunjukkan stabilitas kategori iklim suatu provinsi dari tahun ke tahun. Perubahan warna dalam satu baris menandakan perubahan karakteristik iklim.")

    if df_filtered_copy['Tahun'].nunique() > 1:
        pivot_cluster = df_filtered_copy.pivot(index='Provinsi', columns='Tahun', values='Cluster')
        fig_height = max(6, len(pivot_cluster) * 0.5)

        fig, ax = plt.subplots(figsize=(12, fig_height))
        sns.heatmap(pivot_cluster, annot=True, cmap='viridis', cbar=False,
                   linewidths=1, linecolor='white', ax=ax)

        ax.set_title('Peta Perubahan Cluster per Provinsi', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Provinsi', fontsize=12)
        ax.set_xlabel('Tahun', fontsize=12)
        plt.tight_layout()

        st.pyplot(fig)
        st.info("💡 **Tips Membaca:** Angka dalam kotak mewakili ID Cluster. Jika angka berubah dari tahun ke tahun pada provinsi yang sama, berarti terjadi pergeseran karakteristik iklim yang signifikan.")
    else:
        st.warning("⚠️ Data yang difilter hanya memiliki 1 tahun. Silakan perluas rentang tahun di Sidebar untuk melihat dinamika perubahan cluster.")

# 🤖 PEMODELAN PREDIKTIF
elif selected_analysis == "🤖 Pemodelan Prediktif":
    st.header("🤖 Pemodelan Prediktif Produksi")

    st.info("🎯 Membandingkan performa berbagai model machine learning untuk prediksi total Produksi padi.")

    feature_cols = [
        'Produktivitas (ku/ha)',
        'Luas Panen',
        'Lama Penyinaran Matahari (%)',
        'Kelembapan',
        'Suhu rata-rata (°C)'
    ]

    df_model = df_filtered[feature_cols + ['Produksi']].dropna()

    X = df_model[feature_cols]
    y = df_model['Produksi']

    test_size = st.slider("Ukuran Data Test (%)", 10, 40, 20, 5) / 100

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
            y_pred = model.predict(X_test)

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