import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Library Management",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_excel("Library_Management_Dataset.xlsx")


df = load_data()


# =========================================================
# MAIN TITLE
# =========================================================

st.title("📚 Smart Library Management System")

st.write(
    "Data Mining based Library Usage Analysis & Recommendations"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Apriori Patterns",
        "Student Segmentation (K-Means)",
        "Recommendations"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header("📊 Library Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        len(df)
    )

    c2.metric(
        "Books Borrowed",
        int(df["Books_Borrowed"].sum())
    )

    c3.metric(
        "Digital Resources",
        int(df["Digital_Resources_Accessed"].sum())
    )

    c4.metric(
        "Avg Reading Hours",
        round(
            df["Reading_Duration_Hours"].mean(),
            2
        )
    )


    # ---------------- POPULAR GENRES & PEAK HOURS ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Popular Genres")

        genre = df[
            "Preferred_Genre"
        ].value_counts()

        st.bar_chart(genre)


    with col2:

        st.subheader("Peak Library Hours")

        peak = df[
            "Peak_Library_Hour"
        ].value_counts()

        st.bar_chart(peak)


    # ---------------- REASONS ----------------

    st.subheader(
        "Reasons for Library Visits"
    )

    reasons = df[
        "Reason_For_Visit"
    ].value_counts()

    st.bar_chart(reasons)


    # ---------------- DATASET PREVIEW ----------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# APRIORI PATTERNS
# =========================================================

elif page == "Apriori Patterns":

    st.header(
        "🔗 Apriori Association Rules"
    )


    transactions = df[
        [
            "Preferred_Genre",
            "Reason_For_Visit",
            "Peak_Library_Hour"
        ]
    ].astype(str)


    transactions = transactions.apply(
        lambda x: x.str.strip()
    )


    # ---------------- ONE-HOT ENCODING ----------------

    basket = pd.get_dummies(
        transactions
    )


    # ---------------- FREQUENT ITEMSETS ----------------

    frequent_items = apriori(
        basket,
        min_support=0.02,
        use_colnames=True
    )


    if not frequent_items.empty:

        rules = association_rules(
            frequent_items,
            metric="confidence",
            min_threshold=0.20
        )


        if not rules.empty:

            rules = rules.sort_values(
                by="lift",
                ascending=False
            )


            st.subheader(
                "Top Association Rules"
            )


            display_rules = rules[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ].head(10).copy()


            display_rules[
                "antecedents"
            ] = display_rules[
                "antecedents"
            ].apply(
                lambda x: ", ".join(
                    map(str, x)
                )
            )


            display_rules[
                "consequents"
            ] = display_rules[
                "consequents"
            ].apply(
                lambda x: ", ".join(
                    map(str, x)
                )
            )


            display_rules[
                "support"
            ] = display_rules[
                "support"
            ].round(3)


            display_rules[
                "confidence"
            ] = display_rules[
                "confidence"
            ].round(3)


            display_rules[
                "lift"
            ] = display_rules[
                "lift"
            ].round(3)


            st.dataframe(
                display_rules,
                use_container_width=True
            )


            # ---------------- APRIORI EVALUATION ----------------

            st.subheader(
                "📈 Apriori Model Evaluation"
            )


            e1, e2, e3 = st.columns(3)


            e1.metric(
                "Avg Support",
                round(
                    rules["support"].mean(),
                    3
                )
            )


            e2.metric(
                "Avg Confidence",
                round(
                    rules["confidence"].mean(),
                    3
                )
            )


            e3.metric(
                "Avg Lift",
                round(
                    rules["lift"].mean(),
                    3
                )
            )


            st.success(
                "Apriori successfully identified frequently occurring "
                "library usage patterns and association rules."
            )


        else:

            st.warning(
                "Frequent itemsets were found, but no association "
                "rules were generated. Try lowering the confidence threshold."
            )


    else:

        st.warning(
            "No frequent itemsets found. "
            "Try lowering the support threshold."
        )


# =========================================================
# K-MEANS SEGMENTATION
# =========================================================

elif page == "Student Segmentation (K-Means)":

    st.header(
        "🧩 Student Segmentation (K-Means Clustering)"
    )

    st.write(
        "Groups students into behavioural segments based on their library usage."
    )


    # ---------------- FEATURES ----------------

    features = [
        "Visit_Frequency_Per_Month",
        "Books_Borrowed",
        "Digital_Resources_Accessed",
        "Reading_Duration_Hours"
    ]


    X = df[
        features
    ].copy()


    # ---------------- STANDARDIZATION ----------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )


    # ---------------- SELECT K ----------------

    n_clusters = st.slider(
        "Number of Clusters (k)",
        min_value=2,
        max_value=6,
        value=3
    )


    # ---------------- K-MEANS MODEL ----------------

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )


    df["Cluster"] = kmeans.fit_predict(
        X_scaled
    )


    # =====================================================
    # CLUSTER PROFILE + METRICS
    # =====================================================

    profile_col, metric_col = st.columns(
        [2.2, 1]
    )


    # ---------------- CLUSTER PROFILE ----------------

    with profile_col:

        st.subheader(
            "📋 Cluster Profile"
        )


        profile = (
            df.groupby("Cluster")[
                features
            ]
            .mean()
            .round(2)
        )


        st.dataframe(
            profile,
            use_container_width=True,
            height=175
        )


    # ---------------- METRICS ----------------

    with metric_col:

        score = silhouette_score(
            X_scaled,
            df["Cluster"]
        )


        m1, m2, m3 = st.columns(3)


        m1.metric(
            "Silhouette",
            round(score, 3)
        )


        m2.metric(
            "Clusters",
            n_clusters
        )


        m3.metric(
            "Students",
            len(df)
        )


    # =====================================================
    # VISUALIZATION
    # =====================================================

    chart1, chart2 = st.columns(2)


    # =====================================================
    # ELBOW METHOD
    # =====================================================

    with chart1:

        st.subheader(
            "📉 Elbow Method"
        )


        inertia = []

        k_range = range(
            2,
            7
        )


        for k in k_range:

            km = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )


            km.fit(
                X_scaled
            )


            inertia.append(
                km.inertia_
            )


        # Compact chart

        fig1, ax1 = plt.subplots(
            figsize=(4.8, 2.7)
        )


        ax1.plot(
            list(k_range),
            inertia,
            marker="o",
            markersize=5,
            linewidth=1.5
        )


        ax1.set_title(
            "Elbow Method",
            fontsize=11
        )


        ax1.set_xlabel(
            "Number of Clusters",
            fontsize=8
        )


        ax1.set_ylabel(
            "Inertia",
            fontsize=8
        )


        ax1.tick_params(
            axis="both",
            labelsize=7
        )


        ax1.grid(
            True,
            alpha=0.25
        )


        plt.tight_layout()


        st.pyplot(
            fig1,
            use_container_width=False
        )


    # =====================================================
    # PCA VISUALIZATION
    # =====================================================

    with chart2:

        st.subheader(
            "🎯 PCA Cluster Visualization"
        )


        pca = PCA(
            n_components=2
        )


        X_pca = pca.fit_transform(
            X_scaled
        )


        # Compact PCA chart

        fig2, ax2 = plt.subplots(
            figsize=(4.8, 2.7)
        )


        scatter = ax2.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=df["Cluster"],
            cmap="Set1",
            s=25,
            alpha=0.8
        )


        ax2.set_title(
            "K-Means Clusters",
            fontsize=11
        )


        ax2.set_xlabel(
            "PCA 1",
            fontsize=8
        )


        ax2.set_ylabel(
            "PCA 2",
            fontsize=8
        )


        ax2.tick_params(
            axis="both",
            labelsize=7
        )


        legend1 = ax2.legend(
            *scatter.legend_elements(),
            title="Cluster",
            fontsize=7,
            title_fontsize=8,
            loc="best"
        )


        ax2.add_artist(
            legend1
        )


        ax2.grid(
            True,
            alpha=0.25
        )


        plt.tight_layout()


        st.pyplot(
            fig2,
            use_container_width=False
        )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    st.success(
        f"✅ K-Means segmented students into {n_clusters} groups "
        "based on visit frequency, books borrowed, digital resource "
        "usage, and reading duration."
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

elif page == "Recommendations":

    st.header(
        "📚 Personalized Reading Recommendations"
    )


    reg_no = st.selectbox(
        "Select Student Registration Number",
        df[
            "Registration_No"
        ].tolist()
    )


    student = df[
        df["Registration_No"] == reg_no
    ]


    if not student.empty:

        name = student[
            "Student_Name"
        ].iloc[0]


        genre = student[
            "Preferred_Genre"
        ].mode()[0]


        st.subheader(
            f"Student: {name}"
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Preferred Genre",
            genre
        )


        c2.metric(
            "Books Borrowed",
            int(
                student[
                    "Books_Borrowed"
                ].iloc[0]
            )
        )


        c3.metric(
            "Reading Hours",
            float(
                student[
                    "Reading_Duration_Hours"
                ].iloc[0]
            )
        )


        st.success(
            f"📖 Recommendation: Explore more books from the "
            f"**{genre}** genre."
        )


        st.success(
            f"💻 Digital Recommendation: Explore digital resources "
            f"related to **{genre}**."
        )


    st.divider()


    # ---------------- BOOK PURCHASE ----------------

    st.subheader(
        "🛒 New Books Purchase Recommendation"
    )


    genre_demand = (
        df[
            "Preferred_Genre"
        ].value_counts()
    )


    top_genre = genre_demand.index[0]


    st.info(
        f"The **{top_genre}** genre has the highest demand. "
        f"The library should consider purchasing more books from "
        f"this genre."
    )


    # ---------------- PEAK HOUR ----------------

    st.subheader(
        "🕐 Peak-Hour Recommendation"
    )


    peak_hour = (
        df[
            "Peak_Library_Hour"
        ]
        .value_counts()
        .index[0]
    )


    st.info(
        f"Library usage is highest during **{peak_hour}**. "
        "Consider increasing seating and staff availability "
        "during peak hours."
    )


# =========================================================
# SIDEBAR FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Smart Library Management • Data Mining Project"
)