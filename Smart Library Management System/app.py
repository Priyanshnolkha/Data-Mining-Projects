import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="Smart Library Management",
    page_icon="📚",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_excel("Library_Management_Dataset.xlsx")

df = load_data()

st.title("📚 Smart Library Management System")
st.write("Data Mining based Library Usage Analysis & Recommendations")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Dashboard", "K-Means Clustering", "Apriori Patterns", "Recommendations"]
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.header("📊 Library Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", len(df))
    c2.metric("Books Borrowed", int(df["Books_Borrowed"].sum()))
    c3.metric("Digital Resources", int(df["Digital_Resources_Accessed"].sum()))
    c4.metric("Avg Reading Hours", round(df["Reading_Duration_Hours"].mean(), 2))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Popular Genres")
        genre = df["Preferred_Genre"].value_counts()
        st.bar_chart(genre)

    with col2:
        st.subheader("Peak Library Hours")
        peak = df["Peak_Library_Hour"].value_counts()
        st.bar_chart(peak)

    st.subheader("Reasons for Library Visits")
    reasons = df["Reason_For_Visit"].value_counts()
    st.bar_chart(reasons)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------- K-MEANS ----------------
elif page == "K-Means Clustering":

    st.header("👥 K-Means Student Segmentation")

    features = [
        "Visit_Frequency_Per_Month",
        "Books_Borrowed",
        "Digital_Resources_Accessed",
        "Reading_Duration_Hours"
    ]

    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df_cluster = df.copy()
    df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, df_cluster["Cluster"])

    st.metric("Silhouette Score", round(score, 3))

    st.subheader("Cluster Profile")
    profile = df_cluster.groupby("Cluster")[features].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots()
    ax.scatter(
        df_cluster["Visit_Frequency_Per_Month"],
        df_cluster["Reading_Duration_Hours"],
        c=df_cluster["Cluster"]
    )
    ax.set_xlabel("Visits Per Month")
    ax.set_ylabel("Reading Duration (Hours)")
    ax.set_title("Library User Clusters")
    st.pyplot(fig)

    st.info(
        "Clusters represent different student library-usage behaviour. "
        "The library can use these groups to plan targeted services."
    )

# ---------------- APRIORI ----------------
elif page == "Apriori Patterns":

    st.header("🔗 Apriori Association Rules")

    transactions = df[
        ["Preferred_Genre", "Reason_For_Visit", "Peak_Library_Hour"]
    ].astype(str)

    transactions = transactions.apply(lambda x: x.str.strip())

    # Convert categorical data into one-hot encoded basket
    basket = pd.get_dummies(transactions)

    # Lower threshold to find more useful patterns
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

            st.subheader("Top Association Rules")

            display_rules = rules[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ].head(10).copy()

            display_rules["antecedents"] = display_rules[
                "antecedents"
            ].apply(lambda x: ", ".join(map(str, x)))

            display_rules["consequents"] = display_rules[
                "consequents"
            ].apply(lambda x: ", ".join(map(str, x)))

            display_rules["support"] = display_rules[
                "support"
            ].round(3)

            display_rules["confidence"] = display_rules[
                "confidence"
            ].round(3)

            display_rules["lift"] = display_rules[
                "lift"
            ].round(3)

            st.dataframe(
                display_rules,
                use_container_width=True
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
            "No frequent itemsets found. Try lowering the support threshold."
        )
# ---------------- RECOMMENDATIONS ----------------
elif page == "Recommendations":

    st.header("📚 Personalized Reading Recommendations")

    reg_no = st.selectbox(
        "Select Student Registration Number",
        df["Registration_No"].tolist()
    )

    student = df[df["Registration_No"] == reg_no]

    if not student.empty:

        name = student["Student_Name"].iloc[0]
        genre = student["Preferred_Genre"].mode()[0]

        st.subheader(f"Student: {name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Preferred Genre", genre)
        c2.metric(
            "Books Borrowed",
            int(student["Books_Borrowed"].iloc[0])
        )
        c3.metric(
            "Reading Hours",
            float(student["Reading_Duration_Hours"].iloc[0])
        )

        st.success(
            f"📖 Recommendation: Explore more books from the **{genre}** genre."
        )

        st.success(
            f"💻 Digital Recommendation: Explore digital resources related to **{genre}**."
        )

    st.divider()

    st.subheader("🛒 New Books Purchase Recommendation")

    genre_demand = df["Preferred_Genre"].value_counts()
    top_genre = genre_demand.index[0]

    st.info(
        f"The **{top_genre}** genre has the highest demand. "
        f"The library should consider purchasing more books from this genre."
    )

    st.subheader("🕐 Peak-Hour Recommendation")

    peak_hour = df["Peak_Library_Hour"].value_counts().index[0]

    st.info(
        f"Library usage is highest during **{peak_hour}**. "
        "Consider increasing seating and staff availability during peak hours."
    )

st.sidebar.divider()
st.sidebar.caption("Smart Library Management • Data Mining Project")
