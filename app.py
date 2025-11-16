
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Unified ML Dashboard")

df = pd.read_csv("survey_data.csv")

st.sidebar.header("Filters")
num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    vals = st.sidebar.multiselect(f"{col}", df[col].unique())
    if vals:
        df = df[df[col].isin(vals)]

st.subheader("Filtered Data Preview")
st.dataframe(df.head())

st.header("1. Regression")
if len(num_cols) >= 2:
    x_col = st.selectbox("Select feature", num_cols)
    y_col = st.selectbox("Select target", num_cols)
    if x_col != y_col:
        X = df[[x_col]]
        y = df[y_col]
        model = LinearRegression()
        model.fit(X, y)
        st.write("Coefficient:", model.coef_[0])
        st.write("Intercept:", model.intercept_)
        fig, ax = plt.subplots()
        ax.scatter(X, y)
        ax.plot(X, model.predict(X))
        st.pyplot(fig)

st.header("2. Clustering (KMeans)")
if len(num_cols) >= 2:
    k = st.slider("Clusters", 2, 6, 3)
    km = KMeans(n_clusters=k)
    cl = km.fit_predict(df[num_cols].fillna(0))
    df["cluster"] = cl
    st.write(df[["cluster"]].head())

st.header("3. Association Rule Mining")
cat_df = df[cat_cols].astype(str)
encoded = pd.get_dummies(cat_df)
freq = apriori(encoded, min_support=0.1, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.5)
st.write(rules.head())

st.header("4. Classification (Label Encoding + KMeans as pseudo classifier)")
if cat_cols.any():
    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df[cat_cols[0]])
    km2 = KMeans(n_clusters=2)
    df["pred_cluster"] = km2.fit_predict(df[num_cols].fillna(0))
    st.write(df[["label_encoded", "pred_cluster"]].head())
