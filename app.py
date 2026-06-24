import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Support Intelligence",
    layout="centered"
)

st.title("Customer Support Intelligence")
st.markdown("Analyze customer tweets — detect sentiment and predict response time.")

companies = [
    "AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares",
    "Delta", "AmericanAir", "TMobileHelp", "comcastcares",
    "British_Airways", "Tesco", "askvisa", "SCsupport",
    "Postmates_Help", "JetBlue", "VerizonSupport", "unknown"
]

with st.form("predict_form"):
    tweet = st.text_area("Customer Tweet", placeholder="Type or paste a customer tweet here...", height=120)
    company = st.selectbox("Company", options=companies)
    submitted = st.form_submit_button("Analyze", use_container_width=True)

if submitted:
    if not tweet.strip():
        st.error("Please enter a tweet.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"text": tweet, "company": company}
                )
                result = response.json()

                col1, col2 = st.columns(2)

                with col1:
                    sentiment = result["sentiment"]
                    if sentiment == "negative":
                        st.error(f"Sentiment: {sentiment.upper()}")
                    else:
                        st.success(f"Sentiment: {sentiment.upper()}")
                    st.caption(f"Score: {result['sentiment_score']}")

                with col2:
                    st.info(f"Estimated Response: {result['estimated_response_mins']:.0f} mins")
                    st.caption(result["estimated_response_label"])

            except Exception as e:
                st.error(f"API error: {e}. Make sure the API is running.")