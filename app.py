import streamlit as st
import requests
import pandas as pd
from datetime import datetime, time

st.set_page_config(
    page_title="Support Intelligence Platform",
    layout="wide"
)

st.title("Support Intelligence Platform")
st.caption("Real-time sentiment triage and response time prediction for customer support operations.")

PRIORITY_MAP = {
    "negative": "URGENT",
    "non-negative": "STANDARD"
}

tab1, tab2 = st.tabs(["Live Analysis", "Batch Evaluation"])

with tab1:
    st.subheader("Analyze a Tweet")

    with st.form("predict_form"):
        tweet = st.text_area(
            "Tweet Content",
            placeholder="e.g. @AmazonHelp my package never arrived and support has not responded",
            height=100
        )
        col_a, col_b = st.columns(2)
        with col_a:
            tweet_date = st.date_input("Date", value=datetime.now().date())
        with col_b:
            tweet_time = st.time_input("Time", value=time(datetime.now().hour, 0))

        submitted = st.form_submit_button("Analyze", use_container_width=True, type="primary")

    if submitted:
        if not tweet.strip():
            st.warning("Please enter a tweet.")
        else:
            with st.spinner("Running inference..."):
                try:
                    timestamp = f"{tweet_date} {tweet_time}"
                    response = requests.post(
                        "http://localhost:8000/predict",
                        json={"text": tweet, "timestamp": timestamp}
                    )
                    result = response.json()

                    st.divider()

                    col1, col2, col3, col4 = st.columns(4)

                    priority = PRIORITY_MAP[result["sentiment"]]

                    with col1:
                        st.metric("Priority", priority)
                        if result["sentiment"] == "negative":
                            st.error("Requires immediate attention")
                        else:
                            st.success("Standard queue")

                    with col2:
                        st.metric("Estimated Response", f"{result['estimated_response_mins']:.0f} min")
                        st.caption(result["estimated_response_label"])

                    with col3:
                        st.metric("Company", result["company"])
                        st.caption("Extracted from mention")

                    with col4:
                        st.metric("Tweet Hour", f"{result['hour']}:00")
                        st.caption("Weekend" if result["is_weekend"] else "Weekday")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the server is running on http://localhost:8000")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

with tab2:
    st.subheader("Batch Evaluation")
    st.caption("Upload a labeled CSV to evaluate model performance against ground truth.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        with st.expander("Preview data"):
            st.dataframe(
                df[["customer_text", "company_name", "created_at_customer", "sentiment", "response_time_capped"]].head(10),
                use_container_width=True
            )

        st.write(f"{len(df):,} rows loaded — evaluating first 1,000")

        if st.button("Run Evaluation", use_container_width=True, type="primary"):
            with st.spinner("Running batch inference..."):
                try:
                    tweets = [
                        {
                            "text": str(row.get("customer_text", "")),
                            "company": str(row.get("company_name", "")),
                            "timestamp": str(row.get("created_at_customer", ""))
                        }
                        for _, row in df.head(1000).iterrows()
                    ]

                    response = requests.post(
                        "http://localhost:8000/predict/batch",
                        json={"tweets": tweets}
                    )
                    results = response.json()

                    results_df = pd.DataFrame(results)

                    results_df["actual_sentiment_raw"] = df["sentiment"].values[:len(results)]
                    results_df["actual_priority"] = df["sentiment"].map(PRIORITY_MAP).values[:len(results)]
                    results_df["actual_response_mins"] = df["response_time_capped"].values[:len(results)]
                    results_df["priority"] = results_df["sentiment"].map(PRIORITY_MAP)
                    results_df["sentiment_correct"] = results_df["sentiment"] == results_df["actual_sentiment_raw"]
                    results_df["response_error_mins"] = (
                        results_df["estimated_response_mins"] - results_df["actual_response_mins"]
                    ).abs()

                    st.divider()
                    st.subheader("Evaluation Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Tweets Evaluated", f"{len(results_df):,}")
                    col2.metric("Sentiment Accuracy", f"{results_df['sentiment_correct'].mean():.1%}")
                    col3.metric("Response Time MAE", f"{results_df['response_error_mins'].mean():.0f} min")
                    col4.metric("Urgent Flagged", f"{(results_df['sentiment']=='negative').sum():,}")

                    st.subheader("Predictions")
                    st.dataframe(
                        results_df[[
                            "priority",
                            "actual_priority",
                            "sentiment_correct",
                            "estimated_response_mins",
                            "actual_response_mins",
                            "response_error_mins",
                            "company"
                        ]].rename(columns={
                            "priority": "Predicted Priority",
                            "actual_priority": "Actual Priority",
                            "sentiment_correct": "Correct",
                            "estimated_response_mins": "Predicted Response (min)",
                            "actual_response_mins": "Actual Response (min)",
                            "response_error_mins": "Error (min)",
                            "company": "Company"
                        }),
                        use_container_width=True
                    )

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Export Results",
                        csv,
                        "evaluation_results.csv",
                        "text/csv",
                        use_container_width=True
                    )

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the server is running on http://localhost:8000")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")