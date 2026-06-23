# Customer Support Intelligence Platform
 
An end-to-end machine learning project that analyzes customer support conversations and generates operational insights for customer service teams.
 
## Problem Statement
 
Organizations receive a large volume of customer support requests through social media and digital channels. Understanding customer concerns, identifying recurring issues, and monitoring support performance is difficult to do manually at scale.
 
This project explores how machine learning can be used to analyze real customer support interactions and support data-driven decision-making in support operations.
 
## Objectives
 
- Predict expected response time for incoming customer requests
- Discover common complaint categories through unsupervised clustering
- Analyze customer sentiment and interaction patterns
- Generate actionable insights about support performance across companies and issue types
## Dataset
 
**Twitter Customer Support Dataset (Kaggle)** — 2.8M real customer-company conversations from Twitter, including customer messages, company responses, and conversation timestamps.
 
Additional preprocessing was applied to pair customer messages with company replies and engineer response time metrics.
 
 
## Key Findings from EDA
 
- Companies respond to 82% of customer tweets; median response time is 21 minutes
- Financial companies (Visa, PayPal) are significantly slower than tech companies
- Tweet activity peaks 3pm–7pm on weekdays — useful for staffing and resource planning
- Tweet length jumped after Twitter raised the character limit to 280 in Nov 2017 — visible in the data
- Tweet length has near-zero correlation with response time — not a useful feature
## Machine Learning Tasks
 
### Regression
Predict expected response time for a customer request.
- Target: `response_time_mins`
- Models: Linear Regression, Random Forest Regressor, Gradient Boosting
### Clustering
Group customer complaints into common issue categories without manual labeling.
- Approach: TF-IDF + K-Means
### Sentiment Analysis
Analyze customer sentiment and its relationship with response behavior and company performance.
 
## MLOps Components
 
- Git and GitHub — version control from day one
- uv — environment and dependency management
- MLflow — experiment tracking, model registry, and versioning
- Configuration management — hyperparameters and paths in config files
- FastAPI — prediction endpoint
- Docker — containerized deployment
- Model card — documented limitations and intended use
## Project Structure
 
```
notebooks/        # EDA and experimentation
src/
  data/           # data loading and preprocessing
  features/       # feature engineering
  models/         # model training and evaluation
  api/            # FastAPI endpoint
  utils/          # shared utilities
data/raw/         # raw data (gitignored)
tests/            # unit tests
config.yaml       # hyperparameters and paths
```
 
## Setup
 
```bash
git clone <repo-url>
cd support-ticket-triage
uv sync
```
 
## Run API
 
```bash
uv run uvicorn src.api.main:app --reload
```
 
## Milestones
 
- [x] Environment setup — repo, virtual environment, dependencies pinned
- [x] EDA — data quality, distributions, response time analysis, text exploration
- [ ] Feature engineering pipeline
- [ ] Baseline model + problem framing
- [ ] Model comparison with MLflow tracking
- [ ] Notebook → .py modules
- [ ] Config and reproducibility
- [ ] FastAPI /predict endpoint
- [ ] Docker deployment
- [ ] Model registry and model card
- [ ] Final demo and retrospective
## Expected Business Value
 
The platform can help organizations:
 
- Monitor customer support performance across channels and teams
- Identify recurring customer issues before they escalate
- Estimate support workload and predict response behavior
- Benchmark support performance against industry peers
- Make data-driven decisions about staffing and prioritization