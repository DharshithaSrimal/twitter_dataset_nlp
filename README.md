# Travel Tweet Clustering: Analyzing Traveler Interests through Social Media

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-Natural%20Language%20Processing-green.svg)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

## 📋 Table of Contents
- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [Results & Analysis](#results--analysis)
- [Key Findings](#key-findings)
- [Performance Comparison](#performance-comparison)
- [Insights & Recommendations](#insights--recommendations)
- [Future Work](#future-work)
- [Installation & Usage](#installation--usage)
- [Contributing](#contributing)

## 🎯 Overview

This research project focuses on analyzing Twitter data from travelers to identify their interests and provide valuable insights to the tourism and hospitality industry. By applying various clustering algorithms to travel-related tweets, we can understand traveler preferences, emerging trends, and provide data-driven recommendations for tourism organizations.

### Why This Matters
- **For Travelers**: Get personalized travel recommendations based on community interests
- **For Tourism Industry**: Understand customer preferences and market trends
- **For Researchers**: Novel approach to social media analytics in tourism domain

## 🎯 Research Objectives

1. **Primary Objective**: Identify interest areas of travelers by analyzing their tweets
2. **Secondary Objectives**:
   - Provide actionable recommendations to travel and tourism organizations
   - Deliver insights about traveler interests and preferences
   - Compare effectiveness of different clustering algorithms for travel data

### Research Hypotheses

**Hypothesis 1:**
- H₀: Twitter data of travelers will not have significant influence towards providing insightful recommendations
- H₁: Twitter data of travelers will have significant influence towards providing insightful recommendations

**Hypothesis 2:**
- H₀: Organizations in travel and tourism industry will not benefit significantly from traveler interest insights
- H₁: Organizations in travel and tourism industry will benefit significantly from traveler interest insights

## 🔬 Methodology

### Research Framework
```
Data Collection → Preprocessing → Feature Extraction → Clustering → Analysis → Validation
```

### Clustering Algorithms Evaluated
1. **K-Means Clustering**
2. **Mini-Batch K-Means Clustering**
3. **K-Medoids Clustering**

### Data Processing Pipeline

#### 1. Data Collection
- **Tool Used**: Scweet (open-source Twitter scraping tool)
- **Data Source**: Twitter API-free scraping
- **Time Period**: January 1, 2015 - July 29, 2021
- **Search Criteria**: Tweets with '#travel' hashtag
- **Total Tweets Collected**: 125,246

#### 2. Data Fields Extracted
- User screen name
- Username
- Timestamp
- Tweet text
- Embedded text
- Emojis
- Comments count
- Likes count
- Retweets count
- Images
- Links
- Tweet URL

#### 3. Preprocessing Steps
```python
# Key preprocessing steps implemented:
- Remove duplicate tweets
- Remove numbers, web links, and email addresses
- Tokenization using NLTK
- Stop words removal
- Lemmatization for word normalization
```

#### 4. Feature Engineering
- **Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Purpose**: Transform text data into numerical feature vectors
- **Implementation**: `fit_transform()` method

## 📊 Dataset

### Dataset Statistics
- **Total Tweets**: 125,246
- **Time Span**: ~6.5 years
- **Language**: English
- **Domain**: Travel-related content
- **Preprocessing**: Cleaned and normalized text data

### Data Quality Challenges
- Misspellings in social media text
- Colloquial expressions and slang
- Abbreviations and acronyms
- Emoji and special character handling

## 🛠️ Implementation

### Technologies Used
- **Python 3.x** - Primary programming language
- **NLTK** - Natural language processing
- **scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization

### Key Libraries
```python
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
```

## 📈 Results & Analysis

### Optimal Cluster Analysis

#### K-Means Clustering
- **Optimal Clusters**: 4
- **Silhouette Score**: 0.0017068
- **Method**: Elbow method for cluster determination

#### Mini-Batch K-Means Clustering
- **Optimal Clusters**: 3
- **Silhouette Score**: 0.000881
- **Batch Size**: 32

#### K-Medoids Clustering
- **Optimal Clusters**: 3
- **Silhouette Score**: 0.00902
- **Sample Size**: 2,500 tweets (due to computational constraints)
- **Features**: 6,554

### Cluster Distribution & Insights

#### K-Means (4 Clusters)
| Cluster | Tweet Count | Percentage | Main Themes |
|---------|-------------|------------|-------------|
| 1 | 1,950 | 1.6% | Rules, regulations, policies |
| 2 | 93,095 | 74.3% | Experiences, destinations |
| 3 | 7,265 | 5.8% | International travel |
| 4 | 22,936 | 18.3% | Food, social experiences |

#### Mini-Batch K-Means (3 Clusters)
| Cluster | Tweet Count | Percentage | Main Themes |
|---------|-------------|------------|-------------|
| 1 | 10,919 | 8.3% | Activities, entertainment |
| 2 | 103,984 | 83.0% | Destinations, nature |
| 3 | 10,344 | 8.7% | Experiences, luxury travel |

#### K-Medoids (3 Clusters)
| Cluster | Tweet Count | Percentage | Main Themes |
|---------|-------------|------------|-------------|
| 1 | 998 | 40.0% | Experiences, adventure |
| 2 | 662 | 26.5% | Travel planning |
| 3 | 840 | 33.5% | Destinations, nature |

### Detailed Cluster Analysis

#### Cluster Themes Identified

**Cluster 1 (K-Means)**: Policy & Regulations
- Keywords: block, covid, restriction, court, state, federal, case, law, official, refugee, american, report, trump, biden, legal, political, leader

**Cluster 2 (K-Means)**: Travel Experiences & Destinations
- Keywords: experience, explore, country, adventure, destination, luxury, photography, business, beach, vacation, nature

**Cluster 3 (K-Means)**: International Travel
- Keywords: international, tourism, australia, scotland, japan, london, uk, canada, mexico, paris, airport

**Cluster 4 (K-Means)**: Food & Social Experiences
- Keywords: food, yummy, coffee, friends, family, restaurant, weekend, girlfriend, boyfriend, eat, night, partner, pizza, wine

## 📊 Performance Comparison

### Accuracy Analysis (5-Fold Cross Validation)

#### K-Means Performance
| Iteration | Error Rate | Accuracy |
|-----------|------------|----------|
| 1 | 0.507 | 0.493 |
| 2 | 0.293 | 0.707 |
| 3 | 0.117 | 0.883 |
| 4 | 0.111 | 0.889 |
| 5 | 0.204 | 0.796 |
| **Average** | **0.246** | **0.754** |

#### Mini-Batch K-Means Performance
| Iteration | Error Rate | Accuracy |
|-----------|------------|----------|
| 1 | 0.182 | 0.818 |
| 2 | 0.295 | 0.705 |
| 3 | 0.077 | 0.923 |
| 4 | 0.087 | 0.913 |
| 5 | 0.086 | 0.914 |
| **Average** | **0.146** | **0.854** |

#### K-Medoids Performance
| Iteration | Error Rate | Accuracy |
|-----------|------------|----------|
| 1 | 0.412 | 0.588 |
| 2 | 0.228 | 0.772 |
| 3 | 0.242 | 0.758 |
| 4 | 0.390 | 0.610 |
| 5 | 0.282 | 0.718 |
| **Average** | **0.311** | **0.689** |

### Algorithm Ranking by Performance
1. **Mini-Batch K-Means**: 85.4% accuracy ⭐
2. **K-Means**: 75.4% accuracy
3. **K-Medoids**: 68.9% accuracy

## 💡 Key Findings

### Survey Results (228 Respondents)
- **79.8%** prefer getting travel information through social media
- **Twitter** ranks as the 2nd most popular platform for travel information
- **High engagement** with travel-related posts on social media
- **Daily usage** of social media is common among travelers

### Travel Interest Categories Identified
1. **Experience-Focused Travel** (74.3% of tweets)
   - Adventure activities
   - Photography and luxury experiences
   - Beach and nature destinations

2. **Food & Social Travel** (18.3% of tweets)
   - Restaurant experiences
   - Social dining
   - Family and friend gatherings

3. **International Destinations** (5.8% of tweets)
   - Country-specific travel
   - Airport and transit experiences
   - Cultural tourism

4. **Policy & Regulations** (1.6% of tweets)
   - Travel restrictions
   - Legal and regulatory updates
   - COVID-related travel policies

## 🎯 Insights & Recommendations

### For Tourism Organizations
1. **Focus on Experience Marketing**: 74.3% of travel discussions center around experiences rather than destinations alone
2. **Leverage Food Tourism**: 18.3% of conversations involve culinary experiences - significant opportunity for restaurant and food tour partnerships
3. **International Market Potential**: Active discussions about international destinations suggest strong market for overseas travel packages
4. **Social Media Strategy**: High engagement rates indicate social media is crucial for travel marketing

### For Travelers
1. **Experience-Based Planning**: Community shows strong interest in experiential travel over traditional sightseeing
2. **Food Integration**: Consider culinary experiences as a major component of travel planning
3. **Social Sharing**: Active community engagement suggests value in sharing travel experiences

### For Researchers
1. **Mini-Batch K-Means** proves most effective for large-scale social media text clustering
2. **Computational Efficiency**: Important consideration when choosing algorithms for large datasets
3. **Domain-Specific Preprocessing**: Travel-related text requires specialized handling of abbreviations and colloquialisms

## 🚀 Future Work

### Proposed Extensions
1. **Multi-Platform Analysis**: Extend to Instagram, Facebook, TikTok
2. **Multi-Language Support**: Add Spanish, Sinhala, and other languages
3. **Real-Time Analysis**: Implement streaming data processing
4. **Sentiment Analysis Integration**: Combine clustering with sentiment scoring
5. **Predictive Modeling**: Forecast travel trends based on social media patterns

### Technical Improvements
1. **Deep Learning Integration**: Explore BERT, GPT-based embeddings
2. **Advanced Clustering**: Investigate hierarchical and density-based clustering
3. **Feature Engineering**: Incorporate user demographics and temporal features
4. **Scalability Enhancement**: Optimize for larger datasets

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
pip install scweet  # For Twitter scraping
pip install scikit-learn-extra  # For K-Medoids
```

### Basic Usage
```python
# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

# Load your tweet data
df = pd.read_csv('travel_tweets.csv')

# Preprocess text data
# (Add your preprocessing functions here)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(processed_tweets)

# Apply clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

# Analyze results
df['cluster'] = clusters
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 👥 Author

**Dharshitha Senevirathne**


---

⭐ **Star this repository if you found this research helpful!**

*Keywords: Twitter Analysis, Travel Tourism, Social Media Mining, Clustering Algorithms, NLP, Machine Learning, Travel Recommendations*
