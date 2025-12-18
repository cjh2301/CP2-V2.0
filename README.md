# Movie Recommendation System

A content-based movie recommendation system designed to solve the cold start problem for new users. Built with Python and Streamlit.

## Features

- **Content-Based Filtering**: Recommends movies based on movie features (genres, cast, crew, keywords, plot)
- **No Cold Start Problem**: Works without user rating history
- **Interactive UI**: Clean Streamlit interface for easy preference collection
- **Personalized Recommendations**: Top-5 movie suggestions based on user preferences
- **Fast Poster Loading**: Optimized concurrent fetching with connection pooling for quick image loading
- **Movie Posters**: High-quality poster images from TMDB API
- **Diversity**: Ensures genre diversity in recommendations

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Data Processing**: Pandas, NumPy
- **NLP/ML**: scikit-learn, TF-IDF vectorization
- **Dataset**: TMDB 5000 Movie Dataset (Kaggle)

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn streamlit requests urllib3
```

## Usage

### 1. Process the Data

First, process the TMDB dataset to extract and clean movie features:

```bash
python data_processing.py
```

This will create `processed_movies.pkl` containing cleaned movie data.

### 2. Run the Web Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Evaluate the System (Optional)

To evaluate recommendation quality metrics:

```bash
python evaluation.py
```

This will generate an evaluation report with Precision, NDCG and Diversity.

## Project Structure

```
├── app.py                      # Streamlit web application
├── data_processing.py          # Data loading and preprocessing
├── recommendation_engine.py    # Content-based recommendation logic
├── evaluation.py              # System evaluation and metrics
├── tmdb_5000_movies.csv       # Movie metadata dataset
├── tmdb_5000_credits.csv      # Cast and crew dataset
├── processed_movies.pkl       # Processed movie data (generated)
├── recommendation_model.pkl   # Trained model (generated)
└── README.md                  # This file
```

## How It Works

### User Flow:
1. **Welcome Page**: Introduction to the recommendation system
2. **Preferences Collection**: 
   - Select favorite genres (required)
   - Select favorite movies (required)
   - Optionally add favorite actors and directors
3. **Recommendations**: View 5 personalized movie suggestions with posters and details

### Technical Process:
1. **Feature Extraction**: System processes movie metadata (genres, keywords, cast, crew, plot)
2. **TF-IDF Vectorization**: Converts text features into numerical vectors
3. **User Profile Creation**: Combines user preferences into a profile vector
4. **Similarity Calculation**: Uses cosine similarity to find movies similar to user preferences
5. **Concurrent Poster Loading**: Fetches movie posters in parallel for fast display
6. **Recommendation**: Returns top-5 movies with diversity across genres

## Performance Metrics

The system is evaluated using industry-standard information retrieval metrics:
- **Precision@5**: Proportion of relevant movies in top-5 recommendations (target: ≥0.3)
- **NDCG@5**: Normalized Discounted Cumulative Gain - measures ranking quality with position-based discounting (target: ≥0.4)
- **Diversity**: Intra-list diversity score measuring how different recommended movies are from each other (target: ≥0.3)

Evaluation uses 50 simulated test cases that mimic real user onboarding scenarios with 1 favorite movie, 1-2 genres, and optional actors/directors.

## API Configuration

The app uses TMDB API for movie posters. API key is configured in `app.py`:
```python
TMDB_API_KEY = 'd5601cac98ebb129144ff2b31d5133c4'
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- streamlit
- requests
- urllib3 (for connection pooling optimization)

## Future Enhancements

- Hybrid filtering (content + collaborative)
- User accounts and preference saving
- More detailed movie information
- Trailer integration
- Mobile-responsive design improvements

## License

This project uses the TMDB 5000 Movie Dataset from Kaggle.

## Author

Built following the Product Requirements Document for a movie recommendation system addressing the cold start problem.
