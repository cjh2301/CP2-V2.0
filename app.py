"""
Streamlit Web Application for Movie Recommendation System
"""

import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_processing import DataProcessor
from recommendation_engine import RecommendationEngine
import pickle
import os

# TMDB API Configuration
TMDB_API_KEY = 'd5601cac98ebb129144ff2b31d5133c4'
TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'

# Create a session with connection pooling for faster requests
@st.cache_resource
def get_requests_session():
    """Create and cache a requests session with connection pooling"""
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.1)
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        border-radius: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #b20710;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache processed movie data"""
    if os.path.exists('processed_movies.pkl'):
        processor = DataProcessor()
        df = processor.load_processed_data()
        return df, processor
    else:
        # Process data if not already done
        processor = DataProcessor()
        processor.load_data()
        processor.process_features()
        processor.save_processed_data()
        return processor.processed_df, processor


@st.cache_resource
def load_recommendation_engine(_processed_df):
    """Load and cache recommendation engine"""
    return RecommendationEngine(_processed_df)


@st.cache_data
def get_all_actors(_processor):
    """Get and cache all actors for autocomplete"""
    return _processor.get_all_actors()


@st.cache_data
def get_all_directors(_processor):
    """Get and cache all directors for autocomplete"""
    return _processor.get_all_directors()


def get_poster_url(movie_id, movie_title):
    """Get poster URL from TMDB API using movie ID or title"""
    try:
        session = get_requests_session()
        # Try to fetch movie details from TMDB API
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = session.get(search_url, timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        return None
    except Exception as e:
        return None


@st.cache_data(ttl=3600)
def get_poster_url_cached(movie_id, movie_title):
    """Cached version of get_poster_url to avoid repeated API calls"""
    return get_poster_url(movie_id, movie_title)


def fetch_posters_batch(movies_df):
    """Fetch multiple posters concurrently for faster loading"""
    posters = {}
    
    def fetch_single(row):
        movie_id = row['id']
        movie_title = row['title']
        poster_url = get_poster_url_cached(movie_id, movie_title)
        return movie_title, poster_url
    
    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single, row): row['title'] 
                  for _, row in movies_df.iterrows()}
        
        for future in as_completed(futures):
            try:
                title, url = future.result()
                posters[title] = url
            except Exception:
                pass
    
    return posters


def display_movie_grid(movies_df, key_prefix="movie"):
    """Display movies in a grid with selection capability using session state"""
    cols_per_row = 5
    
    # Prefetch all posters concurrently for faster loading
    posters = fetch_posters_batch(movies_df)
    
    for i in range(0, len(movies_df), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(movies_df):
                movie = movies_df.iloc[idx]
                movie_title = movie['title']
                with col:
                    poster_url = posters.get(movie_title)
                    
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x450?text=No+Poster", 
                               use_container_width=True)
                    
                    st.markdown(f"**{movie['title']}**")
                    st.caption(f"{movie['year']} • ⭐ {movie['vote_average']:.1f}")
                    
                    # Check if movie is currently selected
                    is_currently_selected = movie_title in st.session_state.selected_movies
                    
                    is_selected = st.checkbox(
                        "Select",
                        key=f"{key_prefix}_{movie_title}_{idx}",
                        value=is_currently_selected
                    )
                    
                    # Handle selection/deselection
                    if is_selected and movie_title not in st.session_state.selected_movies:
                        st.session_state.selected_movies.append(movie_title)
                        st.rerun()
                    elif not is_selected and movie_title in st.session_state.selected_movies:
                        st.session_state.selected_movies.remove(movie_title)
                        st.rerun()


def display_selection_panel():
    """Display a panel showing currently selected movies with clear all option"""
    st.markdown("""<style>
    .selection-panel {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        border: 2px solid #E50914;
    }
    .selection-title {
        color: #E50914;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .selected-movie-item {
        background-color: #16213e;
        padding: 8px 12px;
        border-radius: 5px;
        margin: 5px 0;
        color: white;
    }
    </style>""", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("###  Your Selected Movies")
        
        if len(st.session_state.selected_movies) > 0:
            st.markdown(f"**{len(st.session_state.selected_movies)} movie(s) selected**")
            
            # Display selected movies as a list
            for i, movie in enumerate(st.session_state.selected_movies):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f" {movie}")
                with col2:
                    if st.button("✕", key=f"remove_{i}", help=f"Remove {movie}"):
                        st.session_state.selected_movies.remove(movie)
                        st.rerun()
            
            st.markdown("---")
            
            # Clear all button
            if st.button(" Clear All Selections", use_container_width=True):
                st.session_state.selected_movies = []
                st.rerun()
        else:
            st.info("No movies selected yet. Select movies from the grid.")


def display_recommendations(recommendations_df, selected_movies=None):
    """Display recommended movies with details"""
    st.markdown("---")
    st.markdown("##  Your Personalized Recommendations")
    
    # Display "Because you liked" message
    if selected_movies and len(selected_movies) > 0:
        movies_text = ", ".join(selected_movies)
        st.markdown(f"**Because you liked:** {movies_text}")
    
    st.markdown("")
    
    # Prefetch all posters concurrently
    posters = fetch_posters_batch(recommendations_df)
    
    for idx, movie in recommendations_df.iterrows():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            poster_url = posters.get(movie['title'])
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x450?text=No+Poster",
                       use_container_width=True)
        
        with col2:
            st.markdown(f"### {movie['title']}")
            
            if movie['year'] > 0:
                st.markdown(f"**Year:** {int(movie['year'])}")
            
            st.markdown(f"**Rating:** ⭐ {movie['vote_average']:.1f}/10")
            
            if movie['genres_list']:
                genres_str = " • ".join(movie['genres_list'])
                st.markdown(f"**Genres:** {genres_str}")
            
            if movie['director']:
                st.markdown(f"**Director:** {movie['director']}")
            
            if movie['cast_list']:
                cast_str = ", ".join(movie['cast_list'][:5])
                st.markdown(f"**Cast:** {cast_str}")
            
            if pd.notna(movie['overview']) and movie['overview']:
                st.markdown(f"**Overview:** {movie['overview']}")
            
            if 'similarity_score' in movie:
                st.caption(f"Similarity Score: {movie['similarity_score']:.2%}")
        
        st.markdown("---")


def initialize_session_state():
    """Initialize session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'selected_genres' not in st.session_state:
        st.session_state.selected_genres = []
    if 'selected_movies' not in st.session_state:
        st.session_state.selected_movies = []
    if 'favorite_actors' not in st.session_state:
        st.session_state.favorite_actors = []
    if 'favorite_directors' not in st.session_state:
        st.session_state.favorite_directors = []
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None


def home_page():
    """Display home/welcome page"""
    st.markdown('<div class="main-header"> Movie Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discover your next favorite movie with personalized recommendations</div>', 
                unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### How it works:
        
        1. **Select Your Favorite Genres** - Choose the types of movies you love
        2. **Pick Your Favorite Movies** - Select movies you've enjoyed
        3. **Add Personal Touches** (Optional) - Tell us about your favorite actors or directors
        4. **Get Recommendations** - Receive 5 personalized movie suggestions
        
        No account needed. No rating history required. Just your preferences!
        """)
        
        st.markdown("")
        st.markdown("")
        
        if st.button(" Get Started", use_container_width=True):
            st.session_state.page = 'preferences'
            st.rerun()


def preferences_page(movies_df, processor):
    """Display preferences collection page"""
    st.markdown('<div class="main-header"> Tell Us What You Like</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select your preferences to get personalized recommendations</div>', 
                unsafe_allow_html=True)
    
    # Progress indicator
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    with progress_col1:
        st.markdown("**1. Genres**")
    with progress_col2:
        st.markdown("**2. Movies**")
    with progress_col3:
        st.markdown("**3. Get Results**")
    
    st.markdown("---")
    
    # Genre Selection
    st.markdown("###  Step 1: Select Your Favorite Genres *")
    st.caption("Choose at least 1 genre")
    
    all_genres = processor.get_all_genres()
    
    # Display genres in columns
    genre_cols = st.columns(4)
    selected_genres = []
    
    for i, genre in enumerate(all_genres):
        with genre_cols[i % 4]:
            if st.checkbox(genre, key=f"genre_{genre}", 
                          value=genre in st.session_state.selected_genres):
                selected_genres.append(genre)
    
    st.session_state.selected_genres = selected_genres
    
    st.markdown("---")
    
    # Movie Selection
    st.markdown("###  Step 2: Select Your Favorite Movies *")
    st.caption("Choose at least 1 movie you've enjoyed")
    
    # Create two columns: one for selection panel, one for movie grid
    selection_col, movie_col = st.columns([1, 3])
    
    # Display selection panel on the left
    with selection_col:
        display_selection_panel()
    
    # Display movie grid on the right
    with movie_col:
        # Get popular movies
        popular_movies = processor.get_popular_movies(n=20)
        
        # Movie search
        search_query = st.text_input(" Search for movies", placeholder="Type to search...")
        
        if search_query:
            # Filter movies by search
            search_results = movies_df[
                movies_df['title'].str.contains(search_query, case=False, na=False)
            ].sort_values('popularity', ascending=False).head(20)
            display_movies = search_results
        else:
            display_movies = popular_movies
        
        display_movie_grid(
            display_movies,
            key_prefix="select"
        )
    
    st.markdown("---")
    
    # Optional preferences
    st.markdown("###  Step 3: Optional Preferences")
    st.caption("Add more details to improve recommendations (optional)")
    
    # Get all actors and directors for autocomplete
    all_actors = get_all_actors(processor)
    all_directors = get_all_directors(processor)
    
    col1, col2 = st.columns(2)
    
    with col1:
        favorite_actors = st.multiselect(
            "Favorite Actors",
            options=all_actors,
            default=st.session_state.favorite_actors if st.session_state.favorite_actors else [],
            placeholder="Start typing to search actors...",
            help="Search and select your favorite actors"
        )
        st.session_state.favorite_actors = favorite_actors
    
    with col2:
        favorite_directors = st.multiselect(
            "Favorite Directors",
            options=all_directors,
            default=st.session_state.favorite_directors if st.session_state.favorite_directors else [],
            placeholder="Start typing to search directors...",
            help="Search and select your favorite directors"
        )
        st.session_state.favorite_directors = favorite_directors
    
    st.markdown("---")
    
    # Validation and submit
    can_submit = len(st.session_state.selected_genres) > 0 and len(st.session_state.selected_movies) > 0
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = 'home'
            st.rerun()
    
    with col3:
        if can_submit:
            if st.button("Get Recommendations →", type="primary", use_container_width=True):
                st.session_state.page = 'results'
                st.rerun()
        else:
            st.button("Get Recommendations →", disabled=True, use_container_width=True)
            if len(st.session_state.selected_genres) == 0:
                st.error("Please select at least 1 genre")
            if len(st.session_state.selected_movies) == 0:
                st.error("Please select at least 1 movie")


def results_page(engine):
    """Display recommendations results page"""
    st.markdown('<div class="main-header"> Your Movie Recommendations</div>', unsafe_allow_html=True)
    
    # Generate recommendations
    with st.spinner(" Finding the perfect movies for you..."):
        # Get actors and directors from session state (already lists)
        actors_list = st.session_state.favorite_actors if st.session_state.favorite_actors else []
        directors_list = st.session_state.favorite_directors if st.session_state.favorite_directors else []
        
        # Create user profile
        user_profile = engine.create_user_profile(
            favorite_movies=st.session_state.selected_movies,
            favorite_genres=st.session_state.selected_genres,
            favorite_actors=actors_list if actors_list else None,
            favorite_directors=directors_list if directors_list else None
        )
        
        # Get recommendations
        recommendations = engine.get_recommendations(
            user_profile,
            favorite_movies=st.session_state.selected_movies,
            n=5,
            diversity_factor=0.3
        )
        
        st.session_state.recommendations = recommendations
    
    # Display recommendations
    if st.session_state.recommendations is not None and len(st.session_state.recommendations) > 0:
        display_recommendations(st.session_state.recommendations, st.session_state.selected_movies)
        
        # Action buttons
        st.markdown("")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("← Modify Preferences", use_container_width=True):
                st.session_state.page = 'preferences'
                st.rerun()
        
        with col2:
            if st.button(" Start Over", use_container_width=True):
                # Reset session state
                st.session_state.selected_genres = []
                st.session_state.selected_movies = []
                st.session_state.favorite_actors = []
                st.session_state.favorite_directors = []
                st.session_state.recommendations = None
                st.session_state.page = 'home'
                st.rerun()
    else:
        st.error("Sorry, we couldn't find suitable recommendations. Please try different preferences.")
        if st.button("← Back to Preferences"):
            st.session_state.page = 'preferences'
            st.rerun()


def main():
    """Main application function"""
    initialize_session_state()
    
    # Load data
    with st.spinner("Loading movie database..."):
        movies_df, processor = load_data()
        engine = load_recommendation_engine(movies_df)
    
    # Route to appropriate page
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'preferences':
        preferences_page(movies_df, processor)
    elif st.session_state.page == 'results':
        results_page(engine)


if __name__ == "__main__":
    main()
