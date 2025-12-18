"""
Recommendation Engine Module
Implements content-based filtering for movie recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class RecommendationEngine:
    """Content-based recommendation engine for movies"""
    
    def __init__(self, processed_df):
        """Initialize the recommendation engine with processed movie data"""
        self.movies_df = processed_df
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.movie_indices = None
        self._build_tfidf_matrix()
    
    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix from combined movie features"""
        print("Building TF-IDF matrix...")
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
              
        )
        
        # Fit and transform the combined features
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies_df['combined_features']
        )
        
        # Create movie title to index mapping
        self.movie_indices = pd.Series(
            self.movies_df.index,
            index=self.movies_df['title']
        ).drop_duplicates()
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def create_user_profile(self, favorite_movies=None, favorite_genres=None, 
                           favorite_actors=None, favorite_directors=None):
        """
        Create a user profile vector based on preferences
        
        Args:
            favorite_movies: List of movie titles the user likes
            favorite_genres: List of genres the user likes
            favorite_actors: List of actors the user likes
            favorite_directors: List of directors the user likes
            
        Returns:
            User profile vector in TF-IDF space
        """
        profile_text = []
        
        # Add favorite movies' features with high weight
        if favorite_movies:
            for movie in favorite_movies:
                if movie in self.movie_indices:
                    idx = self.movie_indices[movie]
                    movie_features = self.movies_df.loc[idx, 'combined_features']
                    # Add movie features multiple times for higher weight
                    profile_text.extend([movie_features] * 6)
        
        # Add favorite genres
        if favorite_genres:
            genres_text = ' '.join([g.lower().replace(' ', '') for g in favorite_genres])
            profile_text.extend([genres_text] * 3)
        
        # Add favorite actors
        if favorite_actors:
            actors_text = ' '.join([a.lower().replace(' ', '') for a in favorite_actors])
            profile_text.extend([actors_text] * 2)
        
        # Add favorite directors
        if favorite_directors:
            directors_text = ' '.join([d.lower().replace(' ', '') for d in favorite_directors])
            profile_text.extend([directors_text] * 2)
        
        # Combine all profile text
        combined_profile = ' '.join(profile_text)
        
        # Transform to TF-IDF vector
        user_vector = self.tfidf_vectorizer.transform([combined_profile])
        
        return user_vector
    
    def get_recommendations(self, user_vector, favorite_movies=None, n=5, diversity_factor=0.3):
        """
        Get movie recommendations based on user profile
        
        Args:
            user_vector: User profile in TF-IDF space
            favorite_movies: List of movies to exclude from recommendations
            n: Number of recommendations to return
            diversity_factor: Factor to promote genre diversity (0-1)
            
        Returns:
            DataFrame with recommended movies
        """
        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # Get movie indices sorted by similarity
        movie_scores = list(enumerate(similarity_scores))
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out favorite movies
        favorite_titles = set(favorite_movies) if favorite_movies else set()
        
        recommendations = []
        used_genres = []
        
        for idx, score in movie_scores:
            if len(recommendations) >= n * 3:  # Get more candidates for diversity
                break
            
            movie_title = self.movies_df.loc[idx, 'title']
            
            # Skip if already in favorites
            if movie_title in favorite_titles:
                continue
            
            movie_genres = self.movies_df.loc[idx, 'genres_list']
            
            recommendations.append({
                'index': idx,
                'score': score,
                'genres': movie_genres
            })
        
        # Apply diversity by selecting movies with different genres
        if diversity_factor > 0 and len(recommendations) > n:
            diverse_recommendations = []
            genre_counts = {}
            
            for rec in recommendations:
                # Calculate genre overlap penalty
                penalty = 0
                for genre in rec['genres']:
                    penalty += genre_counts.get(genre, 0)
                
                # Adjusted score with diversity
                adjusted_score = rec['score'] - (penalty * diversity_factor * 0.1)
                rec['adjusted_score'] = adjusted_score
                
                diverse_recommendations.append(rec)
            
            # Re-sort by adjusted score
            diverse_recommendations = sorted(
                diverse_recommendations,
                key=lambda x: x['adjusted_score'],
                reverse=True
            )[:n]
            
            # Update genre counts
            for rec in diverse_recommendations:
                for genre in rec['genres']:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            final_indices = [rec['index'] for rec in diverse_recommendations]
        else:
            final_indices = [rec['index'] for rec in recommendations[:n]]
        
        # Create result dataframe
        result_df = self.movies_df.loc[final_indices].copy()
        result_df['similarity_score'] = [
            similarity_scores[idx] for idx in final_indices
        ]
        
        return result_df
    
    def get_similar_movies(self, movie_title, n=5):
        """
        Get movies similar to a given movie
        
        Args:
            movie_title: Title of the movie
            n: Number of similar movies to return
            
        Returns:
            DataFrame with similar movies
        """
        if movie_title not in self.movie_indices:
            return pd.DataFrame()
        
        # Get movie index
        idx = self.movie_indices[movie_title]
        
        # Calculate similarity scores
        movie_vector = self.tfidf_matrix[idx]
        similarity_scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        # Get top similar movies (excluding the movie itself)
        movie_scores = list(enumerate(similarity_scores))
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        
        indices = [i[0] for i in movie_scores]
        
        result_df = self.movies_df.iloc[indices].copy()
        result_df['similarity_score'] = [similarity_scores[i] for i in indices]
        
        return result_df
    
    def search_movies(self, query, limit=20):
        """
        Search for movies by title
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            DataFrame with matching movies
        """
        query_lower = query.lower()
        matches = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(query_lower, na=False)
        ]
        
        # Sort by popularity
        matches = matches.sort_values('popularity', ascending=False)
        
        return matches.head(limit)
    
    def save_model(self, filepath='recommendation_model.pkl'):
        """Save the recommendation model"""
        model_data = {
            'tfidf_matrix': self.tfidf_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'movie_indices': self.movie_indices
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='recommendation_model.pkl'):
        """Load the recommendation model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.movie_indices = model_data['movie_indices']
        
        print(f"Model loaded from {filepath}")


def main():
    """Test the recommendation engine"""
    from data_processing import DataProcessor
    
    # Load processed data
    processor = DataProcessor()
    processor.load_processed_data()
    
    # Create recommendation engine
    engine = RecommendationEngine(processor.processed_df)
    
    # Test with sample preferences
    favorite_movies = ['The Dark Knight', 'Inception']
    favorite_genres = ['Action', 'Thriller']
    
    user_profile = engine.create_user_profile(
        favorite_movies=favorite_movies,
        favorite_genres=favorite_genres
    )
    
    recommendations = engine.get_recommendations(
        user_profile,
        favorite_movies=favorite_movies,
        n=5
    )
    
    print("\nRecommendations:")
    print(recommendations[['title', 'genres_list', 'year', 'similarity_score']])
    
    # Save model
    engine.save_model()


if __name__ == "__main__":
    main()
