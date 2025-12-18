"""
Data Processing Module for Movie Recommendation System
Handles loading, cleaning, and preprocessing of TMDB dataset
"""

import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import re

class DataProcessor:
    """Handles all data processing operations for the movie recommendation system"""
    
    def __init__(self, movies_path='tmdb_5000_movies.csv', credits_path='tmdb_5000_credits.csv'):
        """Initialize the data processor with dataset paths"""
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.movies_df = None
        self.processed_df = None
        
    def load_data(self):
        """Load the TMDB datasets"""
        print("Loading datasets...")
        self.movies_df = pd.read_csv(self.movies_path)
        credits_df = pd.read_csv(self.credits_path)
        
        # Drop title from credits to avoid duplicate columns
        credits_df = credits_df.drop('title', axis=1)
        
        # Merge datasets on id and movie_id
        self.movies_df = self.movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')
        print(f"Loaded {len(self.movies_df)} movies")
        
        return self.movies_df
    
    def parse_json_column(self, column_data):
        """Safely parse JSON-like string columns"""
        try:
            if pd.isna(column_data):
                return []
            return ast.literal_eval(column_data)
        except (ValueError, SyntaxError):
            return []
    
    def extract_names(self, data_list, key='name', limit=None):
        """Extract names from parsed JSON data"""
        if not isinstance(data_list, list):
            return []
        names = [item[key] for item in data_list if key in item]
        return names[:limit] if limit else names
    
    def extract_director(self, crew_list):
        """Extract director from crew data"""
        if not isinstance(crew_list, list):
            return ''
        for person in crew_list:
            if isinstance(person, dict) and person.get('job') == 'Director':
                return person.get('name', '')
        return ''
    
    def clean_text(self, text):
        """Clean text data - remove spaces, convert to lowercase"""
        if pd.isna(text) or text == '':
            return ''
        # Remove spaces between words to treat multi-word items as single tokens
        text = str(text).lower()
        text = re.sub(r'\s+', '', text)
        return text
    
    def process_features(self):
        """Process and extract all features from the dataset"""
        print("Processing movie features...")
        
        df = self.movies_df.copy()
        
        # Extract year from release_date
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        df['year'] = df['year'].fillna(0).astype(int)
        
        # Parse JSON columns
        df['genres_parsed'] = df['genres'].apply(self.parse_json_column)
        df['keywords_parsed'] = df['keywords'].apply(self.parse_json_column)
        df['cast_parsed'] = df['cast'].apply(self.parse_json_column)
        df['crew_parsed'] = df['crew'].apply(self.parse_json_column)
        
        # Extract specific information
        df['genres_list'] = df['genres_parsed'].apply(lambda x: self.extract_names(x))
        df['keywords_list'] = df['keywords_parsed'].apply(lambda x: self.extract_names(x))
        df['cast_list'] = df['cast_parsed'].apply(lambda x: self.extract_names(x, limit=10))
        df['director'] = df['crew_parsed'].apply(self.extract_director)
        
        # Clean text fields
        df['overview_clean'] = df['overview'].fillna('')
        
        # Create combined feature string for content-based filtering
        df['genres_str'] = df['genres_list'].apply(lambda x: ' '.join([self.clean_text(g) for g in x]))
        df['keywords_str'] = df['keywords_list'].apply(lambda x: ' '.join([self.clean_text(k) for k in x]))
        df['cast_str'] = df['cast_list'].apply(lambda x: ' '.join([self.clean_text(c) for c in x]))
        df['director_str'] = df['director'].apply(self.clean_text)
        
        # Combine all features with weighted importance
        # Genres and keywords get higher weight (repeated 3 times)
        df['combined_features'] = (
            df['genres_str'] + ' ' + df['genres_str'] + ' ' + df['genres_str'] + ' ' +
            df['keywords_str'] + ' ' + df['keywords_str'] + ' ' + df['keywords_str'] + ' ' +
            df['cast_str'] + ' ' + df['cast_str'] + ' ' +
            df['director_str'] + ' ' + df['director_str'] + ' ' +
            df['overview_clean']
        )
        
        # Select relevant columns for the final dataframe
        columns_to_keep = [
            'id', 'title', 'genres_list', 'keywords_list', 'cast_list', 
            'director', 'overview', 'year', 'vote_average', 'popularity',
            'combined_features'
        ]
        
        # Keep only the columns that exist
        df = df[columns_to_keep].copy()
        
        # Create poster_path column (will be fetched from TMDB API using movie id)
        df['poster_path'] = None
        
        # Remove duplicates and missing titles
        df = df.dropna(subset=['title'])
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        self.processed_df = df
        print(f"Processed {len(df)} movies successfully")
        
        return df
    
    def get_all_genres(self):
        """Get list of all unique genres"""
        if self.processed_df is None:
            raise ValueError("Data must be processed first")
        
        all_genres = set()
        for genres in self.processed_df['genres_list']:
            all_genres.update(genres)
        
        return sorted(list(all_genres))
    
    def get_popular_movies(self, n=100):
        """Get top N popular movies across different genres"""
        if self.processed_df is None:
            raise ValueError("Data must be processed first")
        
        # Sort by popularity and vote_average
        df = self.processed_df.copy()
        df = df[df['vote_average'] > 6.0]  # Filter for decent ratings
        df = df.sort_values(['popularity', 'vote_average'], ascending=[False, False])
        
        return df.head(n)
    
    def get_all_actors(self):
        """Get list of all unique actors sorted by frequency"""
        if self.processed_df is None:
            raise ValueError("Data must be processed first")
        
        from collections import Counter
        actor_counts = Counter()
        for cast_list in self.processed_df['cast_list']:
            if isinstance(cast_list, list):
                actor_counts.update(cast_list)
        
        # Return actors sorted by frequency (most common first)
        return [actor for actor, count in actor_counts.most_common()]
    
    def get_all_directors(self):
        """Get list of all unique directors sorted by frequency"""
        if self.processed_df is None:
            raise ValueError("Data must be processed first")
        
        from collections import Counter
        director_counts = Counter()
        for director in self.processed_df['director']:
            if director and isinstance(director, str) and director.strip():
                director_counts[director] += 1
        
        # Return directors sorted by frequency (most common first)
        return [director for director, count in director_counts.most_common()]
    
    def save_processed_data(self, filepath='processed_movies.pkl'):
        """Save processed dataframe to pickle file"""
        if self.processed_df is None:
            raise ValueError("Data must be processed first")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.processed_df, f)
        
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath='processed_movies.pkl'):
        """Load processed dataframe from pickle file"""
        with open(filepath, 'rb') as f:
            self.processed_df = pickle.load(f)
        
        print(f"Loaded {len(self.processed_df)} processed movies")
        return self.processed_df


def main():
    """Main function to process data"""
    processor = DataProcessor()
    processor.load_data()
    processor.process_features()
    processor.save_processed_data()
    
    print("\nSample of processed data:")
    print(processor.processed_df[['title', 'genres_list', 'director', 'year']].head())
    
    print("\nAll genres:", processor.get_all_genres())


if __name__ == "__main__":
    main()
