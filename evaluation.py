"""
Evaluation Module for Movie Recommendation System
Evaluates using Precision@5, NDCG@5, and Diversity (intra-list diversity) metrics
"""

import pandas as pd
import numpy as np
from recommendation_engine import RecommendationEngine
from data_processing import DataProcessor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class RecommendationEvaluator:
    """Evaluates recommendation system using Precision@5, NDCG@5, and Diversity"""
    
    def __init__(self, engine, movies_df):
        """
        Initialize evaluator
        
        Args:
            engine: RecommendationEngine instance
            movies_df: Processed movies dataframe
        """
        self.engine = engine
        self.movies_df = movies_df
        self.k = 5  # Evaluate at top-5 recommendations
    
    def create_test_cases(self, n_cases=50):
        """
        Create test cases that simulate REAL user onboarding behavior.
        Each test case: 1 favorite movie + 1-2 genres + optional actors/director.
        
        Args:
            n_cases: Number of test cases to create
            
        Returns:
            List of test case dictionaries
        """
        test_cases = []
        np.random.seed(42)
        
        # Use all movies with at least some metadata
        candidate_movies = self.movies_df[
            (self.movies_df['genres_list'].apply(len) > 0)
        ].copy()
        
        # Create test cases
        for _ in range(n_cases):
            # Simulate real onboarding: pick 1 random favorite movie
            seed_movie_df = candidate_movies.sample(n=1)
            seed_movie_title = seed_movie_df['title'].iloc[0]
            seed_movie_data = seed_movie_df.iloc[0]
            
            # Extract 1-2 genres from the seed movie
            available_genres = seed_movie_data['genres_list']
            if len(available_genres) > 0:
                n_genres = min(np.random.choice([1, 2]), len(available_genres))
                user_genres = list(np.random.choice(available_genres, size=n_genres, replace=False))
            else:
                user_genres = []
            
            # Optionally extract 1-2 actors (50% chance)
            user_actors = []
            if np.random.random() < 0.5 and len(seed_movie_data['cast_list']) > 0:
                n_actors = min(np.random.choice([1, 2]), len(seed_movie_data['cast_list']))
                user_actors = list(np.random.choice(seed_movie_data['cast_list'], size=n_actors, replace=False))
            
            # Optionally extract director (30% chance)
            user_director = None
            if np.random.random() < 0.3 and seed_movie_data['director']:
                user_director = seed_movie_data['director']
            
            # Get ground truth relevant movies using attribute-based rules
            relevant_movies = self.get_relevant_movies(
                seed_movie_title=seed_movie_title,
                seed_movie_data=seed_movie_data,
                user_genres=user_genres,
                user_actors=user_actors,
                user_director=user_director
            )
            
            test_cases.append({
                'seed_movies': [seed_movie_title],
                'seed_genres': user_genres,
                'user_actors': user_actors,
                'user_director': user_director,
                'seed_movie_data': seed_movie_data,
                'relevant_movies': relevant_movies
            })
        
        return test_cases
    
    def get_relevant_movies(self, seed_movie_title, seed_movie_data, user_genres, user_actors, user_director):
        """
        Calculate ground truth relevant movies using ATTRIBUTE-BASED RULES.
        Independent from the recommender's TF-IDF/similarity calculation.
        
        A movie is relevant if:
        - It shares at least 1 user genre AND
        - At least 1 of the following:
          * same director
          * OR at least 1 actor overlap
          * OR at least 1 keyword overlap
        
        Args:
            seed_movie_title: Title of the seed movie
            seed_movie_data: Full data row for the seed movie
            user_genres: List of user's selected genres
            user_actors: List of user's selected actors (may be empty)
            user_director: User's selected director (may be None)
            
        Returns:
            Dict of {movie_title: relevance_score}
        """
        relevant_movies = []
        user_genres_set = set(user_genres)
        user_actors_set = set(user_actors) if user_actors else set()
        seed_keywords_set = set(seed_movie_data['keywords_list'])
        
        for idx, movie in self.movies_df.iterrows():
            # Skip the seed movie itself
            if movie['title'] == seed_movie_title:
                continue
            
            # Skip movies with no genres
            if len(movie['genres_list']) == 0:
                continue
            
            # Check if movie shares at least 1 user genre
            movie_genres_set = set(movie['genres_list'])
            genre_overlap = len(user_genres_set.intersection(movie_genres_set))
            
            if genre_overlap == 0:
                continue  # Must share at least 1 genre
            
            # Check additional relevance criteria
            has_additional_match = False
            relevance_score = genre_overlap  # Base score from genre overlap
            
            # Check director match
            if user_director and movie['director'] == user_director:
                has_additional_match = True
                relevance_score += 3  # Director match is highly relevant
            
            # Check actor overlap
            movie_actors_set = set(movie['cast_list'])
            actor_overlap = len(user_actors_set.intersection(movie_actors_set))
            if actor_overlap > 0:
                has_additional_match = True
                relevance_score += actor_overlap * 2  # Each actor match adds relevance
            
            # Check keyword overlap
            movie_keywords_set = set(movie['keywords_list'])
            keyword_overlap = len(seed_keywords_set.intersection(movie_keywords_set))
            if keyword_overlap >= 1:
                has_additional_match = True
                relevance_score += min(keyword_overlap, 3) * 0.5  # Cap keyword contribution
            
            # Movie is relevant if it has genre match AND at least one additional match
            if has_additional_match:
                relevant_movies.append({
                    'title': movie['title'],
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance score and return as dict
        relevant_movies = sorted(relevant_movies, key=lambda x: x['relevance_score'], reverse=True)
        
        return {rm['title']: rm['relevance_score'] for rm in relevant_movies}
    
    def calculate_precision_at_k(self, recommendations, relevant_movies, k=5):
        """
        Calculate Precision@k
        
        Args:
            recommendations: List of recommended movie titles
            relevant_movies: Dict of relevant movie titles and their relevance scores
            k: Number of top recommendations to consider
            
        Returns:
            Precision@k score (0-1)
        """
        if len(recommendations) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_in_top_k = sum(1 for movie in top_k if movie in relevant_movies)
        
        precision = relevant_in_top_k / k
        return precision
    
    def calculate_dcg_at_k(self, recommendations, relevant_movies, k=5):
        """
        Calculate Discounted Cumulative Gain at k
        
        Args:
            recommendations: List of recommended movie titles
            relevant_movies: Dict of relevant movie titles and their relevance scores
            k: Number of top recommendations to consider
            
        Returns:
            DCG@k score
        """
        dcg = 0.0
        for i, movie in enumerate(recommendations[:k]):
            if movie in relevant_movies:
                relevance = relevant_movies[movie]
                # DCG formula: rel_i / log2(i + 2)
                dcg += relevance / np.log2(i + 2)
        
        return dcg
    
    def calculate_ndcg_at_k(self, recommendations, relevant_movies, k=5):
        """
        Calculate Normalized Discounted Cumulative Gain at k
        
        Args:
            recommendations: List of recommended movie titles
            relevant_movies: Dict of relevant movie titles and their relevance scores
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@k score (0-1)
        """
        if len(relevant_movies) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = self.calculate_dcg_at_k(recommendations, relevant_movies, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_order = sorted(relevant_movies.items(), key=lambda x: x[1], reverse=True)
        ideal_movies = [movie for movie, _ in ideal_order]
        idcg = self.calculate_dcg_at_k(ideal_movies, relevant_movies, k)
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg
    
    def calculate_intra_list_diversity(self, recommendations_df):
        """
        Calculate intra-list diversity using genre dissimilarity
        Measures how diverse the recommended movies are from each other
        
        Args:
            recommendations_df: DataFrame of recommended movies
            
        Returns:
            Diversity score (0-1), higher is more diverse
        """
        if len(recommendations_df) <= 1:
            return 0.0
        
        # Create genre vectors for each movie
        all_genres = set()
        for _, movie in recommendations_df.iterrows():
            all_genres.update(movie['genres_list'])
        
        all_genres = sorted(list(all_genres))
        genre_vectors = []
        
        for _, movie in recommendations_df.iterrows():
            vector = [1 if genre in movie['genres_list'] else 0 for genre in all_genres]
            genre_vectors.append(vector)
        
        # Calculate pairwise dissimilarity
        if len(genre_vectors) == 0 or all(sum(v) == 0 for v in genre_vectors):
            return 0.0
        
        similarities = cosine_similarity(genre_vectors)
        
        # Calculate average dissimilarity (1 - similarity)
        n = len(similarities)
        total_dissimilarity = 0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                dissimilarity = 1 - similarities[i][j]
                total_dissimilarity += dissimilarity
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_diversity = total_dissimilarity / count
        return avg_diversity
    
    def evaluate_system(self, n_test_cases=50):
        """
        Evaluate the entire recommendation system across three scenarios
        
        Args:
            n_test_cases: Number of test cases to evaluate
            
        Returns:
            Dictionary with scenario-specific metrics
        """
        print(f"Evaluating recommendation system with {n_test_cases} test cases...")
        print(f"Scenarios: A (Genre Only), B (Genre + Movie), C (Enhanced)")
        print(f"Metrics: Precision@{self.k}, NDCG@{self.k}, Diversity (intra-list)\n")
        
        test_cases = self.create_test_cases(n_test_cases)
        
        # Separate tracking for each scenario
        scenarios = {
            'A': {'name': 'Genre Only (Baseline)', 'precisions': [], 'ndcgs': [], 'diversities': [], 'successful': 0},
            'B': {'name': 'Genre + Movie', 'precisions': [], 'ndcgs': [], 'diversities': [], 'successful': 0},
            'C': {'name': 'Enhanced (w/ Actor/Director)', 'precisions': [], 'ndcgs': [], 'diversities': [], 'successful': 0}
        }
        
        for i, test_case in enumerate(test_cases):
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{n_test_cases} test cases...")
            
            # Scenario A: Genre Only (Baseline)
            user_profile_a = self.engine.create_user_profile(
                favorite_movies=[],  # Empty - genre only
                favorite_genres=test_case['seed_genres'],
                favorite_actors=None,
                favorite_directors=None
            )
            
            recommendations_a = self.engine.get_recommendations(
                user_profile_a,
                favorite_movies=[],
                n=self.k,
                diversity_factor=0.3
            )
            
            if len(recommendations_a) > 0:
                scenarios['A']['successful'] += 1
                recommended_titles_a = recommendations_a['title'].tolist()
                
                scenarios['A']['precisions'].append(
                    self.calculate_precision_at_k(recommended_titles_a, test_case['relevant_movies'], k=self.k)
                )
                scenarios['A']['ndcgs'].append(
                    self.calculate_ndcg_at_k(recommended_titles_a, test_case['relevant_movies'], k=self.k)
                )
                scenarios['A']['diversities'].append(
                    self.calculate_intra_list_diversity(recommendations_a)
                )
            
            # Scenario B: Genre + Movie (Mandatory)
            user_profile_b = self.engine.create_user_profile(
                favorite_movies=test_case['seed_movies'],
                favorite_genres=test_case['seed_genres'],
                favorite_actors=None,
                favorite_directors=None
            )
            
            recommendations_b = self.engine.get_recommendations(
                user_profile_b,
                favorite_movies=test_case['seed_movies'],
                n=self.k,
                diversity_factor=0.3
            )
            
            if len(recommendations_b) > 0:
                scenarios['B']['successful'] += 1
                recommended_titles_b = recommendations_b['title'].tolist()
                
                scenarios['B']['precisions'].append(
                    self.calculate_precision_at_k(recommended_titles_b, test_case['relevant_movies'], k=self.k)
                )
                scenarios['B']['ndcgs'].append(
                    self.calculate_ndcg_at_k(recommended_titles_b, test_case['relevant_movies'], k=self.k)
                )
                scenarios['B']['diversities'].append(
                    self.calculate_intra_list_diversity(recommendations_b)
                )
            
            # Scenario C: Enhanced (Genre + Movie + Actor/Director)
            # Always include actors/directors from seed movie for this scenario
            seed_movie_data = test_case['seed_movie_data']
            enhanced_actors = test_case['user_actors'] if test_case['user_actors'] else (
                list(seed_movie_data['cast_list'][:2]) if len(seed_movie_data['cast_list']) > 0 else None
            )
            enhanced_directors = [test_case['user_director']] if test_case['user_director'] else (
                [seed_movie_data['director']] if seed_movie_data['director'] else None
            )
            
            user_profile_c = self.engine.create_user_profile(
                favorite_movies=test_case['seed_movies'],
                favorite_genres=test_case['seed_genres'],
                favorite_actors=enhanced_actors,
                favorite_directors=enhanced_directors
            )
            
            recommendations_c = self.engine.get_recommendations(
                user_profile_c,
                favorite_movies=test_case['seed_movies'],
                n=self.k,
                diversity_factor=0.3
            )
            
            if len(recommendations_c) > 0:
                scenarios['C']['successful'] += 1
                recommended_titles_c = recommendations_c['title'].tolist()
                
                scenarios['C']['precisions'].append(
                    self.calculate_precision_at_k(recommended_titles_c, test_case['relevant_movies'], k=self.k)
                )
                scenarios['C']['ndcgs'].append(
                    self.calculate_ndcg_at_k(recommended_titles_c, test_case['relevant_movies'], k=self.k)
                )
                scenarios['C']['diversities'].append(
                    self.calculate_intra_list_diversity(recommendations_c)
                )
        
        # Calculate overall metrics for each scenario
        results = {
            'total_cases': n_test_cases,
            'scenarios': {}
        }
        
        for scenario_key, scenario_data in scenarios.items():
            results['scenarios'][scenario_key] = {
                'name': scenario_data['name'],
                'precision_at_5': np.mean(scenario_data['precisions']) if scenario_data['precisions'] else 0,
                'ndcg_at_5': np.mean(scenario_data['ndcgs']) if scenario_data['ndcgs'] else 0,
                'diversity': np.mean(scenario_data['diversities']) if scenario_data['diversities'] else 0,
                'successful_cases': scenario_data['successful'],
                'success_rate': scenario_data['successful'] / n_test_cases
            }
        
        return results
    
    def print_evaluation_report(self, results):
        """Print a formatted evaluation report with scenario comparison"""
        print("\n" + "="*80)
        print("RECOMMENDATION SYSTEM EVALUATION REPORT")
        print("="*80)
        print(f"\nTest Configuration:")
        print(f"  Total Test Cases: {results['total_cases']}")
        print("\n" + "="*80)
        print("SCENARIO PERFORMANCE COMPARISON")
        print("="*80)
        
        # Header
        print(f"{'Scenario':<30} | {'Precision@5':>12} | {'NDCG@5':>10} | {'Diversity':>10}")
        print("-" * 80)
        
        # Scenario rows
        scenario_order = ['A', 'B', 'C']
        for scenario_key in scenario_order:
            scenario = results['scenarios'][scenario_key]
            scenario_label = f"{scenario_key}: {scenario['name']}"
            print(f"{scenario_label:<30} | {scenario['precision_at_5']:>12.4f} | {scenario['ndcg_at_5']:>10.4f} | {scenario['diversity']:>10.4f}")
        
        print("="*80)
        
        # Success rates
        print("\nSuccess Rates (Generated Recommendations):")
        for scenario_key in scenario_order:
            scenario = results['scenarios'][scenario_key]
            print(f"  {scenario_key}: {scenario['name']:<30} {scenario['successful_cases']:>3}/{results['total_cases']} ({scenario['success_rate']:>6.1%})")
        
        print("\n" + "="*80)
        print("Metric Interpretations:")
        print("  • Precision@5: Proportion of relevant movies in top-5 recommendations")
        print("  • NDCG@5: Ranking quality with position-based discounting")
        print("  • Diversity: How different recommended movies are from each other")
        print("="*80)
        
        # Performance assessment for each scenario
        print("\nPerformance Assessment:")
        print("-" * 80)
        
        for scenario_key in scenario_order:
            scenario = results['scenarios'][scenario_key]
            print(f"\nScenario {scenario_key}: {scenario['name']}")
            
            # Precision assessment
            if scenario['precision_at_5'] >= 0.4:
                print("  ✓ Precision@5: EXCELLENT (≥0.4)")
            elif scenario['precision_at_5'] >= 0.3:
                print("  ✓ Precision@5: GOOD (≥0.3)")
            elif scenario['precision_at_5'] >= 0.2:
                print("  ~ Precision@5: ACCEPTABLE (≥0.2)")
            else:
                print("  ✗ Precision@5: NEEDS IMPROVEMENT (<0.2)")
            
            # NDCG assessment
            if scenario['ndcg_at_5'] >= 0.5:
                print("  ✓ NDCG@5: EXCELLENT (≥0.5)")
            elif scenario['ndcg_at_5'] >= 0.4:
                print("  ✓ NDCG@5: GOOD (≥0.4)")
            elif scenario['ndcg_at_5'] >= 0.3:
                print("  ~ NDCG@5: ACCEPTABLE (≥0.3)")
            else:
                print("  ✗ NDCG@5: NEEDS IMPROVEMENT (<0.3)")
            
            # Diversity assessment
            if scenario['diversity'] >= 0.4:
                print("  ✓ Diversity: EXCELLENT (≥0.4)")
            elif scenario['diversity'] >= 0.3:
                print("  ✓ Diversity: GOOD (≥0.3)")
            elif scenario['diversity'] >= 0.2:
                print("  ~ Diversity: ACCEPTABLE (≥0.2)")
            else:
                print("  ✗ Diversity: NEEDS IMPROVEMENT (<0.2)")
        
        print("\n" + "="*80)
        
        # Comparative insights
        print("\nComparative Insights:")
        prec_a = results['scenarios']['A']['precision_at_5']
        prec_b = results['scenarios']['B']['precision_at_5']
        prec_c = results['scenarios']['C']['precision_at_5']
        
        if prec_b > prec_a:
            improvement = ((prec_b - prec_a) / prec_a * 100) if prec_a > 0 else 0
            print(f"  • Adding movie input (B vs A): +{improvement:.1f}% Precision improvement")
        
        if prec_c > prec_b:
            improvement = ((prec_c - prec_b) / prec_b * 100) if prec_b > 0 else 0
            print(f"  • Adding actor/director (C vs B): +{improvement:.1f}% Precision improvement")
        
        print("="*80 + "\n")


def main():
    """Run evaluation"""
    print("Loading data and model...")
    
    # Load processed data
    processor = DataProcessor()
    processor.load_processed_data()
    
    # Create recommendation engine
    engine = RecommendationEngine(processor.processed_df)
    
    # Create evaluator
    evaluator = RecommendationEvaluator(engine, processor.processed_df)
    
    # Run evaluation
    results = evaluator.evaluate_system(n_test_cases=50)
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    # Save results to CSV with scenario breakdown
    results_list = []
    for scenario_key in ['A', 'B', 'C']:
        scenario = results['scenarios'][scenario_key]
        results_list.append({
            'scenario': scenario_key,
            'scenario_name': scenario['name'],
            'precision_at_5': scenario['precision_at_5'],
            'ndcg_at_5': scenario['ndcg_at_5'],
            'diversity': scenario['diversity'],
            'successful_cases': scenario['successful_cases'],
            'total_cases': results['total_cases'],
            'success_rate': scenario['success_rate']
        })
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nResults saved to evaluation_results.csv")


if __name__ == "__main__":
    main()
