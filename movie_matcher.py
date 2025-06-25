if len(st.session_state.selected_movies) == 10:
        # Create user picks dictionary
        user_picks = {movie: rank+1 for rank, movie in enumerate(st.session_state.selected_movies)}
        
        # Validate that all movies are in the dataset
        invalid_movies = [movie for movie in user_picks if movie not in unique_movies]
        if invalid_movies:
            st.error(f"The following movies are not in the dataset: {', '.join(invalid_movies)}")
            st.info("Please remove these movies and select from the dropdown list.")
        else:
            # Find similar voters
            similar_voters = find_similar_voters(user_picks, voter_movie_matrix, top_n=20)
            
            st.markdown("### Most Similar Reviewers:")
            
            has_matches = False
            for i, (voter, similarity, common_movies) in enumerate(similar_voters):
                if similarity > 0:  # Only show voters with some similarity
                    has_matches = True
                    with st.expander(f"{i+1}. **{voter}** - Similarity: {similarity:.2%}"):
                        st.write(f"**Movies in common:** {len(common_movies)}")
                        
                        # Show which movies matched and their rankings
                        if common_movies:
                            match_df = []
                            for movie in common_movies:
                                try:
                                    user_rank = user_picks[movie]
                                    voter_rank = int(voter_movie_matrix.loc[voter, movie])
                                    match_df.append({
                                        'Movie': movie,
                                        'Your Rank': user_rank,
                                        'Their Rank': voter_rank,
                                        'Difference': abs(user_rank - voter_rank)
                                    })
                                except:
                                    continue
                            
                            if match_df:
                                match_df = pd.DataFrame(match_df)
                                match_df = match_df.sort_values('Your Rank')
                                st.dataframe(match_df, hide_index=True)
                            
                            # Show voter's other highly ranked movies not in user's list
                            try:
                                voter_all_picks = voter_movie_matrix.loc[voter]
                                voter_all_picks = voter_all_picks[voter_all_picks > 0].sort_values()
                                
                                other_movies = [m for m in voter_all_picks.index if m not in user_picks]
                                if other_movies:
                                    st.markdown("**Their other top movies you might like:**")
                                    for movie in other_movies[:5]:  # Show top 5
                                        rank = int(voter_all_picks[movie])
                                        st.write(f"#{rank}: {movie}")
                            except:
                                pass
            
            if not has_matches:
                st.warning("No reviewers found with matching movies. This might happen if your selected movies are not in the NYT dataset.")import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

st.set_page_config(page_title="Find Your Movie Match", layout="wide")

@st.cache_data
def load_data():
    """Load and preprocess the movie votes data."""
    df = pd.read_csv('Find your match_ NYT_2025_100Films_Votes - Votes.csv')
    
    # Get list of unique movies for dropdown
    unique_movies = sorted(df['Movie'].unique())
    
    # Create a pivot table: voters as rows, movies as columns, picks as values
    voter_movie_matrix = df.pivot_table(
        index='Voter', 
        columns='Movie', 
        values='Pick',
        fill_value=0  # 0 means movie wasn't ranked
    )
    
    return df, unique_movies, voter_movie_matrix

def calculate_similarity(user_picks: Dict[str, int], voter_picks: pd.Series) -> float:
    """
    Calculate similarity between user's picks and a voter's picks.
    Uses a weighted similarity score that considers both overlap and ranking differences.
    """
    # Find common movies
    common_movies = []
    for movie, user_rank in user_picks.items():
        try:
            if movie in voter_picks.index and voter_picks[movie] > 0:
                common_movies.append(movie)
        except:
            # Skip if movie causes any issues
            continue
    
    if not common_movies:
        return 0.0
    
    # Calculate similarity based on ranking differences
    total_score = 0
    max_possible_score = 0
    
    for movie in common_movies:
        try:
            user_rank = user_picks[movie]
            voter_rank = voter_picks[movie]
            
            # Inverse weight: higher ranked movies (lower numbers) get more weight
            weight = 1 / user_rank + 1 / voter_rank
            
            # Penalty for ranking difference (0 if same rank, increases with difference)
            rank_diff = abs(user_rank - voter_rank)
            score = weight * (1 / (1 + rank_diff))
            
            total_score += score
            max_possible_score += weight
        except:
            # Skip this movie if there's any calculation error
            continue
    
    # Normalize by number of common movies and maximum possible score
    if max_possible_score > 0:
        similarity = (total_score / max_possible_score) * (len(common_movies) / 10)
    else:
        similarity = 0
    
    return similarity

def find_similar_voters(user_picks: Dict[str, int], voter_movie_matrix: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, float, List[str]]]:
    """Find the most similar voters based on movie preferences."""
    similarities = []
    
    for voter in voter_movie_matrix.index:
        voter_picks = voter_movie_matrix.loc[voter]
        similarity = calculate_similarity(user_picks, voter_picks)
        
        # Get common movies between user and voter
        common_movies = []
        for movie, user_rank in user_picks.items():
            try:
                if movie in voter_picks.index and voter_picks[movie] > 0:
                    common_movies.append(movie)
            except:
                # Skip if movie causes any issues
                continue
        
        similarities.append((voter, similarity, common_movies))
    
    # Sort by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]

# Main app
st.title("🎬 Find Your Movie Match")
st.markdown("Select and rank 10 movies to find NYT reviewers with similar taste!")

# Load data
try:
    df, unique_movies, voter_movie_matrix = load_data()
except FileNotFoundError:
    st.error("Please ensure 'Find your match_ NYT_2025_100Films_Votes  Votes.csv' is in the same directory as this app.")
    st.stop()

# Initialize session state for selected movies
if 'selected_movies' not in st.session_state:
    st.session_state.selected_movies = []

# Create two columns for the interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Select Your Top 10 Movies")
    
    # Movie selection
    available_movies = [m for m in unique_movies if m not in st.session_state.selected_movies]
    
    if len(st.session_state.selected_movies) < 10:
        selected_movie = st.selectbox(
            f"Select movie #{len(st.session_state.selected_movies) + 1}:",
            [""] + available_movies,
            key=f"movie_select_{len(st.session_state.selected_movies)}"
        )
        
        if selected_movie and selected_movie != "":
            if st.button("Add Movie", key="add_movie"):
                st.session_state.selected_movies.append(selected_movie)
                st.rerun()
    
    # Display selected movies with option to remove
    if st.session_state.selected_movies:
        st.markdown("### Your Rankings:")
        for i, movie in enumerate(st.session_state.selected_movies):
            col_rank, col_movie, col_remove = st.columns([1, 8, 2])
            with col_rank:
                st.write(f"**#{i+1}**")
            with col_movie:
                st.write(movie)
            with col_remove:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.selected_movies.pop(i)
                    st.rerun()
    
    # Clear all button
    if st.session_state.selected_movies:
        if st.button("Clear All Selections"):
            st.session_state.selected_movies = []
            st.rerun()

with col2:
    st.subheader("Your Movie Matches")
    
    if len(st.session_state.selected_movies) == 10:
        # Create user picks dictionary
        user_picks = {movie: rank+1 for rank, movie in enumerate(st.session_state.selected_movies)}
        
        # Find similar voters
        similar_voters = find_similar_voters(user_picks, voter_movie_matrix, top_n=20)
        
        st.markdown("### Most Similar Reviewers:")
        
        for i, (voter, similarity, common_movies) in enumerate(similar_voters):
            if similarity > 0:  # Only show voters with some similarity
                with st.expander(f"{i+1}. **{voter}** - Similarity: {similarity:.2%}"):
                    st.write(f"**Movies in common:** {len(common_movies)}")
                    
                    # Show which movies matched and their rankings
                    if common_movies:
                        match_df = []
                        for movie in common_movies:
                            user_rank = user_picks[movie]
                            voter_rank = int(voter_movie_matrix.loc[voter, movie])
                            match_df.append({
                                'Movie': movie,
                                'Your Rank': user_rank,
                                'Their Rank': voter_rank,
                                'Difference': abs(user_rank - voter_rank)
                            })
                        
                        match_df = pd.DataFrame(match_df)
                        match_df = match_df.sort_values('Your Rank')
                        st.dataframe(match_df, hide_index=True)
                        
                        # Show voter's other highly ranked movies not in user's list
                        voter_all_picks = voter_movie_matrix.loc[voter]
                        voter_all_picks = voter_all_picks[voter_all_picks > 0].sort_values()
                        
                        other_movies = [m for m in voter_all_picks.index if m not in user_picks]
                        if other_movies:
                            st.markdown("**Their other top movies you might like:**")
                            for movie in other_movies[:5]:  # Show top 5
                                rank = int(voter_all_picks[movie])
                                st.write(f"#{rank}: {movie}")
    
    else:
        remaining = 10 - len(st.session_state.selected_movies)
        st.info(f"Please select {remaining} more movie{'s' if remaining > 1 else ''} to find your matches!")
        
        # Show some stats about the dataset
        st.markdown("### About the Dataset")
        st.write(f"- **Total Reviewers:** {len(voter_movie_matrix)}")
        st.write(f"- **Total Movies:** {len(unique_movies)}")
        st.write(f"- **Most voters ranked:** 10 movies")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How it works:
1. Select movies from the dropdown in order of preference (1st = favorite)
2. Add exactly 10 movies to your list
3. The app will find NYT reviewers with similar taste based on:
   - Which movies you both ranked
   - How similarly you ranked them
   - The total overlap in your selections
""")