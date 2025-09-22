import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

class BetterMoodFashionEngine:
    def __init__(self, csv_path):
        """Initialize the mood fashion engine with better matching"""
        st.write("🔄 Loading fashion data...")
        
        # Load data
        self.df = pd.read_csv(csv_path)
        st.write(f"✅ Loaded {len(self.df)} fashion items")
        
        # Create comprehensive mood mappings
        self.mood_mappings = {
            'happy': ['happy', 'joyful', 'cheerful', 'bright', 'vibrant', 'sunny', 'festive', 'celebratory'],
            'sad': ['sad', 'melancholy', 'somber', 'dark', 'muted', 'subdued', 'calm', 'peaceful'],
            'excited': ['excited', 'energetic', 'dynamic', 'bold', 'striking', 'dramatic', 'vibrant'],
            'calm': ['calm', 'serene', 'peaceful', 'soft', 'gentle', 'minimal', 'clean', 'simple'],
            'romantic': ['romantic', 'elegant', 'feminine', 'delicate', 'soft', 'flowing', 'graceful'],
            'confident': ['confident', 'powerful', 'strong', 'bold', 'sharp', 'structured', 'professional'],
            'casual': ['casual', 'relaxed', 'comfortable', 'easy', 'laid-back', 'informal', 'everyday'],
            'party': ['party', 'festive', 'glamorous', 'sparkly', 'bold', 'eye-catching', 'fun'],
            'cozy': ['cozy', 'warm', 'comfortable', 'soft', 'snug', 'comfortable', 'relaxed'],
            'elegant': ['elegant', 'sophisticated', 'refined', 'classy', 'formal', 'polished', 'graceful'],
            'trendy': ['trendy', 'fashionable', 'stylish', 'modern', 'contemporary', 'hip', 'cool'],
            'vintage': ['vintage', 'retro', 'classic', 'timeless', 'nostalgic', 'old-fashioned'],
            'sporty': ['sporty', 'athletic', 'active', 'dynamic', 'energetic', 'performance'],
            'bohemian': ['bohemian', 'boho', 'artistic', 'free-spirited', 'eclectic', 'creative'],
            'minimalist': ['minimalist', 'minimal', 'clean', 'simple', 'understated', 'basic']
        }
        
        # Create reverse mappings (mood -> keywords)
        self.reverse_mappings = {}
        for mood, keywords in self.mood_mappings.items():
            for keyword in keywords:
                if keyword not in self.reverse_mappings:
                    self.reverse_mappings[keyword] = []
                self.reverse_mappings[keyword].append(mood)
        
        # Initialize TF-IDF vectorizer
        st.write("🔄 Setting up mood matching...")
        self.setup_mood_matching()
        
        st.write("✅ Ready to find your perfect mood-to-fashion matches!")
    
    def setup_mood_matching(self):
        """Setup the mood matching system"""
        # Create mood tags for each item
        self.df['mood_tags'] = self.df['mood'].fillna('').apply(self.expand_mood_tags)
        
        # Combine all text features for better matching
        self.df['combined_text'] = (
            self.df['productDisplayName'].fillna('') + ' ' +
            self.df['articleType'].fillna('') + ' ' +
            self.df['baseColour'].fillna('') + ' ' +
            self.df['subCategory'].fillna('') + ' ' +
            self.df['mood_tags'].fillna('')
        )
        
        # Initialize TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit TF-IDF on all text
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_text'])
        
        # Create mood vectors
        self.mood_vectors = {}
        for mood, keywords in self.mood_mappings.items():
            mood_text = ' '.join(keywords)
            mood_vector = self.tfidf.transform([mood_text])
            self.mood_vectors[mood] = mood_vector
    
    def expand_mood_tags(self, mood_text):
        """Expand mood text with related keywords"""
        if pd.isna(mood_text) or mood_text == '':
            return ''
        
        mood_text = str(mood_text).lower()
        expanded_tags = [mood_text]
        
        # Add related keywords
        for keyword, moods in self.reverse_mappings.items():
            if keyword in mood_text:
                expanded_tags.extend(moods)
        
        return ' '.join(set(expanded_tags))
    
    def find_mood_matches(self, user_mood, top_k=20):
        """Find fashion items that match the user's mood"""
        user_mood = user_mood.lower().strip()
        
        # Find best matching mood category
        best_mood = None
        best_score = 0
        
        for mood, keywords in self.mood_mappings.items():
            score = 0
            for keyword in keywords:
                if keyword in user_mood:
                    score += 1
            if score > best_score:
                best_score = score
                best_mood = mood
        
        # If no direct match, try fuzzy matching
        if best_mood is None:
            for mood, keywords in self.mood_mappings.items():
                for keyword in keywords:
                    if keyword in user_mood or user_mood in keyword:
                        best_mood = mood
                        break
                if best_mood:
                    break
        
        # Get matches using multiple methods
        matches = []
        
        # Method 1: Direct mood matching
        if best_mood:
            mood_matches = self.df[
                self.df['mood_tags'].str.contains(best_mood, case=False, na=False)
            ].index.tolist()
            matches.extend(mood_matches)
        
        # Method 2: TF-IDF similarity
        user_vector = self.tfidf.transform([user_mood])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-50:][::-1]  # Top 50
        matches.extend(top_indices)
        
        # Method 3: Text matching in product names
        text_matches = self.df[
            self.df['combined_text'].str.contains(user_mood, case=False, na=False)
        ].index.tolist()
        matches.extend(text_matches)
        
        # Method 4: Color and category matching
        color_matches = self.df[
            self.df['baseColour'].str.contains(user_mood, case=False, na=False)
        ].index.tolist()
        matches.extend(color_matches)
        
        # Remove duplicates and get results
        unique_matches = list(dict.fromkeys(matches))[:top_k]
        
        # Ensure indices are valid
        valid_indices = [idx for idx in unique_matches if idx < len(self.df)]
        if valid_indices:
            return self.df.iloc[valid_indices]
        else:
            return self.df.head(top_k)
    
    def get_category_analysis(self, results_df):
        """Analyze category distribution in recommendations"""
        category_counts = results_df['subCategory'].value_counts().head(10)
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Category Distribution in Recommendations"
        )
        return fig
    
    def get_color_analysis(self, results_df):
        """Analyze color distribution in recommendations"""
        color_counts = results_df['baseColour'].value_counts().head(10)
        
        fig = px.bar(
            x=color_counts.index,
            y=color_counts.values,
            title="Top Colors in Recommendations",
            labels={'x': 'Color', 'y': 'Count'}
        )
        fig.update_layout(xaxis_tickangle=45)
        return fig

def main():
    st.set_page_config(
        page_title="Mood-to-Fashion Engine",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Mood-to-Fashion Discovery Engine")
    st.markdown("**Find fashion items that match your mood perfectly!**")
    
    # Load the engine
    @st.cache_data
    def load_engine():
        return BetterMoodFashionEngine('dataset_with_moods.csv')
    
    engine = load_engine()
    
    # Sidebar for mood input
    st.sidebar.header("🎯 Your Mood")
    
    # Mood input options
    mood_input = st.sidebar.text_input(
        "How are you feeling?",
        placeholder="e.g., happy, cozy, elegant, party mood..."
    )
    
    # Quick mood buttons
    st.sidebar.markdown("**Quick Moods:**")
    quick_moods = ['happy', 'cozy', 'elegant', 'party', 'casual', 'romantic', 'confident', 'trendy']
    
    cols = st.sidebar.columns(2)
    for i, mood in enumerate(quick_moods):
        with cols[i % 2]:
            if st.button(mood.title(), key=f"mood_{mood}"):
                mood_input = mood
    
    # Number of results
    num_results = st.sidebar.slider("Number of items to show", 5, 50, 20)
    
    # Find matches
    if mood_input:
        st.header(f"🎯 Fashion for '{mood_input}' mood")
        
        with st.spinner("Finding your perfect matches..."):
            results = engine.find_mood_matches(mood_input, top_k=num_results)
        
        if len(results) > 0:
            st.success(f"Found {len(results)} items matching your mood!")
            
            # Display results in a grid
            cols_per_row = 3
            for i in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        item = results.iloc[i + j]
                        
                        with col:
                            st.markdown(f"### {item['productDisplayName']}")
                            
                            # Display image if link exists
                            if pd.notna(item['link']) and item['link'] != '':
                                try:
                                    st.image(item['link'], width=200, caption=item['productDisplayName'])
                                except:
                                    st.write("🖼️ Image not available")
                            
                            # Product details
                            st.write(f"**Category:** {item['subCategory']}")
                            st.write(f"**Color:** {item['baseColour']}")
                            st.write(f"**Type:** {item['articleType']}")
                            st.write(f"**Mood:** {item['mood']}")
                            
                            # Link to product
                            if pd.notna(item['link']) and item['link'] != '':
                                st.markdown(f"[🛍️ View Product]({item['link']})")
                            
                            st.markdown("---")
            
            # Analysis section
            st.header("📊 Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                category_fig = engine.get_category_analysis(results)
                st.plotly_chart(category_fig, use_container_width=True)
            
            with col2:
                color_fig = engine.get_color_analysis(results)
                st.plotly_chart(color_fig, use_container_width=True)
            
            # Show raw data
            if st.checkbox("Show raw data"):
                st.dataframe(results[['productDisplayName', 'subCategory', 'baseColour', 'articleType', 'mood', 'link']])
        
        else:
            st.warning("No items found for this mood. Try a different mood!")
    
    else:
        st.info("👆 Enter your mood in the sidebar to get started!")
        
        # Show sample moods
        st.header("💡 Sample Moods to Try:")
        sample_moods = [
            "happy and bright", "cozy and warm", "elegant and sophisticated",
            "party and festive", "casual and relaxed", "romantic and soft",
            "confident and bold", "trendy and modern", "vintage and classic",
            "sporty and active", "bohemian and artistic", "minimalist and clean"
        ]
        
        for mood in sample_moods:
            if st.button(f"🎭 {mood}", key=f"sample_{mood}"):
                st.session_state.mood_input = mood
                st.rerun()

if __name__ == "__main__":
    main()
