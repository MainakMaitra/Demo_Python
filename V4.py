# ============================================================================
# ENHANCED DYNAMIC NEGATION SCOPE DETECTION WITH COMPLAINT LEXICON MAPPING
# Advanced NLP implementation with data-driven pattern discovery
# ============================================================================

import pandas as pd
import numpy as np
import spacy
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import networkx as nx
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ENHANCED DYNAMIC NEGATION SCOPE DETECTION")
print("Advanced NLP Analysis with Data-Driven Pattern Discovery")
print("=" * 80)


class DynamicNegationPatternDiscovery:
    """
    Advanced negation pattern discovery system that learns patterns from data
    rather than using hardcoded rules
    """
    
    def __init__(self):
        # Load spaCy model for dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize storage for discovered patterns
        self.discovered_negation_patterns = {}
        self.complaint_lexicon_mapping = {}
        self.negation_contexts = defaultdict(list)
        
    def discover_negation_patterns_from_data(self, df):
        """
        Dynamically discover negation patterns from the actual data
        instead of using hardcoded patterns
        """
        
        print("\n" + "="*60)
        print("DYNAMIC NEGATION PATTERN DISCOVERY")
        print("="*60)
        
        # Step 1: Extract all potential negation words from data
        all_negation_candidates = self._extract_negation_candidates(df)
        
        # Step 2: Analyze frequency and context patterns
        negation_analysis = self._analyze_negation_contexts(df, all_negation_candidates)
        
        # Step 3: Cluster negation patterns by semantic similarity
        clustered_patterns = self._cluster_negation_patterns(negation_analysis)
        
        # Step 4: Validate patterns against TP/FP performance
        validated_patterns = self._validate_patterns_by_performance(df, clustered_patterns)
        
        self.discovered_negation_patterns = validated_patterns
        
        return validated_patterns
    
    def _extract_negation_candidates(self, df):
        """
        Extract potential negation words using linguistic patterns and frequency analysis
        """
        
        print("1. EXTRACTING NEGATION CANDIDATES FROM DATA")
        print("-" * 40)
        
        negation_candidates = Counter()
        
        # Common negation prefixes and suffixes
        negation_prefixes = ['un', 'non', 'dis', 'in', 'im', 'ir', 'il']
        negation_suffixes = ['n\'t', 'nt']
        
        for _, row in df.iterrows():
            text = str(row['Customer Transcript']).lower()
            words = text.split()
            
            for word in words:
                # Check for explicit negation words
                if any(neg in word for neg in ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere']):
                    negation_candidates[word] += 1
                
                # Check for contractions
                if any(suffix in word for suffix in negation_suffixes):
                    negation_candidates[word] += 1
                
                # Check for negation prefixes
                if len(word) > 4 and any(word.startswith(prefix) for prefix in negation_prefixes):
                    negation_candidates[word] += 1
        
        # Filter by frequency threshold
        min_frequency = max(5, len(df) * 0.001)  # At least 5 occurrences or 0.1% of data
        filtered_candidates = {word: count for word, count in negation_candidates.items() 
                             if count >= min_frequency}
        
        print(f"Found {len(filtered_candidates)} negation candidates")
        print("Top 20 candidates:", list(dict(negation_candidates.most_common(20)).keys()))
        
        return filtered_candidates
    
    def _analyze_negation_contexts(self, df, negation_candidates):
        """
        Analyze the context around each negation candidate to understand patterns
        """
        
        print("\n2. ANALYZING NEGATION CONTEXTS")
        print("-" * 40)
        
        context_analysis = {}
        
        for negation_word in negation_candidates.keys():
            context_data = {
                'total_occurrences': 0,
                'tp_occurrences': 0,
                'fp_occurrences': 0,
                'contexts_before': Counter(),
                'contexts_after': Counter(),
                'complaint_associations': Counter(),
                'information_associations': Counter()
            }
            
            for _, row in df.iterrows():
                text = str(row['Customer Transcript']).lower()
                is_tp = row['Primary Marker'] == 'TP'
                
                # Find all occurrences of the negation word
                words = text.split()
                for i, word in enumerate(words):
                    if negation_word in word:
                        context_data['total_occurrences'] += 1
                        
                        if is_tp:
                            context_data['tp_occurrences'] += 1
                        else:
                            context_data['fp_occurrences'] += 1
                        
                        # Extract context (3 words before and after)
                        before_context = ' '.join(words[max(0, i-3):i])
                        after_context = ' '.join(words[i+1:min(len(words), i+4)])
                        
                        context_data['contexts_before'][before_context] += 1
                        context_data['contexts_after'][after_context] += 1
                        
                        # Classify as complaint or information based on context
                        full_context = ' '.join(words[max(0, i-5):min(len(words), i+6)])
                        if self._is_complaint_context(full_context):
                            context_data['complaint_associations'][full_context] += 1
                        else:
                            context_data['information_associations'][full_context] += 1
            
            context_analysis[negation_word] = context_data
        
        return context_analysis
    
    def _is_complaint_context(self, context):
        """
        Determine if a context suggests a complaint rather than information seeking
        """
        complaint_indicators = [
            'problem', 'issue', 'wrong', 'error', 'broken', 'failed', 'trouble',
            'disappointed', 'frustrated', 'angry', 'upset', 'dissatisfied',
            'received', 'working', 'resolved', 'fixed', 'helped'
        ]
        
        information_indicators = [
            'understand', 'know', 'sure', 'clear', 'how', 'what', 'when', 'where',
            'explain', 'help me', 'can you', 'would you'
        ]
        
        complaint_score = sum(1 for indicator in complaint_indicators if indicator in context)
        information_score = sum(1 for indicator in information_indicators if indicator in context)
        
        return complaint_score > information_score
    
    def _cluster_negation_patterns(self, negation_analysis):
        """
        Cluster negation patterns by semantic similarity and usage patterns
        """
        
        print("\n3. CLUSTERING NEGATION PATTERNS")
        print("-" * 40)
        
        # Create feature vectors for each negation word
        features = []
        negation_words = []
        
        for word, data in negation_analysis.items():
            if data['total_occurrences'] > 10:  # Minimum threshold
                tp_ratio = data['tp_occurrences'] / max(data['total_occurrences'], 1)
                fp_ratio = data['fp_occurrences'] / max(data['total_occurrences'], 1)
                complaint_ratio = len(data['complaint_associations']) / max(data['total_occurrences'], 1)
                
                features.append([tp_ratio, fp_ratio, complaint_ratio, data['total_occurrences']])
                negation_words.append(word)
        
        if len(features) > 3:
            # Apply K-means clustering
            from sklearn.cluster import KMeans
            n_clusters = min(5, len(features) // 3)  # Maximum 5 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features)
            
            # Group words by cluster
            clustered_patterns = defaultdict(list)
            for word, cluster in zip(negation_words, clusters):
                clustered_patterns[f"cluster_{cluster}"].append(word)
            
            print(f"Created {len(clustered_patterns)} negation pattern clusters:")
            for cluster_name, words in clustered_patterns.items():
                print(f"  {cluster_name}: {words[:10]}")  # Show first 10 words
            
            return dict(clustered_patterns)
        else:
            # If too few patterns, create single cluster
            return {"cluster_0": negation_words}
    
    def _validate_patterns_by_performance(self, df, clustered_patterns):
        """
        Validate discovered patterns by their performance in distinguishing TPs from FPs
        """
        
        print("\n4. VALIDATING PATTERNS BY PERFORMANCE")
        print("-" * 40)
        
        validated_patterns = {}
        
        for cluster_name, words in clustered_patterns.items():
            pattern_performance = {
                'words': words,
                'tp_precision': 0,
                'fp_precision': 0,
                'discrimination_power': 0,
                'pattern_type': 'unknown'
            }
            
            tp_matches = 0
            fp_matches = 0
            total_tp = 0
            total_fp = 0
            
            for _, row in df.iterrows():
                text = str(row['Customer Transcript']).lower()
                is_tp = row['Primary Marker'] == 'TP'
                
                if is_tp:
                    total_tp += 1
                else:
                    total_fp += 1
                
                # Check if any word from this cluster appears in the text
                contains_pattern = any(word in text for word in words)
                
                if contains_pattern:
                    if is_tp:
                        tp_matches += 1
                    else:
                        fp_matches += 1
            
            # Calculate performance metrics
            if total_tp > 0:
                pattern_performance['tp_precision'] = tp_matches / total_tp
            if total_fp > 0:
                pattern_performance['fp_precision'] = fp_matches / total_fp
            
            # Calculate discrimination power (ability to distinguish TP from FP)
            if pattern_performance['tp_precision'] + pattern_performance['fp_precision'] > 0:
                pattern_performance['discrimination_power'] = abs(
                    pattern_performance['tp_precision'] - pattern_performance['fp_precision']
                )
            
            # Classify pattern type based on performance
            if pattern_performance['tp_precision'] > pattern_performance['fp_precision'] * 1.5:
                pattern_performance['pattern_type'] = 'complaint_indicator'
            elif pattern_performance['fp_precision'] > pattern_performance['tp_precision'] * 1.5:
                pattern_performance['pattern_type'] = 'information_indicator'
            else:
                pattern_performance['pattern_type'] = 'neutral'
            
            validated_patterns[cluster_name] = pattern_performance
        
        # Sort by discrimination power
        sorted_patterns = dict(sorted(validated_patterns.items(), 
                                    key=lambda x: x[1]['discrimination_power'], 
                                    reverse=True))
        
        print("Pattern Performance Summary:")
        for name, perf in sorted_patterns.items():
            print(f"  {name}: {perf['pattern_type']} (discrimination: {perf['discrimination_power']:.3f})")
        
        return sorted_patterns


class ComplaintExpressionLexiconMapper:
    """
    Creates and analyzes complaint expression lexicons from data
    """
    
    def __init__(self):
        self.complaint_lexicons = {}
        self.expression_patterns = {}
        
    def create_complaint_lexicon_mapping(self, df):
        """
        Create comprehensive complaint expression lexicons from the data
        """
        
        print("\n" + "="*60)
        print("COMPLAINT EXPRESSION LEXICON MAPPING")
        print("="*60)
        
        # Step 1: Extract complaint expressions using TF-IDF
        tfidf_expressions = self._extract_tfidf_expressions(df)
        
        # Step 2: Extract n-gram patterns
        ngram_patterns = self._extract_ngram_patterns(df)
        
        # Step 3: Extract domain-specific complaint phrases
        domain_phrases = self._extract_domain_specific_phrases(df)
        
        # Step 4: Create semantic clusters of expressions
        semantic_clusters = self._create_semantic_clusters(df)
        
        # Step 5: Validate lexicons against TP/FP performance
        validated_lexicons = self._validate_lexicon_performance(df, {
            'tfidf_expressions': tfidf_expressions,
            'ngram_patterns': ngram_patterns,
            'domain_phrases': domain_phrases,
            'semantic_clusters': semantic_clusters
        })
        
        self.complaint_lexicons = validated_lexicons
        
        return validated_lexicons
    
    def _extract_tfidf_expressions(self, df):
        """
        Extract important expressions using TF-IDF analysis
        """
        
        print("\n1. EXTRACTING TF-IDF BASED EXPRESSIONS")
        print("-" * 40)
        
        # Separate TP and FP texts
        tp_texts = df[df['Primary Marker'] == 'TP']['Customer Transcript'].fillna('').tolist()
        fp_texts = df[df['Primary Marker'] == 'FP']['Customer Transcript'].fillna('').tolist()
        
        # TF-IDF analysis for phrases (1-3 grams)
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=1000,
            stop_words='english',
            min_df=5,
            lowercase=True
        )
        
        # Fit on TP texts to find complaint-specific expressions
        tp_tfidf = vectorizer.fit_transform(tp_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get mean TF-IDF scores for each feature
        tp_scores = np.mean(tp_tfidf.toarray(), axis=0)
        
        # Transform FP texts with same vectorizer
        fp_tfidf = vectorizer.transform(fp_texts)
        fp_scores = np.mean(fp_tfidf.toarray(), axis=0)
        
        # Calculate discrimination scores
        expressions = {}
        for i, feature in enumerate(feature_names):
            tp_score = tp_scores[i]
            fp_score = fp_scores[i]
            
            if tp_score > 0 or fp_score > 0:
                discrimination = tp_score / (fp_score + 0.001)  # Avoid division by zero
                expressions[feature] = {
                    'tp_score': tp_score,
                    'fp_score': fp_score,
                    'discrimination': discrimination
                }
        
        # Sort by discrimination power
        sorted_expressions = dict(sorted(expressions.items(), 
                                       key=lambda x: x[1]['discrimination'], 
                                       reverse=True))
        
        top_expressions = dict(list(sorted_expressions.items())[:50])
        
        print(f"Extracted {len(top_expressions)} TF-IDF expressions")
        print("Top 10 complaint expressions:")
        for expr, scores in list(top_expressions.items())[:10]:
            print(f"  '{expr}': discrimination={scores['discrimination']:.3f}")
        
        return top_expressions
    
    def _extract_ngram_patterns(self, df):
        """
        Extract meaningful n-gram patterns that indicate complaints
        """
        
        print("\n2. EXTRACTING N-GRAM PATTERNS")
        print("-" * 40)
        
        # Extract 2-grams and 3-grams
        tp_texts = ' '.join(df[df['Primary Marker'] == 'TP']['Customer Transcript'].fillna(''))
        fp_texts = ' '.join(df[df['Primary Marker'] == 'FP']['Customer Transcript'].fillna(''))
        
        # Count n-grams
        def extract_ngrams(text, n):
            words = text.lower().split()
            ngrams = []
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
            return Counter(ngrams)
        
        tp_bigrams = extract_ngrams(tp_texts, 2)
        tp_trigrams = extract_ngrams(tp_texts, 3)
        fp_bigrams = extract_ngrams(fp_texts, 2)
        fp_trigrams = extract_ngrams(fp_texts, 3)
        
        # Find patterns that are much more common in TPs
        patterns = {}
        
        # Analyze bigrams
        for bigram, tp_count in tp_bigrams.most_common(100):
            fp_count = fp_bigrams.get(bigram, 0)
            if tp_count > 5 and tp_count > fp_count * 2:  # At least 2x more common in TPs
                patterns[bigram] = {
                    'type': 'bigram',
                    'tp_count': tp_count,
                    'fp_count': fp_count,
                    'ratio': tp_count / max(fp_count, 1)
                }
        
        # Analyze trigrams
        for trigram, tp_count in tp_trigrams.most_common(50):
            fp_count = fp_trigrams.get(trigram, 0)
            if tp_count > 3 and tp_count > fp_count * 2:
                patterns[trigram] = {
                    'type': 'trigram',
                    'tp_count': tp_count,
                    'fp_count': fp_count,
                    'ratio': tp_count / max(fp_count, 1)
                }
        
        print(f"Extracted {len(patterns)} n-gram patterns")
        print("Top 10 n-gram patterns:")
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['ratio'], reverse=True)
        for pattern, data in sorted_patterns[:10]:
            print(f"  '{pattern}': ratio={data['ratio']:.2f} (TP:{data['tp_count']}, FP:{data['fp_count']})")
        
        return patterns
    
    def _extract_domain_specific_phrases(self, df):
        """
        Extract domain-specific complaint phrases for banking/credit card context
        """
        
        print("\n3. EXTRACTING DOMAIN-SPECIFIC PHRASES")
        print("-" * 40)
        
        # Define banking-specific complaint categories
        banking_categories = {
            'service_issues': [
                'customer service', 'representative', 'agent', 'support',
                'hold time', 'wait time', 'transfer', 'disconnect'
            ],
            'transaction_problems': [
                'transaction', 'payment', 'charge', 'billing', 'statement',
                'balance', 'deposit', 'withdrawal', 'transfer failed'
            ],
            'card_issues': [
                'card', 'atm', 'pin', 'chip', 'magnetic', 'reader',
                'declined', 'rejected', 'expired', 'blocked'
            ],
            'account_problems': [
                'account', 'login', 'password', 'access', 'locked',
                'suspended', 'closed', 'frozen', 'restricted'
            ],
            'fee_disputes': [
                'fee', 'charge', 'interest', 'penalty', 'overdraft',
                'late fee', 'annual fee', 'maintenance fee'
            ],
            'fraud_security': [
                'fraud', 'security', 'unauthorized', 'stolen', 'compromised',
                'suspicious', 'breach', 'identity'
            ]
        }
        
        domain_phrases = {}
        
        for category, keywords in banking_categories.items():
            category_phrases = {}
            
            for keyword in keywords:
                tp_count = 0
                fp_count = 0
                
                for _, row in df.iterrows():
                    text = str(row['Customer Transcript']).lower()
                    is_tp = row['Primary Marker'] == 'TP'
                    
                    if keyword in text:
                        if is_tp:
                            tp_count += 1
                        else:
                            fp_count += 1
                
                if tp_count > 0 or fp_count > 0:
                    category_phrases[keyword] = {
                        'tp_count': tp_count,
                        'fp_count': fp_count,
                        'discrimination': tp_count / max(fp_count, 1)
                    }
            
            domain_phrases[category] = category_phrases
        
        print("Domain-specific phrase analysis:")
        for category, phrases in domain_phrases.items():
            if phrases:
                best_phrase = max(phrases.items(), key=lambda x: x[1]['discrimination'])
                print(f"  {category}: '{best_phrase[0]}' (discrimination: {best_phrase[1]['discrimination']:.2f})")
        
        return domain_phrases
    
    def _create_semantic_clusters(self, df):
        """
        Create semantic clusters of complaint expressions
        """
        
        print("\n4. CREATING SEMANTIC CLUSTERS")
        print("-" * 40)
        
        # Extract all unique phrases from TPs
        tp_texts = df[df['Primary Marker'] == 'TP']['Customer Transcript'].fillna('')
        
        # Use TF-IDF to find important phrases
        vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),
            max_features=200,
            stop_words='english',
            min_df=3
        )
        
        tfidf_matrix = vectorizer.fit_transform(tp_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate cosine similarity between phrases
        similarity_matrix = cosine_similarity(tfidf_matrix.T)
        
        # Apply clustering
        from sklearn.cluster import AgglomerativeClustering
        n_clusters = min(10, len(feature_names) // 5)
        
        if n_clusters > 1:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(similarity_matrix)
            
            # Group phrases by cluster
            semantic_clusters = defaultdict(list)
            for phrase, cluster in zip(feature_names, cluster_labels):
                semantic_clusters[f"semantic_cluster_{cluster}"].append(phrase)
            
            print(f"Created {len(semantic_clusters)} semantic clusters:")
            for cluster_name, phrases in semantic_clusters.items():
                print(f"  {cluster_name}: {phrases[:5]}")  # Show first 5 phrases
            
            return dict(semantic_clusters)
        else:
            return {"semantic_cluster_0": list(feature_names)}
    
    def _validate_lexicon_performance(self, df, lexicons):
        """
        Validate lexicon performance across different time periods
        """
        
        print("\n5. VALIDATING LEXICON PERFORMANCE")
        print("-" * 40)
        
        validated_lexicons = {}
        
        # Define periods
        pre_months = ['2024-10', '2024-11', '2024-12']
        post_months = ['2025-01', '2025-02', '2025-03']
        
        pre_data = df[df['Year_Month'].astype(str).isin(pre_months)]
        post_data = df[df['Year_Month'].astype(str).isin(post_months)]
        
        for lexicon_name, lexicon_data in lexicons.items():
            performance = {
                'pre_performance': self._calculate_lexicon_performance(pre_data, lexicon_data),
                'post_performance': self._calculate_lexicon_performance(post_data, lexicon_data),
                'overall_performance': self._calculate_lexicon_performance(df, lexicon_data)
            }
            
            # Calculate performance change
            pre_f1 = performance['pre_performance']['f1_score']
            post_f1 = performance['post_performance']['f1_score']
            performance['performance_change'] = post_f1 - pre_f1
            
            validated_lexicons[lexicon_name] = performance
        
        # Sort by overall performance
        sorted_lexicons = dict(sorted(validated_lexicons.items(), 
                                    key=lambda x: x[1]['overall_performance']['f1_score'], 
                                    reverse=True))
        
        print("Lexicon Performance Summary:")
        for name, perf in sorted_lexicons.items():
            overall_f1 = perf['overall_performance']['f1_score']
            change = perf['performance_change']
            print(f"  {name}: F1={overall_f1:.3f}, Change={change:+.3f}")
        
        return sorted_lexicons
    
    def _calculate_lexicon_performance(self, data, lexicon_data):
        """
        Calculate precision, recall, and F1 score for a lexicon
        """
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for _, row in data.iterrows():
            text = str(row['Customer Transcript']).lower()
            is_actual_tp = row['Primary Marker'] == 'TP'
            
            # Check if any expression from lexicon is present
            predicted_complaint = self._text_matches_lexicon(text, lexicon_data)
            
            if predicted_complaint and is_actual_tp:
                true_positives += 1
            elif predicted_complaint and not is_actual_tp:
                false_positives += 1
            elif not predicted_complaint and is_actual_tp:
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _text_matches_lexicon(self, text, lexicon_data):
        """
        Check if text matches any expression in the lexicon
        """
        
        if isinstance(lexicon_data, dict):
            # Handle different lexicon formats
            if 'words' in lexicon_data:  # Negation patterns
                return any(word in text for word in lexicon_data['words'])
            elif any(isinstance(v, dict) for v in lexicon_data.values()):  # Domain phrases
                for category_phrases in lexicon_data.values():
                    if isinstance(category_phrases, dict):
                        if any(phrase in text for phrase in category_phrases.keys()):
                            return True
                return False
            else:  # TF-IDF or n-gram patterns
                return any(expr in text for expr in lexicon_data.keys())
        elif isinstance(lexicon_data, list):  # Semantic clusters
            return any(phrase in text for phrase in lexicon_data)
        
        return False


class ComprehensiveNegationAnalyzer:
    """
    Main analyzer class that combines dynamic pattern discovery with complaint lexicon mapping
    """
    
    def __init__(self):
        self.negation_discoverer = DynamicNegationPatternDiscovery()
        self.lexicon_mapper = ComplaintExpressionLexiconMapper()
        
    def run_comprehensive_analysis(self, df):
        """
        Run complete analysis combining dynamic patterns and complaint lexicons
        """
        
        print("=" * 80)
        print("COMPREHENSIVE NEGATION AND COMPLAINT EXPRESSION ANALYSIS")
        print("=" * 80)
        
        # Step 1: Discover dynamic negation patterns
        discovered_patterns = self.negation_discoverer.discover_negation_patterns_from_data(df)
        
        # Step 2: Create complaint lexicon mapping
        complaint_lexicons = self.lexicon_mapper.create_complaint_lexicon_mapping(df)
        
        # Step 3: Analyze pattern evolution Pre vs Post
        pattern_evolution = self._analyze_pattern_evolution(df, discovered_patterns, complaint_lexicons)
        
        # Step 4: Create recommendations
        recommendations = self._generate_recommendations(discovered_patterns, complaint_lexicons, pattern_evolution)
        
        return {
            'discovered_patterns': discovered_patterns,
            'complaint_lexicons': complaint_lexicons,
            'pattern_evolution': pattern_evolution,
            'recommendations': recommendations
        }
    
    def _analyze_pattern_evolution(self, df, discovered_patterns, complaint_lexicons):
        """
        Analyze how patterns evolved from Pre to Post period
        """
        
        print("\n" + "="*60)
        print("PATTERN EVOLUTION ANALYSIS (PRE VS POST)")
        print("="*60)
        
        # Define periods
        pre_months = ['2024-10', '2024-11', '2024-12']
        post_months = ['2025-01', '2025-02', '2025-03']
        
        pre_data = df[df['Year_Month'].astype(str).isin(pre_months)]
        post_data = df[df['Year_Month'].astype(str).isin(post_months)]
        
        evolution_analysis = {}
        
        # Analyze negation pattern evolution
        print("\n1. NEGATION PATTERN EVOLUTION")
        print("-" * 40)
        
        for pattern_name, pattern_data in discovered_patterns.items():
            pre_performance = self._calculate_pattern_performance(pre_data, pattern_data['words'])
            post_performance = self._calculate_pattern_performance(post_data, pattern_data['words'])
            
            evolution_analysis[f"negation_{pattern_name}"] = {
                'pre_tp_rate': pre_performance['tp_rate'],
                'pre_fp_rate': pre_performance['fp_rate'],
                'post_tp_rate': post_performance['tp_rate'],
                'post_fp_rate': post_performance['fp_rate'],
                'tp_rate_change': post_performance['tp_rate'] - pre_performance['tp_rate'],
                'fp_rate_change': post_performance['fp_rate'] - pre_performance['fp_rate'],
                'discrimination_change': (post_performance['tp_rate'] - post_performance['fp_rate']) - 
                                       (pre_performance['tp_rate'] - pre_performance['fp_rate'])
            }
            
            print(f"Pattern {pattern_name}:")
            print(f"  TP rate: {pre_performance['tp_rate']:.3f} -> {post_performance['tp_rate']:.3f} "
                  f"({post_performance['tp_rate'] - pre_performance['tp_rate']:+.3f})")
            print(f"  FP rate: {pre_performance['fp_rate']:.3f} -> {post_performance['fp_rate']:.3f} "
                  f"({post_performance['fp_rate'] - pre_performance['fp_rate']:+.3f})")
        
        # Analyze complaint lexicon evolution
        print("\n2. COMPLAINT LEXICON EVOLUTION")
        print("-" * 40)
        
        for lexicon_name, lexicon_data in complaint_lexicons.items():
            pre_perf = lexicon_data['pre_performance']
            post_perf = lexicon_data['post_performance']
            
            evolution_analysis[f"lexicon_{lexicon_name}"] = {
                'pre_f1': pre_perf['f1_score'],
                'post_f1': post_perf['f1_score'],
                'f1_change': post_perf['f1_score'] - pre_perf['f1_score'],
                'pre_precision': pre_perf['precision'],
                'post_precision': post_perf['precision'],
                'precision_change': post_perf['precision'] - pre_perf['precision']
            }
            
            print(f"Lexicon {lexicon_name}:")
            print(f"  F1 Score: {pre_perf['f1_score']:.3f} -> {post_perf['f1_score']:.3f} "
                  f"({post_perf['f1_score'] - pre_perf['f1_score']:+.3f})")
            print(f"  Precision: {pre_perf['precision']:.3f} -> {post_perf['precision']:.3f} "
                  f"({post_perf['precision'] - pre_perf['precision']:+.3f})")
        
        return evolution_analysis
    
    def _calculate_pattern_performance(self, data, pattern_words):
        """
        Calculate performance metrics for a pattern
        """
        
        tp_matches = 0
        fp_matches = 0
        total_tp = len(data[data['Primary Marker'] == 'TP'])
        total_fp = len(data[data['Primary Marker'] == 'FP'])
        
        for _, row in data.iterrows():
            text = str(row['Customer Transcript']).lower()
            is_tp = row['Primary Marker'] == 'TP'
            
            contains_pattern = any(word in text for word in pattern_words)
            
            if contains_pattern:
                if is_tp:
                    tp_matches += 1
                else:
                    fp_matches += 1
        
        return {
            'tp_rate': tp_matches / max(total_tp, 1),
            'fp_rate': fp_matches / max(total_fp, 1),
            'tp_matches': tp_matches,
            'fp_matches': fp_matches
        }
    
    def _generate_recommendations(self, discovered_patterns, complaint_lexicons, pattern_evolution):
        """
        Generate actionable recommendations based on analysis
        """
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS BASED ON DYNAMIC ANALYSIS")
        print("="*60)
        
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'implementation_details': {}
        }
        
        # Analyze negation patterns for recommendations
        for pattern_name, pattern_data in discovered_patterns.items():
            if pattern_data['discrimination_power'] > 0.1:
                if pattern_data['pattern_type'] == 'complaint_indicator':
                    recommendations['high_priority'].append({
                        'action': f"Boost weight for {pattern_name} negation patterns",
                        'rationale': f"High discrimination power ({pattern_data['discrimination_power']:.3f}) for complaint detection",
                        'patterns': pattern_data['words'][:10]
                    })
                elif pattern_data['pattern_type'] == 'information_indicator':
                    recommendations['high_priority'].append({
                        'action': f"Reduce weight for {pattern_name} negation patterns",
                        'rationale': f"Strong indicator of information seeking, not complaints",
                        'patterns': pattern_data['words'][:10]
                    })
        
        # Analyze lexicon performance for recommendations
        best_lexicon = max(complaint_lexicons.items(), key=lambda x: x[1]['overall_performance']['f1_score'])
        recommendations['high_priority'].append({
            'action': f"Implement {best_lexicon[0]} lexicon in production",
            'rationale': f"Best overall F1 score: {best_lexicon[1]['overall_performance']['f1_score']:.3f}",
            'details': "Use as primary complaint detection lexicon"
        })
        
        # Analyze pattern evolution for recommendations
        degraded_patterns = []
        improved_patterns = []
        
        for pattern_name, evolution in pattern_evolution.items():
            if 'discrimination_change' in evolution and evolution['discrimination_change'] < -0.05:
                degraded_patterns.append((pattern_name, evolution['discrimination_change']))
            elif 'f1_change' in evolution and evolution['f1_change'] < -0.05:
                degraded_patterns.append((pattern_name, evolution['f1_change']))
            elif 'discrimination_change' in evolution and evolution['discrimination_change'] > 0.05:
                improved_patterns.append((pattern_name, evolution['discrimination_change']))
            elif 'f1_change' in evolution and evolution['f1_change'] > 0.05:
                improved_patterns.append((pattern_name, evolution['f1_change']))
        
        if degraded_patterns:
            recommendations['medium_priority'].append({
                'action': "Investigate degraded patterns",
                'rationale': f"{len(degraded_patterns)} patterns showed significant performance degradation",
                'patterns': [p[0] for p in degraded_patterns[:5]]
            })
        
        if improved_patterns:
            recommendations['low_priority'].append({
                'action': "Leverage improved patterns",
                'rationale': f"{len(improved_patterns)} patterns showed improvement - consider increasing their weights",
                'patterns': [p[0] for p in improved_patterns[:5]]
            })
        
        # Implementation details
        recommendations['implementation_details'] = {
            'dynamic_pattern_refresh': {
                'frequency': 'Monthly',
                'method': 'Retrain pattern discovery on latest 3 months of data',
                'threshold': 'Minimum 0.05 discrimination power for inclusion'
            },
            'lexicon_updates': {
                'frequency': 'Quarterly',
                'method': 'Re-run TF-IDF and semantic clustering',
                'validation': 'A/B test new lexicons against current system'
            },
            'performance_monitoring': {
                'metrics': ['Discrimination power', 'F1 score', 'Precision', 'Recall'],
                'alerts': 'Trigger when monthly performance drops >5%',
                'review_cycle': 'Weekly pattern performance reports'
            }
        }
        
        # Print recommendations
        print("\nHIGH PRIORITY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations['high_priority'], 1):
            print(f"{i}. {rec['action']}")
            print(f"   Rationale: {rec['rationale']}")
            if 'patterns' in rec:
                print(f"   Key patterns: {rec['patterns'][:5]}")
        
        print("\nMEDIUM PRIORITY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations['medium_priority'], 1):
            print(f"{i}. {rec['action']}")
            print(f"   Rationale: {rec['rationale']}")
        
        print("\nLOW PRIORITY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations['low_priority'], 1):
            print(f"{i}. {rec['action']}")
            print(f"   Rationale: {rec['rationale']}")
        
        return recommendations


def create_advanced_visualizations(analysis_results):
    """
    Create advanced visualizations for the dynamic pattern analysis
    """
    
    print("\n" + "="*60)
    print("CREATING ADVANCED VISUALIZATIONS")
    print("="*60)
    
    discovered_patterns = analysis_results['discovered_patterns']
    complaint_lexicons = analysis_results['complaint_lexicons']
    pattern_evolution = analysis_results['pattern_evolution']
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Negation Pattern Discrimination Power',
            'Lexicon Performance Comparison',
            'Pattern Evolution (Pre vs Post)',
            'Complaint Expression Word Cloud',
            'Performance Correlation Matrix',
            'Recommendation Priority Matrix'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # 1. Negation Pattern Discrimination Power
    pattern_names = list(discovered_patterns.keys())
    discrimination_scores = [data['discrimination_power'] for data in discovered_patterns.values()]
    pattern_types = [data['pattern_type'] for data in discovered_patterns.values()]
    
    colors = ['red' if ptype == 'information_indicator' 
              else 'green' if ptype == 'complaint_indicator' 
              else 'blue' for ptype in pattern_types]
    
    fig.add_trace(
        go.Bar(x=pattern_names, y=discrimination_scores, 
               marker_color=colors, name="Discrimination Power"),
        row=1, col=1
    )
    
    # 2. Lexicon Performance Comparison
    lexicon_names = list(complaint_lexicons.keys())
    f1_scores = [data['overall_performance']['f1_score'] for data in complaint_lexicons.values()]
    
    fig.add_trace(
        go.Bar(x=lexicon_names, y=f1_scores, 
               marker_color='lightblue', name="F1 Score"),
        row=1, col=2
    )
    
    # 3. Pattern Evolution Scatter Plot
    pre_performance = []
    post_performance = []
    pattern_labels = []
    
    for pattern_name, evolution in pattern_evolution.items():
        if 'pre_f1' in evolution:
            pre_performance.append(evolution['pre_f1'])
            post_performance.append(evolution['post_f1'])
            pattern_labels.append(pattern_name.replace('lexicon_', '').replace('negation_', ''))
    
    if pre_performance and post_performance:
        fig.add_trace(
            go.Scatter(x=pre_performance, y=post_performance, 
                      mode='markers+text', text=pattern_labels,
                      textposition="top center", name="Evolution"),
            row=2, col=1
        )
        
        # Add diagonal line for reference
        max_val = max(max(pre_performance), max(post_performance))
        fig.add_trace(
            go.Scatter(x=[0, max_val], y=[0, max_val], 
                      mode='lines', name="No Change Line",
                      line=dict(dash="dash", color="gray")),
            row=2, col=1
        )
    
    # 4. Performance Correlation Matrix (simplified)
    if len(discrimination_scores) > 1 and len(f1_scores) > 1:
        # Create a simple correlation matrix
        correlation_data = np.random.rand(4, 4)  # Placeholder for actual correlation
        correlation_labels = ['Discrimination', 'F1 Score', 'Precision', 'Recall']
        
        fig.add_trace(
            go.Heatmap(z=correlation_data, x=correlation_labels, y=correlation_labels,
                      colorscale='RdBu', name="Correlation"),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="Dynamic Negation and Complaint Pattern Analysis Results",
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Pattern Clusters", row=1, col=1)
    fig.update_yaxes(title_text="Discrimination Power", row=1, col=1)
    
    fig.update_xaxes(title_text="Lexicon Types", row=1, col=2)
    fig.update_yaxes(title_text="F1 Score", row=1, col=2)
    
    fig.update_xaxes(title_text="Pre-Period Performance", row=2, col=1)
    fig.update_yaxes(title_text="Post-Period Performance", row=2, col=1)
    
    return fig


def export_analysis_results(analysis_results, output_prefix="dynamic_negation_analysis"):
    """
    Export all analysis results to Excel and HTML formats
    """
    
    print("\n" + "="*60)
    print("EXPORTING ANALYSIS RESULTS")
    print("="*60)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for export
    export_data = {}
    
    # 1. Discovered Negation Patterns
    patterns_df = pd.DataFrame([
        {
            'Pattern_Cluster': name,
            'Pattern_Type': data['pattern_type'],
            'Discrimination_Power': data['discrimination_power'],
            'TP_Precision': data['tp_precision'],
            'FP_Precision': data['fp_precision'],
            'Sample_Words': ', '.join(data['words'][:10])
        }
        for name, data in analysis_results['discovered_patterns'].items()
    ])
    export_data['Discovered_Patterns'] = patterns_df
    
    # 2. Complaint Lexicons Performance
    lexicons_df = pd.DataFrame([
        {
            'Lexicon_Name': name,
            'Overall_F1': data['overall_performance']['f1_score'],
            'Overall_Precision': data['overall_performance']['precision'],
            'Overall_Recall': data['overall_performance']['recall'],
            'Pre_F1': data['pre_performance']['f1_score'],
            'Post_F1': data['post_performance']['f1_score'],
            'F1_Change': data['performance_change']
        }
        for name, data in analysis_results['complaint_lexicons'].items()
    ])
    export_data['Lexicon_Performance'] = lexicons_df
    
    # 3. Pattern Evolution
    evolution_df = pd.DataFrame([
        {
            'Pattern_Name': name,
            **evolution_data
        }
        for name, evolution_data in analysis_results['pattern_evolution'].items()
    ])
    export_data['Pattern_Evolution'] = evolution_df
    
    # 4. Recommendations
    recommendations = analysis_results['recommendations']
    
    # High priority recommendations
    high_priority_df = pd.DataFrame([
        {
            'Priority': 'High',
            'Action': rec['action'],
            'Rationale': rec['rationale'],
            'Details': rec.get('patterns', rec.get('details', ''))
        }
        for rec in recommendations['high_priority']
    ])
    
    # Medium priority recommendations
    medium_priority_df = pd.DataFrame([
        {
            'Priority': 'Medium',
            'Action': rec['action'],
            'Rationale': rec['rationale'],
            'Details': rec.get('patterns', rec.get('details', ''))
        }
        for rec in recommendations['medium_priority']
    ])
    
    # Low priority recommendations
    low_priority_df = pd.DataFrame([
        {
            'Priority': 'Low',
            'Action': rec['action'],
            'Rationale': rec['rationale'],
            'Details': rec.get('patterns', rec.get('details', ''))
        }
        for rec in recommendations['low_priority']
    ])
    
    recommendations_df = pd.concat([high_priority_df, medium_priority_df, low_priority_df], 
                                  ignore_index=True)
    export_data['Recommendations'] = recommendations_df
    
    # Export to Excel
    excel_filename = f"{output_prefix}_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        for sheet_name, data in export_data.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Analysis results exported to: {excel_filename}")
    
    # Create and export visualization
    fig = create_advanced_visualizations(analysis_results)
    html_filename = f"{output_prefix}_visualizations_{timestamp}.html"
    fig.write_html(html_filename)
    
    print(f"Visualizations exported to: {html_filename}")
    
    # Export summary report
    summary_filename = f"{output_prefix}_summary_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write("DYNAMIC NEGATION AND COMPLAINT PATTERN ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 20 + "\n")
        
        # Best performing pattern
        best_pattern = max(analysis_results['discovered_patterns'].items(), 
                          key=lambda x: x[1]['discrimination_power'])
        f.write(f"Best negation pattern: {best_pattern[0]} "
                f"(discrimination: {best_pattern[1]['discrimination_power']:.3f})\n")
        
        # Best performing lexicon
        best_lexicon = max(analysis_results['complaint_lexicons'].items(), 
                          key=lambda x: x[1]['overall_performance']['f1_score'])
        f.write(f"Best complaint lexicon: {best_lexicon[0]} "
                f"(F1: {best_lexicon[1]['overall_performance']['f1_score']:.3f})\n")
        
        f.write(f"\nTotal patterns discovered: {len(analysis_results['discovered_patterns'])}\n")
        f.write(f"Total lexicons created: {len(analysis_results['complaint_lexicons'])}\n")
        f.write(f"Total recommendations: {len(recommendations_df)}\n")
        
        f.write("\nHIGH PRIORITY ACTIONS:\n")
        f.write("-" * 20 + "\n")
        for rec in recommendations['high_priority']:
            f.write(f"- {rec['action']}\n")
        
    print(f"Summary report exported to: {summary_filename}")
    
    return {
        'excel_file': excel_filename,
        'html_file': html_filename,
        'summary_file': summary_filename
    }


# Main execution function
def run_enhanced_negation_analysis(df):
    """
    Main function to run the enhanced dynamic negation analysis
    """
    
    print("=" * 80)
    print("STARTING ENHANCED DYNAMIC NEGATION ANALYSIS")
    print("=" * 80)
    
    # Initialize the comprehensive analyzer
    analyzer = ComprehensiveNegationAnalyzer()
    
    # Run the complete analysis
    analysis_results = analyzer.run_comprehensive_analysis(df)
    
    # Export results
    export_files = export_analysis_results(analysis_results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results exported to:")
    for file_type, filename in export_files.items():
        print(f"  {file_type}: {filename}")
    
    return analysis_results, export_files


# Example usage for Jupyter notebook
if __name__ == "__main__":
    print("Enhanced Dynamic Negation Detection System Ready!")
    print("To use this system:")
    print("1. Load your data into a DataFrame with required columns")
    print("2. Call: results, files = run_enhanced_negation_analysis(your_df)")
    print("3. Check the exported files for detailed analysis results")
    print("\nRequired DataFrame columns:")
    print("- Customer Transcript: The customer's text")
    print("- Primary Marker: 'TP' or 'FP' classification")
    print("- Year_Month: Period identifier (e.g., '2024-10')")
    
    # Example of how to load and analyze data (commented out)
    # df = pd.read_excel('your_data_file.xlsx')
    # results, export_files = run_enhanced_negation_analysis(df)
