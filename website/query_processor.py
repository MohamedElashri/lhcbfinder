# query_processor.py
import re
import numpy as np
from typing import Tuple, Optional

class QueryProcessor:
    def __init__(self):
        # LHCb-related keywords and topics
        self.lhcb_keywords = {
        # General
        'lhcb', 'cern', 'particle physics', 'detector', 'collision', 'physics', 
        'analysis', 'measurement', 'result', 'phenomenology', 'standard model', 'SM',

        # Particles
        'hadron', 'meson', 'b meson', 'charm', 'beauty', 'quark', 'lepton', 'boson', 
        'neutrino', 'dark matter', 'DM', 'higgs', 'H', 'supersymmetry', 'SUSY', 
        'quantum chromodynamics', 'qcd', 'electroweak', 'EW', 'flavor', 'flavour',

        # Decay and Properties
        'decay', 'branching ratio', 'BR', 'cp violation', 'CPV', 'cross section', 'σ', 
        'production', 'luminosity', 'L', 'violation', 'magnetic moment', 'electric charge',
        'magnetic field', 'electric field', 'magnetic flux', 'electric potential', 
        'magnetic flux density',

        # Experimental Techniques
        'trigger', 'reconstruction', 'reco', 'vertex', 'tracking', 'calorimeter', 'calo',
        'muon', 'event generator', 'gen', 'throughput', 'data', 'run', 'dataset', 'DST',

        # Computing and AI
        'neural', 'deep', 'learning', 'artificial intelligence', 'AI', 'machine learning', 'ML', 
        'Allen', 'LLM', 'computer', 'intelligence', 'software', 'algorithm', 'GPU', 
        'high-performance computing', 'HPC',

        # LHC-specific
        'LHC', 'Large Hadron Collider', 'collision energy', 'beam', 'proton', 
        'interaction point', 'IP', 'beam energy', '√s',

        # Symbols and Representations
        'cc', 'cc_bar', 'bar', 'b', 'b_bar', 'bb', 'bb_bar', 'bb_bar_bar',

        # Advanced Topics
        'quantum', 'quantum mechanics', 'QM', 'dark energy', 'Monte Carlo', 'MC', 
        'simulation', 'sim', 'likelihood', 'statistical analysis', 'stats',

        # Auxiliary Concepts
        'particle interaction', 'high energy physics', 'HEP', 
        'statistical uncertainty', 'systematic uncertainty', 'stat', 'syst', 'interaction'
        }
        
        # Common physics units and notation
        self.physics_notation = {
            'gev', 'tev', 'mev', 'kev', 'ev', 'fb', 'pb', 'nb', 'mb',
            'sigma', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta',
            'lambda', 'mu', 'tau', 'phi', 'psi', 'omega', 'pi', 'rho'
        }
        
        # Offensive or inappropriate terms (basic list, can be expanded)
        self.blocked_terms = {
            'hack', 'crack', 'exploit', 'vulnerability', 'password', 'credentials',
            'private', 'confidential', 'secret', 'classified', 'proprietary', 'politics'
            'racism', 'hate', 'sexism', 'sex', 'homosexuality', 'religion', 'best paper'
        }

    def clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Convert to lowercase
        query = query.lower()
        # Remove special characters except common physics notation
        query = re.sub(r'[^a-z0-9\s\-\+\_\.\(\)\[\]\{\}\^]', '', query)
        # Normalize whitespace
        query = ' '.join(query.split())
        return query

    def is_physics_related(self, query: str) -> bool:
        """Check if the query is related to physics/LHCb research."""
        words = set(query.lower().split())
        
        # Check for physics notation
        has_notation = any(term in words for term in self.physics_notation)
        
        # Check for LHCb/physics keywords
        has_keywords = any(keyword in query.lower() for keyword in self.lhcb_keywords)
        
        # Check for common physics patterns (e.g., B -> K mu+ mu-)
        has_physics_notation = bool(re.search(r'[A-Za-z]\s*[-→>]\s*[A-Za-z]', query))
        
        return has_notation or has_keywords or has_physics_notation

    def contains_blocked_terms(self, query: str) -> bool:
        """Check if the query contains any blocked terms."""
        return any(term in query.lower() for term in self.blocked_terms)

    def validate_query(self, query: str, embedding: list) -> Tuple[bool, Optional[str]]:
        """
        Validate the query and its embedding.
        Returns (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty."

        if len(query) > 200:
            return False, "Query length cannot exceed 200 characters."

        clean_query = self.clean_query(query)
        
        if self.contains_blocked_terms(clean_query):
            return False, "Query contains inappropriate terms."
            
        if not self.is_physics_related(clean_query):
            return False, "Query should be related to particle physics or LHCb research."
            
        # Check if embedding is reasonable
        if embedding:
            embedding_array = np.array(embedding)
            if np.isnan(embedding_array).any():
                return False, "Invalid query embedding."
            
            # Check if the embedding magnitude is reasonable
            magnitude = np.linalg.norm(embedding_array)
            if magnitude < 0.1 or magnitude > 100:
                return False, "Query embedding has unusual magnitude."

        return True, None

    def get_minimum_similarity_threshold(self, query: str) -> float:
        """
        Determine the minimum similarity threshold based on query characteristics.
        Returns a value between 0 and 1.
        """
        # More strict threshold for shorter queries
        if len(query.split()) <= 3:
            return 0.6
        
        # More lenient threshold for specific physics notation
        if any(term in query.lower() for term in self.physics_notation):
            return 0.4
            
        # Default threshold
        return 0.5
