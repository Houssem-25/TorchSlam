import logging
import time
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from .keyframe import Keyframe, KeyframeManager
from .odometry.base import FramePose


class LoopCandidate:
    """Represents a potential loop closure candidate."""

    def __init__(self, query_id: int, match_id: int, score: float):
        """
        Initialize a loop candidate.

        Args:
            query_id: ID of the query keyframe
            match_id: ID of the matched keyframe
            score: Matching score
        """
        self.query_id = query_id
        self.match_id = match_id
        self.score = score
        self.inlier_matches = []  # List of (query_idx, match_idx) tuples
        self.transform = None  # Transformation from match to query
        self.inlier_ratio = 0.0
        self.verified = False

    def set_verified(
        self,
        verified: bool,
        inlier_matches: List[Tuple[int, int]],
        transform: Optional[FramePose] = None,
        inlier_ratio: float = 0.0,
    ):
        """
        Set verification result.

        Args:
            verified: Whether the loop candidate is verified
            inlier_matches: List of inlier matches
            transform: Estimated transformation
            inlier_ratio: Ratio of inliers to total matches
        """
        self.verified = verified
        self.inlier_matches = inlier_matches
        self.transform = transform
        self.inlier_ratio = inlier_ratio


class InvertedIndex:
    """Inverted index for Bag of Words model."""

    def __init__(self, num_words: int):
        """
        Initialize inverted index.

        Args:
            num_words: Number of visual words in the vocabulary
        """
        self.index = defaultdict(list)  # word_id -> [(keyframe_id, weight), ...]
        self.num_words = num_words
        self.keyframe_word_weights = {}  # keyframe_id -> {word_id: weight, ...}
        self.keyframe_norm = {}  # keyframe_id -> L2 norm of the BoW vector

    def add_keyframe(self, keyframe_id: int, bow_vector: Dict[int, float]):
        """
        Add a keyframe to the inverted index.

        Args:
            keyframe_id: ID of the keyframe
            bow_vector: Bag of Words vector as {word_id: weight, ...}
        """
        # Add entries to inverted index
        for word_id, weight in bow_vector.items():
            self.index[word_id].append((keyframe_id, weight))

        # Store BoW vector weights for this keyframe
        self.keyframe_word_weights[keyframe_id] = bow_vector

        # Compute and store L2 norm
        self.keyframe_norm[keyframe_id] = np.sqrt(
            sum(w * w for w in bow_vector.values())
        )

    def remove_keyframe(self, keyframe_id: int):
        """
        Remove a keyframe from the inverted index.

        Args:
            keyframe_id: ID of the keyframe to remove
        """
        # Remove keyframe from index entries
        if keyframe_id in self.keyframe_word_weights:
            for word_id in self.keyframe_word_weights[keyframe_id]:
                self.index[word_id] = [
                    (kf_id, w)
                    for kf_id, w in self.index[word_id]
                    if kf_id != keyframe_id
                ]

                # Remove word if no keyframes use it
                if not self.index[word_id]:
                    del self.index[word_id]

            # Remove keyframe data
            del self.keyframe_word_weights[keyframe_id]
            del self.keyframe_norm[keyframe_id]

    def query(
        self, bow_vector: Dict[int, float], top_k: int = 10, min_score: float = 0.2
    ) -> List[Tuple[int, float]]:
        """
        Query the index for similar keyframes.

        Args:
            bow_vector: Query BoW vector
            top_k: Number of top matches to return
            min_score: Minimum similarity score

        Returns:
            List of (keyframe_id, score) tuples
        """
        # Calculate query vector norm
        query_norm = np.sqrt(sum(w * w for w in bow_vector.values()))

        if query_norm == 0:
            return []

        # Compute similarity scores for all keyframes containing at least one word from the query
        scores = defaultdict(float)
        keyframes_to_score = set()

        for word_id, query_weight in bow_vector.items():
            # Get all keyframes containing this word
            for kf_id, kf_weight in self.index.get(word_id, []):
                keyframes_to_score.add(kf_id)

                # Accumulate dot product
                scores[kf_id] += query_weight * kf_weight

        # Normalize scores by vector norms (cosine similarity)
        results = []
        for kf_id in keyframes_to_score:
            kf_norm = self.keyframe_norm.get(kf_id, 1.0)
            similarity = scores[kf_id] / (query_norm * kf_norm)

            if similarity >= min_score:
                results.append((kf_id, similarity))

        # Sort by score (descending) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_word_frequency(self) -> Dict[int, int]:
        """
        Get the frequency of each word in the index.

        Returns:
            Dictionary mapping word_id to frequency
        """
        return {word_id: len(entries) for word_id, entries in self.index.items()}


class VocabularyTree:
    """
    Hierarchical vocabulary tree for Bag of Words.

    This class implements a simplified version of a vocabulary tree.
    The actual implementation would involve training on a large dataset
    of image features.
    """

    def __init__(
        self, branching_factor: int = 10, depth: int = 6, descriptor_dim: int = 128
    ):
        """
        Initialize vocabulary tree.

        Args:
            branching_factor: Number of children for each node
            depth: Maximum depth of the tree
            descriptor_dim: Dimensionality of feature descriptors
        """
        self.branching_factor = branching_factor
        self.depth = depth
        self.descriptor_dim = descriptor_dim

        # Total number of words (leaves)
        self.num_words = branching_factor**depth

        # Initialize cluster centers (simplified)
        # In a real implementation, these would be learned from data
        self.centers = self._initialize_centers()

        # Word weights for tf-idf scoring
        self.word_weights = torch.ones(self.num_words)

        # IDF weights
        self.idf_weights = torch.ones(self.num_words)

    def _initialize_centers(self) -> Dict[int, torch.Tensor]:
        """
        Initialize cluster centers for the vocabulary tree.
        This is a simplified placeholder implementation.

        Returns:
            Dictionary mapping node ID to cluster center
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        centers = {}

        # Root node has ID 1
        centers[1] = torch.zeros(
            self.branching_factor, self.descriptor_dim, device=device
        )

        # For a real implementation, these would be learned from data
        # Here we just initialize with random values for demonstration
        for i in range(self.branching_factor):
            centers[1][i] = torch.randn(self.descriptor_dim, device=device)
            centers[1][i] = F.normalize(centers[1][i], p=2, dim=0)

        return centers

    def quantize(self, descriptor: torch.Tensor) -> int:
        """
        Quantize a descriptor to a visual word.

        Args:
            descriptor: Feature descriptor

        Returns:
            Word ID
        """
        # Ensure descriptor is normalized
        descriptor = F.normalize(descriptor, p=2, dim=0)

        # Start at root node
        node_id = 1

        # Traverse tree to a leaf
        for level in range(self.depth):
            # Find closest center
            if node_id in self.centers:
                similarities = torch.matmul(self.centers[node_id], descriptor)
                child_idx = torch.argmax(similarities).item()

                # Compute child node ID
                node_id = self.branching_factor * node_id + child_idx + 1
            else:
                # If center not available, use a simple hash function
                hash_val = torch.sum(descriptor * (level + 1)).item()
                child_idx = int(hash_val * 10) % self.branching_factor
                node_id = self.branching_factor * node_id + child_idx + 1

        # Convert to word ID (0-indexed)
        word_id = node_id - self.num_words

        return max(0, min(word_id, self.num_words - 1))  # Ensure valid range

    def compute_bow_vector(self, descriptors: torch.Tensor) -> Dict[int, float]:
        """
        Compute Bag of Words vector for a set of descriptors.

        Args:
            descriptors: Feature descriptors (N, descriptor_dim)

        Returns:
            BoW vector as {word_id: weight, ...}
        """
        bow_vector = defaultdict(float)

        # Quantize each descriptor
        for i in range(descriptors.shape[0]):
            word_id = self.quantize(descriptors[i])
            bow_vector[word_id] += 1.0

        # Apply tf-idf weighting
        for word_id, count in list(bow_vector.items()):
            # Term frequency
            tf = count / descriptors.shape[0]

            # Inverse document frequency
            idf = self.idf_weights[word_id].item()

            # TF-IDF weight
            bow_vector[word_id] = tf * idf

        return dict(bow_vector)

    def update_idf_weights(self, inverted_index: InvertedIndex, num_documents: int):
        """
        Update IDF weights based on word frequencies.

        Args:
            inverted_index: Inverted index containing word frequencies
            num_documents: Total number of documents (keyframes)
        """
        # Get word frequencies
        word_freq = inverted_index.get_word_frequency()

        # Compute IDF weights
        for word_id in range(self.num_words):
            freq = word_freq.get(word_id, 0)
            if freq > 0:
                self.idf_weights[word_id] = torch.log(num_documents / freq)
            else:
                self.idf_weights[word_id] = 1.0


class DBoWVocabulary:
    """
    Vocabulary for DBoW (Bag of Binary Words).

    This is a simplified implementation for binary feature descriptors like ORB.
    """

    def __init__(self, branching_factor: int = 9, depth: int = 6, seed: int = 42):
        """
        Initialize vocabulary.

        Args:
            branching_factor: Number of children for each node
            depth: Maximum depth of the tree
            seed: Random seed
        """
        self.branching_factor = branching_factor
        self.depth = depth
        self.seed = seed

        # Number of leaf nodes (words)
        self.num_words = branching_factor**depth

        # Tree structure (simplified)
        self.nodes = {}  # node_id -> descriptor (center)

        # Word weights for scoring
        self.word_weights = {}  # word_id -> weight

    def train(self, descriptors: List[torch.Tensor]):
        """
        Train vocabulary from descriptors.

        Args:
            descriptors: List of binary descriptors
        """
        # Placeholder for vocabulary training
        # In a real implementation, this would involve hierarchical clustering
        pass

    def transform(self, descriptors: torch.Tensor) -> Dict[int, float]:
        """
        Transform descriptors to BoW vector.

        Args:
            descriptors: Binary descriptors

        Returns:
            BoW vector as {word_id: weight, ...}
        """
        # Simplified implementation
        bow_vector = {}

        for i in range(descriptors.shape[0]):
            # Hash descriptor to a word ID
            descriptor = descriptors[i]
            word_id = self._hash_descriptor(descriptor)

            # Update count
            if word_id in bow_vector:
                bow_vector[word_id] += 1.0
            else:
                bow_vector[word_id] = 1.0

        # Normalize
        total = sum(bow_vector.values())
        if total > 0:
            for word_id in bow_vector:
                bow_vector[word_id] /= total

        return bow_vector

    def _hash_descriptor(self, descriptor: torch.Tensor) -> int:
        """
        Hash a binary descriptor to a word ID.
        This is a placeholder for the actual tree traversal.

        Args:
            descriptor: Binary descriptor

        Returns:
            Word ID
        """
        # Compute a simple hash based on the binary descriptor
        if descriptor.dtype == torch.uint8:
            # For binary descriptors, sum the bytes and hash
            hash_val = int(torch.sum(descriptor).item())
        else:
            # For float descriptors, use a different approach
            hash_val = int(torch.sum(descriptor > 0.5).item())

        # Map to word ID
        word_id = hash_val % self.num_words

        return word_id


class LoopDetector:
    """Loop closure detection in SLAM system using Bag of Words approach."""

    def __init__(self, config: Dict = None):
        """
        Initialize loop detector.

        Args:
            config: Configuration dictionary with the following keys:
                - bow_vocabulary_path: Path to BoW vocabulary (optional)
                - bow_branching_factor: Branching factor for vocabulary tree
                - bow_tree_depth: Depth of vocabulary tree
                - min_loop_interval: Minimum interval between keyframes for loop detection
                - similarity_threshold: Threshold for considering a loop candidate
                - min_inlier_ratio: Minimum ratio of inliers for loop verification
                - ransac_threshold: Threshold for RANSAC in loop verification
                - temporal_constraint: Minimum temporal distance for loop candidates
        """
        self.config = config if config is not None else {}

        # Extract configuration
        self.bow_branching_factor = self.config.get("bow_branching_factor", 10)
        self.bow_tree_depth = self.config.get("bow_tree_depth", 6)
        self.min_loop_interval = self.config.get("min_loop_interval", 50)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.75)
        self.min_inlier_ratio = self.config.get("min_inlier_ratio", 0.3)
        self.ransac_threshold = self.config.get("ransac_threshold", 4.0)
        self.temporal_constraint = self.config.get("temporal_constraint", 30)

        # Database of keyframes
        self.keyframe_timestamps = {}  # keyframe_id -> timestamp

        # Initialize BoW components
        self._initialize_bow()

        # Recent loop candidates
        self.recent_candidates = OrderedDict()  # query_id -> LoopCandidate
        self.max_recent_candidates = 10

        # Verified loop closures
        self.verified_loops = {}  # query_id -> LoopCandidate

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_bow(self):
        """Initialize Bag of Words components."""
        # Initialize vocabulary tree
        self.vocabulary_tree = VocabularyTree(
            branching_factor=self.bow_branching_factor, depth=self.bow_tree_depth
        )

        # Initialize inverted index
        self.inverted_index = InvertedIndex(self.vocabulary_tree.num_words)

        # Load vocabulary if provided
        vocabulary_path = self.config.get("bow_vocabulary_path")
        if vocabulary_path:
            try:
                self._load_vocabulary(vocabulary_path)
            except Exception as e:
                self.logger.warning(f"Failed to load vocabulary: {e}")

    def _load_vocabulary(self, path: str):
        """
        Load vocabulary tree from file.

        Args:
            path: Path to vocabulary file
        """
        # Placeholder for loading vocabulary
        self.logger.info(f"Loading vocabulary from {path}")
        # In a real implementation, this would load the vocabulary tree parameters

    def process_keyframe(self, keyframe: Keyframe) -> Optional[LoopCandidate]:
        """
        Process a keyframe for loop detection.

        Args:
            keyframe: Keyframe to process

        Returns:
            Loop candidate if a loop is detected, None otherwise
        """
        # Store keyframe timestamp
        self.keyframe_timestamps[keyframe.id] = keyframe.timestamp

        # Skip if no descriptors
        if keyframe.descriptors is None or keyframe.descriptors.shape[0] == 0:
            return None

        # Compute BoW vector if not already available
        if keyframe.bow_vector is None:
            bow_vector = self.vocabulary_tree.compute_bow_vector(keyframe.descriptors)
            keyframe.set_bow_vector(bow_vector)
        else:
            bow_vector = keyframe.bow_vector

        # Add to inverted index
        self.inverted_index.add_keyframe(keyframe.id, bow_vector)

        # Query for similar keyframes
        loop_candidates = self._query_bow(keyframe)

        # Verify loop candidates
        return self._verify_candidates(keyframe, loop_candidates)

    def _query_bow(self, keyframe: Keyframe) -> List[Tuple[int, float]]:
        """
        Query for similar keyframes using BoW.

        Args:
            keyframe: Query keyframe

        Returns:
            List of (keyframe_id, similarity_score) tuples
        """
        # Skip if no bow vector
        if not hasattr(keyframe, "bow_vector") or not keyframe.bow_vector:
            return []

        # Query inverted index
        candidates = self.inverted_index.query(keyframe.bow_vector)

        # Apply temporal constraint
        filtered_candidates = self._apply_temporal_constraint(keyframe.id, candidates)

        return filtered_candidates

    def _apply_temporal_constraint(
        self, query_id: int, candidates: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Filter loop candidates based on temporal constraint.

        Args:
            query_id: ID of the query keyframe
            candidates: List of (keyframe_id, score) tuples

        Returns:
            Filtered list of candidates
        """
        # Skip if no timestamps
        if query_id not in self.keyframe_timestamps:
            return candidates

        query_time = self.keyframe_timestamps[query_id]
        filtered_candidates = []

        for kf_id, score in candidates:
            # Skip if timestamp not available
            if kf_id not in self.keyframe_timestamps:
                continue

            # Check keyframe ID distance
            if abs(query_id - kf_id) < self.min_loop_interval:
                continue

            # Check timestamp distance
            kf_time = self.keyframe_timestamps[kf_id]
            time_diff = abs(query_time - kf_time)

            # Convert time difference to frame count (assuming 30fps)
            frame_diff = time_diff * 30

            if frame_diff < self.temporal_constraint:
                continue

            filtered_candidates.append((kf_id, score))

        return filtered_candidates

    def _verify_candidates(
        self, query_keyframe: Keyframe, candidates: List[Tuple[int, float]]
    ) -> Optional[LoopCandidate]:
        """
        Verify loop candidates by geometric validation.

        Args:
            query_keyframe: Query keyframe
            candidates: List of (keyframe_id, score) tuples

        Returns:
            Verified loop candidate or None
        """
        if not candidates:
            return None

        # Process top candidates
        for match_id, score in candidates[:3]:  # Check top 3 candidates
            # Create loop candidate
            candidate = LoopCandidate(query_keyframe.id, match_id, score)

            # Find feature matches
            inlier_matches, transform, inlier_ratio = self._find_matches(
                query_keyframe, match_id
            )

            # Check if enough inliers
            if inlier_ratio >= self.min_inlier_ratio:
                candidate.set_verified(True, inlier_matches, transform, inlier_ratio)

                # Store in recent candidates
                self._add_recent_candidate(candidate)

                # Store as verified loop
                self.verified_loops[query_keyframe.id] = candidate

                return candidate

        return None

    def _find_matches(
        self, query_keyframe: Keyframe, match_id: int
    ) -> Tuple[List[Tuple[int, int]], Optional[FramePose], float]:
        """
        Find feature matches between keyframes and estimate transformation.

        Args:
            query_keyframe: Query keyframe
            match_id: ID of match keyframe

        Returns:
            Tuple of (inlier_matches, transform, inlier_ratio)
        """
        # Placeholder implementation
        # In a real system, this would perform feature matching and RANSAC
        inlier_matches = []
        transform = None
        inlier_ratio = 0.0

        # Simulate some matches
        inlier_matches = [(i, i) for i in range(10)]
        transform = FramePose.identity()
        inlier_ratio = 0.5

        return inlier_matches, transform, inlier_ratio

    def _add_recent_candidate(self, candidate: LoopCandidate):
        """
        Add a loop candidate to recent candidates cache.

        Args:
            candidate: Loop candidate to add
        """
        # Add to recent candidates
        self.recent_candidates[candidate.query_id] = candidate

        # Remove oldest if exceeding maximum
        if len(self.recent_candidates) > self.max_recent_candidates:
            self.recent_candidates.popitem(last=False)

    def get_recent_loops(self, n: int = 5) -> List[LoopCandidate]:
        """
        Get recent loop closures.

        Args:
            n: Number of recent loops to return

        Returns:
            List of recent verified loop candidates
        """
        # Filter for verified loops
        verified_loops = [
            candidate
            for candidate in self.recent_candidates.values()
            if candidate.verified
        ]

        # Sort by query ID (latest first)
        verified_loops.sort(key=lambda x: x.query_id, reverse=True)

        return verified_loops[:n]

    def get_all_verified_loops(self) -> Dict[int, LoopCandidate]:
        """
        Get all verified loop closures.

        Returns:
            Dictionary mapping query keyframe ID to loop candidate
        """
        return self.verified_loops

    def get_loop_connections(self) -> List[Tuple[int, int]]:
        """
        Get all loop connections.

        Returns:
            List of (query_id, match_id) tuples
        """
        return [(loop.query_id, loop.match_id) for loop in self.verified_loops.values()]

    def reset(self):
        """Reset loop detector state."""
        self.keyframe_timestamps = {}
        self.inverted_index = InvertedIndex(self.vocabulary_tree.num_words)
        self.recent_candidates = OrderedDict()
        self.verified_loops = {}

    def remove_keyframe(self, keyframe_id: int):
        """
        Remove a keyframe from the loop detector.

        Args:
            keyframe_id: ID of the keyframe to remove
        """
        # Remove from timestamp database
        if keyframe_id in self.keyframe_timestamps:
            del self.keyframe_timestamps[keyframe_id]

        # Remove from inverted index
        self.inverted_index.remove_keyframe(keyframe_id)

        # Remove from verified loops
        if keyframe_id in self.verified_loops:
            del self.verified_loops[keyframe_id]

        # Remove from recent candidates
        if keyframe_id in self.recent_candidates:
            del self.recent_candidates[keyframe_id]
