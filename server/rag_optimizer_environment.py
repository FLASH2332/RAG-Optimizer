# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rag Optimizer Environment Implementation.
The agent acts as a Data Engineer to un-block a broken RAG pipeline.
"""

from copy import deepcopy
from uuid import uuid4
from typing import Dict, Any, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Import scikit-learn for our Grader (fallback/legacy)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Hybrid Search imports
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

try:
    from models import RagOptimizerAction, RagOptimizerObservation
except ImportError:
    from models import RagOptimizerAction, RagOptimizerObservation


class RagOptimizerEnvironment(Environment):
    """
    RAG Optimizer Engine.
    Maintains a simulated Knowledge Base and grades it using TF-IDF.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.kb = {}
        self.test_suite = []
        self.dense_vectors = {}
        self._setup_task("easy")
        
    def _setup_task(self, task_id: str):
        if task_id == "easy":
            self.kb = {
                "doc_pricing_legacy": {
                    "text": "Pricing for 2021: Enterprise tier is $1000/mo. Standard is $500/mo. All plans include 10 users.",
                    "metadata": {"type": "pricing"}
                },
                "doc_pricing_current_v2": {
                    "text": "Current Pricing 2024: Enterprise is $1500/mo. Standard is $750/mo. Refunds are not permitted on the enterprise tier.",
                    "metadata": {}
                },
            }
            # Add 20 random distractors
            for i in range(20):
                self.kb[f"doc_distractor_{i}"] = {"text": f"Weekly team update notes. Discussed the weather and the upcoming launch {i}.", "metadata":{}}

            self.test_suite = [
                {"query": "What is the current 2024 price for standard?", "target_concept": "750/mo"},
                {"query": "What is the refund policy for enterprise?", "target_concept": "Refunds are not permitted"}
            ]
            
        elif task_id == "medium":
            self.kb = {
                "doc_messy_support_ticket_1": {
                    "text": "User complained the button disappeared on the frontend. Another user said the database latency was high. The frontend team fixed the button by updating CSS.",
                    "metadata": {}
                },
                "doc_messy_support_ticket_2": {
                    "text": "Email integration is failing with error 401 Unauthorized. The API key was rotated on Tuesday.",
                    "metadata": {}
                },
            }
            # Add 20 engineering distractors
            for i in range(20):
                self.kb[f"doc_distractor_eng_{i}"] = {"text": f"Engineering architecture decision record {i}. We decided to use Postgres because of scaling concerns.", "metadata":{}}

            self.test_suite = [
                {"query": "UI rendering failure missing elements", "target_concept": "fixed the button by updating CSS", "required_metadata_key": "status", "required_metadata_value": "resolved"},
                {"query": "Authentication rejection credentials", "target_concept": "API key was rotated", "required_metadata_key": "status", "required_metadata_value": "investigating"}
            ]
            
        elif task_id == "hard":
            self.kb = {
                "doc_monolithic_onboarding": {
                    "text": "Welcome to the company! Here are some rules. 1) VPN access requires DUO. 2) The cafetaria opens at 8 AM. 3) For HR issues, email hr@company.com. 4) The 2024 holiday schedule includes Dec 25, Jan 1, and July 4. 5) Parking passes must be renewed annually in March. 6) All internal shipments to remote branch offices take 5-7 business days. Overnight shipping is only available for C-suite.",
                    "metadata": {}
                }
            }
            # Vector Poisoning: Add 50 adversarial distractors that mention "shipping", "holidays", "parking" but contain the WRONG semantic logic.
            for i in range(25):
                self.kb[f"adv_shipping_{i}"] = {"text": f"Shipping update {i}: We typically do not send to branch offices unless it takes 1-{i} days. Overnight is default.", "metadata":{}}
                self.kb[f"adv_parking_{i}"] = {"text": f"Parking passes are usually handled in {['January', 'February', 'April', 'May'][i%4]}. Renewals {i} are manual.", "metadata":{}}

            self.test_suite = [
                {"query": "How long does logistics transit take to branch offices?", "target_concept": "5-7 business days"},
                {"query": "When are vehicular parking permits processed?", "target_concept": "annually in March"},
                {"query": "Which festive days are non-working?", "target_concept": "July 4"}
            ]
            
        else:
            self._setup_task("easy")

        self._rebuild_cache()

    def _rebuild_cache(self):
        """Called whenever KB documents are added, removed, or updated."""
        if not self.kb:
            self.dense_vectors = {}
            return
            
        doc_ids = list(self.kb.keys())
        # Append metadata to text for embedding
        doc_texts = [(self.kb[d]["text"] + " " + " ".join(self.kb[d]["metadata"].values())).strip() for d in doc_ids]
        
        vectors = self.embedding_model.encode(doc_texts, convert_to_tensor=False)
        self.dense_vectors = {doc_id: vectors[i] for i, doc_id in enumerate(doc_ids)}

    def _get_kb_summary(self) -> Dict[str, Dict]:
        """Returns a summary of the KB for the observation."""
        summary = {}
        for k, v in self.kb.items():
            summary[k] = {"metadata": v.get("metadata", {}), "length": len(v.get("text", ""))}
        return summary

    def reset(self, **kwargs) -> RagOptimizerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        task_id = kwargs.get("task_id") or kwargs.get("task") or "easy"
        self._setup_task(task_id)
        
        return RagOptimizerObservation(
            message=f"RagOptimizerEnv Initialized for task: {task_id}. Resolve conflicts, append metadata, or splinter chunks to win.",
            current_docs=self._get_kb_summary(),
            done=False,
            reward=self._evaluate_kb()
        )

    def _evaluate_kb(self) -> float:
        """The Grader: Evaluates the current KB using Hybrid RRF (BM25 + Semantic MRR)."""
        if not self.kb or not self.test_suite:
            return 0.0
            
        doc_ids = list(self.kb.keys())
        doc_texts = [(self.kb[d]["text"] + " " + " ".join(self.kb[d]["metadata"].values())).strip() for d in doc_ids]
        
        # 1. BM25 Corpus Preparation
        tokenized_corpus = [doc.lower().split() for doc in doc_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Dense Matrix
        doc_vectors = np.array([self.dense_vectors[d] for d in doc_ids])
        
        mrr_sum = 0.0
        
        for case in self.test_suite:
            # BM25 Search
            tokenized_query = case["query"].lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_ranks = bm25_scores.argsort()[::-1]
            
            # Dense Search
            query_vec = self.embedding_model.encode(case["query"])
            dense_scores = cosine_similarity([query_vec], doc_vectors)[0]
            dense_ranks = dense_scores.argsort()[::-1]
            
            # Reciprocal Rank Fusion (RRF)
            rrf_scores = {d: 0.0 for d in doc_ids}
            k = 60
            for rank, idx in enumerate(bm25_ranks):
                rrf_scores[doc_ids[idx]] += 1.0 / (k + rank + 1)
            for rank, idx in enumerate(dense_ranks):
                rrf_scores[doc_ids[idx]] += 1.0 / (k + rank + 1)
                
            # Grade Top-K fused list using MRR
            ranked_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            
            req_meta = case.get("required_metadata_key")
            req_meta_val = case.get("required_metadata_value")
            
            case_mrr = 0.0
            for i, doc_id in enumerate(ranked_doc_ids):
                # Is this the true doc?
                doc_text = self.kb[doc_id]["text"].lower()
                if case["target_concept"].lower() in doc_text:
                    valid = True
                    
                    # Medium Task Semantic Check
                    if req_meta and req_meta_val:
                        if self.kb[doc_id]["metadata"].get(req_meta) != req_meta_val:
                            valid = False
                            
                    if valid:
                        case_mrr = 1.0 / (i + 1)  # MRR formula starts at rank 1
                        break
            
            mrr_sum += case_mrr
            
        base_reward = float(mrr_sum / len(self.test_suite))
        
        # Step Cost Penalty calculation (-0.01 per step)
        cost_penalty = self._state.step_count * 0.01
        
        return max(0.0, base_reward - cost_penalty)

    def step(self, action: RagOptimizerAction) -> RagOptimizerObservation:  # type: ignore[override]
        self._state.step_count += 1
        
        msg = ""
        done = False
        reward = 0.0
        
        try:
            if action.action_type == "read_document":
                if action.doc_id in self.kb:
                    msg = f"Content of {action.doc_id}: {self.kb[action.doc_id]['text']}"
                else:
                    msg = f"Error: doc_id {action.doc_id} not found."
            
            elif action.action_type == "delete_document":
                if action.doc_id in self.kb:
                    del self.kb[action.doc_id]
                    self._rebuild_cache()
                    msg = f"Deleted {action.doc_id}."
                else:
                    msg = f"Error: doc_id {action.doc_id} not found."
                    
            elif action.action_type == "update_document":
                if not action.doc_id or not action.text:
                    msg = "Error: doc_id and text required for update_document."
                else:
                    if action.doc_id not in self.kb:
                        self.kb[action.doc_id] = {"text": "", "metadata": {}}
                    self.kb[action.doc_id]["text"] = action.text
                    self._rebuild_cache()
                    msg = f"Updated text for {action.doc_id}."
                    
            elif action.action_type == "add_metadata":
                if not action.doc_id or not action.metadata_key or not action.metadata_value:
                    msg = "Error: doc_id, metadata_key, and metadata_value required."
                else:
                    if action.doc_id not in self.kb:
                        msg = f"Error: doc_id {action.doc_id} not found."
                    else:
                        self.kb[action.doc_id]["metadata"][action.metadata_key] = action.metadata_value
                        self._rebuild_cache()
                        msg = f"Added metadata to {action.doc_id}."
                        
            elif action.action_type == "submit":
                done = True
                reward = self._evaluate_kb()
                msg = f"Evaluation complete. Final reward: {reward:.2f}"
                
        except Exception as e:
            msg = f"Action failed: {str(e)}"

        if not done:
            reward = self._evaluate_kb()

        return RagOptimizerObservation(
            message=msg,
            current_docs=self._get_kb_summary(),
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
