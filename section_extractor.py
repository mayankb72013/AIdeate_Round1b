import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import re
from collections import defaultdict
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SectionExtractor:
    def __init__(self, model_name: str = "./models/models--cross-encoder--ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Section Extractor with cross-encoder model for relevance scoring.
        
        Args:
            model_name: Cross-encoder model for relevance scoring
        """
        self.cross_encoder = None
        self.sentence_model = None
        
        try:
            # Try to load cross-encoder model if available
            if os.path.exists(model_name):
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(model_name)
                logger.info(f"Loaded cross-encoder model: {model_name}")
            else:
                logger.warning(f"Cross-encoder model not found at {model_name}, using fallback scoring")
                
        except Exception as e:
            logger.warning(f"Error loading cross-encoder model: {e}, using fallback scoring")
            self.cross_encoder = None
        
        # REMOVED: This line was causing network calls
        # self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("SectionExtractor initialized successfully")
    
    def extract_relevant_sections(self, documents_data: Dict, persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """
        Extract and rank document sections based on persona and job-to-be-done.
        
        Args:
            documents_data: Processed document data from document_processor.py
            persona: Description of the user persona
            job_to_be_done: Specific task the persona needs to accomplish
            
        Returns:
            Dictionary with extracted sections and analysis
        """
        try:
            logger.info("Starting section extraction and ranking")
            
            # Generate relevance queries from persona and job
            relevance_queries = self._generate_relevance_queries(persona, job_to_be_done)
            
            # Extract all sections from documents
            all_sections = self._extract_all_sections(documents_data)
            
            # Score sections based on relevance
            scored_sections = self._score_sections(all_sections, relevance_queries, persona, job_to_be_done)
            
            # Rank and filter top sections
            top_sections = self._rank_and_filter_sections(scored_sections)
            
            # Perform sub-section analysis
            sub_section_analysis = self._analyze_sub_sections(top_sections, documents_data, persona, job_to_be_done)
            
            # Generate output in required format
            output = self._format_output(
                documents_data, persona, job_to_be_done, 
                top_sections, sub_section_analysis
            )
            
            logger.info(f"Extracted {len(top_sections)} relevant sections")
            return output
            
        except Exception as e:
            logger.error(f"Error in section extraction: {e}")
            raise
    
    def _generate_relevance_queries(self, persona: str, job_to_be_done: str) -> List[str]:
        """
        Generate multiple query variations for better relevance matching.
        """
        queries = [
            f"{persona} needs to {job_to_be_done}",
            job_to_be_done,
            persona,
        ]
        
        # Extract key terms from job description
        job_keywords = self._extract_keywords(job_to_be_done)
        persona_keywords = self._extract_keywords(persona)
        
        # Add keyword-based queries
        if job_keywords:
            queries.append(" ".join(job_keywords))
        if persona_keywords:
            queries.append(" ".join(persona_keywords))
        
        # Add domain-specific queries based on persona
        if "researcher" in persona.lower() or "phd" in persona.lower():
            queries.extend([
                "methodology research methods",
                "results findings analysis",
                "literature review background",
                "datasets benchmarks performance",
                "experimental design approach"
            ])
        elif "student" in persona.lower():
            queries.extend([
                "learning concepts fundamentals",
                "examples practice problems",
                "key topics important points"
            ])
        elif "analyst" in persona.lower() or "business" in persona.lower():
            queries.extend([
                "data analysis trends",
                "performance metrics results",
                "strategic recommendations"
            ])
        
        return list(set(queries))  # Remove duplicates
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Return most frequent keywords (up to 5)
        from collections import Counter
        return [word for word, _ in Counter(keywords).most_common(5)]
    
    def _extract_all_sections(self, documents_data: Dict) -> List[Dict]:
        """
        Extract all sections from all documents with their content.
        """
        all_sections = []
        
        for doc_key, doc_data in documents_data.get("documents", {}).items():
            if "sections" not in doc_data:
                continue
                
            for section in doc_data["sections"]:
                # Combine all content from all pages of this section
                section_content = ""
                content_pages = section.get("content_pages", {})
                
                for page_key, page_data in content_pages.items():
                    section_content += page_data.get("content", "") + " "
                
                section_content = section_content.strip()
                
                if section_content and len(section_content) > 30:
                    section_info = {
                        "document": doc_key,
                        "section_title": section["heading"]["text"],
                        "level": section["heading"]["level"],
                        "page_number": section["heading"]["page"],
                        "content": section_content,
                        "word_count": len(section_content.split()),
                        "content_pages": content_pages
                    }
                    all_sections.append(section_info)
        
        logger.info(f"Extracted {len(all_sections)} sections from all documents")
        return all_sections
    
    def _score_sections(self, sections: List[Dict], queries: List[str], persona: str, job_to_be_done: str) -> List[Dict]:
        """
        Score sections based on relevance to persona and job.
        Uses cross-encoder if available, otherwise falls back to keyword matching.
        """
        scored_sections = []
        
        # Primary query combining persona and job
        primary_query = f"{persona}: {job_to_be_done}"
        
        for section in sections:
            try:
                # Prepare text for scoring (title + content preview)
                section_text = f"{section['section_title']}. {section['content'][:300]}"
                
                if self.cross_encoder:
                    # Use cross-encoder if available
                    primary_score = self._score_with_cross_encoder(primary_query, section_text)
                    query_scores = [self._score_with_cross_encoder(query, section_text) for query in queries[:3]]
                else:
                    # Use fallback keyword scoring
                    primary_score = self._score_with_keywords(primary_query, section_text)
                    query_scores = [self._score_with_keywords(query, section_text) for query in queries[:3]]
                
                # Calculate combined relevance score
                avg_query_score = np.mean(query_scores) if query_scores else 0.0
                combined_score = (primary_score * 0.6) + (avg_query_score * 0.4)
                
                # Add section-level importance
                section_importance = self._calculate_section_importance(section, persona, job_to_be_done)
                
                # Final score
                final_score = (combined_score * 0.7) + (section_importance * 0.3)
                
                section_copy = section.copy()
                section_copy.update({
                    "relevance_score": final_score,
                    "primary_score": primary_score,
                    "query_scores": query_scores,
                    "section_importance": section_importance
                })
                
                scored_sections.append(section_copy)
                
            except Exception as e:
                logger.warning(f"Error scoring section '{section['section_title']}': {e}")
                # Add with minimal score to avoid losing sections
                section_copy = section.copy()
                section_copy["relevance_score"] = 0.2
                scored_sections.append(section_copy)
        
        return scored_sections
    
    def _score_with_cross_encoder(self, query: str, text: str) -> float:
        """Score using cross-encoder model."""
        try:
            score_raw = self.cross_encoder.predict([query, text])
            if isinstance(score_raw, (list, np.ndarray)):
                score = float(score_raw[0])
            else:
                score = float(score_raw)
            
            # Normalize score to 0-1 range
            if score < 0:
                score = 1.0 / (1.0 + np.exp(-score))  # Sigmoid for negative scores
            elif score > 1:
                score = min(1.0, score / 10.0)  # Scale down if too large
                
            return score
        except Exception as e:
            logger.warning(f"Cross-encoder prediction failed: {e}")
            return self._score_with_keywords(query, text)
    
    def _score_with_keywords(self, query: str, text: str) -> float:
        """Fallback keyword-based scoring."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        # Calculate overlap
        overlap = len(query_words.intersection(text_words))
        max_possible = len(query_words)
        
        if max_possible == 0:
            return 0.0
        
        return overlap / max_possible
    
    def _calculate_section_importance(self, section: Dict, persona: str, job_to_be_done: str) -> float:
        """
        Calculate intrinsic importance of a section based on its characteristics.
        """
        importance = 0.1  # Base importance for all sections
        title_lower = section["section_title"].lower()
        
        # High importance sections for academic/research personas
        if any(term in persona.lower() for term in ["researcher", "phd", "academic", "scientist"]):
            if any(term in title_lower for term in ["methodology", "method", "approach", "results", "findings", "analysis", "discussion", "conclusion"]):
                importance += 0.7
            elif any(term in title_lower for term in ["introduction", "background", "literature", "related work"]):
                importance += 0.5
            elif any(term in title_lower for term in ["abstract", "summary"]):
                importance += 0.6
            elif any(term in title_lower for term in ["dataset", "benchmark", "performance", "evaluation", "experiment"]):
                importance += 0.8
        
        # High importance sections for business/analyst personas
        elif any(term in persona.lower() for term in ["analyst", "business", "investment", "financial"]):
            if any(term in title_lower for term in ["financial", "revenue", "performance", "market", "strategy", "analysis", "results"]):
                importance += 0.7
            elif any(term in title_lower for term in ["executive summary", "overview", "conclusion", "recommendation"]):
                importance += 0.6
        
        # High importance sections for student personas
        elif "student" in persona.lower():
            if any(term in title_lower for term in ["example", "practice", "exercise", "summary", "key", "important", "fundamental"]):
                importance += 0.7
            elif any(term in title_lower for term in ["introduction", "basic", "overview", "concept"]):
                importance += 0.5
        
        # Job-specific keywords
        job_keywords = self._extract_keywords(job_to_be_done)
        for keyword in job_keywords:
            if keyword.lower() in title_lower:
                importance += 0.4
        
        # Section level importance (H1 > H2 > H3)
        if section["level"] == "H1":
            importance += 0.2
        elif section["level"] == "H2":
            importance += 0.1
        
        # Content length bonus
        if section["word_count"] > 500:
            importance += 0.1
        elif section["word_count"] > 200:
            importance += 0.05
        
        return min(1.0, importance)
    
    def _rank_and_filter_sections(self, scored_sections: List[Dict], max_sections: int = 20) -> List[Dict]:
        """
        Rank sections by relevance score and filter to top sections.
        """
        # Sort by relevance score (descending)
        ranked_sections = sorted(scored_sections, key=lambda x: x["relevance_score"], reverse=True)
        
        # Lower threshold for better coverage
        min_score = 0.15
        filtered_sections = [s for s in ranked_sections if s["relevance_score"] >= min_score]
        
        # If still no sections, take top 10 regardless of score
        if not filtered_sections and ranked_sections:
            logger.warning("No sections passed minimum threshold, taking top 10 by score")
            filtered_sections = ranked_sections[:10]
        
        # Limit to max_sections
        top_sections = filtered_sections[:max_sections]
        
        # Add importance rank
        for i, section in enumerate(top_sections, 1):
            section["importance_rank"] = i
        
        if top_sections:
            logger.info(f"Ranked sections, top score: {top_sections[0]['relevance_score']:.3f}, threshold: {min_score}")
        else:
            logger.warning("No sections found after filtering")
        
        return top_sections
    
    def _analyze_sub_sections(self, top_sections: List[Dict], documents_data: Dict, persona: str, job_to_be_done: str) -> List[Dict]:
        """
        Perform detailed sub-section analysis on top sections.
        """
        sub_section_analysis = []
        
        for section in top_sections:
            try:
                # Extract key sentences/paragraphs from the section content
                refined_text = self._extract_key_content(section["content"], persona, job_to_be_done)
                
                analysis = {
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "refined_text": refined_text,
                    "page_number": section["page_number"],
                    "relevance_score": section["relevance_score"],
                    "word_count": len(refined_text.split())
                }
                
                sub_section_analysis.append(analysis)
                
            except Exception as e:
                logger.warning(f"Error in sub-section analysis for '{section['section_title']}': {e}")
                # Add basic analysis
                analysis = {
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "refined_text": section["content"][:500] + "..." if len(section["content"]) > 500 else section["content"],
                    "page_number": section["page_number"]
                }
                sub_section_analysis.append(analysis)
        
        return sub_section_analysis
    
    def _extract_key_content(self, content: str, persona: str, job_to_be_done: str, max_sentences: int = 3) -> str:
        """
        Extract the most relevant sentences from section content.
        """
        try:
            # Split content into sentences
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            # If content is reasonable length, return as-is
            if len(sentences) <= max_sentences or len(content) <= 800:
                return content[:800] + "..." if len(content) > 800 else content
            
            # Use keyword matching for sentence selection
            query_keywords = set(self._extract_keywords(f"{persona} {job_to_be_done}"))
            
            sentence_scores = []
            for sentence in sentences:
                sentence_keywords = set(self._extract_keywords(sentence))
                # Simple overlap score
                overlap = len(query_keywords.intersection(sentence_keywords))
                sentence_scores.append((sentence, overlap))
            
            # Sort by score and take top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
            
            # Reconstruct text maintaining original order
            selected_sentences = [s[0] for s in top_sentences]
            original_order_sentences = []
            
            for sentence in sentences:
                if sentence in selected_sentences:
                    original_order_sentences.append(sentence)
            
            result = ". ".join(original_order_sentences) + "."
            return result[:800] + "..." if len(result) > 800 else result
            
        except Exception as e:
            logger.warning(f"Error extracting key content: {e}")
            # Return truncated content as fallback
            return content[:800] + "..." if len(content) > 800 else content
    
    def _format_output(self, documents_data: Dict, persona: str, job_to_be_done: str, 
                      top_sections: List[Dict], sub_section_analysis: List[Dict]) -> Dict[str, Any]:
        """
        Format the output according to the required JSON structure.
        """
        from datetime import datetime
        
        # Get input document names
        input_documents = list(documents_data.get("documents", {}).keys())
        if input_documents:
            input_documents = [f"{doc}.pdf" for doc in input_documents]
        
        # Format extracted sections
        extracted_sections = []
        for section in top_sections:
            extracted_sections.append({
                "document": f"{section['document']}.pdf",
                "page_number": section["page_number"],
                "section_title": section["section_title"],
                "importance_rank": section["importance_rank"]
            })
        
        # Format sub-section analysis
        formatted_sub_analysis = []
        for analysis in sub_section_analysis:
            formatted_sub_analysis.append({
                "document": f"{analysis['document']}.pdf",
                "section_title": analysis["section_title"],
                "refined_text": analysis["refined_text"],
                "page_number": analysis["page_number"]
            })
        
        output = {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat() + "Z"
            },
            "extracted_sections": extracted_sections,
            "sub_section_analysis": formatted_sub_analysis
        }
        
        return output