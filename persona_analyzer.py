import json
import logging
import re
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaAnalyzer:
    def __init__(self):
        """
        Initialize the Persona Analyzer for generating relevant queries and understanding persona requirements.
        """
        # Define persona categories and their characteristics
        self.persona_categories = {
            "researcher": {
                "keywords": ["researcher", "phd", "scientist", "academic", "postdoc", "professor"],
                "focus_areas": ["methodology", "results", "analysis", "literature", "findings", "conclusions", "data", "experiments"],
                "priority_sections": ["abstract", "methodology", "results", "discussion", "conclusion", "related work"],
                "query_patterns": [
                    "What are the key methodologies used?",
                    "What are the main findings and results?",
                    "How does this relate to existing research?",
                    "What are the limitations and future work?",
                    "What datasets and benchmarks were used?"
                ]
            },
            "student": {
                "keywords": ["student", "undergraduate", "graduate", "learner", "pupil"],
                "focus_areas": ["concepts", "examples", "fundamentals", "basics", "theory", "practice", "exercises"],
                "priority_sections": ["introduction", "examples", "summary", "key concepts", "exercises", "fundamentals"],
                "query_patterns": [
                    "What are the key concepts I need to understand?",
                    "Can you provide examples and illustrations?",
                    "What are the fundamental principles?",
                    "How does this apply in practice?",
                    "What should I focus on for exams?"
                ]
            },
            "analyst": {
                "keywords": ["analyst", "business", "investment", "financial", "consultant", "advisor"],
                "focus_areas": ["trends", "performance", "metrics", "strategy", "recommendations", "data", "insights"],
                "priority_sections": ["executive summary", "financial analysis", "market analysis", "recommendations", "conclusions"],
                "query_patterns": [
                    "What are the key performance indicators?",
                    "What trends and patterns emerge?",
                    "What are the strategic implications?",
                    "What recommendations can be made?",
                    "How does this compare to competitors?"
                ]
            },
            "practitioner": {
                "keywords": ["engineer", "developer", "practitioner", "professional", "specialist", "expert"],
                "focus_areas": ["implementation", "practical", "applications", "tools", "techniques", "solutions"],
                "priority_sections": ["implementation", "applications", "case studies", "best practices", "tools"],
                "query_patterns": [
                    "How can this be implemented in practice?",
                    "What tools and techniques are available?",
                    "What are the practical applications?",
                    "What are the best practices?",
                    "How do I solve specific problems?"
                ]
            }
        }
        
        # Define job category patterns
        self.job_categories = {
            "literature_review": {
                "patterns": ["literature review", "survey", "comprehensive review", "systematic review"],
                "focus": ["methodology", "findings", "gaps", "trends", "comparative analysis"],
                "sections": ["related work", "background", "methodology", "results", "discussion"]
            },
            "exam_preparation": {
                "patterns": ["exam", "test", "study", "preparation", "review for"],
                "focus": ["key concepts", "important topics", "examples", "practice problems"],
                "sections": ["summary", "key points", "examples", "exercises", "fundamentals"]
            },
            "business_analysis": {
                "patterns": ["analyze", "assessment", "evaluation", "performance", "trends"],
                "focus": ["metrics", "performance", "trends", "recommendations", "insights"],
                "sections": ["analysis", "results", "conclusions", "recommendations", "summary"]
            },
            "implementation": {
                "patterns": ["implement", "build", "develop", "create", "design"],
                "focus": ["methods", "techniques", "tools", "procedures", "best practices"],
                "sections": ["methodology", "implementation", "approach", "design", "architecture"]
            }
        }
        
        logger.info("PersonaAnalyzer initialized with predefined categories")
    
    def analyze_persona_requirements(self, persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """
        Analyze persona and job requirements to generate relevant queries and priorities.
        
        Args:
            persona: Description of the user persona
            job_to_be_done: Specific task the persona needs to accomplish
            
        Returns:
            Dictionary containing persona analysis and generated queries
        """
        try:
            logger.info(f"Analyzing persona: {persona}")
            logger.info(f"Job to be done: {job_to_be_done}")
            
            # Identify persona category
            persona_category = self._identify_persona_category(persona)
            
            # Identify job category
            job_category = self._identify_job_category(job_to_be_done)
            
            # Extract domain and expertise level
            domain_info = self._extract_domain_info(persona)
            
            # Generate relevant queries
            relevant_queries = self._generate_relevant_queries(persona, job_to_be_done, persona_category, job_category)
            
            # Determine section priorities
            section_priorities = self._determine_section_priorities(persona_category, job_category)
            
            # Generate keyword filters
            keyword_filters = self._generate_keyword_filters(persona, job_to_be_done, domain_info)
            
            analysis_result = {
                "persona_category": persona_category,
                "job_category": job_category,
                "domain_info": domain_info,
                "relevant_queries": relevant_queries,
                "section_priorities": section_priorities,
                "keyword_filters": keyword_filters,
                "search_strategy": self._generate_search_strategy(persona_category, job_category)
            }
            
            logger.info(f"Generated {len(relevant_queries)} relevant queries")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in persona analysis: {e}")
            # Return basic analysis as fallback
            return {
                "persona_category": "general",
                "job_category": "general",
                "relevant_queries": [job_to_be_done, persona],
                "section_priorities": ["introduction", "conclusion", "summary"],
                "keyword_filters": self._extract_keywords(persona + " " + job_to_be_done)
            }
    
    def _identify_persona_category(self, persona: str) -> str:
        """
        Identify the category of the persona based on keywords.
        """
        persona_lower = persona.lower()
        
        for category, info in self.persona_categories.items():
            for keyword in info["keywords"]:
                if keyword in persona_lower:
                    logger.info(f"Identified persona category: {category}")
                    return category
        
        # Try to infer from context
        if any(term in persona_lower for term in ["phd", "research", "academic", "study"]):
            return "researcher"
        elif any(term in persona_lower for term in ["business", "financial", "investment", "analyst"]):
            return "analyst"
        elif "student" in persona_lower:
            return "student"
        elif any(term in persona_lower for term in ["engineer", "developer", "professional"]):
            return "practitioner"
        
        logger.info("Using general persona category")
        return "general"
    
    def _identify_job_category(self, job_to_be_done: str) -> str:
        """
        Identify the category of the job based on patterns.
        """
        job_lower = job_to_be_done.lower()
        
        for category, info in self.job_categories.items():
            for pattern in info["patterns"]:
                if pattern in job_lower:
                    logger.info(f"Identified job category: {category}")
                    return category
        
        # Try to infer from context
        if any(term in job_lower for term in ["review", "survey", "comprehensive", "literature"]):
            return "literature_review"
        elif any(term in job_lower for term in ["study", "exam", "test", "learn"]):
            return "exam_preparation"
        elif any(term in job_lower for term in ["analyze", "assessment", "evaluate", "trends"]):
            return "business_analysis"
        elif any(term in job_lower for term in ["implement", "build", "develop", "create"]):
            return "implementation"
        
        logger.info("Using general job category")
        return "general"
    
    def _extract_domain_info(self, persona: str) -> Dict[str, Any]:
        """
        Extract domain and expertise information from persona description.
        """
        domain_info = {
            "domain": "general",
            "expertise_level": "intermediate",
            "specializations": []
        }
        
        persona_lower = persona.lower()
        
        # Identify domain
        domains = {
            "biology": ["biology", "biological", "life sciences", "biotechnology", "bioinformatics"],
            "chemistry": ["chemistry", "chemical", "molecular", "biochemistry"],
            "computer_science": ["computer science", "computing", "software", "ai", "machine learning", "data science"],
            "finance": ["finance", "financial", "investment", "banking", "economics"],
            "medicine": ["medicine", "medical", "healthcare", "clinical", "pharmaceutical"],
            "engineering": ["engineering", "technical", "mechanical", "electrical", "civil"],
            "business": ["business", "management", "marketing", "operations", "strategy"]
        }
        
        for domain, keywords in domains.items():
            if any(keyword in persona_lower for keyword in keywords):
                domain_info["domain"] = domain
                break
        
        # Identify expertise level
        if any(term in persona_lower for term in ["phd", "doctor", "professor", "senior", "expert", "lead"]):
            domain_info["expertise_level"] = "expert"
        elif any(term in persona_lower for term in ["master", "graduate", "experienced", "professional"]):
            domain_info["expertise_level"] = "advanced"
        elif any(term in persona_lower for term in ["undergraduate", "student", "junior", "beginner"]):
            domain_info["expertise_level"] = "beginner"
        
        # Extract specializations (words that might indicate specific areas)
        specialization_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', persona)
        domain_info["specializations"] = specialization_words[:3]  # Limit to top 3
        
        return domain_info
    
    def _generate_relevant_queries(self, persona: str, job_to_be_done: str, 
                                 persona_category: str, job_category: str) -> List[str]:
        """
        Generate multiple relevant queries based on persona and job analysis.
        """
        queries = []
        
        # Add primary query
        queries.append(f"{persona}: {job_to_be_done}")
        
        # Add job-specific queries
        if job_category in self.job_categories:
            job_info = self.job_categories[job_category]
            for focus_area in job_info["focus"]:
                queries.append(f"{focus_area} for {persona}")
                queries.append(f"{focus_area} {job_to_be_done}")
        
        # Add persona-specific queries
        if persona_category in self.persona_categories:
            persona_info = self.persona_categories[persona_category]
            for focus_area in persona_info["focus_areas"]:
                queries.append(f"{focus_area} relevant to {job_to_be_done}")
            
            # Add pattern-based queries
            for pattern in persona_info["query_patterns"]:
                queries.append(pattern)
        
        # Add keyword-based queries
        persona_keywords = self._extract_keywords(persona)
        job_keywords = self._extract_keywords(job_to_be_done)
        
        # Combine keywords for additional queries
        if persona_keywords and job_keywords:
            queries.append(" ".join(persona_keywords[:3] + job_keywords[:3]))
        
        # Add domain-specific queries
        domain_queries = self._generate_domain_specific_queries(persona, job_to_be_done)
        queries.extend(domain_queries)
        
        # Remove duplicates and filter out very short queries
        unique_queries = []
        seen = set()
        for query in queries:
            query_clean = query.strip().lower()
            if query_clean not in seen and len(query_clean) > 5:
                unique_queries.append(query)
                seen.add(query_clean)
        
        return unique_queries[:10]  # Limit to top 10 queries
    
    def _generate_domain_specific_queries(self, persona: str, job_to_be_done: str) -> List[str]:
        """
        Generate domain-specific queries based on identified domain.
        """
        queries = []
        persona_lower = persona.lower()
        job_lower = job_to_be_done.lower()
        
        # Academic/Research domain
        if any(term in persona_lower for term in ["researcher", "phd", "academic", "scientist"]):
            queries.extend([
                "research methodology and approaches",
                "experimental design and methods",
                "data analysis and results",
                "literature review and background",
                "conclusions and future work"
            ])
        
        # Business/Finance domain
        elif any(term in persona_lower for term in ["business", "analyst", "financial", "investment"]):
            queries.extend([
                "financial performance and metrics",
                "market analysis and trends",
                "strategic recommendations",
                "competitive analysis",
                "risk assessment and management"
            ])
        
        # Educational domain
        elif "student" in persona_lower:
            queries.extend([
                "key concepts and definitions",
                "examples and case studies",
                "practice problems and exercises",
                "summary and review materials",
                "fundamental principles"
            ])
        
        # Technical/Engineering domain
        elif any(term in persona_lower for term in ["engineer", "developer", "technical"]):
            queries.extend([
                "implementation details and methods",
                "technical specifications",
                "best practices and guidelines",
                "tools and technologies",
                "troubleshooting and solutions"
            ])
        
        return queries
    
    def _determine_section_priorities(self, persona_category: str, job_category: str) -> List[Dict[str, float]]:
        """
        Determine priority weights for different section types.
        """
        priorities = []
        
        # Base priorities for all personas
        base_priorities = {
            "abstract": 0.8,
            "summary": 0.8,
            "introduction": 0.7,
            "conclusion": 0.8,
            "conclusions": 0.8
        }
        
        # Add persona-specific priorities
        if persona_category in self.persona_categories:
            persona_sections = self.persona_categories[persona_category]["priority_sections"]
            for section in persona_sections:
                base_priorities[section] = base_priorities.get(section, 0.0) + 0.6
        
        # Add job-specific priorities
        if job_category in self.job_categories:
            job_sections = self.job_categories[job_category]["sections"]
            for section in job_sections:
                base_priorities[section] = base_priorities.get(section, 0.0) + 0.5
        
        # Convert to list format with normalized weights
        max_weight = max(base_priorities.values()) if base_priorities else 1.0
        for section, weight in base_priorities.items():
            normalized_weight = min(1.0, weight / max_weight)
            priorities.append({
                "section_type": section,
                "priority_weight": normalized_weight
            })
        
        # Sort by priority weight
        priorities.sort(key=lambda x: x["priority_weight"], reverse=True)
        
        return priorities
    
    def _generate_keyword_filters(self, persona: str, job_to_be_done: str, domain_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate keyword filters for content relevance.
        """
        filters = {
            "high_priority": [],
            "medium_priority": [],
            "domain_specific": [],
            "exclude": []
        }
        
        # Extract keywords from persona and job
        persona_keywords = self._extract_keywords(persona)
        job_keywords = self._extract_keywords(job_to_be_done)
        
        filters["high_priority"] = job_keywords[:5]
        filters["medium_priority"] = persona_keywords[:5]
        
        # Add domain-specific keywords
        domain = domain_info.get("domain", "general")
        domain_keywords = {
            "biology": ["gene", "protein", "cell", "molecular", "organism", "evolution", "dna", "rna"],
            "chemistry": ["molecule", "reaction", "compound", "synthesis", "catalyst", "bond", "element"],
            "computer_science": ["algorithm", "data", "model", "system", "network", "software", "programming"],
            "finance": ["revenue", "profit", "investment", "market", "stock", "financial", "economic"],
            "medicine": ["patient", "treatment", "diagnosis", "clinical", "therapy", "medical", "health"],
            "engineering": ["design", "system", "process", "optimization", "performance", "efficiency"]
        }
        
        if domain in domain_keywords:
            filters["domain_specific"] = domain_keywords[domain]
        
        # Add exclusion keywords (common noise)
        filters["exclude"] = [
            "figure", "table", "appendix", "bibliography", "reference", "citation",
            "page", "chapter", "section", "subsection", "paragraph",
            "copyright", "published", "journal", "conference", "proceedings"
        ]
        
        return filters
    
    def _generate_search_strategy(self, persona_category: str, job_category: str) -> Dict[str, Any]:
        """
        Generate a search strategy based on persona and job analysis.
        """
        strategy = {
            "search_approach": "balanced",
            "content_depth": "medium",
            "section_focus": "broad",
            "scoring_weights": {
                "content_relevance": 0.4,
                "section_importance": 0.3,
                "keyword_match": 0.2,
                "structural_position": 0.1
            }
        }
        
        # Adjust strategy based on persona
        if persona_category == "researcher":
            strategy.update({
                "search_approach": "comprehensive",
                "content_depth": "deep",
                "section_focus": "methodology_results",
                "scoring_weights": {
                    "content_relevance": 0.5,
                    "section_importance": 0.3,
                    "keyword_match": 0.15,
                    "structural_position": 0.05
                }
            })
        elif persona_category == "student":
            strategy.update({
                "search_approach": "educational",
                "content_depth": "medium",
                "section_focus": "examples_concepts",
                "scoring_weights": {
                    "content_relevance": 0.3,
                    "section_importance": 0.4,
                    "keyword_match": 0.2,
                    "structural_position": 0.1
                }
            })
        elif persona_category == "analyst":
            strategy.update({
                "search_approach": "targeted",
                "content_depth": "medium",
                "section_focus": "analysis_results",
                "scoring_weights": {
                    "content_relevance": 0.45,
                    "section_importance": 0.25,
                    "keyword_match": 0.25,
                    "structural_position": 0.05
                }
            })
        
        # Adjust strategy based on job
        if job_category == "literature_review":
            strategy["content_depth"] = "comprehensive"
            strategy["section_focus"] = "all_sections"
        elif job_category == "exam_preparation":
            strategy["section_focus"] = "key_concepts"
            strategy["scoring_weights"]["keyword_match"] = 0.3
        
        return strategy
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text using improved techniques.
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Extract words (3+ characters, exclude pure numbers)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words
        keywords = [w for w in words if w not in stop_words]
        
        # Get word frequency and return most common
        from collections import Counter
        word_counts = Counter(keywords)
        
        # Return most frequent keywords (up to 8)
        return [word for word, _ in word_counts.most_common(8)]
    
    def generate_section_queries(self, persona: str, job_to_be_done: str, section_title: str, section_content: str = "") -> List[str]:
        """
        Generate specific queries for evaluating a particular section's relevance.
        
        Args:
            persona: User persona description
            job_to_be_done: Specific task description
            section_title: Title of the section being evaluated
            section_content: Preview of section content (optional)
            
        Returns:
            List of queries tailored for this specific section
        """
        queries = []
        
        # Primary query
        queries.append(f"How relevant is '{section_title}' for {persona} to {job_to_be_done}?")
        
        # Content-based queries
        if section_content:
            content_keywords = self._extract_keywords(section_content)
            if content_keywords:
                queries.append(f"Does this section about {' '.join(content_keywords[:3])} help with {job_to_be_done}?")
        
        # Section-specific queries based on title
        title_lower = section_title.lower()
        
        if any(term in title_lower for term in ["method", "approach", "technique"]):
            queries.append(f"Are these methods relevant for {persona}'s work on {job_to_be_done}?")
        elif any(term in title_lower for term in ["result", "finding", "outcome"]):
            queries.append(f"Do these results support {persona}'s need to {job_to_be_done}?")
        elif any(term in title_lower for term in ["introduction", "background", "overview"]):
            queries.append(f"Does this background information help {persona} understand {job_to_be_done}?")
        elif any(term in title_lower for term in ["conclusion", "discussion", "summary"]):
            queries.append(f"Are these conclusions relevant to {persona}'s goal of {job_to_be_done}?")
        
        return queries[:5]  # Limit to top 5 queries
    
    def evaluate_section_match(self, persona_analysis: Dict[str, Any], section_title: str, section_content: str = "") -> float:
        """
        Evaluate how well a section matches the persona requirements.
        
        Args:
            persona_analysis: Result from analyze_persona_requirements
            section_title: Title of the section
            section_content: Content of the section (optional)
            
        Returns:
            Relevance score between 0 and 1
        """
        score = 0.0
        title_lower = section_title.lower()
        content_lower = section_content.lower() if section_content else ""
        
        # Check section priorities
        section_priorities = persona_analysis.get("section_priorities", [])
        for priority in section_priorities:
            section_type = priority["section_type"]
            if section_type in title_lower:
                score += priority["priority_weight"] * 0.4
        
        # Check keyword filters
        keyword_filters = persona_analysis.get("keyword_filters", {})
        
        # High priority keywords
        high_priority = keyword_filters.get("high_priority", [])
        for keyword in high_priority:
            if keyword.lower() in title_lower:
                score += 0.3
            elif keyword.lower() in content_lower:
                score += 0.15
        
        # Medium priority keywords
        medium_priority = keyword_filters.get("medium_priority", [])
        for keyword in medium_priority:
            if keyword.lower() in title_lower:
                score += 0.2
            elif keyword.lower() in content_lower:
                score += 0.1
        
        # Domain-specific keywords
        domain_specific = keyword_filters.get("domain_specific", [])
        for keyword in domain_specific:
            if keyword.lower() in title_lower or keyword.lower() in content_lower:
                score += 0.1
        
        # Exclude keywords (negative scoring)
        exclude_keywords = keyword_filters.get("exclude", [])
        for keyword in exclude_keywords:
            if keyword.lower() in title_lower:
                score -= 0.2
        
        return min(1.0, max(0.0, score))


# Usage example and testing
# if __name__ == "__main__":
#     # Test the persona analyzer
#     analyzer = PersonaAnalyzer()
    
#     # Test case 1: Academic researcher
#     persona1 = "PhD Researcher in Computational Biology"
#     job1 = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    
#     print("="*60)
#     print("TEST CASE 1: Academic Researcher")
#     print("="*60)
    
#     result1 = analyzer.analyze_persona_requirements(persona1, job1)
#     print(f"Persona Category: {result1['persona_category']}")
#     print(f"Job Category: {result1['job_category']}")
#     print(f"Domain Info: {result1['domain_info']}")
#     print(f"Generated Queries ({len(result1['relevant_queries'])}):")
#     for i, query in enumerate(result1['relevant_queries'], 1):
#         print(f"  {i}. {query}")
    
#     print(f"\nSection Priorities:")
#     for priority in result1['section_priorities'][:5]:
#         print(f"  {priority['section_type']}: {priority['priority_weight']:.2f}")
    
#     # Test case 2: Business analyst
#     persona2 = "Investment Analyst"
#     job2 = "Analyze revenue trends, R&D investments, and market positioning strategies"
    
#     print("\n" + "="*60)
#     print("TEST CASE 2: Business Analyst")
#     print("="*60)
    
#     result2 = analyzer.analyze_persona_requirements(persona2, job2)
#     print(f"Persona Category: {result2['persona_category']}")
#     print(f"Job Category: {result2['job_category']}")
#     print(f"Generated Queries ({len(result2['relevant_queries'])}):")
#     for i, query in enumerate(result2['relevant_queries'], 1):
#         print(f"  {i}. {query}")
    
#     # Test section evaluation
#     print("\n" + "="*60)
#     print("SECTION EVALUATION TEST")
#     print("="*60)
    
#     test_sections = [
#         ("Methodology", "Our research methodology combines quantitative analysis with machine learning approaches..."),
#         ("Financial Performance", "The company's revenue grew by 15% year-over-year, with significant improvements in..."),
#         ("References", "1. Smith, J. (2023). Advanced techniques in computational biology...")
#     ]
    
#     for section_title, section_content in test_sections:
#         score1 = analyzer.evaluate_section_match(result1, section_title, section_content)
#         score2 = analyzer.evaluate_section_match(result2, section_title, section_content)
#         print(f"\nSection: '{section_title}'")
#         print(f"  Researcher relevance: {score1:.3f}")
#         print(f"  Analyst relevance: {score2:.3f}")