import fitz  # PyMuPDF
import json
import os
import re
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional


class DocumentProcessor:
    def __init__(self):
        # Enhanced heading patterns from your 1A code
        self.strong_heading_patterns = [
            r'^([IVXLC]+)\.\s*[A-Z]',           # Roman numerals: I. INTRODUCTION
            r'^([IVXLC]+)\s+[A-Z]',             # Roman without dot: I INTRODUCTION  
            r'^(\d+)\.\s*[A-Z]',                # Numbers: 1. Introduction, 2. Overview
            r'^Chapter\s+(\d+)',                # Chapter 1, Chapter 2
            r'^Section\s+(\d+)',                # Section 1, Section 2
            r'^Part\s+([IVXLC\d]+)',            # Part I, Part II
        ]
        
        self.medium_heading_patterns = [
            r'^(\d+\.\d+)\s*[A-Z]',             # Sub-numbers: 1.1 Overview, 2.3 Details
            r'^(\d+\.\d+\.\d+)\s*[A-Z]',        # Sub-sub: 1.1.1 Details
        ]
        
        self.weak_heading_patterns = [
            r'^([A-Z])\.\s*[A-Z]',              # Letters: A. Background, B. Methods
            r'^(\d+)\)\s*[A-Z]',                # 1) Introduction
            r'^\((\d+)\)\s*[A-Z]',              # (1) Introduction
            r'^([A-Z])\)\s*[A-Z]',              # A) Background
            r'^\(([A-Z])\)\s*[A-Z]',            # (A) Background
        ]
        
        # Enhanced section keywords with academic focus
        self.section_keywords = {
            'abstract': 0.8, 'introduction': 0.9, 'background': 0.7,
            'literature review': 0.8, 'related work': 0.7,
            'methodology': 0.8, 'method': 0.8, 'methods': 0.8,
            'approach': 0.6, 'framework': 0.6, 'model': 0.6,
            'implementation': 0.7, 'design': 0.7, 'architecture': 0.7,
            'experiment': 0.7, 'experiments': 0.7, 'evaluation': 0.7,
            'results': 0.8, 'findings': 0.7, 'analysis': 0.7,
            'discussion': 0.8, 'conclusion': 0.9, 'conclusions': 0.9,
            'future work': 0.7, 'limitations': 0.6,
            'references': 0.8, 'bibliography': 0.7,
            'acknowledgment': 0.6, 'acknowledgments': 0.6,
            'appendix': 0.6, 'appendices': 0.6,
        }
        
        # Title-specific keywords
        self.title_keywords = {
            'analysis', 'study', 'research', 'investigation', 'approach', 'system',
            'framework', 'model', 'design', 'development', 'implementation',
            'evaluation', 'assessment', 'review', 'survey', 'comprehensive',
            'novel', 'new', 'improved', 'enhanced', 'advanced', 'efficient',
            'application', 'using', 'based', 'machine learning', 'deep learning'
        }
        
        # Exclusion patterns
        self.exclusion_patterns = [
            r'.*@.*\.(com|org|edu|gov)',         # Email addresses
            r'.*\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # Phone numbers
            r'.*(university|college|institute|department).*',  # Institutions
            r'.*(copyright|©|\(c\)).*',          # Copyright
            r'.*(published|received|accepted|revised).*',  # Publication dates
            r'^page\s+\d+',                      # Page numbers
            r'^\d+\s*$',                         # Standalone numbers
            r'^figure\s+\d+',                    # Figure captions
            r'^table\s+\d+',                     # Table captions
            r'^www\.',                           # Websites
            r'^http[s]?://',                     # URLs
        ]

    def clean_special_characters(self, text: str) -> str:
        """Remove special characters from text while preserving basic punctuation."""
        cleaned = re.sub(r'[^\w\s\.\,\:\;\-\'\(\)]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def is_code_or_formula(self, text: str) -> bool:
        """Check if text contains parentheses with non-numeric content (likely code/formula)."""
        parentheses_content = re.findall(r'\(([^)]+)\)', text)
        if not parentheses_content:
            return False
            
        for content in parentheses_content:
            content = content.strip()
            if not (content.isdigit() or (len(content) == 1 and content.isalpha())):
                return True
        return False

    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract comprehensive text blocks with detailed metadata including line spacing."""
        doc = fitz.open(pdf_path)
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            page_width = page.rect.width
            
            page_blocks = []
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if line["spans"]:
                            line_text = ""
                            main_span = line["spans"][0]
                            
                            for span in line["spans"]:
                                line_text += span["text"]
                            
                            line_text = line_text.strip()
                            if line_text and len(line_text) > 1:
                                bbox = main_span["bbox"]
                                rel_y = bbox[1] / page_height if page_height > 0 else 0
                                rel_x = bbox[0] / page_width if page_width > 0 else 0
                                
                                block_data = {
                                    "text": line_text,
                                    "page": page_num + 1,
                                    "font_size": main_span["size"],
                                    "font_name": main_span["font"],
                                    "flags": main_span["flags"],
                                    "bbox": bbox,
                                    "x": bbox[0],
                                    "y": bbox[1],
                                    "width": bbox[2] - bbox[0],
                                    "height": bbox[3] - bbox[1],
                                    "rel_y": rel_y,
                                    "rel_x": rel_x,
                                    "is_bold": bool(main_span["flags"] & 16),
                                    "is_italic": bool(main_span["flags"] & 2),
                                    "page_height": page_height,
                                    "page_width": page_width
                                }
                                page_blocks.append(block_data)
            
            # Sort blocks by Y position for this page
            page_blocks.sort(key=lambda x: (x["y"], x["x"]))
            
            # Calculate line spacing for each block
            for i, block in enumerate(page_blocks):
                spacing_before = 0
                if i > 0:
                    prev_block = page_blocks[i-1]
                    if abs(block["x"] - prev_block["x"]) < 50:
                        spacing_before = block["y"] - (prev_block["y"] + prev_block["height"])
                
                spacing_after = 0
                if i < len(page_blocks) - 1:
                    next_block = page_blocks[i+1]
                    if abs(block["x"] - next_block["x"]) < 50:
                        spacing_after = next_block["y"] - (block["y"] + block["height"])
                
                block["spacing_before"] = max(0, spacing_before)
                block["spacing_after"] = max(0, spacing_after)
                
                text_blocks.append(block)
        
        doc.close()
        return sorted(text_blocks, key=lambda x: (x["page"], x["y"], x["x"]))

    def establish_baseline(self, blocks: List[Dict]) -> Dict[str, Any]:
        """Establish baseline text characteristics with improved font analysis and spacing."""
        if not blocks:
            return {
                "body_font_size": 12,
                "body_font_family": "default",
                "median_font_size": 12,
                "std_font_size": 0,
                "heading_thresholds": {"title": 18, "h1": 16, "h2": 14, "h3": 12},
                "all_font_sizes": [12],
                "normal_line_spacing": 2,
                "heading_spacing_threshold": 8
            }
        
        valid_blocks = [b for b in blocks if 6 <= b["font_size"] <= 40]
        font_sizes = [block["font_size"] for block in valid_blocks]
        
        spacings_before = []
        for block in valid_blocks:
            spacing = block.get("spacing_before", 0)
            if 0 < spacing < 100:
                spacings_before.append(spacing)
        
        if not font_sizes:
            font_sizes = [12]
        
        # Find the most common font size (body text)
        size_counts = Counter([round(size, 1) for size in font_sizes])
        body_font_size = size_counts.most_common(1)[0][0]
        
        median_font_size = np.median(font_sizes)
        std_font_size = np.std(font_sizes) if len(font_sizes) > 1 else 0
        
        normal_line_spacing = 3
        if spacings_before:
            normal_line_spacing = max(2, np.median(spacings_before))
        
        heading_spacing_threshold = max(normal_line_spacing * 2.5, 10)
        
        unique_sizes = sorted(list(set([round(size, 1) for size in font_sizes])), reverse=True)
        
        # Dynamic thresholds based on actual distribution
        heading_thresholds = {
            "title": body_font_size + 6.0,
            "h1": body_font_size + 3.0,
            "h2": body_font_size + 1.5,
            "h3": body_font_size + 0.5
        }
        
        if len(unique_sizes) >= 3:
            largest_size = unique_sizes[0]
            if largest_size > body_font_size + 4:
                heading_thresholds["title"] = largest_size
            if len(unique_sizes) >= 2:
                second_largest = unique_sizes[1]
                if second_largest > body_font_size + 2:
                    heading_thresholds["h1"] = second_largest
        
        return {
            "body_font_size": body_font_size,
            "body_font_family": Counter([b["font_name"] for b in valid_blocks]).most_common(1)[0][0],
            "median_font_size": median_font_size,
            "std_font_size": std_font_size,
            "heading_thresholds": heading_thresholds,
            "all_font_sizes": unique_sizes,
            "normal_line_spacing": normal_line_spacing,
            "heading_spacing_threshold": heading_spacing_threshold
        }

    def identify_headers_footers(self, blocks: List[Dict]) -> set:
        """Identify headers and footers by position and repetition."""
        headers_footers = set()
        
        if len(blocks) < 10:
            return headers_footers
        
        position_groups = defaultdict(list)
        
        for block in blocks:
            rel_y_rounded = round(block['rel_y'], 1)
            position_groups[rel_y_rounded].append(block)
        
        for rel_y, group in position_groups.items():
            if len(group) >= 3:
                pages = set(b['page'] for b in group)
                if len(pages) >= 3:
                    if rel_y < 0.08 or rel_y > 0.92:
                        sample_texts = [b['text'].lower().strip() for b in group[:5]]
                        is_likely_header_footer = True
                        
                        for text in sample_texts:
                            if any(keyword in text for keyword in ['abstract', 'introduction', 'conclusion', 'references']):
                                is_likely_header_footer = False
                                break
                            if re.match(r'^[ivx]+\.|^\d+\.', text):
                                is_likely_header_footer = False
                                break
                        
                        if is_likely_header_footer:
                            for block in group:
                                headers_footers.add(block['text'])
        
        return headers_footers

    def calculate_title_score(self, text_blocks: List[Dict], baseline: Dict, headers_footers: set, 
                             first_page_blocks: List[Dict]) -> float:
        """Enhanced title detection with multi-line support."""
        combined_text = self.combine_multi_line_text(text_blocks)
        main_block = text_blocks[0]
        
        if any(block["text"] in headers_footers for block in text_blocks):
            return 0.0
        
        if any(re.match(pattern, combined_text.lower()) for pattern in self.exclusion_patterns):
            return 0.0
        
        if self.is_code_or_formula(combined_text):
            return 0.0
        
        word_count = len(combined_text.split())
        if word_count < 3 or word_count > 25:
            return 0.0
        
        alpha_ratio = sum(c.isalpha() for c in combined_text) / len(combined_text) if combined_text else 0
        if alpha_ratio < 0.6:
            return 0.0
        
        score = 0.0
        text_lower = combined_text.lower()
        
        # Font Size Score (40% weight)
        size_diff = main_block["font_size"] - baseline["body_font_size"]
        if size_diff >= 6:
            font_size_score = 1.0
        elif size_diff >= 4:
            font_size_score = 0.8
        elif size_diff >= 2:
            font_size_score = 0.6
        elif size_diff >= 1:
            font_size_score = 0.3
        else:
            font_size_score = 0.1
        
        # Position Score (25% weight)
        position_score = 0.0
        if main_block["page"] == 1:
            if main_block["rel_y"] < 0.2:
                position_score = 1.0
            elif main_block["rel_y"] < 0.35:
                position_score = 0.7
            elif main_block["rel_y"] < 0.5:
                position_score = 0.4
        else:
            position_score = 0.1
        
        # Content Score (20% weight)
        content_score = 0.0
        title_keyword_count = sum(1 for keyword in self.title_keywords 
                                 if keyword in text_lower)
        if title_keyword_count > 0:
            content_score += min(0.6, title_keyword_count * 0.2)
        
        section_keywords_found = [kw for kw in self.section_keywords.keys() if kw in text_lower]
        if section_keywords_found:
            content_score -= 0.3
        
        words = combined_text.split()
        if len(words) >= 3:
            capitalized_count = sum(1 for word in words 
                                  if word[0].isupper() and len(word) > 2 and not word.isupper())
            if capitalized_count >= len(words) * 0.6:
                content_score += 0.3
        
        content_score = max(0.0, min(1.0, content_score))
        
        # Line Spacing Score (15% weight)
        spacing_score = 0.0
        spacing_before = main_block.get("spacing_before", 0)
        spacing_after = text_blocks[-1].get("spacing_after", 0)
        
        if spacing_after > baseline["heading_spacing_threshold"]:
            spacing_score += 0.6
        elif spacing_after > baseline["normal_line_spacing"] * 2:
            spacing_score += 0.3
        
        if spacing_before > baseline["normal_line_spacing"] * 2:
            spacing_score += 0.4
        
        spacing_score = min(1.0, spacing_score)
        
        # Combine scores
        total_score = (
            font_size_score * 0.40 +
            position_score * 0.25 +
            content_score * 0.20 +
            spacing_score * 0.15
        )
        
        # Additional bonuses and penalties
        if main_block["page"] == 1 and first_page_blocks:
            font_sizes = [b["font_size"] for b in first_page_blocks]
            max_font_size = max(font_sizes)
            if main_block["font_size"] >= max_font_size * 0.9:
                total_score += 0.15
        
        if len(text_blocks) > 1:
            total_score += 0.1
        
        if combined_text.endswith('.'):
            total_score *= 0.5
        
        if re.match(r'^(\d+\.|[IVXLC]+\.)', combined_text):
            total_score *= 0.2
        
        if ',' in combined_text and len(combined_text.split(',')) >= 3:
            total_score *= 0.1
        
        return min(1.0, total_score)

    def calculate_heading_score(self, text_blocks: List[Dict], baseline: Dict, headers_footers: set, 
                               doc_title: str = "") -> float:
        """Enhanced heading score calculation with multi-line support."""
        combined_text = self.combine_multi_line_text(text_blocks)
        main_block = text_blocks[0]
        
        if any(block["text"] in headers_footers for block in text_blocks) or combined_text == doc_title:
            return 0.0
        
        if any(re.match(pattern, combined_text.lower()) for pattern in self.exclusion_patterns):
            return 0.0
        
        if self.is_code_or_formula(combined_text):
            return 0.0
        
        word_count = len(combined_text.split())
        if word_count < 1 or word_count > 30:
            return 0.0
        
        alpha_ratio = sum(c.isalpha() for c in combined_text) / len(combined_text) if combined_text else 0
        if alpha_ratio < 0.4:
            return 0.0
        
        score = 0.0
        text_lower = combined_text.lower().strip()
        text_upper = combined_text.upper()
        
        # Section Keyword Score (30% weight)
        keyword_score = 0.0
        for keyword, weight in self.section_keywords.items():
            if keyword == text_lower or keyword in text_lower:
                if keyword == text_lower:
                    keyword_score = weight
                elif text_lower.startswith(keyword) or text_lower.endswith(keyword):
                    keyword_score = max(keyword_score, weight * 0.9)
                else:
                    keyword_score = max(keyword_score, weight * 0.7)
        
        # Pattern Matching Score (30% weight)
        pattern_score = 0.0
        pattern_strength = 0.0
        
        for pattern in self.strong_heading_patterns:
            if re.match(pattern, combined_text):
                pattern_score = 1.0
                pattern_strength = 1.0
                break
        
        if pattern_score == 0.0:
            for pattern in self.medium_heading_patterns:
                if re.match(pattern, combined_text):
                    pattern_score = 0.8
                    pattern_strength = 0.8
                    break
        
        if pattern_score == 0.0:
            for pattern in self.weak_heading_patterns:
                if re.match(pattern, combined_text):
                    pattern_score = 0.4
                    pattern_strength = 0.4
                    break
        
        # Font Size Score (20% weight)
        font_size_score = 0.0
        size_diff = main_block["font_size"] - baseline["body_font_size"]
        if size_diff >= 3:
            font_size_score = 1.0
        elif size_diff >= 2:
            font_size_score = 0.8
        elif size_diff >= 1:
            font_size_score = 0.6
        elif size_diff >= 0.5:
            font_size_score = 0.4
        elif size_diff >= 0:
            font_size_score = 0.2
        
        # Formatting Score (10% weight)
        format_score = 0.0
        if main_block["is_bold"]:
            format_score += 0.6
        if combined_text.isupper() and 2 <= word_count <= 8:
            format_score += 0.4
        format_score = min(1.0, format_score)
        
        # Position and Spacing Score (10% weight)
        position_score = 0.0
        spacing_before = main_block.get("spacing_before", 0)
        if spacing_before > baseline["heading_spacing_threshold"]:
            position_score += 0.5
        elif spacing_before > baseline["normal_line_spacing"] * 1.5:
            position_score += 0.3
        
        if main_block["rel_x"] < 0.1:
            position_score += 0.3
        
        position_score = min(1.0, position_score)
        
        # Combine scores
        total_score = (
            keyword_score * 0.30 +
            pattern_score * 0.30 +
            font_size_score * 0.20 +
            format_score * 0.10 +
            position_score * 0.10
        )
        
        # Apply bonuses and penalties
        core_sections = ['ABSTRACT', 'INTRODUCTION', 'CONCLUSION', 'CONCLUSIONS', 
                        'METHODOLOGY', 'METHODS', 'DISCUSSION', 'REFERENCES', 
                        'ACKNOWLEDGMENT', 'ACKNOWLEDGMENTS', 'BACKGROUND']
        if text_upper in core_sections:
            total_score += 0.15
        
        if len(text_blocks) > 1:
            total_score += 0.05
        
        if combined_text.endswith('.') and word_count > 2:
            total_score *= 0.3
        
        if word_count > 20:
            total_score *= 0.4
        elif word_count > 15:
            total_score *= 0.6
        elif word_count > 10:
            total_score *= 0.8
        
        if pattern_strength < 0.5:
            words = combined_text.split()
            lowercase_count = sum(1 for word in words if word.islower())
            if lowercase_count > word_count * 0.6:
                total_score *= 0.6
        
        return min(1.0, total_score)

    def find_multi_line_text_blocks(self, blocks: List[Dict], block_index: int, max_distance: float = 20) -> List[Dict]:
        """Find text blocks that might be part of a multi-line title or heading."""
        current_block = blocks[block_index]
        multi_line_blocks = [current_block]
        
        if block_index + 1 < len(blocks):
            next_block = blocks[block_index + 1]
            
            if (next_block["page"] == current_block["page"] and
                abs(next_block["font_size"] - current_block["font_size"]) <= 1 and
                abs(next_block["x"] - current_block["x"]) <= 50 and
                next_block["y"] - (current_block["y"] + current_block["height"]) <= max_distance):
                
                multi_line_blocks.append(next_block)
        
        return multi_line_blocks

    def combine_multi_line_text(self, blocks: List[Dict]) -> str:
        """Combine multiple text blocks into a single text string."""
        combined_text = " ".join(block["text"].strip() for block in blocks)
        return self.clean_special_characters(combined_text)

    def extract_content_between_headings(self, blocks: List[Dict], heading_positions: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract content between headings, organized by page and section."""
        content_sections = {}
        
        # Sort heading positions by page and y-coordinate
        sorted_headings = sorted(heading_positions, key=lambda x: (x["page"], x["y_position"]))
        
        for i, heading in enumerate(sorted_headings):
            section_key = f"{heading['text']}||{heading['page']}||{heading['level']}"
            content_sections[section_key] = {
                "heading": heading,
                "content_blocks": [],
                "pages": set()
            }
            
            # Find start and end boundaries for content extraction
            start_page = heading["page"]
            start_y = heading["y_position"]
            
            # Determine end boundary (next heading or end of document)
            if i + 1 < len(sorted_headings):
                next_heading = sorted_headings[i + 1]
                end_page = next_heading["page"]
                end_y = next_heading["y_position"]
            else:
                # Last heading - extract till end of document
                end_page = max(block["page"] for block in blocks)
                end_y = float('inf')
            
            # Extract content blocks
            for block in blocks:
                # Skip if it's a heading itself
                if any(block["text"].strip() == h["text"] for h in sorted_headings):
                    continue
                
                # Check if block falls within this section's boundaries
                if (block["page"] == start_page and block["y"] > start_y) or \
                   (start_page < block["page"] < end_page) or \
                   (block["page"] == end_page and block["y"] < end_y):
                    
                    # Additional filtering for content quality
                    text = block["text"].strip()
                    if (len(text) > 5 and  # Minimum length
                        not re.match(r'^\d+$', text) and  # Not just numbers
                        not re.match(r'^[^\w\s]*$', text)):  # Not just special chars
                        
                        content_sections[section_key]["content_blocks"].append({
                            "text": text,
                            "page": block["page"],
                            "font_size": block["font_size"],
                            "is_bold": block["is_bold"],
                            "y_position": block["y"]
                        })
                        content_sections[section_key]["pages"].add(block["page"])
        
        return content_sections

    def classify_heading_levels(self, heading_blocks: List[Dict], baseline: Dict) -> List[Dict]:
        """Enhanced heading level classification."""
        if not heading_blocks:
            return []
        
        result = []
        
        for heading_data in heading_blocks:
            if isinstance(heading_data, dict) and "text_blocks" in heading_data:
                text_blocks = heading_data["text_blocks"]
                combined_text = self.combine_multi_line_text(text_blocks)
                main_block = text_blocks[0]
                font_size = main_block["font_size"]
                page = main_block["page"]
                y_position = main_block["y"]
            else:
                combined_text = self.clean_special_characters(heading_data["text"].strip())
                font_size = heading_data["font_size"]
                page = heading_data["page"]
                y_position = heading_data["y"]
            
            text_lower = combined_text.lower().strip()
            text_upper = combined_text.upper()
            
            # Start with font-based classification
            if font_size >= baseline["heading_thresholds"]["h1"]:
                level = "H1"
            elif font_size >= baseline["heading_thresholds"]["h2"]:
                level = "H2"
            else:
                level = "H3"
            
            # Override based on content and patterns
            if (re.match(r'^[IVXLC]+\.?\s*[A-Z]', combined_text) or
                re.match(r'^CHAPTER\s+\d+', text_upper) or
                re.match(r'^PART\s+[IVXLC\d]+', text_upper) or
                re.match(r'^\d+\.?\s*[A-Z]', combined_text) or
                text_upper in ['ABSTRACT', 'INTRODUCTION', 'CONCLUSION', 'CONCLUSIONS', 
                              'METHODOLOGY', 'METHODS', 'DISCUSSION', 
                              'REFERENCES', 'ACKNOWLEDGMENT', 'ACKNOWLEDGMENTS',
                              'BACKGROUND'] or
                any(keyword in text_lower for keyword in 
                    ['literature review', 'future work'])):
                level = "H1"
            
            elif (re.match(r'^\d+\.\d+\.?\s*[A-Z]', combined_text) or
                  re.match(r'^[A-Z]\.?\s*[A-Z]', combined_text) or
                  re.match(r'^SECTION\s+\d+', text_upper)):
                level = "H2"
            
            elif (re.match(r'^\d+\.\d+\.\d+', combined_text) or
                  re.match(r'^\([a-zA-Z0-9]+\)', combined_text) or
                  re.match(r'^[a-z]\)', combined_text)):
                level = "H3"
            
            result.append({
                "level": level,
                "text": combined_text,
                "page": page,
                "font_size": font_size,
                "y_position": y_position
            })
        
        return result

    def extract_document_title(self, blocks: List[Dict], baseline: Dict) -> str:
        """Enhanced title extraction with multi-line support."""
        first_page_blocks = [b for b in blocks if b["page"] == 1 and b["rel_y"] < 0.6]
        
        if not first_page_blocks:
            return ""
        
        headers_footers = self.identify_headers_footers(blocks)
        title_candidates = []
        
        i = 0
        while i < len(first_page_blocks):
            current_block = first_page_blocks[i]
            multi_line_blocks = self.find_multi_line_text_blocks(first_page_blocks, i)
            
            score = self.calculate_title_score(multi_line_blocks, baseline, headers_footers, first_page_blocks)
            if score > 0.2:
                combined_text = self.combine_multi_line_text(multi_line_blocks)
                title_candidates.append((combined_text, score, multi_line_blocks))
            
            i += len(multi_line_blocks)
        
        if title_candidates:
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            best_title = title_candidates[0][0]
            return best_title
        
        return ""

    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """Main processing method that extracts document structure and content."""
        try:
            print(f"Processing document: {os.path.basename(pdf_path)}")
            
            # Step 1: Extract text blocks
            blocks = self.extract_text_blocks(pdf_path)
            if not blocks:
                return {"title": "", "sections": [], "pages": {}}
            
            print(f"  Extracted {len(blocks)} text blocks")
            
            # Step 2: Establish baseline characteristics
            baseline = self.establish_baseline(blocks)
            print(f"  Body font size: {baseline['body_font_size']}")
            
            # Step 3: Identify and remove headers/footers
            headers_footers = self.identify_headers_footers(blocks)
            clean_blocks = [b for b in blocks if b["text"] not in headers_footers]
            print(f"  Removed {len(blocks) - len(clean_blocks)} header/footer items")
            
            # Step 4: Extract document title
            title = self.extract_document_title(clean_blocks, baseline)
            print(f"  Title: '{title}'")
            
            # Step 5: Find potential headings
            potential_headings = []
            i = 0
            while i < len(clean_blocks):
                current_block = clean_blocks[i]
                
                if current_block["text"] != title:
                    multi_line_blocks = self.find_multi_line_text_blocks(clean_blocks, i)
                    score = self.calculate_heading_score(multi_line_blocks, baseline, headers_footers, title)
                    
                    if score > 0.2:
                        combined_text = self.combine_multi_line_text(multi_line_blocks)
                        heading_data = {
                            "text_blocks": multi_line_blocks,
                            "text": combined_text,
                            "page": multi_line_blocks[0]["page"],
                            "font_size": multi_line_blocks[0]["font_size"],
                            "heading_score": score,
                            "y": multi_line_blocks[0]["y"]
                        }
                        potential_headings.append(heading_data)
                
                i += len(multi_line_blocks) if current_block["text"] != title else 1
            
            print(f"  Found {len(potential_headings)} potential headings")
            
            # Step 6: Classify heading levels
            classified_headings = self.classify_heading_levels(potential_headings, baseline)
            
            # Step 7: Extract content between headings
            content_sections = self.extract_content_between_headings(clean_blocks, classified_headings)
            
            # Step 8: Organize by pages
            pages_data = {}
            for page_num in range(1, max(block["page"] for block in blocks) + 1):
                page_blocks = [b for b in clean_blocks if b["page"] == page_num]
                page_headings = [h for h in classified_headings if h["page"] == page_num]
                
                pages_data[f"page_{page_num}"] = {
                    "page_number": page_num,
                    "headings": page_headings,
                    "text_blocks": len(page_blocks),
                    "content_summary": f"Page {page_num} contains {len(page_headings)} headings and {len(page_blocks)} text blocks"
                }
            
            # Step 9: Create structured sections with content
            structured_sections = []
            for section_key, section_data in content_sections.items():
                heading_text, page, level = section_key.split('||')
                
                # Combine content text by page
                content_by_page = {}
                for content_block in section_data["content_blocks"]:
                    page_num = content_block["page"]
                    if page_num not in content_by_page:
                        content_by_page[page_num] = []
                    content_by_page[page_num].append(content_block["text"])
                
                # Create content text for each page
                page_contents = {}
                for page_num in sorted(content_by_page.keys()):
                    page_text = " ".join(content_by_page[page_num])
                    # Clean and filter content
                    if len(page_text.strip()) > 20:  # Only include substantial content
                        page_contents[f"page_{page_num}"] = {
                            "content": page_text.strip(),
                            "word_count": len(page_text.split())
                        }
                
                if page_contents:  # Only add sections with content
                    structured_sections.append({
                        "heading": {
                            "text": heading_text,
                            "level": level,
                            "page": int(page)
                        },
                        "content_pages": page_contents,
                        "total_pages": len(page_contents),
                        "content_summary": f"Section spans {len(page_contents)} pages with content"
                    })
            
            print(f"  Created {len(structured_sections)} structured sections")
            
            result = {
                "document_info": {
                    "filename": os.path.basename(pdf_path),
                    "title": title,
                    "total_pages": max(block["page"] for block in blocks),
                    "total_sections": len(structured_sections)
                },
                "sections": structured_sections,
                "pages": pages_data,
                "extraction_stats": {
                    "total_text_blocks": len(blocks),
                    "clean_text_blocks": len(clean_blocks),
                    "potential_headings": len(potential_headings),
                    "final_headings": len(classified_headings),
                    "sections_with_content": len(structured_sections)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"title": "", "sections": [], "pages": {}, "error": str(e)}

    def process_multiple_documents(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """Process multiple PDF documents and return structured data."""
        all_documents = {}
        
        for pdf_path in pdf_paths:
            document_data = self.process_document(pdf_path)
            doc_key = os.path.splitext(os.path.basename(pdf_path))[0]
            all_documents[doc_key] = document_data
        
        # Create summary statistics
        total_sections = sum(len(doc.get("sections", [])) for doc in all_documents.values())
        total_pages = sum(doc.get("document_info", {}).get("total_pages", 0) for doc in all_documents.values())
        
        return {
            "collection_summary": {
                "total_documents": len(all_documents),
                "total_pages": total_pages,
                "total_sections": total_sections,
                "documents_processed": list(all_documents.keys())
            },
            "documents": all_documents
        }

    def get_section_content(self, document_data: Dict, section_heading: str, page_number: Optional[int] = None) -> Optional[str]:
        """Helper method to get content for a specific section."""
        for section in document_data.get("sections", []):
            if section["heading"]["text"].lower() == section_heading.lower():
                if page_number:
                    page_key = f"page_{page_number}"
                    return section["content_pages"].get(page_key, {}).get("content")
                else:
                    # Return all content combined
                    all_content = []
                    for page_data in section["content_pages"].values():
                        all_content.append(page_data["content"])
                    return " ".join(all_content) if all_content else None
        return None

    def get_headings_by_level(self, document_data: Dict, level: str) -> List[Dict]:
        """Helper method to get all headings of a specific level."""
        headings = []
        for section in document_data.get("sections", []):
            if section["heading"]["level"] == level:
                headings.append(section["heading"])
        return headings

    def get_document_outline(self, document_data: Dict) -> List[Dict]:
        """Helper method to get a clean outline structure similar to 1A output."""
        outline = []
        for section in document_data.get("sections", []):
            outline.append({
                "level": section["heading"]["level"],
                "text": section["heading"]["text"],
                "page": section["heading"]["page"]
            })
        return outline


# Example usage and testing
# if __name__ == "__main__":
#     processor = DocumentProcessor()
    
#     # Test with a single document
#     pdf_path = "input/documents/sample.pdf"  # Replace with actual path
    
#     if os.path.exists(pdf_path):
#         result = processor.process_document(pdf_path)
        
#         print("\n" + "="*60)
#         print("DOCUMENT PROCESSING RESULTS")
#         print("="*60)
        
#         print(f"Title: {result['document_info']['title']}")
#         print(f"Total Sections: {result['document_info']['total_sections']}")
#         print(f"Total Pages: {result['document_info']['total_pages']}")
        
#         print("\nSECTIONS WITH CONTENT:")
#         for i, section in enumerate(result['sections'], 1):
#             heading = section['heading']
#             print(f"\n{i}. {heading['level']}: {heading['text']} (Page {heading['page']})")
#             print(f"   Content spans {section['total_pages']} page(s)")
            
#             # Show first 100 characters of content from first page
#             if section['content_pages']:
#                 first_page_content = list(section['content_pages'].values())[0]['content']
#                 preview = first_page_content[:100] + "..." if len(first_page_content) > 100 else first_page_content
#                 print(f"   Preview: {preview}")
        
#         print(f"\nExtraction Stats: {result['extraction_stats']}")
        
#     else:
#         print(f"PDF file not found: {pdf_path}")
#         print("Please ensure you have PDF files in the input/documents/ directory")