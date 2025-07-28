#!/usr/bin/env python3
"""
Adobe Hackathon Round 1B - Persona-Driven Document Intelligence
Main integration script that processes PDFs based on persona and job-to-be-done

This script integrates:
- document_processor.py: PDF processing and structure extraction
- section_extractor.py: Section relevance scoring and extraction
- persona_analyzer.py: Persona requirements analysis

Author: Team
Date: 2025-01-27
"""

import json
import os
import sys
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import our custom modules
try:
    from document_processor import DocumentProcessor
    from section_extractor import SectionExtractor
    from persona_analyzer import PersonaAnalyzer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure document_processor.py, section_extractor.py, and persona_analyzer.py are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/processing.log') if os.path.exists('/app') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

class PersonaDrivenDocumentProcessor:
    """
    Main orchestrator class that integrates all components for persona-driven document analysis.
    """
    
    def __init__(self):
        """Initialize all processing components."""
        try:
            logger.info("Initializing PersonaDrivenDocumentProcessor...")
            
            # Initialize components
            self.document_processor = DocumentProcessor()
            logger.info("✓ Document processor initialized")
            
            self.section_extractor = SectionExtractor()
            logger.info("✓ Section extractor initialized")
            
            self.persona_analyzer = PersonaAnalyzer()
            logger.info("✓ Persona analyzer initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def load_configuration(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from config.json file.
        
        Args:
            config_path: Path to config.json file
            
        Returns:
            Dictionary containing persona and job_to_be_done
        """
        try:
            logger.info(f"Loading configuration from: {config_path}")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['persona', 'job_to_be_done']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field in config: {field}")
            
            logger.info(f"Configuration loaded successfully")
            logger.info(f"Persona: {config['persona']}")
            logger.info(f"Job to be done: {config['job_to_be_done']}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def find_pdf_files(self, documents_dir: str) -> List[str]:
        """
        Find all PDF files in the documents directory.
        
        Args:
            documents_dir: Path to documents directory
            
        Returns:
            List of PDF file paths
        """
        try:
            logger.info(f"Searching for PDF files in: {documents_dir}")
            
            if not os.path.exists(documents_dir):
                raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
            
            pdf_files = []
            for file_name in os.listdir(documents_dir):
                if file_name.lower().endswith('.pdf'):
                    pdf_path = os.path.join(documents_dir, file_name)
                    if os.path.isfile(pdf_path):
                        pdf_files.append(pdf_path)
            
            if not pdf_files:
                raise ValueError(f"No PDF files found in {documents_dir}")
            
            # Sort for consistent processing order
            pdf_files.sort()
            
            logger.info(f"Found {len(pdf_files)} PDF files:")
            for pdf_file in pdf_files:
                logger.info(f"  - {os.path.basename(pdf_file)}")
            
            return pdf_files
            
        except Exception as e:
            logger.error(f"Error finding PDF files: {e}")
            raise
    
    def process_documents(self, pdf_files: List[str]) -> Dict[str, Any]:
        """
        Process all PDF documents to extract structure and content.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            Processed documents data
        """
        try:
            logger.info("Starting document processing...")
            start_time = time.time()
            
            # Process all documents
            documents_data = self.document_processor.process_multiple_documents(pdf_files)
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            # Log processing statistics
            collection_summary = documents_data.get("collection_summary", {})
            logger.info(f"Processing summary:")
            logger.info(f"  - Documents processed: {collection_summary.get('total_documents', 0)}")
            logger.info(f"  - Total pages: {collection_summary.get('total_pages', 0)}")
            logger.info(f"  - Total sections: {collection_summary.get('total_sections', 0)}")
            
            return documents_data
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    def analyze_persona_requirements(self, persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """
        Analyze persona requirements and generate search strategy.
        
        Args:
            persona: User persona description
            job_to_be_done: Specific task description
            
        Returns:
            Persona analysis results
        """
        try:
            logger.info("Analyzing persona requirements...")
            
            persona_analysis = self.persona_analyzer.analyze_persona_requirements(
                persona, job_to_be_done
            )
            
            logger.info(f"Persona analysis completed:")
            logger.info(f"  - Persona category: {persona_analysis.get('persona_category', 'unknown')}")
            logger.info(f"  - Job category: {persona_analysis.get('job_category', 'unknown')}")
            logger.info(f"  - Generated {len(persona_analysis.get('relevant_queries', []))} relevant queries")
            
            return persona_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing persona requirements: {e}")
            raise
    
    def extract_relevant_sections(self, documents_data: Dict[str, Any], 
                                persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """
        Extract and rank relevant sections based on persona and job requirements.
        
        Args:
            documents_data: Processed documents data
            persona: User persona description
            job_to_be_done: Specific task description
            
        Returns:
            Extracted sections with relevance ranking
        """
        try:
            logger.info("Extracting relevant sections...")
            start_time = time.time()
            
            # Extract relevant sections using the section extractor
            extraction_results = self.section_extractor.extract_relevant_sections(
                documents_data, persona, job_to_be_done
            )
            
            extraction_time = time.time() - start_time
            logger.info(f"Section extraction completed in {extraction_time:.2f} seconds")
            
            # Log extraction statistics
            num_extracted = len(extraction_results.get("extracted_sections", []))
            num_sub_sections = len(extraction_results.get("sub_section_analysis", []))
            
            logger.info(f"Extraction results:")
            logger.info(f"  - Extracted sections: {num_extracted}")
            logger.info(f"  - Sub-section analyses: {num_sub_sections}")
            
            return extraction_results
            
        except Exception as e:
            logger.error(f"Error extracting relevant sections: {e}")
            raise
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """
        Validate the output data format against requirements.
        
        Args:
            output_data: Output data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required top-level keys
            required_keys = ["metadata", "extracted_sections", "sub_section_analysis"]
            for key in required_keys:
                if key not in output_data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            # Validate metadata
            metadata = output_data["metadata"]
            metadata_keys = ["input_documents", "persona", "job_to_be_done", "processing_timestamp"]
            for key in metadata_keys:
                if key not in metadata:
                    logger.error(f"Missing metadata key: {key}")
                    return False
            
            # Validate extracted sections structure
            for section in output_data["extracted_sections"]:
                section_keys = ["document", "page_number", "section_title", "importance_rank"]
                for key in section_keys:
                    if key not in section:
                        logger.error(f"Missing section key: {key}")
                        return False
            
            # Validate sub-section analysis structure
            for analysis in output_data["sub_section_analysis"]:
                analysis_keys = ["document", "section_title", "refined_text", "page_number"]
                for key in analysis_keys:
                    if key not in analysis:
                        logger.error(f"Missing sub-section analysis key: {key}")
                        return False
            
            logger.info("Output validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating output: {e}")
            return False
    
    def save_output(self, output_data: Dict[str, Any], output_path: str) -> None:
        """
        Save the output data to JSON file.
        
        Args:
            output_data: Data to save
            output_path: Path to save the output file
        """
        try:
            logger.info(f"Saving output to: {output_path}")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Validate output before saving
            if not self.validate_output(output_data):
                raise ValueError("Output validation failed")
            
            # Save with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Verify file was created and get size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Output saved successfully ({file_size} bytes)")
            else:
                raise IOError("Failed to create output file")
                
        except Exception as e:
            logger.error(f"Error saving output: {e}")
            raise
    
    def process_full_pipeline(self, input_dir: str, output_dir: str) -> None:
        """
        Execute the complete processing pipeline.
        
        Args:
            input_dir: Input directory containing config.json and documents/
            output_dir: Output directory for results
        """
        start_time = time.time()
        
        try:
            logger.info("="*60)
            logger.info("STARTING PERSONA-DRIVEN DOCUMENT PROCESSING PIPELINE")
            logger.info("="*60)
            
            # Step 1: Load configuration
            config_path = os.path.join(input_dir, "config.json")
            config = self.load_configuration(config_path)
            persona = config["persona"]
            job_to_be_done = config["job_to_be_done"]
            
            # Step 2: Find PDF files
            documents_dir = os.path.join(input_dir, "documents")
            pdf_files = self.find_pdf_files(documents_dir)
            
            # Step 3: Process documents
            logger.info("\n" + "="*40)
            logger.info("STEP 1: PROCESSING DOCUMENTS")
            logger.info("="*40)
            
            documents_data = self.process_documents(pdf_files)
            
            # Step 4: Analyze persona requirements
            logger.info("\n" + "="*40)
            logger.info("STEP 2: ANALYZING PERSONA REQUIREMENTS")
            logger.info("="*40)
            
            persona_analysis = self.analyze_persona_requirements(persona, job_to_be_done)
            
            # Step 5: Extract relevant sections
            logger.info("\n" + "="*40)
            logger.info("STEP 3: EXTRACTING RELEVANT SECTIONS")
            logger.info("="*40)
            
            extraction_results = self.extract_relevant_sections(
                documents_data, persona, job_to_be_done
            )
            
            # Step 6: Save output
            logger.info("\n" + "="*40)
            logger.info("STEP 4: SAVING RESULTS")
            logger.info("="*40)
            
            output_path = os.path.join(output_dir, "challenge1b_output.json")
            self.save_output(extraction_results, output_path)
            
            # Final statistics
            total_time = time.time() - start_time
            logger.info("\n" + "="*60)
            logger.info("PROCESSING COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Documents processed: {len(pdf_files)}")
            logger.info(f"Sections extracted: {len(extraction_results.get('extracted_sections', []))}")
            logger.info(f"Sub-sections analyzed: {len(extraction_results.get('sub_section_analysis', []))}")
            logger.info(f"Output saved to: {output_path}")
            
            # Performance check
            if total_time > 60:
                logger.warning(f"Processing time ({total_time:.2f}s) exceeded 60s limit")
            else:
                logger.info(f"✓ Processing completed within time limit")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            
            # Create error output file
            error_output = {
                "metadata": {
                    "input_documents": [],
                    "persona": config.get("persona", "unknown") if 'config' in locals() else "unknown",
                    "job_to_be_done": config.get("job_to_be_done", "unknown") if 'config' in locals() else "unknown",
                    "processing_timestamp": datetime.now().isoformat() + "Z",
                    "error": str(e)
                },
                "extracted_sections": [],
                "sub_section_analysis": []
            }
            
            try:
                error_output_path = os.path.join(output_dir, "challenge1b_output.json")
                os.makedirs(output_dir, exist_ok=True)
                with open(error_output_path, 'w', encoding='utf-8') as f:
                    json.dump(error_output, f, indent=2, ensure_ascii=False)
                logger.info(f"Error output saved to: {error_output_path}")
            except:
                pass
            
            raise


def main():
    """
    Main entry point for the application.
    Handles Docker execution environment and command-line usage.
    """
    try:
        # Determine input and output directories
        if os.path.exists('/app/input') and os.path.exists('/app/output'):
            # Docker environment
            input_dir = '/app/input'
            output_dir = '/app/output'
            logger.info("Running in Docker environment")
        else:
            # Local development environment
            input_dir = os.path.join(os.getcwd(), 'input')
            output_dir = os.path.join(os.getcwd(), 'output')
            logger.info("Running in local development environment")
            
            # Create directories if they don't exist
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Verify input directory structure
        config_path = os.path.join(input_dir, "config.json")
        documents_dir = os.path.join(input_dir, "documents")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if not os.path.exists(documents_dir):
            raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
        
        # Initialize processor and run pipeline
        processor = PersonaDrivenDocumentProcessor()
        processor.process_full_pipeline(input_dir, output_dir)
        
        logger.info("Application completed successfully")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()