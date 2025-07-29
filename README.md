# Persona-Driven Document Intelligence

## Team AIdeate - Adobe Hackathon Solution

### Project Overview

This project delivers an intelligent document processing system that adapts to different user personas and their specific job requirements. Instead of providing generic document summaries, our solution understands who the user is and what they need to accomplish, then extracts and prioritizes the most relevant information accordingly.

### The Problem We Solve

Traditional document analysis tools treat all users the same way, providing generic summaries and extractions. However, a PhD researcher looking for methodologies needs different information than a business analyst seeking market trends, even when analyzing the same document.

### Our Solution

Our system performs **persona-driven document intelligence** by:

- **Understanding the User**: Analyzes persona descriptions to identify user type (researcher, student, analyst, practitioner) and expertise level
- **Understanding the Task**: Interprets the specific job-to-be-done to determine what information is most valuable
- **Smart Content Extraction**: Uses advanced NLP models to score and rank document sections based on relevance to the persona and task
- **Adaptive Prioritization**: Adjusts the importance of different document sections based on user needs

### Key Features

- **Multi-Persona Support**: Optimized for researchers, students, business analysts, and practitioners
- **Intelligent Section Ranking**: Uses cross-encoder models for accurate relevance scoring
- **Content Refinement**: Extracts key sentences and insights from relevant sections
- **Comprehensive Document Processing**: Handles complex PDF structures, headings, and multi-page content
- **Scalable Architecture**: Processes multiple documents efficiently with detailed extraction statistics

### Use Cases

- **Academic Researchers**: Focus on methodologies, datasets, and benchmarks for literature reviews
- **Students**: Extract key concepts, examples, and fundamental principles for learning
- **Business Analysts**: Identify market trends, performance metrics, and strategic insights
- **Practitioners**: Find implementation details, best practices, and practical applications

### Technical Approach

Our solution combines document structure analysis, semantic understanding, and persona-aware ranking to deliver precisely what each user needs. The system processes PDFs through multiple stages of analysis, from text extraction to intelligent content filtering, ensuring high-quality results tailored to individual requirements.

## How to Build and Run

The project is containerized using Docker and is designed to run according to the competition's execution specifications.

### Prerequisites
- Docker must be installed and running.

### 📁 Prepare the Input Directory

The script requires a specific folder structure for its input.

## Steps

1. Navigate into the `input/` folder.
2. Inside `input/`, create a folder named `documents`.
3. Place all the **PDF files** you want to analyze inside the `input/documents/` folder.
4. Inside the `input/` folder, create a file named `config.json`.

---

## 🛠 The `config.json` File

Our system uses a streamlined `config.json` that only requires the essential inputs: the **persona** and the **job-to-be-done**.

### ✅ Advantage:
Unlike other formats, you do **not** need to manually list every single document filename.  
Our script **intelligently discovers and processes** all PDF files found in the `input/documents/` directory, making it more robust and easier to use.

---

### Example `config.json` structure:

```json
{
  "persona": "PhD Researcher in Computational Biology",
  "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
}

### Build the Docker Image

Navigate to the project's root directory in your terminal and run the following command to build the image:

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier . 
```

### Run the Solution

To process the PDFs, place them inside a folder named input in the current directory, and create an empty output folder where the results will be saved. Then run the following command:

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier
```

- The container will automatically:
- Find all .pdf files in the input directory
- Process each one
- Generate a corresponding .json file in the output directory

---

Let me know if you want the README written to a file for download, or if you'd like to include a sample output format or API spec as well.

### Team AIdeate

Built with passion for intelligent document processing and user-centric AI solutions.

---

*This project was developed for the Adobe Hackathon, demonstrating the power of persona-driven AI in transforming how we interact with documents.*