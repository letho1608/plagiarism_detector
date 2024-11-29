# Plagiarism Detection Web Application

A Streamlit-based web application for detecting plagiarism in text documents and comparing multiple files for similarity.

## Features

### 1. Text Analysis
- Direct text input for plagiarism checking
- Sentence-by-sentence analysis
- Real-time progress tracking
- Google search integration for source detection

### 2. File Support
- Multiple file format support:
  - PDF (.pdf)
  - Word (.docx)
  - Text (.txt)
- File preview functionality
- Multi-file comparison

### 3. Analysis Tools
- Similarity scoring (0-100%)
- Detailed analysis for each comparison
- Source URL tracking
- Customizable similarity threshold

### 4. Visualization
- Interactive charts:
  - Bar charts
  - Scatter plots
  - Progress indicators
  - Metric displays

### 5. Reporting
- Export options:
  - PDF reports
  - Excel spreadsheets
- Customizable report formats
- Timestamp-based file naming

## Technical Components

### Core Functions
- **File Processing**
- **API Integration**
- **UI Components**

### Main Layout
- **Dark theme**
- **Responsive design**
- Three main tabs:
  - Text Input
  - File Check
  - File Comparison

### Analysis Display
- Overview metrics
- Interactive charts
- Detailed results table

## Dependencies
- `streamlit`: Web interface
- `pandas`: Data handling
- `nltk`: Text processing
- `beautifulsoup4`: Web scraping
- `plotly`: Data visualization
- `fpdf`: PDF generation
- `xlsxwriter`: Excel report generation


