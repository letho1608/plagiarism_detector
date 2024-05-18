## Function:

**1. Text input options:**
- Check text: Users can enter text directly into the provided text area. This text is then compared with other texts to find similarities.
- Check File: Users can upload text files (.docx, .pdf, .txt) using the file uploader. The content of the uploaded file is extracted and used to detect plagiarism.
- Check multiple files together: Users can upload multiple text files. The contents of each file are compared to each other to determine similarities.
 
**2. Inspection mechanism:**
- Cosine similarity: This tool uses cosine similarity, a measure of similarity between two non-zero vectors of the internal product space, to compare texts.
- Single text analysis: For input texts or uploaded files, the engine calculates a similarity score between the provided text and content fetched from URLs found in the text.
- Compare multiple files: For multiple uploaded files, the tool calculates a similarity score for each pair of files, displaying the similarity score by pair.
  
**3. Integrated web scraping:**
- URL Extraction: The tool extracts URLs embedded in the provided text to fetch additional content for comparison.
- Content fetching: The content of each URL is collected using the BeautifulSoup library, then compared with uploaded files or input text.
  
**4. User interface:**
- Streamlit Framework: User interface built using Streamlit, a Python library for creating web applications.
- File uploader: Streamlit provides various input widgets such as text area and file uploader for user interaction.
- Display results: Results, including overall similarity score and pairwise similarity score, are displayed using Streamlit's visualization capabilities.

## Demo:
- Link: https://plagiarismdetector.streamlit.app/
