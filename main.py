import streamlit as st
import pandas as pd
import nltk
from nltk import tokenize
from bs4 import BeautifulSoup
import requests
from sklearn.metrics.pairwise import cosine_similarity
import io
import docx2txt
from PyPDF2 import PdfReader
import concurrent.futures
from googlesearch import search
import plotly.express as px
from fpdf import FPDF
import xlsxwriter
from functools import lru_cache
import numpy as np
from datetime import datetime
import contextlib

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')


# Hàm chia văn bản thành các câu
def get_sentences(text):
    sentences = tokenize.sent_tokenize(text)
    return sentences

# Hàm tìm URL kết quả tìm kiếm đầu tiên từ Google cho một câu
def get_url(sentence, timeout=5):
    try:
        base_url = 'https://www.google.com/search?q='
        query = '+'.join(sentence.split())
        url = base_url + query
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        res = requests.get(url, headers=headers, timeout=timeout)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, 'html.parser')
        if url := next((a['href'] for div in soup.find_all('div', class_='yuRUbf') 
                       if (a := div.find('a')) and 'youtube' not in a['href']), None):
            return url
        return None
    except Exception:
        return None

# Hàm đọc nội dung từ tệp văn bản dạng .txt
def read_text_file(file):
    try:
        file.seek(0)
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Lỗi khi đọc file text: {str(e)}")
        return ""

# Hàm đọc nội dung từ tệp văn bản dạng .docx
def read_docx_file(file):
    text = docx2txt.process(file)
    return text

# Hàm đọc nội dung từ tệp văn bản dạng .pdf
def read_pdf_file(file):
    try:
        text = []
        pdf = PdfReader(file)
        for page in pdf.pages:
            if content := page.extract_text():
                text.append(content)
        return "\n".join(text)
    except Exception as e:
        st.error(f"Lỗi khi đọc file PDF: {str(e)}")
        return ""

# Hàm đọc nội dung từ tệp văn bản (hỗ trợ các định dạng .txt, .pdf, .docx)
def get_text_from_file(uploaded_file):
    content = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            content = read_text_file(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            content = read_pdf_file(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = read_docx_file(uploaded_file)
    return content

# Hàm lấy nội dung văn bản từ một trang web
def get_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all(['p', 'div', 'article'])  # Mở rộng tìm kiếm
        text = ' '.join(p.get_text(strip=True) for p in paragraphs)
        return text
    except Exception as e:
        st.error(f"Lỗi khi đọc nội dung từ URL {url}: {str(e)}")
        return ""

# Khởi tạo mô hình Ollama
OLLAMA_API = "http://127.0.0.1:11434/v1"
MODEL_NAME = "Tuanpham/t-visstar-7b:latest"

# Cache cho kết quả tìm kiếm
@lru_cache(maxsize=100)
def get_google_search_results(query):
    results = list(search(query, num_results=3))
    return results

# Hàm gọi API Ollama
def query_ollama(text1, text2):
    response = requests.post(f"{OLLAMA_API}/generate", json={
        "model": MODEL_NAME,
        "prompt": f"So sánh độ tương đồng ngữ nghĩa giữa hai đoạn văn bản sau:\n1. {text1}\n2. {text2}"
    })
    return response.json()['response']

# Cập nhật hàm get_similarity để chỉ sử dụng Ollama
def get_similarity(text1, text2):
    response = requests.post(f"{OLLAMA_API}/generate", json={
        "model": MODEL_NAME,
        "prompt": f"""So sánh độ tương đồng giữa hai đoạn văn bản sau và cho điểm từ 0-1:
        Văn bản 1: {text1}
        Văn bản 2: {text2}
        
        Trả về kết quả theo định dạng JSON:
        {{
            "similarity_score": <điểm số từ 0-1>,
            "analysis": "<phân tích chi tiết>"
        }}
        """
    })
    
    try:
        result = response.json()['response']
        # Parse JSON từ response string
        import json
        parsed_result = json.loads(result)
        
        similarity_score = float(parsed_result['similarity_score'])
        analysis = parsed_result['analysis']
        
        return similarity_score, analysis
    except Exception as e:
        st.error(f"Lỗi khi xử lý response từ Ollama: {str(e)}")
        return 0.0, "Không thể phân tích kết quả"

# Đơn giản hóa hàm process_pair
def process_pair(i, j, texts=None, filenames=None):
    try:
        text1, text2 = texts[i], texts[j]
        file1, file2 = filenames[i], filenames[j]
        
        similarity_score, analysis = get_similarity(text1, text2)
        
        return {
            'file1': file1,
            'file2': file2,
            'similarity': similarity_score,
            'analysis': analysis
        }
    except Exception as e:
        st.error(f"Lỗi khi xử lý cặp {file1} và {file2}: {str(e)}")
        return None

# Đơn giản hóa hàm get_similarity_list
def get_similarity_list(texts, filenames=None):
    if not filenames:
        filenames = [f"Tệp {i+1}" for i in range(len(texts))]
    
    pairs = [(i, j) for i in range(len(texts)) for j in range(i+1, len(texts))]
    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_pair, 
                i, j, 
                texts=texts,
                filenames=filenames
            ) 
            for i, j in pairs
        ]
        
        for future in concurrent.futures.as_completed(futures):
            if result := future.result():
                results.append(result)
    
    return results

# Hàm tính toán độ tương đồng giữa một đoạn văn bản và các văn bản lấy từ danh sách URL
@lru_cache(maxsize=1000)
def get_similarity_cached(text1, text2):
    return get_similarity(text1, text2)

def get_similarity_list2(text, url_list):
    try:
        similarity_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    lambda u: get_similarity_cached(text, get_text(u)),
                    url
                ) for url in url_list if url
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    if result := future.result():
                        similarity_list.append(result)
                except Exception as e:
                    st.warning(f"Lỗi khi xử lý một URL: {str(e)}")
                    continue
                    
        return similarity_list
    except Exception as e:
        st.error(f"Lỗi khi tính toán độ tương đồng: {str(e)}")
        return []

# Thêm hàm xuất báo cáo
@contextlib.contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)
    with src as output:
        with contextlib.redirect_stdout(output):
            yield output_func

def export_report(df, format='pdf'):
    try:
        if format == 'pdf':
            pdf = FPDF()
            pdf.add_page()
            # Sử dụng font Arial thay vì DejaVu để tránh lỗi font
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Bao cao Kiem tra Dao van', 0, 1, 'C')
            
            pdf.set_font('Arial', '', 12)
            for i, row in df.iterrows():
                for col in df.columns:
                    content = str(row[col])
                    pdf.multi_cell(0, 10, f"{col}: {content}")
                pdf.ln(5)
                
            output_path = f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
            pdf.output(output_path)
            return output_path
            
        else:
            output_path = f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Ket qua', index=False)
                
                workbook = writer.book
                worksheet = writer.sheets['Ket qua']
                
                header_format = workbook.add_format({
                    'bold': True,
                    'align': 'center',
                    'valign': 'vcenter',
                    'bg_color': '#D9EAD3'
                })
                
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(col)
                    )
                    worksheet.set_column(idx, idx, max_length + 2)
                    worksheet.write(0, idx, col, header_format)
                    
            return output_path
            
    except Exception as e:
        st.error(f"Lỗi khi xuất báo cáo: {str(e)}")
        return None

# Cập nhật phần giao diện Streamlit
st.set_page_config(
    page_title='Phát hiện Đạo văn',
    page_icon='📚',
    layout='wide',  # Sử dụng toàn bộ chiều rộng màn hình
    initial_sidebar_state='expanded',
    menu_items={
        'Get Help': 'https://github.com/yourusername/plagiarism-checker',
        'Report a bug': "https://github.com/yourusername/plagiarism-checker/issues",
        'About': "# Ứng dụng Kiểm tra Đạo văn\nPhát triển bởi Tho Le"
    }
)

# Cập nhật CSS với theme tối
st.markdown("""
    <style>
    /* Theme tối */
    [data-testid="stAppViewContainer"] {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    .stTextArea textarea {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
        border: 1px solid #404040 !important;
        font-size: 16px !important;
    }
    
    .stAlert {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
        border: 1px solid #404040 !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #2D2D2D;
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
    }
    
    /* Nút với hiệu ứng hover */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #2D2D2D !important;
        border: 1px dashed #404040 !important;
        padding: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #2D2D2D;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4CAF50;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: #2D2D2D;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #404040;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2D2D2D;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #404040;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Header với logo và tiêu đề
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title('🔍 Web Kiểm Tra Đạo Văn')

# Sidebar với các tùy chọn
with st.sidebar:
    st.title("⚙️ Tùy chọn")
    
    with st.expander("Cài đặt báo cáo", expanded=True):
        export_format = st.selectbox(
            "📄 Định dạng xuất báo cáo",
            ['PDF', 'Excel'],
            help="Chọn định dạng file để xuất báo cáo"
        )
        
        if st.button('📥 Xuất báo cáo'):
            if st.session_state.df is not None:
                with st.spinner('Đang tạo báo cáo...'):
                    try:
                        output_path = export_report(st.session_state.df, format=export_format.lower())
                        if output_path:
                            with open(output_path, 'rb') as f:
                                mime_type = 'application/pdf' if export_format.lower() == 'pdf' else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                st.download_button(
                                    f'⬇️ Tải xuống báo cáo {export_format}',
                                    f,
                                    file_name=output_path,
                                    mime=mime_type
                                )
                    except Exception as e:
                        st.error(f"Lỗi khi tạo báo cáo: {str(e)}")
            else:
                st.warning("Vui lòng kiểm tra đạo văn trước khi xuất báo cáo")

# Main content
st.markdown("### 📝 Nhập văn bản hoặc tải lên tệp để kiểm tra đạo văn")

# Tabs thay vì radio buttons
tab1, tab2, tab3 = st.tabs(["✏️ Nhập văn bản", "📁 Kiểm tra tệp văn bản", "🔄 So sánh tệp văn bản"])

# Thêm biến state để lưu trữ DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None

# Xử lý chức năng cho từng tab
with tab1:
    # Tạo form để bắt sự kiện Enter
    with st.form(key='text_input_form', clear_on_submit=False):
        text = st.text_area(
            "Nhập văn bản vào đây",
            height=200,
            placeholder="Nhập văn bản cần kiểm tra và nhấn Ctrl+Enter hoặc nút Kiểm tra đạo văn",
            help="Nhấn Ctrl+Enter hoặc nút Kiểm tra đạo văn để bắt đầu kiểm tra",
            key='text_input'
        )
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            submit_button = st.form_submit_button(
                label='🔍 Kiểm tra đạo văn',
                use_container_width=True,
                type='primary'  # Làm nổi bật nút
            )
        
    if submit_button:
        if not text:
            st.warning("⚠️ Vui lòng nhập văn bản cần kiểm tra")
            st.stop()
            
        with st.spinner('🔄 Đang phân tích văn bản...'):
            sentences = get_sentences(text)
            urls = []
            similarity_scores = []
            analyses = []
            
            # Xử lý từng câu
            progress_bar = st.progress(0)
            for idx, sentence in enumerate(sentences):
                # Tìm URL
                url = get_url(sentence)
                urls.append(url)
                
                if url:
                    # Tính độ tương đồng
                    similarity_score, analysis = get_similarity(sentence, get_text(url))
                    similarity_scores.append(similarity_score)
                    analyses.append(analysis)
                else:
                    similarity_scores.append(0)
                    analyses.append("Không tìm thấy nguồn tương đồng")
                
                progress_bar.progress((idx + 1) / len(sentences))
            
            # Tạo DataFrame kết quả
            st.session_state.df = pd.DataFrame({
                'Câu': sentences,
                'URL': urls,
                'Tỷ lệ': [f"{score*100:.2f}%" for score in similarity_scores],
                'Phân tích': analyses
            })
            st.session_state.df_type = 'text'

with tab2:
    uploaded_file = st.file_uploader(
        "Tải lên tệp (.docx, .pdf, .txt)",
        type=["docx", "pdf", "txt"],
        help="Hỗ trợ các định dạng: DOCX, PDF, TXT"
    )
    
    if uploaded_file:
        with st.expander("👁️ Xem trước nội dung", expanded=False):
            preview_text = get_text_from_file(uploaded_file)
            st.text_area("Nội dung tệp:", value=preview_text, height=150)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            check_button = st.button(
                '🔍 Kiểm tra đạo văn',
                key='check2',
                use_container_width=True
            )
        
        if check_button:
            with st.spinner('🔄 Đang phân tích tệp...'):
                text = get_text_from_file(uploaded_file)
                sentences = get_sentences(text)
                
                urls = []
                similarity_scores = []
                analyses = []
                
                progress_bar = st.progress(0)
                for idx, sentence in enumerate(sentences):
                    url = get_url(sentence)
                    urls.append(url)
                    
                    if url:
                        similarity_score, analysis = get_similarity(sentence, get_text(url))
                        similarity_scores.append(similarity_score)
                        analyses.append(analysis)
                    else:
                        similarity_scores.append(0)
                        analyses.append("Không tìm thấy nguồn tương đồng")
                    
                    progress_bar.progress((idx + 1) / len(sentences))
                
                st.session_state.df = pd.DataFrame({
                    'Câu': sentences,
                    'URL': urls,
                    'Tỷ lệ': [f"{score*100:.2f}%" for score in similarity_scores],
                    'Phân tích': analyses
                })
                st.session_state.df_type = 'file'

with tab3:
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded_files = st.file_uploader(
            "Tải lên nhiều tệp (.docx, .pdf, .txt)",
            type=["docx", "pdf", "txt"],
            accept_multiple_files=True,
            help="Có thể chọn nhiều tệp cùng lúc"
        )
    with col2:
        similarity_threshold = st.slider(
            "🎯 Ngưỡng tương đồng (%)",
            min_value=0,
            max_value=100,
            value=50,
            help="Chỉ hiển thị kết quả có độ tương đồng lớn hơn ngưỡng này"
        )
    
    if uploaded_files and len(uploaded_files) >= 2:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            compare_button = st.button(
                '🔍 So sánh các tệp',
                key='check3',
                use_container_width=True
            )
        
        if compare_button:
            with st.spinner('🔄 Đang so sánh các tệp...'):
                texts = []
                filenames = []
                
                # Đọc nội dung các tệp
                for file in uploaded_files:
                    text = get_text_from_file(file)
                    texts.append(text)
                    filenames.append(file.name)
                
                # So sánh các tệp
                results = get_similarity_list(texts, filenames)
                
                # Lọc kết quả theo ngưỡng
                filtered_results = [
                    r for r in results 
                    if float(r['similarity']) * 100 >= similarity_threshold
                ]
                
                if filtered_results:
                    st.session_state.df = pd.DataFrame(filtered_results)
                    st.session_state.df['Tỷ lệ'] = st.session_state.df['similarity'].apply(
                        lambda x: f"{float(x)*100:.2f}%"
                    )
                    st.session_state.df = st.session_state.df.drop('similarity', axis=1)
                    st.session_state.df_type = 'comparison'
                else:
                    st.warning("❗ Không tìm thấy kết quả nào vượt ngưỡng tương đồng")
    else:
        st.info("ℹ️ Vui lòng tải lên ít nhất 2 tệp để so sánh")

# Cải thiện hàm vẽ biểu đồ
def create_similarity_chart(df_plot, chart_type='bar'):
    if chart_type == 'bar':
        fig = px.bar(
            df_plot,
            x=df_plot.index,
            y='Tỷ lệ',
            title='Phân tích độ tương đồng',
            labels={'index': 'STT', 'Tỷ lệ': 'Độ tương đồng (%)'},
            template='plotly_dark'
        )
    else:  # chart_type == 'scatter'
        fig = px.scatter(
            df_plot,
            x=df_plot.index,
            y='Tỷ lệ',
            title='Phân tích độ tương đồng',
            labels={'index': 'STT', 'Tỷ lệ': 'Độ tương đồng (%)'},
            template='plotly_dark'
        )

    # Cải thiện giao diện biểu đồ
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        hovermode='x unified'
    )

    # Thêm đường ngưỡng tham chiếu
    fig.add_hline(
        y=50,  # Ngưỡng 50%
        line_dash="dash",
        line_color="red",
        annotation_text="Ngưỡng cảnh báo (50%)",
        annotation_position="bottom right"
    )

    fig.update_traces(
        marker_color='#4CAF50',
        marker_line_color='#45a049',
        marker_line_width=1.5,
        opacity=0.8,
        hovertemplate='<b>STT</b>: %{x}<br>' +
                      '<b>Độ tương đồng</b>: %{y:.1f}%<extra></extra>'
    )

    return fig

# Cải thiện hiển thị kết quả
if st.session_state.df is not None:
    with st.container():
        st.markdown("### 📊 Kết quả phân tích")
        
        # Thêm tabs cho phân tích
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
            "📈 Tổng quan", 
            "📊 Biểu đồ phân tích",
            "📋 Chi tiết"
        ])
        
        with analysis_tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.session_state.df_type in ['text', 'file']:
                    total_sentences = len(st.session_state.df)
                    st.metric(
                        "📝 Tổng số câu/đoạn", 
                        f"{total_sentences:,}",
                        delta=None
                    )
                else:
                    st.metric("📑 Số cặp so sánh", len(st.session_state.df))
            
            with col2:
                avg_similarity = st.session_state.df['Tỷ lệ'].str.rstrip('%').astype(float).mean()
                st.metric(
                    "📊 Độ tương đồng trung bình", 
                    f"{avg_similarity:.1f}%",
                    delta=f"{avg_similarity-50:.1f}%" if avg_similarity > 50 else None,
                    delta_color="inverse"
                )
            
            with col3:
                if st.session_state.df_type in ['text', 'file']:
                    unique_sources = len([url for url in st.session_state.df['URL'].unique() if url is not None])
                    st.metric(
                        "🔗 Nguồn phát hiện", 
                        f"{unique_sources:,}",
                        help="Số lượng nguồn độc lập được phát hiện"
                    )
                else:
                    total_files = len(set(st.session_state.df['file1'].unique()) | 
                                    set(st.session_state.df['file2'].unique()))
                    st.metric("📚 Tệp đã so sánh", f"{total_files:,}")

        with analysis_tab2:
            # Thêm tùy chọn loại biểu đồ
            chart_type = st.radio(
                "Chọn loại biểu đồ:",
                ['Bar Chart', 'Scatter Plot'],
                horizontal=True
            )
            
            df_plot = st.session_state.df.copy()
            df_plot['Tỷ lệ'] = df_plot['Tỷ lệ'].str.rstrip('%').astype(float)
            
            fig = create_similarity_chart(
                df_plot, 
                'bar' if chart_type == 'Bar Chart' else 'scatter'
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with analysis_tab3:
            # Thêm bộ lọc và sắp xếp
            col1, col2 = st.columns([2,1])
            with col1:
                search_term = st.text_input(
                    "🔍 Tìm kiếm",
                    placeholder="Nhập từ khóa để lọc..."
                )
            with col2:
                sort_by = st.selectbox(
                    "Sắp xếp theo",
                    ['Tỷ lệ giảm dần', 'Tỷ lệ tăng dần']
                )
            
            # Lọc và sắp xếp DataFrame
            filtered_df = st.session_state.df.copy()
            if search_term:
                mask = filtered_df.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False)
                ).any(axis=1)
                filtered_df = filtered_df[mask]
            
            # Sắp xếp
            filtered_df['sort_value'] = filtered_df['Tỷ lệ'].str.rstrip('%').astype(float)
            filtered_df = filtered_df.sort_values(
                'sort_value',
                ascending=sort_by == 'Tỷ lệ tăng dần'
            ).drop('sort_value', axis=1)
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    'URL': st.column_config.LinkColumn('URL'),
                    'Tỷ lệ': st.column_config.ProgressColumn(
                        'Độ tương đồng',
                        format='%s',
                        min_value='0%',
                        max_value='100%',
                        help="Độ tương đồng với nguồn được phát hiện"
                    )
                }
            )

# Cập nhật footer với theme tối
st.markdown("""
    <div style='text-align: center; color: #808080; padding: 20px;'>
        <hr style='border-color: #404040;'>
        <p>© 2024 Web Kiểm Tra Đạo Văn | Phát triển bởi Tho Le</p>
    </div>
""", unsafe_allow_html=True)

def auto_export_report(df, format='pdf'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.{format}"
    
    if format == 'pdf':
        export_report(df, 'pdf')
        with open(filename, "rb") as f:
            st.download_button(
                "Tải xuống báo cáo PDF",
                f,
                filename,
                "application/pdf"
            )
    else:
        export_report(df, 'xlsx')
        with open(filename, "rb") as f:
            st.download_button(
                "Tải xuống báo cáo Excel",
                f,
                filename,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
