import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt')
from nltk import tokenize
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import docx2txt
from PyPDF2 import PdfReader

def get_sentences(text):
    sentences = tokenize.sent_tokenize(text)
    return sentences

def get_url(sentence):
    base_url = 'https://www.google.com/search?q='
    query = sentence
    query = query.replace(' ', '+')
    url = base_url + query
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    divs = soup.find_all('div', class_='yuRUbf')
    urls = []
    for div in divs:
        a = div.find('a')
        urls.append(a['href'])
    if len(urls) == 0:
        return None
    elif "youtube" in urls[0]:
        return None
    else:
        return urls[0]

def read_text_file(file):
    content = ""
    file.seek(0)  # Đảm bảo bắt đầu đọc từ đầu tệp
    content = file.read().decode('utf-8')  # Đọc nội dung tệp dưới dạng chuỗi
    return content

def read_docx_file(file):
    text = docx2txt.process(file)
    return text

def read_pdf_file(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

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

def get_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text

def get_similarity(text1, text2):
    text_list = [text1, text2]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    similarity = cosine_similarity(count_matrix)[0][1]
    return similarity

def get_similarity_list(texts, filenames=None):
    similarity_list = []
    if filenames is None:
        filenames = [f"Tệp {i+1}" for i in range(len(texts))]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = get_similarity(texts[i], texts[j])
            similarity_list.append((filenames[i], filenames[j], similarity))
    return similarity_list

def get_similarity_list2(text, url_list):
    similarity_list = []
    for url in url_list:
        text2 = get_text(url)
        similarity = get_similarity(text, text2)
        similarity_list.append(similarity)
    return similarity_list

st.set_page_config(page_title='Phát hiện Đạo văn')
st.title('Web kiểm tra đạo văn')

st.write("""
### Nhập văn bản hoặc tải lên tệp để kiểm tra đạo văn hoặc tìm sự Tỷ lệ giữa các file văn bản.
""")
option = st.radio(
    "Chọn chức năng tương ứng:",
    ('Nhập văn bản', 'Kiểm tra tệp văn bản', 'So sánh giữa các tệp văn bản')
)

if option == 'Nhập văn bản':
    text = st.text_area("Nhập văn bản vào đây", height=200)
    uploaded_files = []
elif option == 'Kiểm tra tệp văn bản':
    uploaded_file = st.file_uploader("Tải lên tệp (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
    if uploaded_file is not None:
        text = get_text_from_file(uploaded_file)
        uploaded_files = [uploaded_file]
    else:
        text = ""
        uploaded_files = []
else:
    uploaded_files = st.file_uploader("Tải lên nhiều tệp (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"], accept_multiple_files=True)
    texts = []
    filenames = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            text = get_text_from_file(uploaded_file)
            texts.append(text)
            filenames.append(uploaded_file.name)
    text = " ".join(texts)

if st.button('Kiểm tra đạo văn'):
    if not text:
        st.write("""
        ### Không tìm thấy văn bản để kiểm tra đạo văn hoặc để so sánh.
        """)
        st.stop()
    
    if option == 'So sánh giữa các tệp văn bản':
        similarities = get_similarity_list(texts, filenames)
        df = pd.DataFrame(similarities, columns=['Tệp 1', 'Tệp 2', 'Tỷ lệ'])
        df['Tỷ lệ'] = df['Tỷ lệ'].apply(lambda x: "{:.2f}%".format(x * 100))  # Định dạng tỷ lệ đạo văn thành phần trăm
        df = df.sort_values(by=['Tỷ lệ'], ascending=False)
    else:
        sentences = get_sentences(text)
        url = []
        for sentence in sentences:
            url.append(get_url(sentence))

        if None in url:
            st.write("""
            ### Không phát hiện đạo văn!
            """)
            st.stop()

        similarity_list = get_similarity_list2(text, url)
        total_similarity = sum(similarity_list) / len(similarity_list) * 100 if similarity_list else 0
        st.write("Tổng tỷ lệ đạo văn của toàn bộ văn bản: {:.2f}%".format(total_similarity))
    
        df = pd.DataFrame({'Câu': sentences, 'URL': url, 'Tỷ lệ': similarity_list})
        df['Tỷ lệ'] = df['Tỷ lệ'].apply(lambda x: "{:.2f}%".format(x * 100))  # Định dạng tỷ lệ đạo văn thành phần trăm
        df = df.sort_values(by=['Tỷ lệ'], ascending=True)


    
    df = df.reset_index(drop=True)
    
    # Make URLs clickable in the DataFrame
    if 'URL' in df.columns:
        df['URL'] = df['URL'].apply(lambda x: '<a href="{}">{}</a>'.format(x, x) if x else '')
    
    # Center align URL column header
    df_html = df.to_html(escape=False)
    if 'URL' in df.columns:
        df_html = df_html.replace('<th>URL</th>', '<th style="text-align: center;">URL</th>')
    st.write(df_html, unsafe_allow_html=True)

