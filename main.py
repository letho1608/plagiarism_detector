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
    with io.open(file.name, 'r', encoding='utf-8') as f:
        content = f.read()
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
            content = uploaded_file.getvalue().decode("utf-8")
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
        filenames = [f"File {i+1}" for i in range(len(texts))]
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



st.set_page_config(page_title='Phát hiện đạo văn')
st.title('Công cụ phát hiện đạo văn')

st.write("""
### Nhập văn bản hoặc tải lên tệp để kiểm tra đạo văn
""")
option = st.radio(
    "Chọn tùy chọn đầu vào:",
    ('Kiểm tra văn bản', 'Kiểm tra file', 'Kiểm tra nhiều file với nhau')
)

if option == 'Nhập văn bản':
    text = st.text_area("Nhập văn bản vào đây", height=200)
    uploaded_files = []
elif option == 'Tải lên tệp':
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
        ### Không tìm thấy các tài liệu liên quan
        """)
        st.stop()

    # Tính tổng phần trăm tương đồng
    total_similarity = 0.0

    if option == 'Tìm sự tương đồng giữa các tệp':
        similarities = get_similarity_list(texts, filenames)
        total_similarity = sum(similarity[2] for similarity in similarities) / len(similarities)
    else:
        sentences = get_sentences(text)
        urls = []
        for sentence in sentences:
            urls.append(get_url(sentence))

        if None in urls:
            st.write("""
            ### Không phát hiện đạo văn!
            """)
            st.stop()

        similarity_list = get_similarity_list2(text, urls)
        total_similarity = sum(similarity for similarity in similarity_list) / len(similarity_list)

    st.write(f"### Tỷ lệ đạo văn: {total_similarity:.0%}")


    if option == 'Tìm sự tương đồng giữa các tệp':
        df = pd.DataFrame(similarities, columns=['Tệp 1', 'Tệp 2', 'Giống nhau'])
        df = df.sort_values(by=['Giống nhau'], ascending=False)
    else:
        df = pd.DataFrame({'Câu': sentences, 'URL': urls, 'Giống nhau': similarity_list})
        df = df.sort_values(by=['Giống nhau'], ascending=True)

    df = df.reset_index(drop=True)
    
    # Make URLs clickable in the DataFrame
    if 'URL' in df.columns:
        df['URL'] = df['URL'].apply(lambda x: '<a href="{}">{}</a>'.format(x, x) if x else '')
    
    # Center align URL column header
    df_html = df.to_html(escape=False)
    if 'URL' in df.columns:
        df_html = df_html.replace('<th>URL</th>', '<th style="text-align: center;">URL</th>')
    st.write(df_html, unsafe_allow_html=True)
    
   