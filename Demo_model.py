# Import library
import streamlit as st
import pandas as pd
import numpy as np
import regex
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from string import punctuation 
from underthesea import word_tokenize, sent_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split 
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_option('deprecation.showPyplotGlobalUse', False)


# Data đã xử lý text và gán nhãn lại
df_new = pd.read_csv('Cleaned_data_for_model.csv')

#1. PROCESSING TEXT
files_1 = ['emojicon.txt','teencode.txt', 'english-vnmese.txt']
files_2 = ['wrong-word.txt', 'vietnamese-stopwords.txt']
d1 = []
d2 = []
for i in files_1:
    filename = "Text_processing/" + i
    file = open(filename, 'r', encoding='utf8')
    temp_lst = file.read().split('\n')
    temp_dict = {}
    for line in temp_lst:
        key, value = line.split('\t')
        temp_dict[key] = str(value)
    d1.append(temp_dict)
    file.close()
for j in files_2:
    filename = "Text_processing/" + j
    file = open(filename, 'r', encoding='utf8')
    temp_lst = file.read().split('\n')
    d2.append(temp_lst)
    file.close()
emoji_dict = d1[0]                   
teen_dict = d1[1]                     
english_dict = d1[2]                 
wrong_lst = d2[0]                     
stopwords_lst = d2[1]
stopwords_lst = stopwords_lst + list(punctuation)   

def eng_vn_transform(sentence):
    result = ''
    for word in sentence.split():
        ma_word = regex.match(r'(\w*)(\W*)', word)
        word = ma_word[1]
        word_end = ma_word[2]
        if word in english_dict:
            result += english_dict[word] + word_end + ' '
        else:
            result += word + ' '
    return result

def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', text, flags=re.MULTILINE)
    document = document.lower() 
    document = regex.sub(r'([a-z])\1*\b', r'\1', document) 
    document = regex.sub(r'\.+', ".", document)
    new_sentence = ''
    for sentence in sent_tokenize(document):
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        sentence = eng_vn_transform(sentence)
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence + sentence + '. '                    
    document = new_sentence
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def remove_stopword(text, stopwords):
    document = ' '.join('' if word in stopwords else word for word in text.split())
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def text_transform(document):
    temp = process_text(document, emoji_dict, teen_dict, wrong_lst) 
    temp = covert_unicode(temp) 
    return remove_stopword(temp, stopwords_lst) 

def process_special_word(text):
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text

def process_Underthesea(data):
    data_text = []
    for i in range(data.shape[0]):
        sentence = word_tokenize(data.review[i],format='text')
        sentence = regex.sub(r'\.', "", sentence) 
        data_text.append(sentence)
    data['review'] = data_text
    data_text = []
    for i in range(data.shape[0]):
        sentence = process_special_word(data.review[i])
        data_text.append(sentence)
    data['review'] = data_text
    data['review'] = data['review'].apply(lambda x: remove_stopword(x,stopwords_lst))
    return data

#2. BUILD MODEL LOGISTIC REGRESSION
vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
X_train, X_test, y_train, y_test = train_test_split(df_new['review'], df_new['y'], test_size=0.2)
pipeline = Pipeline(
    [
        ("vect", CountVectorizer(**vectorizer_params)),
        ("tfidf", TfidfTransformer()),
        ('smt', RandomOverSampler()),
        ("clf", LogisticRegression(C=10, solver='liblinear', penalty='l1'))
    ])
model = pipeline.fit(X_train, y_train)

# 3. STREAMLIT 
# Tên project 
st.title('Sentiment Analysis for E-commerce project')
st.write("GVHD : Nguyễn Quan Liêm")
st.write("Học viên : Nguyễn Thị Kim Oanh")

menu = ["Input by row", "Upload Excel file", "Upload CSV file"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Input by row":
    st.write("Bạn vui lòng nhập nội dung cần kiểm tra và nhấn Submit")
    cmt = st.text_input("Review")
    submit = st.button("Submit")
    if submit:
        df_input = pd.DataFrame(data=[cmt],columns=["review_text"])
        df_input['review'] = df_input['review_text'].apply(lambda x: text_transform(x))
        df_input = process_Underthesea(df_input)
        df_input['pred_sentiment'] = model.predict(df_input['review'])
        y_class = {0:'positive', 1:'neutral', 2:'negative'}
        df_input['pred_sentiment']  = [y_class[i] for i in df_input.pred_sentiment]
        st.write("Result : ",df_input['pred_sentiment'][0])
elif choice == "Upload Excel file":
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df_upload = pd.read_excel(uploaded_file, engine='openpyxl')
        filename = uploaded_file.name
        df_new.column = ["review_text"]
        df_upload['review'] = df_upload.iloc[:, 0].apply(lambda x: text_transform(x))
        df_upload = process_Underthesea(df_upload)
        df_upload['sentiment'] = model.predict(df_upload['review'])
        y_class = {0:'positive', 1:'neutral', 2:'negative'}
        df_upload['sentiment']  = [y_class[i] for i in df_upload.sentiment]
        st.subheader("Result & Statistics :")
        st.table(df_upload.iloc[:,[0,2]])

        st.subheader("Biểu đồ Wordcloud trích xuất đặc trưng theo nhóm sentiment: ")
        cmt0 = df_upload[df_upload['sentiment']=='positive']
        cmt1 = df_upload[df_upload['sentiment']=='neutral']
        cmt2 = df_upload[df_upload['sentiment']=='negative']
        comment_words_0 = ''
        comment_words_1 = ''
        comment_words_2 = ''
        for i in range(len(cmt0)):
            for word in cmt0.review.iloc[i]:
                word = word.lower()
                comment_words_0 += " ".join(word) + ""
            comment_words_0 += ' '
    
        for i in range(len(cmt1)):
            for word in cmt1.review.iloc[i]:
                word = word.lower()
                comment_words_1 += " ".join(word) + ""
            comment_words_1 += ' '

        for i in range(len(cmt2)):
            for word in cmt2.review.iloc[i]:
                word = word.lower()
                comment_words_2 += " ".join(word) + ""
            comment_words_2 += ' '
        
        plt.subplot(1,3,1)
        wc0 = WordCloud(background_color = 'white', 
               color_func=lambda *args, **kwargs: "blue").generate(comment_words_0)
        wc1 = WordCloud(background_color = 'white', 
               color_func=lambda *args, **kwargs: "orange").generate(comment_words_1)
        wc2 = WordCloud(background_color = 'white', 
               color_func=lambda *args, **kwargs: "green").generate(comment_words_2)
        #1
        plt.subplot(1,3,1)
        plt.title("POSITIVE")
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(wc0)
        #2
        plt.subplot(1,3,2)
        plt.title("NEUTRAL")
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(wc1)
        #3
        plt.subplot(1,3,3)
        plt.title("NEGATIVE")
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(wc2)
        st.pyplot()
        
        st.subheader("Biểu đồ thống kê số lượng theo nhóm sentiment :")
        fig, ax = plt.subplots()
        ax = sns.countplot(df_upload.sentiment,color='yellow')
        st.pyplot(fig,  figsize=(3, 4))
    else:
        st.warning("Bạn vui lòng upload đúng định dạng file đã chọn")
else:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file, index_col= 0)
        filename = uploaded_file.name
        df_new.column = ["review_text"]
        df_upload['review'] = df_upload.iloc[:, 0].apply(lambda x: text_transform(x))
        df_upload = process_Underthesea(df_upload)
        df_upload['sentiment'] = model.predict(df_upload['review'])
        y_class = {0:'positive', 1:'neutral', 2:'negative'}
        df_upload['sentiment']  = [y_class[i] for i in df_upload.sentiment]
        st.subheader("Result & Statistics :")
        st.table(df_upload.iloc[:,[0,2]])

        st.subheader("Biểu đồ Wordcloud trích xuất đặc trưng theo nhóm sentiment: ")
        cmt0 = df_upload[df_upload['sentiment']=='positive']
        cmt1 = df_upload[df_upload['sentiment']=='neutral']
        cmt2 = df_upload[df_upload['sentiment']=='negative']
        comment_words_0 = ''
        comment_words_1 = ''
        comment_words_2 = ''
        for i in range(len(cmt0)):
            for word in cmt0.review.iloc[i]:
                word = word.lower()
                comment_words_0 += " ".join(word) + ""
            comment_words_0 += ' '
    
        for i in range(len(cmt1)):
            for word in cmt1.review.iloc[i]:
                word = word.lower()
                comment_words_1 += " ".join(word) + ""
            comment_words_1 += ' '

        for i in range(len(cmt2)):
            for word in cmt2.review.iloc[i]:
                word = word.lower()
                comment_words_2 += " ".join(word) + ""
            comment_words_2 += ' '
        
        plt.subplot(1,3,1)
        wc0 = WordCloud(background_color = 'white', 
               color_func=lambda *args, **kwargs: "blue").generate(comment_words_0)
        wc1 = WordCloud(background_color = 'white', 
               color_func=lambda *args, **kwargs: "orange").generate(comment_words_1)
        wc2 = WordCloud(background_color = 'white', 
               color_func=lambda *args, **kwargs: "green").generate(comment_words_2)
        #1
        plt.subplot(1,3,1)
        plt.title("POSITIVE")
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(wc0)
        #2
        plt.subplot(1,3,2)
        plt.title("NEUTRAL")
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(wc1)
        #3
        plt.subplot(1,3,3)
        plt.title("NEGATIVE")
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(wc2)
        st.pyplot()
        
        st.subheader("Biểu đồ thống kê số lượng theo nhóm sentiment :")
        fig, ax = plt.subplots()
        ax = sns.countplot(df_upload.sentiment,color='orange')
        st.pyplot(fig,  figsize=(3, 4))
    else:
        st.warning("Bạn vui lòng upload đúng định dạng file đã chọn")