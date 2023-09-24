import streamlit as st
import pandas as pd 
import io
import json

# for pandas ai
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import PandasAI
import matplotlib

matplotlib.use('TkAgg') # setting matplotlib parameter to use tkinter to display charts

# for llangchain
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader

#from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
#from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain 

#new
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI as openai2

# for pdf reader
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from typing_extensions import Concatenate
#import pickle
#from langchain.callbacks import get_openai_callback
#from langchain.chains.question_answering import load_qa_chain

from langchain.memory import ConversationBufferMemory
#from htmlTemplates import css, bot_template, user_template

# for video
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
import whisper

def page_configuration() -> None:
    st.set_page_config(
    page_title = "RIO-GPT",
    layout="wide")


def initialize_session_state() -> None:
    if "df" not in st.session_state : 
        st.session_state['df'] = pd.DataFrame()
    if "File_uploader_object" not in st.session_state : 
        st.session_state['File_uploader_object'] = None

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    if "outlier_check_trigger" not in st.session_state : 
        st.session_state['outlier_check_trigger'] = False
    if "chat_bot_llama_trigger" not in st.session_state : 
        st.session_state['chat_bot_llama_trigger'] = False
    if "chat_bot_Pandasai_trigger" not in st.session_state : 
        st.session_state['chat_bot_Pandasai_trigger'] = False

    if "button_outlier_check_trigger" not in st.session_state : 
        st.session_state['button_outlier_check_trigger'] = False

    #session state for chat bot
    if 'history' not in st.session_state : 
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        
        st.session_state['generated'] = ['Hi I am Rio GPT. Your friendly neighbourhood genrative-AI powered conversational assistant. I am still in development mode. \n You can simply ask questions to your data in a natural language and I will try to answer it.😊']
    
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hey! ']

    # session state for pdf reader
    if 'raw_text' not in st.session_state:
        st.session_state['raw_text'] = ""

    
    # new
    if "conversation" not in st.session_state:
        st.session_state['conversation'] = None

    #new
    
    if "user_question" not in st.session_state:
        st.session_state['user_question'] = None

def clear_chat() -> None:
    '''
    This function will execute when clear button is clicked
    '''
    st.session_state['history'] = []

    st.session_state['generated'] = ['Hi I am Rio GPT. Your friendly neighbourhood genrative-AI powered conversational assistant. I am still in development mode. \n You can simply ask questions to your data in a natural language and I will try to answer it.😊']
        
    st.session_state['past'] = ['Hey!']

def download_chat()-> None:
    '''
    This funciton will help download th converstaion history
    '''
    download_history = json.dumps(st.session_state['history'])

    return download_history

def read_file() -> pd.DataFrame: 
    # Function to read file and return and dataframe and Fileuploader object
    df_uploader = st.file_uploader("✳️Upload your file here", type = ['csv','pdf','mp4'],accept_multiple_files=False)

    if df_uploader is not None :

        # getting the extension
        file_extension = os.path.splitext(df_uploader.name)[1]

        if file_extension == '.csv':
            df_uploader.seek(0)
            df = pd.read_csv(df_uploader)
        elif file_extension == '.xlsx': 
            df = pd.read_excel(df_uploader)
        elif file_extension == '.pdf':
            pdf_reader = PdfReader(df_uploader)
            raw_text =""
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
                       
            return raw_text, df_uploader
        elif file_extension == '.mp4':
            temp_file_1 = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
            temp_file_1.write(df_uploader.getbuffer())
            # extracting the audio
            audio_file = "audio.mp3"
            ffmpeg_extract_audio(temp_file_1.name, audio_file)
            # audio to transcript
            model = whisper.load_model("base")
            st.video(temp_file_1.name)
            st.audio(audio_file)

            #fileexists = os.path.isfile(audio_file)
            #st.write(fileexists)
            st.write(audio_file)
            
            st.write(audio_file)
            result = model.transcribe(audio_file)
            st.write(result["text"])
            raw_text = result["text"]
            return raw_text, df_uploader
        else :
            pass # saving this for pdf in future

        st.session_state['df'] = df
        
        return st.session_state['df'], df_uploader
    else:
        
        return pd.DataFrame(), None

def get_df_info(df):
    '''
    Function to display df.info() details in dataframe format
    '''
    buffer = io.StringIO ()
    df.info (buf=buffer)
    lines = buffer.getvalue ().split ('\n')
    # lines to print directly
    lines_to_print = [0, 1, 2, -2, -3]
    for i in lines_to_print:
        st.write (lines [i])
    # lines to arrange in a df
    list_of_list = []
    for x in lines [5:-3]:
        list = x.split ()
        list_of_list.append (list)
    info_df = pd.DataFrame (list_of_list, columns=['index', 'Column', 'Non-null-Count', 'null', 'Dtype'])
    info_df.drop (columns=['index', 'null'], axis=1, inplace=True)
    st.dataframe(info_df)

def get_df_info_text(df) -> None:
    '''
    Function to display df.info() details in text format
    '''
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

def display_uploaded_data() -> None:

    # display uploaded file
    st.dataframe(st.session_state['df'], width = 1100)

def display_uploaded_data_info() -> None:
    data = st.session_state['df']

    # display info():
    get_df_info_text(data)

    # display describe:
    st.dataframe(data.describe().T, width = 1100)
    
def chat_bot_Pandasai_api() -> None:
    '''
    This function uses Pandasai library using Open AI to converse with uploaded file data.
    The key is available in .env file against OPENAI_API_KEY
    '''    
    # checking is the .env file exits ( only incase of running file locally) else extract API from streamlit interface
    check_file = os.path.isfile('.env')

    if check_file:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

    else:
        openai_api_key = st.secrets["API_KEY"]

        
    def chat_with_csv(df,prompt):
        llm = OpenAI(api_token=openai_api_key)
        pandas_ai = PandasAI(llm)
        result = pandas_ai.run(df, prompt=prompt)
        #print(result)
        return result

    
    input_text = st.text_area("Enter your query")

    if input_text is not None:
        if st.button("Chat"):
            st.info("Your Query: "+input_text)
            with st.spinner("Generating your response!"):
                result = chat_with_csv(st.session_state['df'], input_text)
                st.write(result)

    return None 

def chat_bot_llangchain_openapi_csv(uploaded_file) -> None:

    
    DB_FAISS_PATH = "vectorestore/db_faiss"

    check_file = os.path.isfile('.env')

    # checking API Key info ( for both local run and for streamlit)
    if check_file:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

    else:
        openai_api_key = st.secrets["API_KEY"]
    

    # create a temporary file object
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','}) # csv loader needs  a filepath hence we created a temporary file path

    
    
    data = loader.load()
    

    # word embedding model ( vector creation)
    # embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2",
    #                                    model_kwargs = {'device' : 'cpu'})
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    db = FAISS.from_documents(data,embeddings)
    # save the db to the path
    db.save_local(DB_FAISS_PATH)

    # load llm model . it will be passed in conversation retrieval chain
    llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

    #chain call
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # function for streamlit chat
    def conversational_chat(query):
        result = chain({'question':query, "chat_history" : st.session_state['history']})
        #st.session_state['history'].append([query, result['answer']])
        st.session_state['history'].append((query, result['answer']))
        return result['answer']
    
    
    # assigning containers for the chat history
    # https://discuss.streamlit.io/t/upload-files-to-streamlit-app/80/48?page=3
    response_container = st.container()
    
    container = st.container()

    with container:
        with st.form(key = "my_form", clear_on_submit = True):
            user_input = st.text_input("Query:", placeholder = "Talk to your CSV Data here", key = 'input')

            submit_buttom = st.form_submit_button(label ='Send')
            

        if submit_buttom and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        
        st.button("Clear Chat", on_click=clear_chat)
    

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user = True,
                        key = str(i) + '_user',
                        avatar_style = "open-peeps")
                message(st.session_state["generated"][i],
                        key = str(i) ,
                        avatar_style = "identicon")
  
def chat_bot_llangchain_openapi_csv2(uploaded_file) -> None:
    ''''
    building a sample
    '''
    
    check_file = os.path.isfile('.env')

    # checking API Key info ( for both local run and for streamlit)
    if check_file:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

    else:
        openai_api_key = st.secrets["API_KEY"]
    
    # create a temporary file object
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    llm = openai2(temperature = 0, openai_api_key = openai_api_key)

    
    agent = create_csv_agent(llm, tmp_file_path,verbose=True)

    # user_question = st.text_input ("Ask a question about your CSV:")

    # if user_question is not None and user_question != "":
    #     response = agent.run(user_question)
    #     st.write(response)

    def conversational_chat(query):
        result = agent.run(query)
        #st.session_state['history'].append([query, result['answer']])
        st.session_state['history'].append((query, result))
        return result
    # builder chat interface
    response_container = st.container()
    container = st.container()
    with container:
        with st.form(key = "my_form", clear_on_submit = True):
            user_input = st.text_input("Query:", placeholder = "Talk to your CSV Data here", key = 'input')

            submit_buttom = st.form_submit_button(label ='Send')
        
        if submit_buttom and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        
        col1,col2 = st.columns([0.2,1])
        # option to clear chat history
        with col1: 
            st.button("Clear Chat", on_click=clear_chat)

        # option to download chat history
        with col2 :
            st.download_button(label="Download chat as text",
                data=download_chat(),
                file_name='chat_history.txt',
                mime='text/csv',)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user = True,
                        key = str(i) + '_user',
                        avatar_style = "open-peeps")
                message(st.session_state["generated"][i],
                        key = str(i) ,
                        avatar_style = "identicon")
              

def chat_bot_llangchain_openapi_pdf():
    '''
    Function to chat with pdf file
    '''
    check_file = os.path.isfile('.env')
    if check_file:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

    else:
        openai_api_key = st.secrets["API_KEY"]
    
    text = st.session_state['raw_text']
    text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
      
    chunks = text_splitter.split_text(text=text)

    
    
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)

    #Store the chunks part in db (vector)
    vectorstore = FAISS.from_texts(texts = chunks,embedding=embeddings)

    
    llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo',openai_api_key=openai_api_key)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    # alternate caht interface
    # st.session_state.conversation = conversation_chain
    
    # user_question = st.text_input("Ask a question about your documents:")
    # if user_question:
    #     response = st.session_state.conversation({'question': user_question})
    #     st.write(response)


    # builder chat interface
    response_container = st.container()
    container = st.container()
    

    def conversational_chat(query):
        result = conversation_chain({'question':query, "chat_history" : st.session_state['history']})
        #st.session_state['history'].append([query, result['answer']])
        st.session_state['history'].append((query, result['answer']))
        return result['answer']

    with container:
        with st.form(key = "my_form", clear_on_submit = True):
            user_input = st.text_input("Query:", placeholder = "Talk to your CSV Data here", key = 'input')

            submit_buttom = st.form_submit_button(label ='Send')
        
        if submit_buttom and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        
        col1,col2 = st.columns([0.2,1])
        # option to clear chat history
        with col1: 
            st.button("Clear Chat", on_click=clear_chat)

        # option to download chat history
        with col2 :
            st.download_button(label="Download chat as text",
                data=download_chat(),
                file_name='chat_history.txt',
                mime='text/csv',)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user = True,
                        key = str(i) + '_user',
                        avatar_style = "open-peeps")
                message(st.session_state["generated"][i],
                        key = str(i) ,
                        avatar_style = "identicon")
    
        # st.session_state.chat_history = response['chat_history']

        # for i, message in enumerate(st.session_state.chat_history):
        #     if i % 2 == 0:
        #         st.write(user_template.replace(
        #             "{{MSG}}", message.content), unsafe_allow_html=True)
        #     else:
        #         st.write(bot_template.replace(
        #             "{{MSG}}", message.content), unsafe_allow_html=True)

    return
    
def chat_bot_llangchain_openapi_video():
    '''
    Function to chat with pdf file
    '''
    check_file = os.path.isfile('.env')
    if check_file:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

    else:
        openai_api_key = st.secrets["API_KEY"]
    
    text = st.session_state['raw_text']
    text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
      
    chunks = text_splitter.split_text(text=text)

    
    
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)

    #Store the chunks part in db (vector)
    vectorstore = FAISS.from_texts(texts = chunks,embedding=embeddings)

    
    llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo',openai_api_key=openai_api_key)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    # alternate caht interface
    # st.session_state.conversation = conversation_chain
    
    # user_question = st.text_input("Ask a question about your documents:")
    # if user_question:
    #     response = st.session_state.conversation({'question': user_question})
    #     st.write(response)


    # builder chat interface
    response_container = st.container()
    container = st.container()
    

    def conversational_chat(query):
        result = conversation_chain({'question':query, "chat_history" : st.session_state['history']})
        #st.session_state['history'].append([query, result['answer']])
        st.session_state['history'].append((query, result['answer']))
        return result['answer']

    with container:
        with st.form(key = "my_form", clear_on_submit = True):
            user_input = st.text_input("Query:", placeholder = "Talk to your CSV Data here", key = 'input')

            submit_buttom = st.form_submit_button(label ='Send')
        
        if submit_buttom and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        
        col1,col2 = st.columns([0.2,1])
        # option to clear chat history
        with col1: 
            st.button("Clear Chat", on_click=clear_chat)

        # option to download chat history
        with col2 :
            st.download_button(label="Download chat as text",
                data=download_chat(),
                file_name='chat_history.txt',
                mime='text/csv',)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user = True,
                        key = str(i) + '_user',
                        avatar_style = "open-peeps")
                message(st.session_state["generated"][i],
                        key = str(i) ,
                        avatar_style = "identicon")
    
        # st.session_state.chat_history = response['chat_history']

        # for i, message in enumerate(st.session_state.chat_history):
        #     if i % 2 == 0:
        #         st.write(user_template.replace(
        #             "{{MSG}}", message.content), unsafe_allow_html=True)
        #     else:
        #         st.write(bot_template.replace(
        #             "{{MSG}}", message.content), unsafe_allow_html=True)

    return
#@st.cache_data
def Table_creation(df):
        
        Final_table = df.groupby("Merchant")["Credit amount","Redemption","Spend"].agg({"Credit amount":'sum',"Redemption":'sum',"Spend":'sum'})

        Final_table["Net Spend"] = Final_table['Spend']- Final_table["Credit amount"]

        Final_table = Final_table.sort_values("Net Spend",ascending= False)
        
        Final_table = Final_table.head(5)
        return Final_table


def bau_report()-> None:
    # Function call : dispalay regular BAU analysis
    df = st.session_state['df']

    # resetting the date column
    df["Date"] = pd.to_datetime(df["Date"])
    st.session_state['df'] = df

    Table = Table_creation(st.session_state['df'])
    
    st.subheader("Summary Report:")
    cola, colb, colc = st.columns(3)
    cola.metric("Total Spend", '$ {:10,d}'.format(df["Spend"].sum()))
    colb.metric("Total Credit", '# {:10,d}'.format(df["Credit amount"].sum()))
    colc.metric("Total Redemption",'# {:10,d}'.format(df["Redemption"].sum()))
    st.write("Merchant Performance:")
    num1 , num2 = st.columns(2)
    num1.dataframe(Table)
    Table.reset_index(inplace=True)
    num2.line_chart(Table,x= 'Merchant',y = ['Spend','Credit amount'],)#color = ["#E3242B","#1E2F97"])
    spend = df.groupby("Date")['Spend'].sum().reset_index()
    spend.sort_values("Date",inplace=True)
    st.write("Month Wise:")
    st.line_chart(spend,x='Date',y='Spend')

def main():

    # page setting configuration
    page_configuration()


    #initializing session state variables
    initialize_session_state()

    # adding page title
    st.markdown("<h1 style='text-align: center; color: Black;'>RIO-GPT</h1>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: Black;'>Your friendly neighbourhood generative-AI powered conversational assistant.</h3>", unsafe_allow_html=True)
    # subheader
    #st.subheader( 'Our friendly neighbourhood chat bot ready to assit RIO whenever there is a need of support.',
                
    st.divider()
    col1, col2 = st.columns(2)

    with col1 :
        # choose solution
        choose_option = st.selectbox("***Choose chat solution:***", ('Chat with csv(Single query)', 'Chat with csv(Conversational Chain)','Chat with pdf','Chat with video'))

        
        
        if choose_option == 'Chat with csv(Single query)' or  choose_option == 'Chat with csv(Conversational Chain)':

            #Read file from column  and  File uploader object
            st.session_state['df'],st.session_state['File_uploader_object'] = read_file()

            if not st.session_state['df'].empty:
            
                # assign to session state
                uploaded_file = st.session_state['File_uploader_object']

                with st.expander('Data Display'):
                    # Function call : display uploaded file on application
                    display_uploaded_data()

                with st.expander("Data Description"):
                    # Function call : display basic details regarding the dataset
                    display_uploaded_data_info()

                with st.expander("BAU Report"):
                    bau_report()
                    
            else:

                st.error("Kindly Upload your file!")

        elif choose_option == 'Chat with pdf':
            st.session_state['raw_text'],st.session_state['File_uploader_object'] = read_file()
            if len(st.session_state['raw_text']) != 0 :
                with st.expander("Data Display"):
                    st.write(st.session_state['raw_text'])
                with st.expander("Data Description"):
                    # Function call : display basic details regarding the dataset
                    st.write('nothing')
            
            else:
                st.error("Kindly Upload your file!")
            
        elif choose_option == 'Chat with video':
            st.session_state['raw_text'],st.session_state['File_uploader_object'] = read_file()
            if len(st.session_state['raw_text']) != 0 :
                with st.expander("Data Display"):
                    st.write(st.session_state['raw_text'])
                with st.expander("Data Description"):
                    # Function call : display basic details regarding the dataset
                    st.write('nothing')
            
            else:
                st.error("Kindly Upload your file!")
                    

    with col2 :
        
        if not st.session_state['df'].empty and choose_option == 'Chat with csv(Single query)':
            chat_bot_Pandasai_api()
        elif not st.session_state['df'].empty and choose_option == 'Chat with csv(Conversational Chain)':
            
            chat_bot_llangchain_openapi_csv2(st.session_state['File_uploader_object'])
            
        elif len(st.session_state['raw_text']) != 0  and choose_option == 'Chat with pdf':
            chat_bot_llangchain_openapi_pdf()


if __name__ == "__main__":
    main()

    