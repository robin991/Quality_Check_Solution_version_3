import streamlit as st
import pandas as pd 
import io

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
        st.session_state['generated'] = ['Hi I am Rio GPT. Your friendly neighbourhood genrative-AI pwoered conversational assistant.']
    
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hey! ']

def read_file(file_name : str) -> pd.DataFrame: 
    # Function to read file and return and dataframe and Fileuploader object
    df_uploader = st.file_uploader("✳️Upload your file here", type = ['xlsx','csv'],accept_multiple_files=False)

    if df_uploader is not None :

        if df_uploader.name[-3:] == 'csv':
            df = pd.read_csv(df_uploader)
        elif df_uploader.name[-3:] == 'xlsx': 
            df = pd.read_excel(df_uploader)
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

def chat_bot_llangchain_openapi() -> None:

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

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user = True,
                        key = str(i) + '_user',
                        avatar_style = "big-smile")
                message(st.session_state["generated"][i],
                        key = str(i) ,
                        avatar_style = "thumbs")
#@st.cache_data
def Table_creation(df):
        
        Final_table = df.groupby("Merchant")["Credit amount","Redemption","Spend"].agg({"Credit amount":'sum',"Redemption":'sum',"Spend":'sum'})

        Final_table["Net Spend"] = Final_table['Spend']- Final_table["Credit amount"]

        Final_table = Final_table.sort_values("Net Spend",ascending= False)
        
        Final_table = Final_table.head(5)
        return Final_table


# page setting configuration
page_configuration()


#initializing session state variables
initialize_session_state()

# adding page title
st.markdown("<h1 style='text-align: center; color: Black;'>RIO-GPT</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: Black;'>Our friendly neighbourhood chat bot ready to assit RIO whenever there is a need of support.</h3>", unsafe_allow_html=True)
# subheader
#st.subheader( 'Our friendly neighbourhood chat bot ready to assit RIO whenever there is a need of support.',
            
st.divider()
col1, col2 = st.columns(2)

with col1 :
    #Read file from column  and  File uploader object
    st.session_state['df'],st.session_state['File_uploader_object'] = read_file(file_name = "File1")
    
    choose_option = st.radio("***Choose chat solution:***", ['Chat with excel(PandasAI)', 'Chat with excel(Conversation Chain)'])

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
            # Function call : dispalay regular BAU analysis
            df = st.session_state['df']

            # resetting the date column
            df["Date"] = pd.to_datetime(df["Date"])
            st.session_state['df'] =df

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
    else:

        st.error("Kindly Upload your file!")
with col2 :
    
    if not st.session_state['df'].empty and choose_option == 'Chat with excel(PandasAI)':
        chat_bot_Pandasai_api()
    elif not st.session_state['df'].empty and choose_option == 'Chat with excel(Conversation Chain)':
        chat_bot_llangchain_openapi()
    

    