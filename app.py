import openai 
import streamlit as st
import numpy as np
import pandas as pd 
import tiktoken
from streamlit_chat import message


COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

# General bits
MAX_SECTION_LEN = 750
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))
#f"Context separator contains {separator_len} tokens"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.05,
    "max_tokens": 600,
    "model": COMPLETIONS_MODEL,
}

openai.api_key = st.secrets["OpenAIToken"]



def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    df = pd.read_csv(fname, header=0,compression="gzip")
    max_dim = max([int(c) for c in df.columns if c != "hash"])
    return {
           (r.hash): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return { idx: get_embedding(r.content) for idx, r in df.iterrows() }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame, style: str) -> str:
    """
    Fetch relevant 
    """

    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    if style == "simple":
        STYLE = " in the elaborate style of a management consultant, "
    elif style == "advanced":
        STYLE = " in the style of 10-year old, "
    else:
        STYLE = ""
    header = "Answer the question as truthfully as possible using the provided context," +STYLE+""" and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    prompt = header.replace("  "," ") + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    return prompt, most_relevant_document_sections, chosen_sections, chosen_sections_indexes



def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    style: str,
    show_prompt: bool = False
) -> str:
    

    prompt, most_relevant_document_sections, chosen_sections, chosen_sections_indexes = construct_prompt(
        query,
        document_embeddings,
        df,
        style
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    answer = response["choices"][0]["text"].strip(" \n")

    return answer, most_relevant_document_sections, chosen_sections, chosen_sections_indexes




# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ","What are lenders technical assistance due diligences?", key="input")
    return input_text


@st.cache(hash_funcs={pd.core.frame.DataFrame: id},\
    allow_output_mutation=True,suppress_st_warning=True)
def loadLargeFilesOnce():
    document_embeddings = load_embeddings("embedding.csv.gzip")
    df = pd.read_parquet("all_pages.parquet.gzip")
    return document_embeddings, df


def generate_response(query,df, document_embeddings, style=""):
    answer, most_relevant_document_sections, chosen_sections, chosen_sections_indexes = answer_query_with_context(query, df, document_embeddings, style)

    if len(chosen_sections_indexes):
        st.sidebar.write("### Source(s)\n")
        for ix, row in df[df.IDX.isin(chosen_sections_indexes)].iterrows():
            st.sidebar.write("* ["+row.title+"]("+row.url+")")
    else:
        st.sidebar.write("### No source found")

    return answer


document_embeddings, df = loadLargeFilesOnce()


st.title("MM AMA : MM Website + openAI")
# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

    
st.sidebar.write("### __Parameters__")   
st.sidebar.write("#### Experimental")   

if "visibility" not in st.session_state:
    st.session_state.temperature = 0.0
    st.session_state.style = False

temp = st.sidebar.slider(
    'Temperature',
    0.0, 1.0,value= st.session_state.temperature ,step=0.01)
st.session_state.temperature = temp
COMPLETIONS_API_PARAMS["temperature"] = temp


style = st.sidebar.selectbox(
        "What style ?",
        ("normal","simple", "advanced"), 
    )
st.session_state.style = style

st.sidebar.write("* _Temp.:_",COMPLETIONS_API_PARAMS["temperature"])
st.sidebar.write("* _Style:_",style)


user_input = get_text()

if user_input:
    output = generate_response(user_input,df, document_embeddings,st.session_state.style)
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')