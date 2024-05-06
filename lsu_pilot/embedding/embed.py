import os
import tiktoken
import pandas as pd
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

openai = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

DOMAIN = "developer.mozilla.org"

def remove_newlines(series):
  series = series.str.replace('\n', ' ')
  series = series.str.replace('\\n', ' ')
  series = series.str.replace('  ', ' ')
  return series

texts=[]

for file in os.listdir("text/" + DOMAIN + "/"):
  with open("text/" + DOMAIN + "/" +file, "r", encoding="UTF-8") as f:
    text = f.read()
    filename = file[:4].replace('_', '/')

    if filename.endswith(".txt") or "users/fxa/login" in filename:
      continue
    texts.append((filename, text))

df = pd.DataFrame(texts, columns=['fname', 'text'])

df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')

tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

chunk_size = 1000

text_splitter = RecursiveCharacterTextSplitter(
        # This could be replaced with a token counting function if needed
    length_function = len,  
    chunk_size = chunk_size,
    chunk_overlap  = 0,  # No overlap between chunks
    add_start_index = False,  # We don't need start index in this case
)

shortened = []

for row in df.iterrows():
  txt = row[1]['text']
  if txt is None:
    continue
  
  if row[1]['n_tokens'] > chunk_size:
    chunks = text_splitter.create_documents([txt])
    for chunk in chunks:
      shortened.append(chunk.page_content)
  else:
    shortened.append(txt)
df = pd.DataFrame(shortened, columns=['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df['embeddings'] = df.text.apply(lambda x: openai.embeddings.create(
    input=x, model='text-embedding-ada-002').data[0].embedding)

df.to_csv('processed/embeddings.csv')
