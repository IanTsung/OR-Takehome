import os
import json
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class DataProcessor:
  def __init__(self, pdf_dir, output_dir) -> None:
    self.pdf_dir = pdf_dir
    self.stopwords = set(stopwords.words("english"))
    self.lemmatiser = WordNetLemmatizer()
    self.output = output_dir
    
  def get_wordnet_pos(self, word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"N": NOUN, "V": VERB, "J": ADJ, "R": ADV}
    return tag_dict.get(tag, NOUN)
  
  def preprocess(self, text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in self.stopwords]
    tokens = [self.lemmatiser.lemmatize(word, self.get_wordnet_pos(word)) for word in tokens]
    return ' '.join(tokens)
  
  def extract_abstracts(self, record_output=False):
    abstracts = {}
    titles = []
    
    for filename in os.listdir(self.pdf_dir):
      if filename.endswith(".pdf"):
        file_path = os.path.join(self.pdf_dir, filename)
        with open(file_path, 'rb') as file:
          reader = PyPDF2.PdfReader(file)
          text = ""
          
          for page in reader.pages:
            text += (page.extract_text() or "").lower()
          
          start = text.find("abstract")
          end = text.find("introduction", start)
          abstract = text[start+8:end].strip() if start != -1 and end != -1 else reader.pages[0].extract_text().strip() # if abstract not found, return the content on the first page
          abstracts[filename] = self.preprocess(abstract)
          titles.append(filename.replace(".pdf", ""))
          
    if record_output:
      with open(self.output, 'w', encoding='utf-8') as output_file:
        json.dump(abstracts, output_file, indent=4)
                 
    return list(abstracts.values()), titles
  
# # Test
# pdf_dir = 'Green Energy Dataset'
# output_dir = 'data/abstracts.json'
# processor = DataProcessor(pdf_dir, output_dir)
# preprocessed_abstracts = processor.extract_abstracts(record_output=True)