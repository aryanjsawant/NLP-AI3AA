import re
import json
import pickle
from collections import Counter
from datasets import load_dataset
import unicodedata

class GujaratiTokenizer:
    def __init__(self):
        # Gujarati Unicode ranges
        # Main Gujarati block: U+0A80-U+0AFF
        # This includes base characters, matras, and other diacritical marks
        self.gujarati_char_pattern = r'[\u0A80-\u0AFF]'
        
        # Enhanced patterns for different token types
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://[^\s]+|www\.[^\s]+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*',
            'decimal_number': r'\b\d+\.\d+\b',
            'integer': r'\b\d+\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'punctuation': r'[।॥.,!?;:"\'()[\]{}\-–—_/\\@#$%^&*+=<>|`~]',
            # Gujarati word: sequence of Gujarati characters (including matras)
            'gujarati_word': self.gujarati_char_pattern + r'+',
            # English word: sequence of English letters
            'english_word': r'[A-Za-z]+',
            # Whitespace
            'whitespace': r'\s+'
        }
        
        # Combine all patterns for tokenization
        self.token_pattern = '|'.join(f'({pattern})' for pattern in self.patterns.values())
        self.compiled_pattern = re.compile(self.token_pattern)
        
        # Sentence boundary patterns for Gujarati
        self.sentence_pattern = re.compile(r'[।॥.!?]+\s*')
    
    def sentence_tokenize(self, text):
        """Tokenize text into sentences"""
        # Split on Gujarati sentence terminators (।, ॥) and common punctuation
        sentences = self.sentence_pattern.split(text.strip())
        # Remove empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def word_tokenize(self, text):
        """Tokenize text into words and other tokens"""
        tokens = []
        matches = self.compiled_pattern.findall(text)
        
        for match in matches:
            # Find the non-empty group in the match tuple
            token = next(group for group in match if group)
            
            # Skip pure whitespace tokens
            if not re.match(r'^\s+$', token):
                tokens.append(token)
        
        return tokens
    
    def classify_token(self, token):
        """Classify a token into its type"""
        if re.match(self.patterns['email'], token):
            return 'email'
        elif re.match(self.patterns['url'], token):
            return 'url'
        elif re.match(self.patterns['decimal_number'], token):
            return 'decimal_number'
        elif re.match(self.patterns['date'], token):
            return 'date'
        elif re.match(self.patterns['integer'], token):
            return 'integer'
        elif re.match(self.patterns['punctuation'], token):
            return 'punctuation'
        elif re.match(self.patterns['gujarati_word'], token):
            return 'gujarati_word'
        elif re.match(self.patterns['english_word'], token):
            return 'english_word'
        else:
            return 'other'
    
    def tokenize_paragraph(self, paragraph):
        """Tokenize a paragraph into sentences and words"""
        sentences = self.sentence_tokenize(paragraph)
        result = {
            'original_text': paragraph,
            'sentences': [],
            'total_words': 0,
            'total_characters': len(paragraph)
        }
        
        for sentence in sentences:
            words = self.word_tokenize(sentence)
            classified_words = [(word, self.classify_token(word)) for word in words]
            
            result['sentences'].append({
                'text': sentence,
                'words': words,
                'classified_words': classified_words,
                'word_count': len(words)
            })
            result['total_words'] += len(words)
        
        return result

def process_dataset(dataset, max_examples=1000):
    """Process the dataset and extract tokenized data"""
    tokenizer = GujaratiTokenizer()
    processed_data = []
    
    print(f"Processing up to {max_examples} examples...")
    
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
            
        if i % 100 == 0:
            print(f"Processed {i} examples...")
        
        # Get the text content
        text = example.get('text', '')
        if not text.strip():
            continue
            
        # Tokenize the paragraph
        tokenized = tokenizer.tokenize_paragraph(text)
        processed_data.append(tokenized)
    
    print(f"Completed processing {len(processed_data)} examples")
    return processed_data

def save_tokenized_data(data, base_filename='gujarati_tokenized'):
    """Save tokenized data in multiple formats"""
    
    # Save as JSON (human readable)
    with open(f'{base_filename}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Save summary statistics as text
    stats = compute_corpus_statistics(data)
    with open(f'{base_filename}_stats.txt', 'w', encoding='utf-8') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Data saved as:")
    print(f"- {base_filename}.json (JSON format)")
    print(f"- {base_filename}.pkl (Pickle format)")  
    print(f"- {base_filename}_stats.txt (Statistics)")

def compute_corpus_statistics(processed_data):
    """Compute corpus statistics"""
    total_sentences = 0
    total_words = 0
    total_characters = 0
    all_words = []
    
    for document in processed_data:
        total_characters += document['total_characters']
        
        for sentence in document['sentences']:
            total_sentences += 1
            sentence_words = sentence['words']
            total_words += len(sentence_words)
            all_words.extend(sentence_words)
    
    # Calculate statistics
    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    
    # Calculate average word length (in characters)
    total_word_chars = sum(len(word) for word in all_words)
    avg_word_length = total_word_chars / total_words if total_words > 0 else 0
    
    # Calculate Type-Token Ratio (TTR)
    unique_words = set(all_words)
    ttr = len(unique_words) / total_words if total_words > 0 else 0
    
    statistics = {
        'Total number of sentences': total_sentences,
        'Total number of words': total_words,
        'Total number of characters': total_characters,
        'Average sentence length (words per sentence)': round(avg_sentence_length, 2),
        'Average word length (characters per word)': round(avg_word_length, 2),
        'Type-Token Ratio (TTR)': round(ttr, 4),
        'Total unique words': len(unique_words)
    }
    
    return statistics

# def main():
#     """Main function to run the complete pipeline"""
    
#     # Step 1: Load the dataset
#     print("Loading Gujarati dataset...")
#     dataset = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="guj_Gujr", streaming=True)
    
#     # Step 2: Process and tokenize the data
#     processed_data = process_dataset(dataset, max_examples=1000)  # Adjust max_examples as needed
    
#     # Step 3: Save the tokenized data
#     print("Saving tokenized data...")
#     save_tokenized_data(processed_data)
    
#     # Step 4: Compute and display statistics
#     print("\nComputing corpus statistics...")
#     stats = compute_corpus_statistics(processed_data)
    
#     print("\n" + "="*50)
#     print("CORPUS STATISTICS")
#     print("="*50)
#     for key, value in stats.items():
#         print(f"{key}: {value}")
    
#     return processed_data, stats

# # Example usage and testing
# if __name__ == "__main__":
#     # Test the tokenizer with some sample text
#     tokenizer = GujaratiTokenizer()
    
#     # Test sentence
#     test_text = "આજે હું દુકાને ગયો. મેં ખાણાની વસ્તુઓ ખરીદી। મારું ઇમેઇલ test@example.com છે. આજની તારીખ 25/07/2025 છે."
    
#     print("Testing tokenizer:")
#     print(f"Input: {test_text}")
#     print("\nSentences:")
#     sentences = tokenizer.sentence_tokenize(test_text)
#     for i, sent in enumerate(sentences, 1):
#         print(f"{i}. {sent}")
    
#     print("\nWords:")
#     words = tokenizer.word_tokenize(test_text)
#     for word in words:
#         word_type = tokenizer.classify_token(word)
#         print(f"'{word}' -> {word_type}")
    
#     # Uncomment the following line to run the main pipeline
# main()