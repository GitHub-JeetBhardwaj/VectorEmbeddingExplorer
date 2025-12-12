# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import PyPDF2
from nltk.corpus import stopwords
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import nltk
import logging
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_and_preprocess(pdf_path):
    corpus = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                flash("Error: The PDF is password-protected or encrypted.")
                return None

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    corpus += text + " "
    except PyPDF2.errors.PdfReadError as e:
        logging.error(f"PDF reading error: {e}")
        flash("Error: The PDF file is corrupted or unreadable.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error reading PDF: {e}")
        flash("Error: Could not read the PDF file.")
        return None

    if not corpus.strip():
        flash("No text could be extracted from the PDF. It might be scanned (image-based) or empty.")
        return None

    corpus = corpus.replace("\n", " ").replace("\r", " ")

    story = []
    for sentence in sent_tokenize(corpus):
        tokens = simple_preprocess(sentence)
        filtered_tokens = [token for token in tokens if token not in stop_words]
        if filtered_tokens:
            story.append(filtered_tokens)
    
    if len(story)<=30:
        flash("No valid sentences found after preprocessing. The document may contain too little text.")
        return None

    return story

def train_word2vec(story):
    model = Word2Vec(window=20, min_count=2, vector_size=100, seed=42)
    model.build_vocab(story)
    model.train(story, total_examples=model.corpus_count, epochs=100)
    return model

def perform_pca(model):
    vectors = model.wv.get_normed_vectors()
    words = model.wv.index_to_key
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(vectors)
    return pca_result, words

def generate_3d_plot(pca_result, words, highlight_words=None, subset_start=0, subset_end=200):
    # Limit to a subset for performance
    pca_subset = pca_result[subset_start:subset_end]
    words_subset = words[subset_start:subset_end]
    
    df = pd.DataFrame(pca_subset, columns=['x', 'y', 'z'])
    df['word'] = words_subset
    df['color'] = 'Other'
    
    if highlight_words:
        df['color'] = df['word'].apply(lambda w: 'Highlight' if w in highlight_words else 'Other')
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', text='word',
                        color_discrete_map={'Other': '#A7D8F0', 'Highlight': 'red'},
                        title='3D PCA Projection of Word Embeddings for first 200 Vectors')
    fig.update_traces(marker=dict(size=5))
    
    # Save to HTML string
    html = fig.to_html(
    full_html=False,
    include_plotlyjs='cdn',
    div_id="plotly-div",
    config={'responsive': True},
    default_width="100%",
    default_height="100%"
)
    return html

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if not file or not allowed_file(file.filename):
            flash('Please upload a valid PDF file.')
            return redirect(request.url)

        filename = file.filename
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        
        try:
            story = extract_and_preprocess(pdf_path)
            if story is None:
                os.unlink(pdf_path)  # Clean up
                return redirect(request.url)

            model = train_word2vec(story)
            
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model.bin')
            model.save(model_path)
            
            flash('PDF processed successfully! Model trained with {} unique words.'.format(len(model.wv)))
            return redirect(url_for('explore', model_path=model_path, pdf_filename=filename))
            
        except Exception as e:
            logging.exception("Error during processing")
            flash(f'Processing failed: {str(e)}')
            # Clean up partial files
            for path in [pdf_path, os.path.join(app.config['UPLOAD_FOLDER'], 'model.bin')]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/explore')
def explore():
    model_path = request.args.get('model_path')
    pdf_filename = request.args.get('pdf_filename')
    
    if not model_path or not os.path.exists(model_path):
        flash('No model found. Please upload a PDF first.')
        return redirect(url_for('index'))
    
    model = Word2Vec.load(model_path)
    pca_result, words = perform_pca(model)
    
    # Generate full plot
    full_plot_html = generate_3d_plot(pca_result, words)
    
    return render_template('explore.html', pdf_filename=pdf_filename, words=words[:500], full_plot_html=full_plot_html)  # Limit words for dropdown

@app.route('/query', methods=['POST'])
def query():
    model_path = request.args.get('model_path')
    word = request.form.get('word')
    
    if not model_path or not os.path.exists(model_path):
        flash('No model found.')
        return redirect(url_for('index'))
    
    model = Word2Vec.load(model_path)
    pca_result, words = perform_pca(model)
    
    try:
        similar = model.wv.most_similar(word, topn=10)
        similar_words = [w for w, _ in similar]
        similar_words.append(word)  # Include the query word
        
        # Generate highlighted plot
        highlight_plot_html = generate_3d_plot(pca_result, words, highlight_words=similar_words)
        
        return render_template('result.html', word=word, similar=similar, highlight_plot_html=highlight_plot_html)
    except KeyError:
        flash(f'Word "{word}" not found in vocabulary.')
        return redirect(url_for('explore', model_path=model_path))

if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=7860, debug=False)
else:
    # For Hugging Face Spaces (gunicorn)
    import gunicorn.app.base
    # This allows gunicorn to import your app
    pass
