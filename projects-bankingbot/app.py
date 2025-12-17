from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag_core import initialize_rag_system
import config
import os

app = Flask(__name__)
CORS(app)

# Global RAG system
rag_chain = None
current_subject = "ASC"

def get_available_subjects():
    """Get list of available subjects from DOCS folder"""
    subjects = ["ASC"]
    if os.path.exists(config.DOCS_ROOT_PATH):
        dirs = [d for d in os.listdir(config.DOCS_ROOT_PATH)
                if os.path.isdir(os.path.join(config.DOCS_ROOT_PATH, d))]
        if dirs:
            subjects = sorted(dirs)
    return subjects

@app.route('/')
def home():
    """Serve the main chat page"""
    subjects = get_available_subjects()
    return render_template('index.html', subjects=subjects, current_subject=current_subject)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    global rag_chain

    try:
        data = request.json
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'error': 'Empty message'}), 400

        if rag_chain is None:
            return jsonify({'error': 'RAG system not initialized'}), 500

        # Get AI response
        response = rag_chain.invoke(message)

        return jsonify({
            'response': response,
            'subject': current_subject
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/change_subject', methods=['POST'])
def change_subject():
    """Change the current subject"""
    global rag_chain, current_subject

    try:
        data = request.json
        new_subject = data.get('subject', 'ASC')

        # Reinitialize RAG system with new subject
        rag_chain = initialize_rag_system(new_subject)
        current_subject = new_subject

        return jsonify({
            'success': True,
            'subject': current_subject
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/subjects', methods=['GET'])
def get_subjects():
    """Get list of available subjects"""
    subjects = get_available_subjects()
    return jsonify({
        'subjects': subjects,
        'current': current_subject
    })

if __name__ == '__main__':
    # Initialize RAG system on startup
    print("Initializing RAG system...")
    rag_chain = initialize_rag_system(current_subject)
    print(f"RAG system initialized for subject: {current_subject}")

    # Run Flask app
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
