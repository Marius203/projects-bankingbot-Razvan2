from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag_core import initialize_rag_system
from document_processor import process_single_document
from werkzeug.utils import secure_filename
import config
import os
import uuid
import threading
import chat_history

chat_history.init_db()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global RAG system
rag_chain = None
current_subject = "ASC"

# Track upload tasks
upload_tasks = {}  # {task_id: {status, progress, message, error}}

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
    data = request.get_json() or {}
    session_id = data.get('session_id') or str(uuid.uuid4())
    message = (data.get('message') or "").strip()
    if not message:
        return jsonify({'error':'Empty message'}), 400

    chat_history.save_message(session_id, 'user', message)

    try:
        if rag_chain is not None:
            ai_resp = rag_chain.invoke(message)  # keep your existing call
            ai_text = str(ai_resp)
        else:
            ai_text = f"Echo: {message}"
    except Exception as e:
        ai_text = f"[error] {e}"

    chat_history.save_message(session_id, 'assistant', ai_text)
    history = chat_history.get_history(session_id)
    return jsonify({'session_id': session_id, 'response': ai_text, 'history': history})

@app.route('/history', methods=['GET'])
def history():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'session_id missing'}), 400
    return jsonify({'session_id': session_id, 'history': chat_history.get_history(session_id)})

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

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Handle document upload and start processing"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'Niciun fișier selectat'}), 400

        file = request.files['file']
        subject = request.form.get('subject', 'ASC')

        if file.filename == '':
            return jsonify({'error': 'Niciun fișier selectat'}), 400

        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Doar fișiere PDF sunt acceptate'}), 400

        # Validate subject exists
        available_subjects = get_available_subjects()
        if subject not in available_subjects:
            return jsonify({'error': 'Materie invalidă'}), 400

        # Secure filename
        filename = secure_filename(file.filename)

        # Create subject folder and Uploaded subfolder if they don't exist
        subject_folder = os.path.join(config.DOCS_ROOT_PATH, subject)
        uploaded_folder = os.path.join(subject_folder, "Uploaded")
        os.makedirs(uploaded_folder, exist_ok=True)

        # Save file to Uploaded subfolder
        file_path = os.path.join(uploaded_folder, filename)
        file.save(file_path)

        # Create task
        task_id = str(uuid.uuid4())
        upload_tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Inițializare...',
            'error': None
        }

        # Start background processing
        thread = threading.Thread(
            target=process_document_background,
            args=(task_id, file_path, subject)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'task_id': task_id,
            'filename': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_progress/<task_id>', methods=['GET'])
def upload_progress(task_id):
    """Get progress of document processing"""
    if task_id not in upload_tasks:
        return jsonify({'error': 'Task nu există'}), 404

    return jsonify(upload_tasks[task_id])

@app.route('/uploaded_files/<subject>', methods=['GET'])
def get_uploaded_files(subject):
    """Get list of uploaded files for a subject"""
    try:
        uploaded_folder = os.path.join(config.DOCS_ROOT_PATH, subject, "Uploaded")

        if not os.path.exists(uploaded_folder):
            return jsonify({'files': []})

        files = []
        for filename in os.listdir(uploaded_folder):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(uploaded_folder, filename)
                file_stats = os.stat(file_path)
                file_size_mb = file_stats.st_size / (1024 * 1024)

                files.append({
                    'name': filename,
                    'size_mb': round(file_size_mb, 2),
                    'uploaded_date': file_stats.st_mtime
                })

        # Sort by upload date (newest first)
        files.sort(key=lambda x: x['uploaded_date'], reverse=True)

        return jsonify({'files': files})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_file/<subject>/<filename>', methods=['DELETE'])
def delete_uploaded_file(subject, filename):
    """Delete an uploaded file"""
    try:
        # Security: only allow deletion from Uploaded subfolder
        uploaded_folder = os.path.join(config.DOCS_ROOT_PATH, subject, "Uploaded")
        safe_name = secure_filename(filename)
        file_path = os.path.join(uploaded_folder, safe_name)

        # Ensure file_path is inside uploaded_folder
        if os.path.commonpath([os.path.abspath(file_path), os.path.abspath(uploaded_folder)]) != os.path.abspath(uploaded_folder):
            return jsonify({'error': 'Acces interzis'}), 403

        if not os.path.exists(file_path):
            return jsonify({'error': 'Fișierul nu există'}), 404

        os.remove(file_path)

        return jsonify({'success': True, 'message': 'Fișier șters'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/conversations', methods=['GET'])
def conversations():
    return jsonify({'conversations': chat_history.list_sessions()})

@app.route('/delete_conversation', methods=['POST'])
def delete_conversation():
    sid = (request.json or {}).get('session_id')
    if not sid: return jsonify({'error':'session_id missing'}), 400
    chat_history.delete_session(sid)
    return jsonify({'success': True})

def process_document_background(task_id, file_path, subject):
    """Background task to process document"""
    global rag_chain, current_subject

    def update_progress(progress, message):
        upload_tasks[task_id]['progress'] = progress
        upload_tasks[task_id]['message'] = message

    try:
        # Process document
        result = process_single_document(file_path, subject, update_progress)

        if result['success']:
            upload_tasks[task_id]['status'] = 'done'
            upload_tasks[task_id]['progress'] = 100
            upload_tasks[task_id]['message'] = f"Document adăugat! ({result['chunks_added']} fragmente)"

            # Auto-reload RAG if processing for current subject
            if subject == current_subject:
                update_progress(100, "Reîncărcare sistem RAG...")
                try:
                    rag_chain = initialize_rag_system(subject)
                except Exception as e:
                    print(f"[app] error reinitializing rag: {e}")

        else:
            upload_tasks[task_id]['status'] = 'error'
            upload_tasks[task_id]['error'] = result.get('error', 'Eroare necunoscută')

    except Exception as e:
        upload_tasks[task_id]['status'] = 'error'
        upload_tasks[task_id]['error'] = str(e)

if __name__ == '__main__':
    # Initialize RAG system on startup
    print("Initializing RAG system...")
    try:
        rag_chain = initialize_rag_system(current_subject)
        print(f"RAG system initialized for subject: {current_subject}")
    except Exception as e:
        print(f"[app] warning: could not initialize RAG system: {e}. continuing with echo fallback.")

    # Run Flask app
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)