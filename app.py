import os
import io
from flask import Flask, request, jsonify, send_from_directory
from google import genai
from google.genai import types
import PyPDF2
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app) # Enable CORS for all routes

# Configure Gemini API with the provided key
API_KEY = "AIzaSyCUEWY7kSv9LM9vDYcx0sOr2vVYJN0HxEo"
client = genai.Client(api_key=API_KEY)

# Define models
MODEL_PRIMARY = 'gemini-2.0-flash'
MODEL_FALLBACK = 'gemini-2.0-flash'

# Store chat sessions per user (in-memory)
# Format: {session_id: {"history": [], "title": "Title", "pdf_context": "extacted text", "pdf_filename": "..." }}
chat_sessions = {}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Backend is running!"})

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Return all active chat sessions for the sidebar."""
    sessions = []
    for sid, data in chat_sessions.items():
        sessions.append({
            "id": sid,
            "title": data.get("title", "New Chat")
        })
    sessions.reverse()
    return jsonify(sessions)

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Return the message history for a specific session."""
    if session_id not in chat_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    chat_data = chat_sessions[session_id]
    history = []
    for msg in chat_data["history"]:
        # Extract text from parts. Content objects in google-genai have a parts list.
        text = ""
        if hasattr(msg, 'parts') and msg.parts:
            text = msg.parts[0].text
        
        history.append({
            "role": msg.role,
            "text": text
        })
    return jsonify(history)

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Receive a PDF, extract text, and store in the session context."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    session_id = request.form.get('session_id', 'default')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    try:
        # Read PDF content
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() + "\n"

        if not extracted_text.strip():
             return jsonify({'error': 'Could not extract any text from the PDF.'}), 400

        # Store in session (Initialize session if needed)
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "history": [],
                "title": f"PDF: {file.filename[:30]}",
                "model_type": 2.5
            }
        
        chat_sessions[session_id]['pdf_context'] = extracted_text
        chat_sessions[session_id]['pdf_filename'] = file.filename
        
        return jsonify({
            'success': True, 
            'filename': file.filename, 
            'context_length': len(extracted_text)
        })
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return jsonify({'error': f"Failed to process PDF: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Ensure session exists
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "history": [],
            "title": user_message[:40] + ("..." if len(user_message) > 40 else ""),
            "model_type": 2.5
        }

    chat_data = chat_sessions[session_id]
    
    # If PDF context is available, inject it into the prompt
    final_prompt = user_message
    if 'pdf_context' in chat_data:
        final_prompt = (
            f"The following is context from an uploaded PDF file ('{chat_data['pdf_filename']}'):\n"
            f"--- START OF PDF CONTEXT ---\n{chat_data['pdf_context']}\n--- END OF PDF CONTEXT ---\n\n"
            f"Based on the text above, please answer the user's question. If the answer is not in the PDF, please say so, but provide any relevant information from the document.\n\n"
            f"User Question: {user_message}"
        )

    try:
        # Create chat session with current history
        chat_session = client.chats.create(
            model=MODEL_PRIMARY,
            history=chat_data["history"]
        )
        response = chat_session.send_message(final_prompt)
        
        # Update history in the session storage using the correct method
        chat_data["history"] = chat_session.get_history()
        
        return jsonify({'response': response.text})

    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            print(f"Quota exceeded for {MODEL_PRIMARY}. Falling back to {MODEL_FALLBACK}...")
            try:
                # Fallback model attempt
                chat_session_fb = client.chats.create(
                    model=MODEL_FALLBACK,
                    history=chat_data["history"]
                )
                response_fb = chat_session_fb.send_message(final_prompt)
                chat_data["history"] = chat_session_fb.get_history()
                
                return jsonify({
                    'response': response_fb.text,
                    'notice': 'Notice: Switched to fallback because primary model quota was exceeded.'
                })
            except Exception as fb_e:
                 return jsonify({'error': f"Fallback Error: {str(fb_e)}"}), 500
        
        print(f"Error calling Gemini API: {error_msg}")
        return jsonify({'error': f"Gemini API Error: {error_msg}"}), 500

@app.route('/api/clear', methods=['POST'])
def clear():
    data = request.json
    session_id = data.get('session_id', 'default')
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:8080")
    app.run(debug=True, port=8080, host='0.0.0.0')
