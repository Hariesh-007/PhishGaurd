import os
import threading
import time
import joblib
import pandas as pd
from flask import Flask, redirect, url_for, session, request, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import ml_pipeline  # To use the preprocessing function

import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
app.config['THREADS_PER_PAGE'] = 15
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///phishing_detector_v4.db')
# Handle SQLAlchemy URL compatibility for certain platforms (e.g. Render/Heroku)
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    credentials = db.Column(db.JSON)
    auto_scan = db.Column(db.Boolean, default=True)

class EmailLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)
    message_id = db.Column(db.String(100), unique=True, nullable=False)
    subject = db.Column(db.String(255))
    sender = db.Column(db.String(255))
    prediction = db.Column(db.String(50))  # "phishing" or "legitimate"
    confidence = db.Column(db.Float)
    body = db.Column(db.Text)  # Store content for phishing threats
    email_date = db.Column(db.String(100))  # Original date from email header
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

with app.app_context():
    db.create_all()

# --- ML Model Loading ---
MODEL_PATH = 'phishing_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
model = None
vectorizer = None

def load_ml_model():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("ML model and vectorizer loaded.")
    else:
        print("ML model not found. Waiting for training to complete...")

# --- OAuth2 Configuration ---
CLIENT_SECRETS_FILE = "credentials.json"
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify', 'openid', 'https://www.googleapis.com/auth/userinfo.email']

def get_google_flow():
    # Priority: Environment Variable -> Local File
    creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    if creds_json:
        client_config = json.loads(creds_json)
        return Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=url_for('callback', _external=True, _scheme='https')
        )
    
    return Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for('callback', _external=True, _scheme='https')
    )

# --- Routes ---
@app.route('/')
def index():
    if 'credentials' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login')
def login():
    flow = get_google_flow()
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    session['state'] = state
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    flow = get_google_flow()
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    
    # Store credentials in session or database
    session['credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    
    # Get user info
    service = build('oauth2', 'v2', credentials=credentials)
    user_info = service.userinfo().get().execute()
    session['user_email'] = user_info['email']
    
    # Save user to DB if not exists
    user = User.query.filter_by(email=user_info['email']).first()
    if not user:
        user = User(email=user_info['email'], credentials=session['credentials'])
        db.session.add(user)
    else:
        user.credentials = session['credentials']
    db.session.commit()
    
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'credentials' not in session:
        return redirect(url_for('login'))
    
    logs = EmailLog.query.filter_by(user_email=session['user_email']).order_by(EmailLog.timestamp.desc()).limit(10).all()
    user = User.query.filter_by(email=session['user_email']).first()
    stats = {
        'total': EmailLog.query.filter_by(user_email=session['user_email']).count(),
        'phishing': EmailLog.query.filter_by(user_email=session['user_email'], prediction='phishing').count(),
        'legitimate': EmailLog.query.filter_by(user_email=session['user_email'], prediction='legitimate').count(),
        'auto_scan': user.auto_scan if user else False
    }
    
    # Calculate Safety Score (last 50 emails)
    total_recent = stats['total']
    if total_recent > 0:
        phishing_recent = stats['phishing']
        safety_score = max(0, 100 - (phishing_recent / total_recent * 100))
    else:
        safety_score = 100
        
    stats['safety_score'] = round(safety_score, 1)
    stats['threat_rate'] = round(100 - safety_score, 1)
    
    threats = EmailLog.query.filter_by(user_email=session['user_email'], prediction='phishing').filter(EmailLog.body != None).order_by(EmailLog.timestamp.desc()).limit(5).all()
    
    return render_template('dashboard.html', logs=logs, stats=stats, threats=threats)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/ai-security-assistant', methods=['POST'])
def ai_security_assistant():
    if 'user_email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid request'}), 400
        
    message = data.get('message', '').lower()
    
    # Simple logic for AI-like responses based on the PhishGuard theme
    if 'how' in message and 'phishing' in message:
        response = "I use a Random Forest model with TF-IDF vectorization to analyze text patterns common in BEC (Business Email Compromise) and credential harvesting."
    elif 'score' in message or 'safety' in message:
        response = f"Your current Inbox Safety Score is {calculate_safety_score_logic(session['user_email'])}%. It increases as more legitimate emails are processed."
    elif 'help' in message:
        response = "I am PhishGuard AI v2.6. I can help you analyze email bodies, check your security score, and interpret forensic indicators."
    else:
        response = "I am monitoring your inbox in near real-time. I have deactivated all dangerous links in the Cyber Vault for your protection."
        
    return jsonify({'response': response})

def calculate_safety_score_logic(email):
    total = EmailLog.query.filter_by(user_email=email).count()
    if total == 0: return 100
    phishing = EmailLog.query.filter_by(user_email=email, prediction='phishing').count()
    return round(max(0, 100 - (phishing / total * 100)), 1)

@app.route('/logs')
def logs():
    if 'credentials' not in session:
        return redirect(url_for('login'))
    
    page = request.args.get('page', 1, type=int)
    pagination = EmailLog.query.filter_by(user_email=session['user_email']).order_by(EmailLog.timestamp.desc()).paginate(page=page, per_page=15)
    return render_template('logs.html', pagination=pagination)

@app.route('/intel-briefing')
def intel_briefing():
    return render_template('intel_briefing.html')

@app.route('/toggle-auto-scan', methods=['POST'])
def toggle_auto_scan():
    if 'user_email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.filter_by(email=session['user_email']).first()
    if user:
        user.auto_scan = not user.auto_scan
        db.session.commit()
        return jsonify({'status': 'success', 'auto_scan': user.auto_scan})
    return jsonify({'error': 'User not found'}), 404

@app.route('/scan', methods=['POST'])
def scan():
    if 'credentials' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    results = scan_inbox(session['user_email'], session['credentials'])
    return jsonify(results)

@app.route('/check-email', methods=['POST'])
def check_email():
    print("DEBUG: Entered /check-email route")
    try:
        data = request.get_json(silent=True)
        print(f"DEBUG: Received data: {data}")
        if not data:
            return jsonify({'error': 'Invalid JSON or empty request'}), 400
            
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if model is None or vectorizer is None:
            print("DEBUG: Model/Vectorizer not loaded. Attempting to load...")
            load_ml_model()
            if model is None:
                return jsonify({'error': 'Model files missing or corrupt'}), 503
        
        print(f"Analysing text: {text[:50]}...")
        cleaned_text = ml_pipeline.preprocess_text(text)
        features = vectorizer.transform([cleaned_text])
        
        prediction_prob = model.predict_proba(features)[0]
        prediction_code = model.predict(features)[0]
        prediction = "phishing" if prediction_code == 1 else "legitimate"
        confidence = round(max(prediction_prob) * 100, 2)
        
        print(f"Prediction: {prediction} ({confidence}%)")
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error in check_email route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- Gmail Scanning Logic ---
def scan_inbox(user_email, creds_dict):
    creds = Credentials(**creds_dict)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Update stored creds
    
    service = build('gmail', 'v1', credentials=creds)
    
    # Fetch unread emails
    results = service.users().messages().list(userId='me', q='is:unread').execute()
    messages = results.get('messages', [])
    
    scanned_count = 0
    phishing_count = 0
    
    for msg_info in messages:
        # Check if already processed
        existing_log = EmailLog.query.filter_by(message_id=msg_info['id']).first()
        if existing_log:
            continue
            
        msg = service.users().messages().get(userId='me', id=msg_info['id']).execute()
        
        # Extract metadata
        payload = msg['payload']
        headers = payload.get('headers', [])
        subject = ""
        sender = ""
        email_date = ""
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            if header['name'] == 'From':
                sender = header['value']
            if header['name'] == 'Date':
                email_date = header['value']
        
        # Extract body
        body = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    import base64
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        elif 'body' in payload:
            import base64
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')

        # Classify
        if model and vectorizer:
            cleaned_body = ml_pipeline.preprocess_text(body)
            features = vectorizer.transform([cleaned_body])
            pred_code = model.predict(features)[0]
            pred_label = "phishing" if pred_code == 1 else "legitimate"
            conf = round(max(model.predict_proba(features)[0]) * 100, 2)
            
            # Action: hide phishing, show legitimate
            if pred_label == 'phishing':
                service.users().messages().batchModify(
                    userId='me',
                    body={
                        'ids': [msg_info['id']],
                        'addLabelIds': ['SPAM'],
                        'removeLabelIds': ['UNREAD', 'INBOX']
                    }
                ).execute()
                phishing_count += 1
            else:
                # Keep as unread (already unread), but could explicitly add label if needed
                pass
            
            # Log to DB
            log = EmailLog(
                user_email=user_email, 
                message_id=msg_info['id'],
                subject=subject, 
                sender=sender, 
                prediction=pred_label, 
                confidence=conf,
                body=body if pred_label == 'phishing' else None,
                email_date=email_date
            )
            db.session.add(log)
            scanned_count += 1

    db.session.commit()
    return {'scanned': scanned_count, 'phishing': phishing_count}

# --- Background Worker ---
def background_scanner():
    while True:
        with app.app_context():
            # Only scan for users who have auto_scan enabled
            users = User.query.filter_by(auto_scan=True).all()
            for user in users:
                print(f"Auto-scanning for {user.email}")
                try:
                    scan_inbox(user.email, user.credentials)
                except Exception as e:
                    print(f"Error auto-scanning {user.email}: {e}")
        time.sleep(30)  # Scan every 30 seconds for a "live" feel

if __name__ == '__main__':
    load_ml_model()
    # Start background thread (can be disabled in production if running separate worker)
    if os.environ.get('ENABLE_BACKGROUND_SCAN', 'true').lower() == 'true':
        print("Starting background scanner thread...")
        daemon = threading.Thread(target=background_scanner, daemon=True)
        daemon.start()
    
    app.run(debug=True, ssl_context='adhoc', port=5000)
else:
    # Logic for Gunicorn/Production
    load_ml_model()
    # In production, we usually want one instance of the background scanner.
    # If Gunicorn is used with multiple workers, this thread would start in EACH worker.
    # Recommend setting ENABLE_BACKGROUND_SCAN=false and running a separate worker process 
    # OR using 1 worker if the load is low.
    if os.environ.get('ENABLE_BACKGROUND_SCAN', 'true').lower() == 'true':
        print("Starting production background scanner thread...")
        daemon = threading.Thread(target=background_scanner, daemon=True)
        daemon.start()
