from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os, json, time, datetime, hashlib
import fitz  # PyMuPDF
import pandas as pd
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import tempfile

# === LangChain / Groq / Embeddings ===
from langchain.vectorstores import FAISS as FAISSStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# === App Setup ===
app = Flask(__name__)
app.secret_key = 'your-secret-key'
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
USERS_FILE = os.path.join(DATA_DIR, "users.json")
FREELANCER_QUERIES_FILE = os.path.join(DATA_DIR, "freelancer_queries.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")

# === Environment Variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === DB Models ===
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.String(100))
    user_email = db.Column(db.String(100))
    role = db.Column(db.String(20))
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

with app.app_context():
    db.create_all()
with app.app_context():
    db.create_all()
# ========== Helpers ==========
def hash_password(password): 
    return hashlib.sha256(password.encode()).hexdigest()

def load_json(path, default):
    if not os.path.exists(path): 
        return default
    with open(path, "r") as f: 
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f: 
        json.dump(data, f, indent=4)

def is_tech(msg):
    keywords = [
        # 👨‍💻 Programming Languages
        "python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "sql", "bash", "shell", "kotlin", "swift", "rust", "go",

        # 🛠️ Tech Topics
        "debug", "error", "exception", "syntax", "logic", "function", "variable", "loop", "class", "method", "oop", "inheritance", "compiler",

        # 🌐 Web Dev
        "react", "node", "express", "api", "rest", "http", "json", "frontend", "backend", "fullstack", "web", "form", "dom", "router",

        # 📊 Data Science
        "pandas", "numpy", "matplotlib", "data", "csv", "dataset", "cleaning", "eda", "exploration", "visualization", "statistics", "mean", "median", "standard deviation",

        # 🤖 Machine Learning & AI
        "model", "ai", "ml", "dl", "tensorflow", "keras", "sklearn", "training", "testing", "accuracy", "loss", "neural network", "regression", "classification", "clustering",

        # 🔠 NLP
        "tokenization", "stemming", "lemmatization", "transformer", "bert", "llm", "text", "nlp", "chatbot",

        # 🔌 Tools / Libraries
        "git", "github", "vscode", "docker", "flask", "django", "streamlit", "fastapi", "firebase", "supabase", "mongodb", "sqlalchemy",

        # 💼 Freelancing
        "client", "proposal", "upwork", "fiverr", "gig", "bid", "project delivery", "communication",

        # 🎓 General Tech Questions
        "project", "code", "script", "automation", "deployment", "prompt", "logic", "workflow"
    ]

    return any(k in msg.lower() for k in keywords)


def should_escalate(message, ai_response):
    uncertain_phrases = [
        "not sure", "refine", "cannot answer", "unclear", 
        "need human", "groq api", "escalate", 
        "i'm confused", "this is hard", 
        "i don’t get it", "please help", "explain more",
        "what should i do", "i need more help", 
        "real person", "manual", "redirect", 
        "chat isn't working", "i want human support"
    ]
    
    msg = message.lower().strip()
    response = ai_response.lower().strip()

    return any(phrase in response for phrase in uncertain_phrases) or "escalate" in msg
def log_escalation_to_dashboard(question, user_email, user_name=None):
    queries = load_json(FREELANCER_QUERIES_FILE, [])
    new_query = {
        "id": int(time.time() * 1000),
        "userId": user_email,
        "userName": user_name or user_email.split("@")[0],
        "question": question,
        "fileAttached": False,
        "note": "Auto-escalated by chatbot due to uncertain response.",
        "time": datetime.datetime.utcnow().isoformat(),
        "status": "open",
        "escalated": True,
        "claimedBy": None,
        "bidPrice": None,
        "bidComment": None
    }
    queries.append(new_query)
    save_json(FREELANCER_QUERIES_FILE, queries)
    return new_query

# ========== Prompt Template ==========
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are SkillBridge AI — a smart, friendly tech mentor who helps students, developers, and freelancers understand technical topics, build projects, and improve skills.

✅ You can explain, fix, or guide users on:

📌 Programming Languages:
- Python, JavaScript, C++, Java, TypeScript, C#
- Syntax, logic, debugging, error fixing

📌 Web Development:
- HTML, CSS, JavaScript, React, Node.js, REST APIs

📌 Data Science & Machine Learning:
- Pandas, NumPy, Matplotlib, Scikit-learn, etc.

📌 Deep Learning & NLP:
- Neural Networks, Transformers, Chatbots, etc.

📌 Software Engineering:
- UML, SDLC, Agile, Microservices, etc.

📌 Freelancing:
- Upwork/Fiverr tips, client communication, project bids

---

📌 If a question is completely irrelevant:
Reply: 
"SkillBridge is focused on tech, freelancing, and smart project guidance. Please refine your question."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ========== AI + Vector DB ==========
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192") if GROQ_API_KEY else None
docs = [Document(page_content=t) for t in [
    "SkillBridge helps with AI, projects, debugging, and freelancing.",
    "You escalate complex queries to human freelancers when needed."
]]
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISSStore.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt_template}) if llm else None

# === Auth Routes ===
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email, password, name = data.get("email"), data.get("password"), data.get("name")
    users = load_json(USERS_FILE, {})
    if email in users:
        return jsonify({"success": False, "message": "User already exists"}), 409
    users[email] = {"password": hash_password(password), "name": name, "premium": False}
    save_json(USERS_FILE, users)
    return jsonify({"success": True, "user": {"email": email, "name": name}})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    email, password = data.get("email"), data.get("password")
    users = load_json(USERS_FILE, {})
    user = users.get(email)
    if user and user["password"] == hash_password(password):
        return jsonify({"success": True, "user": {"email": email, "name": user["name"]}})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

# === PDF Upload (File Context) ===
@app.route("/api/shared-file", methods=["POST"])
def shared_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith(".pdf"):
        return jsonify({"success": False, "message": "Only PDF files are supported"}), 400
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        doc.close()
        session["uploaded_file_context"] = text.strip()[:3000]
        return jsonify({
            "success": True,
            "message": "PDF uploaded and processed.",
            "preview": text.strip()[:500]
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"PDF processing error: {str(e)}"}), 500

# === Chat Route ===
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")
    user_email = data.get("user_email")
    chat_id = data.get("chat_id", f"chat-{int(time.time())}")
    users = load_json(USERS_FILE, {})
    user = users.get(user_email)
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 403

    messages = ChatMessage.query.filter_by(user_email=user_email, chat_id=chat_id)\
        .order_by(ChatMessage.timestamp.desc()).limit(5).all()
    context = "\n".join([f"{m.role}: {m.content}" for m in reversed(messages)])
    personalized_context = f"User's name is {user['name']}. " if "name" in user else ""
    file_context = session.get("uploaded_file_context", "")

    if file_context:
        final_prompt = f"{file_context}\n\n{personalized_context + context}\nUser: {message}"
    else:
        final_prompt = f"{personalized_context + context}\nUser: {message}"

    try:
        ai_response = qa_chain.run(final_prompt) if qa_chain else "⚠️ AI not configured."
    except Exception as e:
        ai_response = f"Error: {str(e)}"

    db.session.add_all([
        ChatMessage(chat_id=chat_id, user_email=user_email, role="user", content=message),
        ChatMessage(chat_id=chat_id, user_email=user_email, role="assistant", content=ai_response)
    ])
    db.session.commit()

    if should_escalate(message, ai_response):
        log_escalation_to_dashboard(message, user_email, user.get("name"))
        ai_response += "\n\n👥 This query has been escalated to a human freelancer."

    return jsonify({"success": True, "response": ai_response, "chat_id": chat_id})
# === Freelancer Queries ===
@app.route("/api/freelancer-queries", methods=["GET"])
def get_freelancer_queries():
    queries = load_json(FREELANCER_QUERIES_FILE, [])
    escalated = [q for q in queries if q.get("escalated") and q.get("status") == "open"]
    return jsonify({"success": True, "queries": escalated})

@app.route("/api/claim-query", methods=["POST"])
def claim_query():
    data = request.get_json()
    query_id = data.get("query_id")
    freelancer = data.get("freelancer_name")
    bid_price = data.get("bid_price")
    bid_comment = data.get("bid_comment", "")
    queries = load_json(FREELANCER_QUERIES_FILE, [])
    for q in queries:
        if q["id"] == query_id:
            q.update({
                "status": "claimed",
                "claimedBy": freelancer,
                "bidPrice": bid_price,
                "bidComment": bid_comment
            })
            break
    save_json(FREELANCER_QUERIES_FILE, queries)
    return jsonify({"success": True})

@app.route("/api/submit-answer", methods=["POST"])
def submit_answer():
    data = request.get_json()
    query_id = data.get("query_id")
    answer = data.get("answer")
    if not query_id or not answer:
        return jsonify({"success": False, "message": "Missing fields"}), 400
    queries = load_json(FREELANCER_QUERIES_FILE, [])
    for q in queries:
        if q["id"] == query_id:
            q["answer"] = answer
            break
    save_json(FREELANCER_QUERIES_FILE, queries)
    return jsonify({"success": True, "message": "Answer submitted"})

# === Feedback ===
def save_feedback(user_email, message, response, rating, comment):
    feedback = {
        "user_email": user_email,
        "message": message,
        "response": response,
        "rating": rating,
        "comment": comment,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(feedback) + "\n")

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()
    required = ["user_email", "message", "response", "rating"]
    if not all(k in data for k in required):
        return jsonify({"success": False, "message": "Missing required fields"}), 400
    save_feedback(data["user_email"], data["message"], data["response"], data["rating"], data.get("comment", ""))
    return jsonify({"success": True, "message": "Feedback submitted"})

@app.route("/api/feedback", methods=["GET"])
def view_feedback():
    feedback = load_json(FEEDBACK_FILE, [])
    return jsonify({"success": True, "feedback": feedback})

# === Admin KPIs ===
@app.route("/api/admin/kpis", methods=["GET"])
def get_kpis():
    total_users = len(load_json(USERS_FILE, {}))
    total_queries = ChatMessage.query.count()
    feedback_count = sum(1 for _ in open(FEEDBACK_FILE)) if os.path.exists(FEEDBACK_FILE) else 0
    escalated = len([q for q in load_json(FREELANCER_QUERIES_FILE, []) if q.get("escalated")])
    return jsonify({
        "success": True,
        "kpis": {
            "total_users": total_users,
            "total_queries": total_queries,
            "total_feedback": feedback_count,
            "total_escalated_queries": escalated
        }
    })

# === CSV/Excel File Support for Chat ===
@app.route("/api/chat-with-file", methods=["POST"])
def chat_with_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files['file']
    prompt = request.form.get('prompt', '').lower()

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({"success": False, "error": "Unsupported file type"}), 400

        if 'head' in prompt:
            return jsonify({"success": True, "response": df.head().to_json()})
        elif 'columns' in prompt:
            return jsonify({"success": True, "response": list(df.columns)})
        elif 'shape' in prompt:
            return jsonify({"success": True, "response": str(df.shape)})
        elif 'null' in prompt:
            return jsonify({"success": True, "response": df.isnull().sum().to_dict()})
        elif 'describe' in prompt:
            return jsonify({"success": True, "response": df.describe().to_dict()})
        else:
            return jsonify({"success": True, "response": "File loaded. Ask about shape, columns, nulls, etc."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
# === User Stats ===
@app.route("/api/user-stats", methods=["GET"])
def get_user_stats():
    user_email = request.args.get("email")
    if not user_email:
        return jsonify({"success": False, "message": "Missing email parameter"}), 400

    total_chats = ChatMessage.query.filter_by(user_email=user_email).count()
    unique_sessions = db.session.query(ChatMessage.chat_id).filter_by(user_email=user_email).distinct().count()
    last_message = ChatMessage.query.filter_by(user_email=user_email).order_by(ChatMessage.timestamp.desc()).first()
    last_active = last_message.timestamp.isoformat() if last_message else None

    return jsonify({
        "success": True,
        "stats": {
            "total_messages": total_chats,
            "unique_sessions": unique_sessions,
            "last_active": last_active
        }
    })

# === Chat Message History ===
@app.route("/api/chat-messages", methods=["GET"])
def get_chat_messages():
    chat_id = request.args.get("chat_id")
    user_email = request.args.get("user_email")
    if not chat_id or not user_email:
        return jsonify({"success": False, "message": "Missing parameters"}), 400

    messages = (
        ChatMessage.query
        .filter_by(user_email=user_email, chat_id=chat_id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
    result = [
        {"id": msg.id, "content": msg.content, "role": msg.role, "timestamp": msg.timestamp.isoformat()}
        for msg in messages
    ]
    return jsonify({"success": True, "messages": result})

@app.route("/api/chat-history", methods=["GET"])
def get_chat_history():
    user_email = request.args.get("user_email")
    if not user_email:
        return jsonify({"success": False, "message": "Missing user_email"}), 400

    sessions = (
        db.session.query(ChatMessage.chat_id, db.func.min(ChatMessage.timestamp).label("timestamp"))
        .filter_by(user_email=user_email)
        .group_by(ChatMessage.chat_id)
        .order_by(db.func.min(ChatMessage.timestamp).desc())
        .all()
    )
    history = [
        {"id": chat_id, "title": f"Chat on {ts.strftime('%Y-%m-%d %H:%M')}", "timestamp": ts.isoformat()}
        for chat_id, ts in sessions
    ]
    return jsonify({"success": True, "history": history})

# === Health Check & 404 Handler ===
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "message": "API endpoint not found"}), 404

from flask import Flask, send_from_directory
from routes.chat import chat_blueprint  # your routes

app = Flask(__name__, static_folder="build", static_url_path="")
app.register_blueprint(chat_blueprint)

# Serve frontend files
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True, port=5000)
