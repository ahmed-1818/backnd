# SkillBridge Backend

Python Flask backend for SkillBridge AI chatbot application.

## Setup

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Run the backend:**
   ```bash
   python app.py
   ```

The backend will start on `http://localhost:5000`

## API Endpoints

- `POST /api/signup` - User registration
- `POST /api/login` - User authentication
- `POST /api/chat` - Send message to AI chatbot
- `POST /api/escalate` - Escalate to human support
- `POST /api/feedback` - Submit user feedback
- `GET /api/chat-history/<email>` - Get user's chat history
- `GET /api/health` - Health check

## Frontend Integration

The React frontend is configured to connect to this backend automatically when running on localhost:5000.

## Data Storage

- User data: `data/users.json`
- Chat history: `data/chats.json`
- Escalations: `data/escalated_queries.csv`
- Feedback: `data/feedback.csv`