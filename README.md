# 🧠 LLaMA 3 Chat App

A modern, interactive chatbot interface built with **Streamlit** and powered by **Groq's LLaMA 3 API**. Experience fast, intelligent conversations with persistent chat history and a clean, user-friendly interface.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white)
![LLaMA](https://img.shields.io/badge/LLaMA-3-green?style=flat)

---

## ✨ Features

- 🤖 **AI-Powered Conversations** - Chat with LLaMA 3 using Groq's lightning-fast API
- 💾 **Persistent Chat History** - Your conversations are saved during the session
- 🔐 **Secure API Management** - Environment variables for safe API key storage
- 🎨 **Clean Interface** - Intuitive Streamlit UI with real-time message display
- ⚡ **Fast Response Times** - Powered by Groq's optimized inference engine

---

## 📁 Project Structure

```
llm-chat-app/
├── 📄 llm_chat.py          # Main Streamlit application
├── 🔐 .env                 # Environment variables (API key)
├── 📋 requirements.txt     # Python dependencies
└── 📖 README.md           # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A Groq API key ([Get one here](https://console.groq.com/))

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ImalkaDilakshan99/llm-chat-app.git
   cd llm-chat-app
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the project root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   > ⚠️ **Important**: Never share or commit your API key. The `.env` file should be added to `.gitignore`.

5. **Launch the Application**
   ```bash
   streamlit run llm_chat.py
   ```

6. **Start Chatting!**
   
   Open your browser and navigate to `http://localhost:8501`

---

## 📦 Dependencies

The application requires the following Python packages:

```txt
streamlit>=1.28.0
llama-index>=0.9.0
llama-index-llms-groq>=0.1.0
python-dotenv>=1.0.0
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🔒 Security & Best Practices

### API Key Protection
- ✅ Store your Groq API key in the `.env` file
- ✅ Add `.env` to your `.gitignore` file
- ❌ Never hardcode API keys in your source code
- ❌ Never commit API keys to version control

### Recommended .gitignore entries:
```gitignore
.env
__pycache__/
*.pyc
.venv/
venv/
```

---

## 🛠️ Usage

1. **Start the Application**: Run `streamlit run llm_chat.py`
2. **Enter Your Message**: Type your question or message in the chat input
3. **Get AI Response**: LLaMA 3 will process your message and respond
4. **Continue Conversation**: Chat history is maintained throughout your session

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key for accessing LLaMA 3 | Yes |

### Customization Options

You can modify the following in `llm_chat.py`:
- Model selection (different LLaMA variants)
- Temperature settings for response creativity
- Maximum token limits
- UI styling and layout

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Streamlit](https://streamlit.io/)** - For the amazing web app framework
- **[Groq](https://groq.com/)** - For providing fast LLM inference
- **[LlamaIndex](https://www.llamaindex.ai/)** - For seamless LLM integration
- **[Meta AI](https://ai.meta.com/)** - For developing the LLaMA model

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'streamlit'`
**Solution**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

**Issue**: `API key not found`
**Solution**: Ensure your `.env` file is in the project root with the correct `GROQ_API_KEY` variable

**Issue**: `Connection timeout`
**Solution**: Check your internet connection and verify your Groq API key is valid

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/llm-chat-app/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ and Python

</div>