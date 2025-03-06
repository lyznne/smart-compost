# 📦 Dependencies

This document outlines all dependencies required for the Smart Compost project.

## 🐍 Python Dependencies

### Core Dependencies
```plaintext
Flask==2.3.3
Werkzeug==2.3.7
Jinja2==3.1.2
MarkupSafe==2.1.3
itsdangerous==2.1.2
click==8.1.7
```

### Database
```plaintext
SQLAlchemy==2.0.23
Flask-SQLAlchemy==3.1.1
Flask-Migrate==4.0.5
```

### Machine Learning
```plaintext
torch==2.1.1
numpy==1.26.2
pandas==2.1.3
scikit-learn==1.3.2
```

### Real-time Communication
```plaintext
Flask-SocketIO==5.3.6
python-socketio==5.10.0
```

### Development & Testing
```plaintext
pytest==7.4.3
coverage==7.3.2
black==23.11.0
flake8==6.1.0
```

## 🌐 Frontend Dependencies

### CSS Framework
- Tailwind CSS

### JavaScript Libraries
- Socket.IO Client
- Chart.js (for analytics)

### Icons
- Font Awesome
- Custom SVG icons

## 🛠️ Development Tools

### Required
- Python 3.12+
- pip
- virtualenv or venv
- Git

### Optional
- Docker
- Docker Compose
- Make

## 📱 Hardware Requirements

### Minimum Specifications
- CPU: 1.0 GHz dual-core
- RAM: 2GB
- Storage: 500MB free space
- Network: WiFi capability

### Recommended Specifications
- CPU: 2.0 GHz quad-core
- RAM: 4GB
- Storage: 1GB free space
- Network: WiFi 5 (802.11ac)

## 🔧 System Requirements

### Operating System Support
- Linux (Ubuntu 20.04+, Debian 11+)
- macOS (10.15+)
- Windows 10/11

### Browser Support
- Chrome 90+
- Firefox 90+
- Safari 14+
- Edge 90+

## ⚙️ Environment Variables

Required environment variables:
```env
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your_secret_key
DATABASE_URL=sqlite:///data/database.sql
```

Optional environment variables:
```env
DEBUG=True
TESTING=False
LOG_LEVEL=INFO
```

## 📝 Version Management

This project uses semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible features
- PATCH: Backwards-compatible bug fixes

## 🔄 Dependency Updates

To update all dependencies to their latest versions:
```bash
pip install --upgrade -r requirements.txt
```

To check for outdated packages:
```bash
pip list --outdated
```
