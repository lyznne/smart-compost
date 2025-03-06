# 🚀 Getting Started with Smart Compost

This guide will help you set up and run the Smart Compost system on your local machine.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.12 or higher
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/lyznne/smart-compost.git
cd smart-compost
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
DATABASE_URL=sqlite:///data/database.sql
```

### 5. Initialize Database

```bash
flask db upgrade
```

### 6. Run the Application

```bash
python run.py
```

The application will be available at `http://localhost:5000`

## 🔑 Default Login Credentials

For testing purposes, use these credentials:
- Email: `emuthiani26@gmail.com`
- Password: `password`

## 📱 Accessing the Dashboard

1. Open your web browser
2. Navigate to `http://localhost:5000`
3. Log in using the credentials above
4. You'll be redirected to the main dashboard

## 🎛️ Basic Operations

### Monitoring Compost Status
- View real-time temperature and moisture levels
- Check compost maturity progress
- Review recent activity logs

### Managing Network Settings
- Configure WiFi connection
- View connected devices
- Monitor network status

### Viewing Analytics
- Access temperature history
- Track moisture trends
- View environmental impact metrics

## 🐳 Docker Deployment

To run using Docker:

```bash
# Build the container
docker build -t smart-compost .

# Run the container
docker run -p 5000:5000 smart-compost
```

Or using Docker Compose:

```bash
docker-compose up
```

## ⚠️ Common Issues

1. **Port Already in Use**
   ```bash
   # Change the port
   flask run --port=5001
   ```

2. **Database Connection Error**
   - Ensure database file exists
   - Check permissions
   - Verify DATABASE_URL in .env

3. **Dependencies Conflicts**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

## 📞 Support

If you encounter any issues:
- Check the [documentation](docs/)
- Open an [issue](https://github.com/lyznne/smart-compost/issues)
- Contact: emuthiani26@gmail.com
