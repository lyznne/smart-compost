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


# Smart Compost Project - Docker Setup

This document provides instructions for setting up and running the Smart Compost project using Docker, with Caddy as the web server, MySQL as the database, and Flask as the application server.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git (to clone the repository)

## Setup Steps

### 1. Environment Configuration

1. Copy the `.env.template` file to create a `.env` file:
   ```bash
   cp .env.template .env
   ```

2. Edit the `.env` file to set secure passwords and your Comet ML API credentials:
   ```
   MYSQL_PASSWORD=your_secure_password
   MYSQL_ROOT_PASSWORD=your_secure_root_password
   COMET_API_KEY=your_comet_api_key
   COMET_PROJECT_NAME=smart_compost
   COMET_WORKSPACE=your_workspace_name
   ```

### 2. Customize Caddy Configuration

Edit the `Caddyfile` to use your actual domain name instead of `example.com`. If you're running locally, you can use `localhost` instead.

### 3. Build and Start the Services

```bash
docker-compose build
docker-compose up -d
```

This will:
- Build the Flask application container
- Start the MySQL database
- Set up the Caddy web server
- Connect all services through a Docker network

### 4. Verify the Setup

1. Check if all containers are running:
   ```bash
   docker-compose ps
   ```

2. Check the logs for any errors:
   ```bash
   docker-compose logs
   ```

3. Access your application:
   - If using a domain: https://yourdomain.com
   - If using localhost: http://localhost

### 5. Database Management

- To connect to the MySQL database:
  ```bash
  docker-compose exec db mysql -u compost_user -p smart_compost
  ```

- You'll be prompted for the password you set in the `.env` file.

### 6. Scaling and Production Considerations

For production:
1. Ensure you're using strong, unique passwords
2. Replace the example.com domain with your actual domain in the Caddyfile
3. Consider setting up database backups
4. Monitor your containers with tools like Prometheus and Grafana

### 7. Stopping the Services

```bash
docker-compose down
```

To remove volumes (database data) as well:
```bash
docker-compose down -v
```

## Project Structure

```
smart-compost/
├── app.py                 # Main Flask application entry point
├── Caddyfile              # Caddy server configuration
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile.app         # Flask application Dockerfile
├── .env                   # Environment variables (not in version control)
├── .env.template          # Template for environment variables
├── init-db/               # SQL scripts for database initialization
│   └── 01-init.sql        # Initial database schema
├── models/                # Directory for saving trained models
├── logs/                  # Directory for application logs
└── requirements.txt       # Python dependencies
```

## 📞 Support

If you encounter any issues:
- Check the [documentation](docs/)
- Open an [issue](https://github.com/lyznne/smart-compost/issues)
- Contact: emuthiani26@gmail.com
