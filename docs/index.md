<div align="center">

<h1>✨ Smart Compost ✨</h1>

<img src="assets/logo.svg" alt="Smart Compost Logo" width="180" height="180"/>

<p>🌿 <strong>Building a sustainable future together</strong></p>

<p>
<a href="https://www.python.org">
    <img src="https://img.shields.io/badge/python-3.12-green.svg" alt="Python">
</a>
<a href="https://flask.palletsprojects.com">
    <img src="https://img.shields.io/badge/flask-2.3.3-black.svg" alt="Flask">
</a>
<a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</a>
</p>

</div>



## 🚀 Overview

Smart Compost is an intelligent composting system that uses machine learning to optimize the composting process. It monitors key parameters like temperature and moisture in real-time, providing automated adjustments and notifications for optimal composting conditions.

### ✨ Key Features

- 🌡️ Real-time temperature monitoring
- 💧 Moisture level tracking
- 🤖 ML-powered composting optimization
- 📱 Mobile & web interface
- 📊 Analytics dashboard
- 🔔 Smart notifications

## 🎯 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/lyznne/smart-compost.git
   cd smart-compost
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python run.py
   ```

5. **Access the web interface**
   ```
   http://localhost:5000
   ```

## 📚 Documentation

Detailed documentation is available in the following sections:

- [Getting Started](./dependencies.md) - Setup guide and basic usage
- [Datasets](./dataset.md) - All Datasets structures for the Compost Model
- [Dependencies](./dependencies.md) - Project dependencies and requirements
- [Machine Learning Model](./model.md) - ML model architecture and training
- [Home Assistance & Flak App Web App](./app.md) - Intergration of the Model with Flask and  Home Assistance.
- [Learn More](./learn.md) - Advanced topics and best practices

## 🏗️ Project Structure

```
smart-compost/
├── app/                    # Main application directory
│   ├── api/               # API endpoints
│   ├── services/          # Business logic services
│   ├── static/            # Static assets (CSS, JS, images)
│   └── templates/         # Jinja2 HTML templates
├── data/                  # Dataset and database files
├── docs/                  # Documentation
├── Models/                # ML model implementation
└── scripts/              # Utility scripts
```

## 🛠️ Technologies

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework**: PyTorch
- **Database**: SQLite
- **Real-time**: WebSockets
- **Containerization**: Docker

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Enos Muthiani**
- Website: [enos.vercel.app](https://enos.vercel.app)
- GitHub: [@lyznne](https://github.com/lyznne)
- Email: emuthiani26@gmail.com

## ⭐ Support

If you find this project helpful, please consider giving it a star ⭐️

<div align="center">
  <sub>Built with ❤️ by Enos Muthiani</sub>
</div>

---

+ Create Instance of HomeAsistance docker conteiner run:
```bash
docker run -d \
  --name homeassistant \
  --privileged \
  --restart=unless-stopped \
  -e TZ=Africa/Nairobi \
  -v /<your-Path>/smart-compost/homeassistant_config:/config \
  -v /run/dbus:/run/dbus:ro \
  --network=host \
  ghcr.io/home-assistant/home-assistant:stable
  ```


+ Checkout this:-

```json
https://demo.home-assistant.io/#/lovelace/home
https://www.homebiogas.com/blog/kitchen-waste-composting/
```
