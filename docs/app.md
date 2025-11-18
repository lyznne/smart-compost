
# 📘 Smart Compost – Flask App & Home Assistant Integration Documentation

This documentation describes the architecture, routes, services, APIs, and IoT integration for the *Smart Compost* backend built using **Flask**, **MQTT**, **PyTorch**, and **Home Assistant**–compatible automation.

---

# 🏗️ Project Structure

```plaintext
app/
├── api.py                # ML training & prediction API
├── views.py              # Frontend routes (UI)
├── channels.py
├── config.py
├── __init__.py
├── models.py
├── services/
│   ├── data_service.py
│   ├── network_manager.py
│   ├── notification.py
│   └── rule_manager.py
├── static/
│   ├── assets/...
│   ├── css/...
│   └── js/...
├── templates/
│   ├── app/...
│   ├── auth/...
│   ├── includes/...
│   └── layouts/...
├── util.py
└── views.py.bak
```

---

# 🖥️ Flask Application Overview

The Flask app provides:

* 🔐 **User authentication**
* 📊 **Dashboard UI** showing compost status
* 📡 **MQTT integration** for real-time sensor data
* 🌐 **WiFi management** for IoT devices
* ⚙️ **Manual system control**
* 🤖 **ML model inference pipeline**
* 📱 **Home Assistant–friendly endpoints**

---

# 🔌 MQTT Integration

The app connects to MQTT to receive:

| Topic              | Purpose                            |
| ------------------ | ---------------------------------- |
| `compost/sensors`  | Raw sensor telemetry               |
| `compost/status`   | Compost state updates              |
| `compost/alerts`   | Real-time alerts                   |
| `compost/commands` | Commands sent *from* the Flask app |

### MQTT Client Lifecycle

```python
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.loop_forever()
```

### Sensor Data Handler

```python
def on_message(client, userdata, message):
    payload = json.loads(message.payload.decode())
    if topic == "compost/sensors":
        latest_sensor_data = {...}
        DataService.store_sensor_reading(...)
        check_alert_conditions(payload)
```

---

# 🧩 Routes Overview (views.py)

## 🏠 Dashboard `/`

Displays the main compost dashboard with:

* Live temperature
* Moisture levels
* WiFi status
* Recent activity
* Notifications
* IoT device information

---

## 📡 WiFi Management `/wifi`

### GET → Scan Networks

Returns a list of available networks:

```python
networks = network_manager.scan_networks()
```

### POST → Connect to WiFi

JSON body:

```json
{
  "ssid": "MyNetwork",
  "password": "mypassword"
}
```

---

## 📈 Stats `/stats`

Displays:

* Temperature history
* Moisture history
* Compost maturity
* Environmental impact

---

## 🛠 Manual Control `/control`

Sends commands to MQTT:

```python
mqtt_client.publish("compost/commands", json.dumps(command))
```

Available actions:

* Mix
* Water
* Aerate
* Harvest

---

## 👤 User Profile `/profile`

Allows:

* Updating personal info
* Managing notifications
* Viewing device activity
* Recent logs

---

# 🔬 ML API Documentation (api.py)

## 🎯 Train Model — `/train`

### Description

Triggers training of LSTM compost prediction model.

### Request

`POST /train`

### Response

```json
{ "message": "Model trained successfully" }
```

### Internals

* Loads dataset from `data/smart_compost_dataset101.csv`
* Splits into train/validation
* Calls:

```python
train_compost_model(train_loader, val_loader)
```

---

## 🔮 Predict — `/predict`

### Request Format

```json
{
  "input_data": [ ... 25 numeric features ... ],
  "user_id": 1
}
```

### Response Example

```json
{
  "predictions": {
    "temperature": 58.2,
    "moisture": 61.5
  },
  "status": "Optimal",
  "confidence": 0.94
}
```

### Stored in Database As:

* temperature
* moisture
* pH
* oxygen
* C:N ratio
* NPK

---

# 🏡 Integrating With Home Assistant

Your Flask backend works well with Home Assistant via:

* REST Sensors
* MQTT Sensors
* Event Automation
* Command Topics

---

## 📡 MQTT Sensor Example (Home Assistant)

```yaml
mqtt:
  sensor:
    - name: "Compost Temperature"
      state_topic: "compost/sensors"
      unit_of_measurement: "°C"
      value_template: "{{ value_json.temperature }}"

    - name: "Compost Moisture"
      state_topic: "compost/sensors"
      unit_of_measurement: "%"
      value_template: "{{ value_json.moisture }}"
```

---

## 🌡️ HA Automation: Overheat Alert

```yaml
automation:
  - alias: Compost Overheat Warning
    trigger:
      - platform: numeric_state
        entity_id: sensor.compost_temperature
        above: 65
    action:
      - service: notify.mobile_app
        data:
          message: "🔥 Compost temperature is too high!"
```

---

## 🛠 Remote Controls via MQTT

```yaml
script:
  mix_compost:
    sequence:
      - service: mqtt.publish
        data:
          topic: "compost/commands"
          payload: '{"user_id":1,"action":"mix"}'
```

---

# 🧱 Backend Services

## 📦 DataService

Handles:

* Storing sensor readings
* Fetching compost stats
* Logging activity
* Reading notification history

---

## 🌐 NetworkManager

Controls IoT WiFi configuration:

* Scan WiFi networks
* Connect to SSID
* Fetch connection status

---

## 🔔 NotificationManager

Creates notifications:

```python
NotificationManager.create_notification(
  user_id, "Low moisture alert"
)
```

---

# 🪪 Authentication

Flask-Login for:

* Sign-in
* Sign-up
* Device registration
* Unauthorized handler

---

# 🚦 Health Check Endpoint `/health`

Returns:

```json
{
  "status": "ok",
  "memory_percent": "47%",
  "cpu_percent": "12%",
  "disk_percent": "55%",
  "mqtt": "connected",
  "database": "connected",
  "gpu": "not available"
}
```

---

# 📈 Future Improvements

* Full MQTT → Home Assistant auto-discovery
* On-device model inferencing
* Device pairing API
* Real-time socket updates
* Admin dashboard
