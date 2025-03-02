/*

---
SMART COMPOST - MODEL PROJECT.

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app

---

*/
-- Create the database
CREATE DATABASE smart_compost;

-- Use the database (MySQL only)
USE smart_compost;

-- Users Table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- SERIAL (PostgreSQL) / AUTO_INCREMENT (MySQL)
    first_name VARCHAR(64) NOT NULL,
    last_name VARCHAR(64) NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    avatar VARCHAR(255) DEFAULT 'default_avatar.png',
    location TEXT,  -- Use TEXT for flexible storage
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Activity Log
CREATE TABLE activity_log (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    activity_type VARCHAR(128) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Devices Table
CREATE TABLE devices (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    device_name VARCHAR(128) NOT NULL,
    device_ip VARCHAR(45) NOT NULL,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Notifications Table
CREATE TABLE notifications (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_status BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Comet ML Experiment Tracking Table
CREATE TABLE comet_experiments (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,  -- Who ran the experiment
    experiment_id VARCHAR(128) UNIQUE NOT NULL,  -- Comet experiment ID
    model_name VARCHAR(128) NOT NULL,  -- Name of the trained model
    parameters JSON NOT NULL,  -- Hyperparameters stored as JSON
    metrics JSON NOT NULL,  -- Training metrics stored as JSON
    status VARCHAR(20) DEFAULT 'running',  -- ["running", "completed", "failed"]
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,  -- NULL until experiment completes
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);


--- initial dataset
CREATE TABLE environmental_variables (
    id SERIAL PRIMARY KEY,
    variable VARCHAR(255) NOT NULL,
    measurement_unit VARCHAR(50),
    optimal_range VARCHAR(50),
    frequency VARCHAR(50),
    dependencies TEXT,
    timing VARCHAR(50),
    quantities VARCHAR(255),
    iot_application VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- insert the values

INSERT INTO environmental_variable (variable, variable_type, measurement_unit, optimal_range, dependencies, introduction_stage, frequency, notes) VALUES
('Temperature', 'Numerical', 'Celsius', '45-65', 'Moisture_Content|Oxygen_Level|Materials_Ratio', 'Initial', 'Continuous', 'Critical for decomposition, affects microbial activity'),
('Moisture_Content', 'Numerical', 'Percentage', '40-60', 'Temperature|Material_Type|Particle_Size', 'Initial', 'Daily', 'Higher for green materials, lower for browns'),
('pH_Level', 'Numerical', 'pH Scale', '6.5-8.0', 'Nitrogen_Content|Temperature|Moisture_Content', 'Initial', 'Weekly', 'Affects microbial growth, decomposition rate'),
('Oxygen_Level', 'Numerical', 'Percentage', '5-15', 'Temperature|Moisture_Content|Bulk_Density', 'Initial', 'Continuous', 'Must be maintained through turning or aeration'),
('Carbon_Nitrogen_Ratio', 'Numerical', 'Ratio', '25-30:1', 'Material_Type|Temperature', 'Initial', 'At_Addition', 'Crucial for proper decomposition, affects smell'),
('Maturity_Index', 'Numerical', 'Scale 1-8', '>7', 'Temperature|Moisture_Content|Time', 'Final_Stage', 'Weekly', 'Indicates completion of composting'),
('Turning_Frequency', 'Numerical', 'Days', '3-7', 'Temperature|Oxygen_Level|Moisture_Content', 'After_Day_3', 'Weekly', 'Based on temperature and oxygen readings');
