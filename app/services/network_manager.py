import os
import logging
import platform
import subprocess
import socket
from typing import List, Optional, Dict
import time


class NetworkManager:
    def __init__(self, env: str = "development"):
        """
        Initialize NetworkManager with environment-specific configuration.

        Args:
            env (str): Environment mode - 'development' or 'production'
        """
        # Configure logging
        self.logger = self._setup_logging(env)

        # Environment-specific configurations
        self.env = env
        self.config = self._load_config()

        # Detect the operating system
        self.os_type = platform.system().lower()
        self.logger.info(f"Running on {self.os_type.capitalize()}")

        # Initialize platform-specific WiFi management
        self.wifi_interface = self._get_wifi_interface()

    def _setup_logging(self, env: str) -> logging.Logger:
        """
        Set up logging based on environment.

        Args:
            env (str): Environment mode

        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger("NetworkManager")

        # Clear any existing handlers
        logger.handlers.clear()

        # Create handler
        if env == "production":
            # Production: Log to file with more detailed format
            handler = logging.FileHandler("network_manager.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setLevel(logging.INFO)
        else:
            # Development: Log to console with debug level
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setLevel(logging.DEBUG)

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _load_config(self) -> Dict:
        """
        Load configuration based on environment.

        Returns:
            dict: Configuration dictionary
        """
        if self.env == "production":
            return {
                "scan_timeout": 5,  # longer timeout in production
                "connection_timeout": 10,
                "log_level": logging.INFO,
            }
        else:
            return {
                "scan_timeout": 3,  # shorter timeout in development
                "connection_timeout": 5,
                "log_level": logging.DEBUG,
            }

    def _get_wifi_interface(self) -> Optional[str]:
        """
        Get the WiFi interface name based on the operating system.

        Returns:
            Optional[str]: WiFi interface name or None if not found
        """
        if self.os_type == "linux":
            try:
                result = subprocess.run(["nmcli", "device"], capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if "wifi" in line.lower():
                        return line.split()[0]
            except Exception as e:
                self.logger.error(f"Error getting WiFi interface on Linux: {e}")

        elif self.os_type == "windows":
            try:
                result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if "name" in line.lower():
                        return line.split(":")[1].strip()
            except Exception as e:
                self.logger.error(f"Error getting WiFi interface on Windows: {e}")

        elif self.os_type == "darwin":  # macOS
            try:
                result = subprocess.run(["networksetup", "-listallhardwareports"], capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if "wi-fi" in line.lower():
                        return line.split(":")[1].strip()
            except Exception as e:
                self.logger.error(f"Error getting WiFi interface on macOS: {e}")

        self.logger.warning("No WiFi interface found")
        return None

    def scan_networks(self) -> List[Dict]:
        """
        Scan and return available WiFi networks.

        Returns:
            List[Dict]: List of network information
        """
        networks = []

        if self.os_type == "linux":
            try:
                result = subprocess.run(["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"],
                                       capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    ssid, signal, security = line.split(":")
                    networks.append({
                        "ssid": ssid,
                        "signal_strength": int(signal),
                        "security": security,
                        "signal_quality": self._get_signal_quality(int(signal)),
                    })
            except Exception as e:
                self.logger.error(f"Error scanning networks on Linux: {e}")

        elif self.os_type == "windows":
            try:
                result = subprocess.run(["netsh", "wlan", "show", "networks"], capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if "SSID" in line:
                        ssid = line.split(":")[1].strip()
                        networks.append({
                            "ssid": ssid,
                            "signal_strength": 0,  # Windows does not provide signal strength in this command
                            "security": "Unknown",
                            "signal_quality": "Unknown",
                        })
            except Exception as e:
                self.logger.error(f"Error scanning networks on Windows: {e}")

        elif self.os_type == "darwin":  # macOS
            try:
                result = subprocess.run(["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-s"],
                                       capture_output=True, text=True)
                for line in result.stdout.splitlines()[1:]:  # Skip header
                    parts = line.split()
                    ssid = parts[0]
                    signal = int(parts[2])
                    security = parts[3]
                    networks.append({
                        "ssid": ssid,
                        "signal_strength": signal,
                        "security": security,
                        "signal_quality": self._get_signal_quality(signal),
                    })
            except Exception as e:
                self.logger.error(f"Error scanning networks on macOS: {e}")

        return networks

    def _get_signal_quality(self, strength: int) -> str:
        """
        Classify signal strength.

        Args:
            strength (int): Signal strength

        Returns:
            str: Signal quality description
        """
        if strength > 70:
            return "Strong"
        elif strength > 50:
            return "Medium"
        return "Weak"

    def connect_to_network(self, ssid: str, password: Optional[str] = None) -> bool:
        """
        Connect to a specific WiFi network.

        Args:
            ssid (str): Network SSID
            password (Optional[str]): Network password

        Returns:
            bool: Connection status
        """
        if not self.wifi_interface:
            self.logger.error("No WiFi interface available")
            return False

        try:
            if self.os_type == "linux":
                if password:
                    subprocess.run(["nmcli", "device", "wifi", "connect", ssid, "password", password], check=True)
                else:
                    subprocess.run(["nmcli", "device", "wifi", "connect", ssid], check=True)

            elif self.os_type == "windows":
                if password:
                    subprocess.run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}", f"key={password}"], check=True)
                else:
                    subprocess.run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"], check=True)

            elif self.os_type == "darwin":  # macOS
                if password:
                    subprocess.run(["networksetup", "-setairportnetwork", self.wifi_interface, ssid, password], check=True)
                else:
                    subprocess.run(["networksetup", "-setairportnetwork", self.wifi_interface, ssid], check=True)

            self.logger.info(f"Successfully connected to {ssid}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to connect to {ssid}: {e}")
            return False

    def get_current_connection(self) -> Optional[Dict]:
        """
        Get details of the current WiFi connection.

        Returns:
            Optional[Dict]: Current connection details or None
        """
        try:
            ip_address = socket.gethostbyname(socket.gethostname())
            return {
                "ssid": self._get_current_ssid(),
                "ip_address": ip_address,
            }
        except Exception as e:
            self.logger.error(f"Error getting current connection: {e}")
            return None

    def _get_current_ssid(self) -> Optional[str]:
        """
        Get the SSID of the current WiFi connection.

        Returns:
            Optional[str]: SSID or None if not connected
        """
        if self.os_type == "linux":
            try:
                result = subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID", "device", "wifi"], capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if "yes" in line:
                        return line.split(":")[1]
            except Exception as e:
                self.logger.error(f"Error getting current SSID on Linux: {e}")

        elif self.os_type == "windows":
            try:
                result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if "SSID" in line:
                        return line.split(":")[1].strip()
            except Exception as e:
                self.logger.error(f"Error getting current SSID on Windows: {e}")

        elif self.os_type == "darwin":  # macOS
            try:
                result = subprocess.run(["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"],
                                       capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if "SSID" in line:
                        return line.split(":")[1].strip()
            except Exception as e:
                self.logger.error(f"Error getting current SSID on macOS: {e}")

        return None
