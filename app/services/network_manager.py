import os
import logging
from typing import List, Optional
import pywifi
from pywifi import const
import time
import netifaces


class NetworkManager:
    def __init__(self, env: str = "development"):
        """
        Initialize NetworkManager with environment-specific configuration

        Args:
            env (str): Environment mode - 'development' or 'production'
        """
        # Configure logging
        self.logger = self._setup_logging(env)

        # Environment-specific configurations
        self.env = env
        self.config = self._load_config()

        try:
            self.wifi = pywifi.PyWiFi()
            interfaces = self.wifi.interfaces()

            if not interfaces:
                self.logger.warning("No WiFi interfaces found")
                self.iface = None
            else:
                self.iface = interfaces[0]
                self.logger.info(f"Using WiFi interface: {self.iface}")

        except Exception as e:
            self.logger.error(f"Error initializing WiFi: {e}")
            self.iface = None

    def _setup_logging(self, env: str) -> logging.Logger:
        """
        Set up logging based on environment

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

    def _load_config(self) -> dict:
        """
        Load configuration based on environment

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

    def scan_networks(self) -> List[dict]:
        """
        Scan and return available WiFi networks

        Returns:
            List[dict]: Sorted list of network information
        """
        if not self.iface:
            self.logger.warning("No WiFi interface available")
            return []

        try:
            self.iface.disconnect()
            time.sleep(1)
            self.iface.scan()
            time.sleep(self.config["scan_timeout"])

            wifi_networks = self.iface.scan_results()
            networks = []

            for network in wifi_networks:
                ssid = network.ssid.rstrip("\x00")
                if not ssid:
                    continue

                network_info = {
                    "ssid": ssid,
                    "signal_strength": abs(network.signal),
                    "security": self._get_security_type(network),
                    "bssid": network.bssid,
                    "signal_quality": self._get_signal_quality(abs(network.signal)),
                }
                networks.append(network_info)

            return sorted(networks, key=lambda x: x["signal_strength"], reverse=True)

        except Exception as e:
            self.logger.error(f"Network scan error: {e}")
            return []

    def _get_signal_quality(self, strength: int) -> str:
        """
        Classify signal strength

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

    def _get_security_type(self, network) -> str:
        """
        Determine network security type

        Args:
            network: WiFi network object

        Returns:
            str: Security type description
        """
        if not network.akm:
            return "Open"

        securities = {
            const.AKM_TYPE_NONE: "Open",
            const.AKM_TYPE_WPA2PSK: "WPA2",
            const.AKM_TYPE_WPA2ENTERPRISE: "WPA2 Enterprise",
            const.AKM_TYPE_WPA1PSK: "WPA1",
            const.AKM_TYPE_WPA1ENTERPRISE: "WPA1 Enterprise",
        }

        return securities.get(network.akm[0], "Unknown")

    def connect_to_network(self, ssid: str, password: str = None) -> bool:
        """
        Connect to a specific WiFi network

        Args:
            ssid (str): Network SSID
            password (str, optional): Network password

        Returns:
            bool: Connection status
        """
        if not self.iface:
            self.logger.error("No WiFi interface available")
            return False

        try:
            self.iface.disconnect()
            time.sleep(1)

            profile = pywifi.Profile()
            profile.ssid = ssid

            if password:
                profile.auth = const.AUTH_ALG_OPEN
                profile.akm.append(const.AKM_TYPE_WPA2PSK)
                profile.cipher = const.CIPHER_TYPE_CCMP
                profile.key = password
            else:
                profile.auth = const.AUTH_ALG_OPEN
                profile.akm.append(const.AKM_TYPE_NONE)

            self.iface.remove_all_network_profiles()
            tmp_profile = self.iface.add_network_profile(profile)
            self.iface.connect(tmp_profile)

            time.sleep(self.config["connection_timeout"])

            connection_status = self.iface.status() == const.IFACE_CONNECTED

            if connection_status:
                self.logger.info(f"Successfully connected to {ssid}")
            else:
                self.logger.warning(f"Failed to connect to {ssid}")

            return connection_status

        except Exception as e:
            self.logger.error(f"Network connection error: {e}")
            return False

    def get_current_connection(self) -> Optional[dict]:
        """
        Get details of current WiFi connection

        Returns:
            Optional[dict]: Current connection details or None
        """
        if not self.iface or self.iface.status() != const.IFACE_CONNECTED:
            return None

        try:
            current_profile = self.iface.current_network()
            return {
                "ssid": current_profile.ssid,
                "ip_address": self._get_ip_address(),
            }
        except Exception as e:
            self.logger.error(f"Error getting current connection: {e}")
            return None

    def _get_ip_address(self) -> str:
        """
        Get current IP address

        Returns:
            str: IP address or 'Unknown'
        """
        try:
            gws = netifaces.gateways()
            default_interface = gws["default"][netifaces.AF_INET][1]
            addresses = netifaces.ifaddresses(default_interface)
            return addresses[netifaces.AF_INET][0]["addr"]
        except Exception as e:
            self.logger.warning(f"Could not retrieve IP address: {e}")
            return "Unknown"

    
