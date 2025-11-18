import os
import subprocess
import logging
import sys


class UdevRuleManager:
    def __init__(self, log_file="/var/log/udev_rule_manager.log"):
        """
        Initialize UdevRuleManager with logging

        Args:
            log_file (str): Path to log file
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=log_file,
            filemode="a",
        )
        self.logger = logging.getLogger(__name__)

        # Udev rule configuration
        self.udev_rule_path = "/etc/udev/rules.d/99-wifi-permissions.rules"

        # Potential network groups
        self.network_groups = [
            "netdev",
            "network",
            "wifi",
            "wireless",
            "networkmanager",
        ]

    def _check_root_privileges(self):
        """
        Check if the script is running with root privileges

        Returns:
            bool: True if running as root, False otherwise
        """
        return os.geteuid() == 0

    def _find_current_user(self):
        """
        Find the non-root user who invoked sudo

        Returns:
            str: Username of the non-root user
        """
        sudo_user = os.environ.get("SUDO_USER")
        if sudo_user:
            return sudo_user

        # Fallback to current user if not found
        return os.getlogin()

    def _detect_network_group(self):
        """
        Detect the first available network-related group

        Returns:
            str or None: First available network group
        """
        try:
            # Get list of existing groups
            with open("/etc/group", "r") as f:
                existing_groups = f.read()

            # Find first matching group
            for group in self.network_groups:
                if group in existing_groups:
                    return group

            return None
        except Exception as e:
            self.logger.error(f"Error detecting network group: {e}")
            return None

    def create_udev_rule(self):
        """
        Create udev rule for automatic network group assignment

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Verify root privileges
            if not self._check_root_privileges():
                self.logger.error("Script must be run with sudo")
                return False

            # Detect network group
            network_group = self._detect_network_group()
            if not network_group:
                self.logger.warning("No suitable network group found")
                return False

            # Find current user
            current_user = self._find_current_user()

            # Construct udev rule
            udev_rule_content = (
                f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{type}}=="1", '
                f'RUN+="/usr/sbin/usermod -a -G {network_group} {current_user}"\n'
            )

            # Write udev rule
            with open(self.udev_rule_path, "w") as f:
                f.write(udev_rule_content)

            self.logger.info(
                f"Created udev rule for {current_user} in {network_group} group"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to create udev rule: {e}")
            return False

    def reload_udev_rules(self):
        """
        Reload udev rules and trigger

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Reload udev rules
            subprocess.run(["udevadm", "control", "--reload-rules"], check=True)

            # Trigger udev events
            subprocess.run(["udevadm", "trigger"], check=True)

            self.logger.info("Udev rules reloaded and triggered successfully")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to reload udev rules: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error reloading udev rules: {e}")
            return False

    def add_user_to_network_group(self):
        """
        Add current user to network group

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Detect network group
            network_group = self._detect_network_group()
            if not network_group:
                self.logger.warning("No suitable network group found")
                return False

            # Find current user
            current_user = self._find_current_user()

            # Add user to group
            subprocess.run(
                ["usermod", "-a", "-G", network_group, current_user], check=True
            )

            self.logger.info(f"Added {current_user} to {network_group} group")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to add user to network group: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error adding user to network group: {e}")
            return False

    def setup_network_permissions(self):
        """
        Comprehensive network permissions setup

        Returns:
            bool: True if all steps successful, False otherwise
        """
        try:
            # Create udev rule
            rule_created = self.create_udev_rule()

            # Add user to network group
            user_added = self.add_user_to_network_group()

            # Reload udev rules
            rules_reloaded = self.reload_udev_rules()

            return rule_created and user_added and rules_reloaded

        except Exception as e:
            self.logger.error(f"Network permissions setup failed: {e}")
            return False


def main():
    """
    Main execution function
    """
    # Initialize UdevRuleManager
    udev_manager = UdevRuleManager()

    # Setup network permissions
    if udev_manager.setup_network_permissions():
        print("Network permissions setup completed successfully.")
        sys.exit(0)
    else:
        print("Network permissions setup failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
