import logging
import apprise
from apprise import NotifyType
from logger import adfs_logger # Removed duplicate import
from config import Config

class PDFProcessingError(Exception):
    """Custom exception for structured error routing in ADFS."""

    # ── Error Types ──
    EXTRACTION_FAILED = "extraction_failed"
    CLASSIFICATION_TIMEOUT = "timeout"
    CORRUPTED_PDF = "corrupted"
    FILE_ALREADY_EXISTS = "collision"
    PERMISSION_DENIED = "permission_denied"
    DISK_FULL = "disk_full"
    UNKNOWN_SYSTEM = "unknown_system"

    def __init__(self, message: str, error_type: str, file_path: str = "Unknown"):
        super().__init__(message)
        self.error_type = error_type
        self.file_path = file_path

class AlertManager:
    """Handles escalation of production errors."""

    """Universal Alert Manager using Apprise for Local Privacy."""

    # 🔴 FIXED: Set to None initially. Do not boot Apprise at import time!
    _apobj = None

    @classmethod
    def get_notifier(cls):
        """Lazy-loader: Only boots Apprise when the first error actually happens."""
        if cls._apobj is None:
            try:
                cls._apobj = apprise.Apprise()
                if Config.APPRISE_URL:
                    cls._apobj.add(Config.APPRISE_URL)
            except Exception as e:
                adfs_logger.error(f"Failed to initialize Apprise: {e}", extra={"stage": "alert_manager"})
        return cls._apobj

    @classmethod
    def send_alert(cls, error_type: str, message: str, severity: str = "WARNING"):
        # ---> CHANGE: In the future, you can replace print with Slack/Discord webhooks or Email alerts here.
        alert_msg = f"🚨 [{severity}] {error_type.upper()}: {message}"
        print(f"\n{alert_msg}")

        # Log critical errors to a file so they aren't lost in the terminal scroll
        if severity == "CRITICAL":
            adfs_logger.critical(message, extra={"stage": "alert_manager", "error_type": error_type})
        elif severity == "HIGH":
            adfs_logger.error(message, extra={"stage": "alert_manager", "error_type": error_type})
        else:
            adfs_logger.warning(message, extra={"stage": "alert_manager", "error_type": error_type})

        # Apprise Universal Notification (Only for HIGH/CRITICAL)
        if severity in ["HIGH", "CRITICAL"] and Config.APPRISE_URL:
            cls._send_via_apprise(error_type, message, severity)

    @classmethod
    def _send_via_apprise(cls, error_type: str, message: str, severity: str):
        """Sends the alert through whatever service is defined in .env"""
        title = f"ADFS {severity}: {error_type.replace('_', ' ').title()}"

        # Determine Apprise notification type
        n_type = apprise.NotifyType.INFO
        if severity == "HIGH": n_type = apprise.NotifyType.WARNING
        if severity == "CRITICAL": n_type = apprise.NotifyType.FAILURE

        try:
            # 🔴 FIXED: Safely fetch the lazy-loaded notifier
            notifier = cls.get_notifier()
            if notifier:
                notifier.notify(
                    title=title,
                    body=message,
                    notify_type=n_type
                )
        except Exception as e:
            print(f"  [AlertManager] Apprise failed: {e}")