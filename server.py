"""

Flask webhook server for Microsoft Graph API change notifications.

Responsibilities:
    - Receive and validate Graph API webhook POST notifications
    - Deduplicate notifications using a persisted processed-IDs file
    - Trigger the agent automatically when a new patching email arrives
    - Manage the Graph subscription lifecycle (create + periodic renewal)

Run:
    python server.py

The webhook must be reachable from the internet (e.g. via ngrok).
Set WEBHOOK_URL in .env to the public URL of this server's /webhook endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from auth import get_headers
from email_agent import run_agent


load_dotenv()

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXCELS_FOLDER  :str=os.environ["EXCELS_FOLDER"]
FOLDER_NAME    : str = os.environ["FOLDER_NAME"]
WEBHOOK_URL    : str = os.environ["WEBHOOK_URL"]
PROCESSED_FILE : str = os.environ.get("PROCESSED_FILE", "processed_ids.txt")


_processed_lock: threading.Lock = threading.Lock()


def _load_processed_ids() -> set[str]:
    """Load the set of already-processed message IDs from disk."""
    path = Path(PROCESSED_FILE)
    if not path.exists():
        return set()
    try:
        ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        logger.debug("Loaded %d processed message IDs from %s", len(ids), PROCESSED_FILE)
        return ids
    except OSError as exc:
        logger.warning("Could not read %s: %s — starting with empty set.", PROCESSED_FILE, exc)
        return set()


def _save_processed_id(message_id: str) -> None:
    """Append a new message ID to the persistent processed-IDs file."""
    try:
        with open(PROCESSED_FILE, "a", encoding="utf-8") as fh:
            fh.write(message_id + "\n")
    except OSError as exc:
        logger.error("Could not persist message ID to %s: %s", PROCESSED_FILE, exc)


# In-memory set for fast O(1) lookup — populated once at startup
_processed_ids: set[str] = _load_processed_ids()


flask_app = Flask(__name__)


@flask_app.route("/webhook", methods=["GET", "POST"])
def webhook():
    """
    Microsoft Graph webhook endpoint.

    GET  → Respond to Graph's validation handshake with the validationToken.
    POST → Process change notifications (new emails in the monitored folder).
    """
    # ---- Validation handshake (GET or POST with validationToken param) ----
    validation_token = request.args.get("validationToken")
    if validation_token:
        logger.info("Webhook validation handshake received — responding OK.")
        return validation_token, 200, {"Content-Type": "text/plain; charset=utf-8"}

    # ---- Change notification (POST) ----------------------------------------
    if request.method != "POST":
        return jsonify({"status": "method not allowed"}), 405

    data = request.get_json(silent=True)
    if not data or "value" not in data:
        logger.warning("Webhook received malformed payload — ignoring.")
        return jsonify({"status": "ignored"}), 200

    for notification in data["value"]:
        message_id = notification.get("resourceData", {}).get("id")
        if not message_id:
            logger.debug("Notification missing resourceData.id — skipping.")
            continue

        # Deduplication — check in-memory set first, then persist
        with _processed_lock:
            if message_id in _processed_ids:
                logger.debug("Duplicate notification for %s — skipping.", message_id)
                continue
            _processed_ids.add(message_id)
            _save_processed_id(message_id)

        logger.info("New mail notification received: %s", message_id)

        # Dispatch agent in a background thread so we return 200 immediately
        # (Graph requires a response within 3 seconds)
        threading.Thread(
            target=_handle_new_mail_notification,
            args=(message_id,),
            daemon=True,
        ).start()

    return jsonify({"status": "ok"}), 200


def _get_message_subject(message_id: str) -> str:
    """Fetch just the subject of a message directly from Graph API."""
    try:
        url  = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}?$select=subject"
        resp = requests.get(url, headers=get_headers(), timeout=15)
        resp.raise_for_status()
        return resp.json().get("subject", "")
    except Exception as exc:
        logger.warning("Could not fetch subject for %s: %s", message_id, exc)
        return ""


# def _handle_new_mail_notification(message_id: str) -> None:
#     """
#     Background task: run the agent in response to a new mail notification.

#     Args:
#         message_id: Graph API message ID (used only for logging here;
#                     the agent fetches the full email via get_latest_mail).
#     """
#     logger.info("Processing notification for message: %s", message_id)

#     query = (
#         "A new patching email just arrived. "
#         "Fetch the latest email and check if its subject contains "
#         "'Maintenance Notification', 'Reschedule Maintenance', or 'Implementation Status'. "
#         "If it does, download any Excel attachments and summarise the Lyric servers "
#         "found in the updated data, including their patch windows and reboot requirements."
#     )

#     try:
#         result = run_agent(query, stream=False)
#         logger.info("\n[Agent Auto-Response]\n%s", result)
#     except Exception as exc:
#         logger.error("Agent failed while handling notification %s: %s", message_id, exc)


def _handle_new_mail_notification(message_id: str) -> None:
    logger.info("Processing notification for message: %s", message_id)

    subject = _get_message_subject(message_id)
    logger.info("Mail subject: %s", subject)

    if "Implementation Status" in subject:
        query = (
            "A new Implementation Status email just arrived. "
            "Fetch the latest email from the Enterprise Patching folder and "
            "download/Save any Excel attachments."
        )
    elif "Reschedule Maintenance" in subject:
        query = (
            "A new Reschedule Maintenance email just arrived. "
            "Fetch the latest email, download any Excel attachments, and summarise "
            "the Lyric servers found in the updated data, including their patch windows "
            "and reboot requirements."
        )
    elif "Maintenance Notification" in subject:
        query = (
            "A new Maintenance Notification email just arrived. "
            "Fetch the latest email, download any Excel attachments, and summarise "
            "the Lyric servers found in the updated data, including their patch windows "
            "and reboot requirements."
        )
    else:
        logger.info(
            "Subject '%s' does not match any known category — skipping agent.", subject
        )
        return

    try:
        result = run_agent(query, stream=False)
        logger.info("\n[Agent Auto-Response]\n%s", result)
    except Exception as exc:
        logger.error(
            "Agent failed while handling notification %s: %s", message_id, exc
        )



def _get_folder_id() -> str | None:
    """Resolve the monitored folder's Graph API ID."""
    url  = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/childFolders"
    resp = requests.get(url, headers=get_headers(), timeout=15)
    resp.raise_for_status()
    for folder in resp.json().get("value", []):
        if folder["displayName"] == FOLDER_NAME:
            return folder["id"]
    return None


def _get_existing_subscriptions(folder_id: str) -> list[dict]:
    """Return active Graph subscriptions that watch the given folder."""
    resp = requests.get(
        "https://graph.microsoft.com/v1.0/subscriptions",
        headers=get_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return [
        s for s in resp.json().get("value", [])
        if f"mailFolders/{folder_id}" in s.get("resource", "")
    ]


def setup_subscription() -> None:
    """
    Create a Graph API change-notification subscription if one does not
    already exist for the monitored folder.

    Graph subscriptions expire after at most ~3 days (4230 minutes is the
    maximum allowed for mail resources). The APScheduler job renew_subscription()
    handles renewal before expiry.

    Called once at startup in a background thread (so Flask is already
    listening before Graph tries to validate the webhook URL).
    """
    # Give Flask a moment to start accepting connections before Graph
    # hits the validation endpoint
    time.sleep(3)
    logger.info("Setting up Graph API subscription…")

    try:
        folder_id = _get_folder_id()
        if not folder_id:
            logger.error("Folder '%s' not found — subscription not created.", FOLDER_NAME)
            return

        existing = _get_existing_subscriptions(folder_id)
        if existing:
            logger.info("Subscription already active (id=%s) — skipping creation.", existing[0]["id"])
            return

        expiration = (datetime.now(timezone.utc) + timedelta(minutes=4200)).isoformat()

        resp = requests.post(
            "https://graph.microsoft.com/v1.0/subscriptions",
            headers=get_headers(),
            json={
                "changeType":         "created",
                "notificationUrl":    WEBHOOK_URL,
                "resource":           f"me/mailFolders/{folder_id}/messages",
                "expirationDateTime": expiration,
                "clientState":        "enterprise_patching_agent",
            },
            timeout=15,
        )

        if resp.status_code in (200, 201):
            sub = resp.json()
            logger.info(
                "Subscription created (id=%s, expires=%s)",
                sub.get("id"),
                sub.get("expirationDateTime"),
            )
        else:
            logger.error("Subscription creation failed (%s): %s", resp.status_code, resp.json())

    except Exception as exc:
        logger.error("setup_subscription failed: %s", exc)


def renew_subscription() -> None:
    """
    Renew all active Graph subscriptions to prevent expiry.
    Called periodically by APScheduler (every 12 hours).
    """
    logger.info("Renewing Graph API subscriptions…")

    try:
        folder_id = _get_folder_id()
        if not folder_id:
            logger.warning("Folder not found during renewal — skipping.")
            return

        subscriptions = _get_existing_subscriptions(folder_id)
        if not subscriptions:
            logger.warning("No active subscriptions found during renewal — recreating.")
            setup_subscription()
            return

        new_expiry = (datetime.now(timezone.utc) + timedelta(minutes=4200)).isoformat()

        for sub in subscriptions:
            sub_id = sub["id"]
            resp   = requests.patch(
                f"https://graph.microsoft.com/v1.0/subscriptions/{sub_id}",
                headers=get_headers(),
                json={"expirationDateTime": new_expiry},
                timeout=15,
            )
            if resp.status_code == 200:
                logger.info("Subscription %s renewed until %s", sub_id, new_expiry)
            else:
                logger.error("Failed to renew subscription %s: %s", sub_id, resp.json())

    except Exception as exc:
        logger.error("renew_subscription failed: %s", exc)




if __name__ == "__main__":
    os.makedirs(EXCELS_FOLDER, exist_ok=True)
    # Start subscription setup in background so Flask binds first
    threading.Thread(target=setup_subscription, daemon=True).start()

    # Schedule periodic subscription renewal every 12 hours
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(renew_subscription, "interval", hours=12, id="renew_sub")
    scheduler.start()
    logger.info("Subscription renewal scheduler started (every 12 hours).")

    logger.info("Starting Flask webhook server on port 5000…")
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)