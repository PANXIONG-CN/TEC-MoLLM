# File: src/utils/notifications.py

import requests
import logging
import os

# 从环境变量中获取token，更安全
WECHAT_BOT_TOKEN = os.environ.get("WECHAT_BOT_TOKEN")

def send_wechat_notification(title: str, content: str):
    """
    Sends a notification to WeChat via AutoDL's service.

    Args:
        title (str): The main title of the message (e.g., experiment name).
        content (str): The detailed content of the message.
    """
    if not WECHAT_BOT_TOKEN:
        logging.warning("WECHAT_BOT_TOKEN not set. Skipping notification.")
        return

    headers = {
        "Authorization": WECHAT_BOT_TOKEN,
        "Content-Type": "application/json",
    }
    payload = {
        "title": title,   # This is required by the API but not shown in WeChat
        "name": content,  # This is the actual content shown in WeChat
    }

    try:
        resp = requests.post(
            "https://www.autodl.com/api/v1/wechat/message/send",
            json=payload,
            headers=headers,
            timeout=10,  # Use a reasonable timeout
        )
        if resp.status_code == 200:
            logging.info(f"微信通知已发送: {title}")
        else:
            logging.error(f"发送微信通知失败: {resp.status_code} - {resp.text}")
    except Exception as e:
        logging.error(f"发送微信通知时发生异常: {e}") 