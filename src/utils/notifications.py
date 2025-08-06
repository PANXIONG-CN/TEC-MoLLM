# File: src/utils/notifications.py

import smtplib
import logging
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from datetime import datetime

# 导入 ServerChan 推送模块
from .serverchan_notify import send_serverchan_notification

# 从环境变量中获取邮箱配置
QQ_EMAIL_USER = os.environ.get("QQ_EMAIL_USER")  # 发送方QQ邮箱
QQ_EMAIL_PASSWORD = os.environ.get("QQ_EMAIL_PASSWORD")  # QQ邮箱授权码
QQ_EMAIL_TO = os.environ.get("QQ_EMAIL_TO")  # 接收方邮箱


def send_qq_email_notification(title: str, content: str):
    """
    通过QQ邮箱SMTP服务发送邮件通知。

    Args:
        title (str): 邮件标题 (e.g., experiment name).
        content (str): 邮件正文内容.
    """
    if not all([QQ_EMAIL_USER, QQ_EMAIL_PASSWORD, QQ_EMAIL_TO]):
        missing_vars = []
        if not QQ_EMAIL_USER:
            missing_vars.append("QQ_EMAIL_USER")
        if not QQ_EMAIL_PASSWORD:
            missing_vars.append("QQ_EMAIL_PASSWORD")
        if not QQ_EMAIL_TO:
            missing_vars.append("QQ_EMAIL_TO")
        logging.warning(f"邮箱配置不完整，缺少环境变量: {', '.join(missing_vars)}。跳过邮件通知。")
        return

    try:
        # 创建邮件对象
        msg = MIMEMultipart()
        msg["From"] = QQ_EMAIL_USER
        msg["To"] = QQ_EMAIL_TO
        msg["Subject"] = Header(f"[TEC-MoLLM] {title}", "utf-8")

        # 添加时间戳到邮件正文
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_content = f"时间: {timestamp}\n\n{content}"

        # 创建HTML格式的邮件正文
        html_content = f"""
        <html>
        <body>
            <h2 style="color: #2E8B57;">{title}</h2>
            <p><strong>时间:</strong> {timestamp}</p>
            <hr>
            <pre style="font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 5px;">{content}</pre>
        </body>
        </html>
        """

        # 添加纯文本和HTML版本
        msg.attach(MIMEText(formatted_content, "plain", "utf-8"))
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        # 尝试连接到QQ邮箱SMTP服务器
        server = None
        try:
            # 方法1: 使用SMTP_SSL
            logging.info("尝试使用SMTP_SSL连接...")
            server = smtplib.SMTP_SSL("smtp.qq.com", 465)
            server.set_debuglevel(0)  # 关闭调试模式避免输出过多信息
            server.login(QQ_EMAIL_USER, QQ_EMAIL_PASSWORD)
        except Exception as ssl_error:
            logging.warning(f"SMTP_SSL连接失败: {ssl_error}")
            # 方法2: 使用SMTP + STARTTLS
            try:
                logging.info("尝试使用SMTP + STARTTLS连接...")
                if server:
                    server.quit()
                server = smtplib.SMTP("smtp.qq.com", 587)
                server.starttls()
                server.login(QQ_EMAIL_USER, QQ_EMAIL_PASSWORD)
            except Exception as tls_error:
                logging.error(f"SMTP + STARTTLS也失败: {tls_error}")
                raise Exception(f"所有连接方式都失败了。SSL错误: {ssl_error}, TLS错误: {tls_error}")

        # 发送邮件
        text = msg.as_string()
        result = server.sendmail(QQ_EMAIL_USER, QQ_EMAIL_TO, text)
        server.quit()

        # 检查发送结果
        if result:
            logging.warning(f"邮件发送部分失败: {result}")
        else:
            logging.info(f"QQ邮箱通知已发送: {title}")
            return True

    except Exception as e:
        logging.error(f"发送QQ邮箱通知时发生异常: {e}")
        return False


# 向后兼容的别名函数
def send_wechat_notification(title: str, content: str):
    """
    发送微信推送通知，使用 ServerChan 服务。
    向后兼容的别名函数，现在使用 ServerChan 发送微信推送。
    """
    return send_serverchan_notification(title, content)
