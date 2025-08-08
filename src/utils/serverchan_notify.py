#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ServerChan 微信推送模块
使用 ServerChan 服务发送微信推送通知
"""

import os
import json
import logging
import requests
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ServerChan 配置
# 从环境变量读取 SendKey，避免硬编码
SERVERCHAN_SENDKEY = os.environ.get("SERVERCHAN_SENDKEY")
SERVERCHAN_URL = (
    f"https://sctapi.ftqq.com/{SERVERCHAN_SENDKEY}.send" if SERVERCHAN_SENDKEY else None
)


def format_content_for_serverchan(content):
    """
    格式化内容为ServerChan可正确显示的Markdown格式

    Args:
        content (str): 原始内容

    Returns:
        str: 格式化后的Markdown内容
    """
    # 将内容按行分割
    lines = content.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if not line:  # 空行
            formatted_lines.append("")
        elif line.startswith("RUN:"):
            # 运行信息用代码块格式
            formatted_lines.append(f"**{line}**")
        elif line.startswith("Epoch"):
            # Epoch信息用粗体
            formatted_lines.append(f"**{line}**")
        elif "•" in line or line.startswith("-"):
            # 列表项保持原样
            formatted_lines.append(line)
        elif ":" in line and ("Loss" in line or "RMSE" in line or "R²" in line):
            # 指标信息用代码格式
            formatted_lines.append(f"`{line}`")
        else:
            # 普通文本
            formatted_lines.append(line)

    # 用两个换行符连接，确保在ServerChan中正确显示
    return "\n\n".join(formatted_lines)


def send_serverchan_notification(title, content):
    """
    使用 ServerChan 发送微信推送通知

    Args:
        title (str): 通知标题
        content (str): 通知内容

    Returns:
        bool: 发送是否成功
    """
    if not SERVERCHAN_SENDKEY:
        logger.warning("ServerChan SendKey 未配置，跳过推送")
        return False

    # 添加时间戳并格式化为Markdown
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 格式化内容为Markdown格式，确保换行正确显示
    formatted_content = format_content_for_serverchan(content)
    content_with_time = f"**时间**: {timestamp}\n\n{formatted_content}"

    try:
        # ServerChan API 数据格式
        payload = {"title": f"[TEC-MoLLM] {title}", "desp": content_with_time}

        # 发送POST请求
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(SERVERCHAN_URL, data=payload, headers=headers, timeout=10)

        # 检查响应
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get("code") == 0:
                    logger.info(f"ServerChan 推送发送成功: {title}")
                    return True
                else:
                    logger.error(f"ServerChan 推送失败: {result.get('message', '未知错误')}")
                    return False
            except json.JSONDecodeError:
                logger.error(f"ServerChan 响应解析失败: {response.text}")
                return False
        else:
            logger.error(f"ServerChan 请求失败: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.Timeout:
        logger.error("ServerChan 请求超时")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"ServerChan 请求异常: {e}")
        return False
    except Exception as e:
        logger.error(f"发送 ServerChan 通知时发生异常: {e}")
        return False


def test_serverchan():
    """
    测试 ServerChan 推送功能
    """
    test_title = "TEC-MoLLM 推送测试"
    test_content = """这是一条来自 TEC-MoLLM 的测试推送消息。

测试内容：
- ServerChan 配置: ✅
- 推送功能: ✅
- 时间戳: 正常显示

如果您收到这条消息，说明 ServerChan 推送配置成功！"""

    print("正在测试 ServerChan 推送...")
    if SERVERCHAN_SENDKEY:
        print(f"SendKey: {SERVERCHAN_SENDKEY[:10]}...{SERVERCHAN_SENDKEY[-4:]}")
    else:
        print("SendKey 未配置")

    if send_serverchan_notification(test_title, test_content):
        print("✅ ServerChan 推送测试成功！请检查微信消息。")
        return True
    else:
        print("❌ ServerChan 推送测试失败！")
        return False


if __name__ == "__main__":
    # 运行测试
    if not SERVERCHAN_SENDKEY:
        print("⚠️ 请设置环境变量 SERVERCHAN_SENDKEY")
    else:
        test_serverchan()
