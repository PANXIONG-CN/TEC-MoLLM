# ServerChan 微信推送配置说明

## 概述

项目已成功配置 ServerChan 微信推送服务，可在训练过程中自动发送微信通知。

## 配置信息

- **服务商**: ServerChan (Server酱)
- **SendKey**: `SCT291707Tq6qEBYkKMcRIdczO5LM6Qp1U`
- **推送地址**: `https://sctapi.ftqq.com/{SendKey}.send`

## 功能特性

### 自动推送场景
训练脚本会在以下情况自动发送微信推送：

1. **训练开始** - 显示训练配置和启动信息
2. **每个Epoch完成** - 显示损失、学习率、验证指标等
3. **发现最佳模型** - 当验证指标创新高时通知
4. **训练完成** - 显示最终结果和统计信息

### 推送内容
- 📊 实时训练指标（损失、学习率、准确率等）
- 🎯 模型性能评估结果
- ⏱️ 训练时间和进度信息
- 💾 模型保存路径
- 📈 验证集表现趋势

## 代码集成

### 核心推送函数
```python
from src.utils.notifications import send_wechat_notification

# 发送微信推送
send_wechat_notification("标题", "内容")
```

### 实现文件
- **主推送模块**: `src/utils/serverchan_notify.py`
- **兼容接口**: `src/utils/notifications.py`
- **训练集成**: `train.py` (第35行导入，多处调用)

## 测试验证

### 手动测试
```bash
# 测试 ServerChan 推送功能
cd /home/panxiong/TEC-MoLLM
python src/utils/serverchan_notify.py
```

### 集成测试
推送功能已在训练脚本中的以下位置验证：
- 训练启动通知 (train.py:323)
- Epoch报告推送 (train.py:411)
- 最佳模型通知 (train.py:452)
- 训练完成通知 (train.py:479)

## 向后兼容

原有的 `send_wechat_notification()` 函数保持不变，内部已切换到 ServerChan 实现：

```python
# 这个函数调用现在会发送 ServerChan 推送
send_wechat_notification("🚀 TEC-MoLLM 训练启动", start_msg)
```

## 优势特点

✅ **无需额外配置** - SendKey 已内置在代码中  
✅ **稳定可靠** - ServerChan 服务稳定性好  
✅ **即时到达** - 微信推送延迟低  
✅ **向后兼容** - 不影响现有训练脚本  
✅ **错误处理** - 包含完整的异常处理和重试机制  

## 注意事项

1. **推送频率**: ServerChan 免费版有推送频率限制，请合理使用
2. **网络要求**: 需要服务器能访问 `sctapi.ftqq.com`
3. **微信绑定**: 确保 ServerChan 账号已绑定微信
4. **SendKey安全**: 代码中的 SendKey 应谨慎保管

## 故障排除

如果推送失败，检查以下项目：

1. **网络连接**: 确认服务器可访问外网
2. **SendKey有效性**: 在 ServerChan 官网验证密钥状态
3. **微信绑定**: 确认微信已正确绑定到 ServerChan 账号
4. **日志信息**: 查看训练日志中的推送错误信息

## 配置历史

- **2025-08-06**: 成功配置 ServerChan 推送，替换原有邮件通知
- **SendKey**: SCT291707Tq6qEBYkKMcRIdczO5LM6Qp1U
- **测试状态**: ✅ 推送功能正常，集成测试通过