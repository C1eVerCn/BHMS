#!/bin/bash
# Kaggle API 设置修复脚本

echo "=========================================="
echo "Kaggle API 设置修复"
echo "=========================================="
echo ""

# 1. 安装kaggle包
echo "步骤 1: 安装 Kaggle API..."
pip install kaggle -U
if [ $? -ne 0 ]; then
    echo "✗ pip安装失败，尝试使用python -m pip..."
    python -m pip install kaggle -U
fi

# 2. 检查安装
echo ""
echo "步骤 2: 验证安装..."
if command -v kaggle &> /dev/null; then
    echo "✓ Kaggle命令已找到"
    kaggle --version
else
    echo "✗ Kaggle命令未找到"
    echo "  尝试查找kaggle安装位置..."
    find ~ -name "kaggle" -type f 2>/dev/null | head -5
fi

# 3. 创建配置目录
echo ""
echo "步骤 3: 创建配置目录..."
mkdir -p ~/.kaggle

# 4. 检查凭证文件
echo ""
echo "步骤 4: 检查凭证文件..."
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✓ 凭证文件已存在: ~/.kaggle/kaggle.json"
    chmod 600 ~/.kaggle/kaggle.json
    echo "✓ 权限已设置为600"
elif [ -f ~/kaggle.json ]; then
    echo "发现凭证文件在 home 目录，移动到正确位置..."
    mv ~/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "✓ 凭证文件已移动并设置权限"
elif [ -f ~/Downloads/kaggle.json ]; then
    echo "发现凭证文件在 Downloads 目录，移动到正确位置..."
    mv ~/Downloads/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "✓ 凭证文件已移动并设置权限"
else
    echo "✗ 未找到凭证文件 kaggle.json"
    echo ""
    echo "请按照以下步骤获取凭证:"
    echo "  1. 访问 https://www.kaggle.com"
    echo "  2. 登录账号"
    echo "  3. 点击头像 → Account"
    echo "  4. 找到 API 部分，点击 'Create New Token'"
    echo "  5. 下载 kaggle.json 文件"
    echo "  6. 重新运行此脚本"
fi

# 5. 验证设置
echo ""
echo "步骤 5: 验证设置..."
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✓ 凭证文件存在"
    ls -la ~/.kaggle/kaggle.json
    
    # 尝试运行kaggle命令
    echo ""
    echo "测试 Kaggle API..."
    kaggle datasets list 2>&1 | head -5
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ Kaggle API 设置成功!"
        echo "=========================================="
    else
        echo ""
        echo "⚠️  Kaggle API 测试失败，请检查错误信息"
    fi
else
    echo "✗ 凭证文件不存在，无法完成设置"
fi

echo ""
echo "设置完成!"
