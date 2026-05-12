#!/usr/bin/env python3
"""
Kaggle Battery Dataset 下载脚本
支持使用Kaggle API下载锂电池相关数据集
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
import argparse


# 推荐的Kaggle电池数据集
KAGGLE_DATASETS: Dict[str, Dict] = {
    "patrickfleith": {
        "name": "lithium-ion-battery-aging-datasets",
        "description": "锂电池老化数据集合集",
        "size": "~500MB",
        "files": ["*.csv", "*.mat"]
    },
    "sahilwagh": {
        "name": "lithium-ion-battery-aging-dataset",
        "description": "NASA和CALCE数据集整合",
        "size": "~200MB",
        "files": ["*.csv"]
    },
    "vinayakshanawad": {
        "name": "lithium-ion-battery-life-prediction",
        "description": "电池寿命预测数据集",
        "size": "~100MB",
        "files": ["*.csv"]
    }
}


class KaggleDataDownloader:
    """Kaggle数据集下载器"""
    
    def __init__(self, save_dir: str = "data/raw/kaggle"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._check_kaggle_api()
    
    def _check_kaggle_api(self) -> bool:
        """检查Kaggle API是否已安装和配置"""
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Kaggle API已安装: {result.stdout.strip()}")
            return True
        except FileNotFoundError:
            print("✗ Kaggle API未安装")
            print("  请运行: pip install kaggle")
            return False
        except subprocess.CalledProcessError:
            print("✗ Kaggle API检查失败")
            return False
    
    def _check_kaggle_credentials(self) -> bool:
        """检查Kaggle API凭证是否配置"""
        # 检查凭证文件位置
        possible_paths = [
            Path.home() / ".kaggle" / "kaggle.json",
            Path.home() / "kaggle.json"
        ]
        
        for cred_path in possible_paths:
            if cred_path.exists():
                print(f"✓ 找到Kaggle凭证: {cred_path}")
                return True
        
        print("✗ 未找到Kaggle API凭证")
        print("  请按照以下步骤配置:")
        print("  1. 访问 https://www.kaggle.com/account")
        print("  2. 点击 'Create New Token'")
        print("  3. 下载 kaggle.json 文件")
        print("  4. 移动到 ~/.kaggle/kaggle.json")
        print("  5. 设置权限: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    def download_dataset(self, owner: str, dataset_name: str, unzip: bool = True) -> bool:
        """
        下载指定数据集
        
        Args:
            owner: 数据集所有者
            dataset_name: 数据集名称
            unzip: 是否自动解压
            
        Returns:
            bool: 下载是否成功
        """
        if not self._check_kaggle_credentials():
            return False
        
        dataset_slug = f"{owner}/{dataset_name}"
        target_dir = self.save_dir / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n⬇️  正在下载数据集: {dataset_slug}")
        print(f"   保存到: {target_dir}")
        
        try:
            cmd = [
                "kaggle", "datasets", "download",
                "-d", dataset_slug,
                "-p", str(target_dir),
                "--force"
            ]
            
            if unzip:
                cmd.append("--unzip")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✓ 下载成功!")
            if result.stdout:
                print(f"   输出: {result.stdout}")
            
            # 列出下载的文件
            files = list(target_dir.glob("*"))
            print(f"   文件列表:")
            for f in files:
                size = f.stat().st_size if f.is_file() else "<dir>"
                print(f"     - {f.name} ({size})")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ 下载失败")
            print(f"   错误: {e.stderr}")
            return False
        except Exception as e:
            print(f"✗ 发生错误: {str(e)}")
            return False
    
    def download_all(self, datasets: List[str] = None) -> Dict[str, bool]:
        """
        下载多个数据集
        
        Args:
            datasets: 要下载的数据集列表，None则下载所有推荐数据集
            
        Returns:
            Dict[str, bool]: 各数据集下载结果
        """
        if datasets is None:
            datasets = list(KAGGLE_DATASETS.keys())
        
        results = {}
        print(f"\n{'='*60}")
        print("Kaggle Battery Dataset 批量下载")
        print(f"{'='*60}\n")
        
        for owner in datasets:
            if owner in KAGGLE_DATASETS:
                info = KAGGLE_DATASETS[owner]
                results[owner] = self.download_dataset(
                    owner=owner,
                    dataset_name=info["name"]
                )
                print()
            else:
                print(f"✗ 未知数据集: {owner}")
                results[owner] = False
        
        # 打印下载统计
        success_count = sum(results.values())
        print(f"{'='*60}")
        print(f"下载完成: {success_count}/{len(datasets)} 个数据集成功")
        print(f"{'='*60}\n")
        
        return results
    
    def list_datasets(self):
        """列出推荐的数据集"""
        print(f"\n{'='*60}")
        print("推荐的Kaggle电池数据集")
        print(f"{'='*60}\n")
        
        for owner, info in KAGGLE_DATASETS.items():
            print(f"📊 {owner}/{info['name']}")
            print(f"   描述: {info['description']}")
            print(f"   大小: {info['size']}")
            print(f"   文件: {', '.join(info['files'])}")
            print()
        
        print(f"{'='*60}")
        print(f"总计: {len(KAGGLE_DATASETS)} 个数据集")
        print(f"{'='*60}\n")
    
    def search_datasets(self, keyword: str = "lithium battery"):
        """
        搜索Kaggle上的电池数据集
        
        Args:
            keyword: 搜索关键词
        """
        if not self._check_kaggle_credentials():
            return
        
        print(f"\n🔍 搜索数据集: '{keyword}'")
        print(f"{'='*60}\n")
        
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "list", "-s", keyword],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ 搜索失败: {e.stderr}")


def setup_kaggle_api():
    """提供Kaggle API设置指导"""
    print(f"\n{'='*60}")
    print("Kaggle API 设置指南")
    print(f"{'='*60}\n")
    
    print("步骤 1: 安装Kaggle API")
    print("  pip install kaggle\n")
    
    print("步骤 2: 获取API Token")
    print("  1. 访问 https://www.kaggle.com")
    print("  2. 注册/登录账号")
    print("  3. 点击右上角头像 → Account")
    print("  4. 向下滚动到 'API' 部分")
    print("  5. 点击 'Create New Token'")
    print("  6. 下载 kaggle.json 文件\n")
    
    print("步骤 3: 配置API凭证")
    print("  # macOS/Linux:")
    print("  mkdir -p ~/.kaggle")
    print("  mv ~/Downloads/kaggle.json ~/.kaggle/")
    print("  chmod 600 ~/.kaggle/kaggle.json\n")
    
    print("  # Windows:")
    print("  mkdir %USERPROFILE%\\.kaggle")
    print("  move %USERPROFILE%\\Downloads\\kaggle.json %USERPROFILE%\\.kaggle\\")
    print("  # 设置文件权限为只读\n")
    
    print("步骤 4: 验证安装")
    print("  kaggle --version")
    print("  kaggle datasets list\n")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='下载Kaggle Battery Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看设置指南
  python download_kaggle_data.py --setup
  
  # 列出推荐数据集
  python download_kaggle_data.py --list
  
  # 下载所有推荐数据集
  python download_kaggle_data.py --all
  
  # 下载指定数据集
  python download_kaggle_data.py --dataset patrickfleith
  
  # 搜索数据集
  python download_kaggle_data.py --search "battery aging"
  
  # 指定保存目录
  python download_kaggle_data.py --all --save-dir ./my_kaggle_data
        """
    )
    
    parser.add_argument('--setup', action='store_true', help='显示Kaggle API设置指南')
    parser.add_argument('--list', action='store_true', help='列出推荐的数据集')
    parser.add_argument('--all', action='store_true', help='下载所有推荐数据集')
    parser.add_argument('--dataset', help='下载指定数据集 (如: patrickfleith)')
    parser.add_argument('--search', metavar='KEYWORD', help='搜索数据集')
    parser.add_argument('--save-dir', default='data/raw/kaggle', help='数据保存目录')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_kaggle_api()
        return
    
    downloader = KaggleDataDownloader(save_dir=args.save_dir)
    
    if args.list:
        downloader.list_datasets()
    elif args.search:
        downloader.search_datasets(args.search)
    elif args.all:
        downloader.download_all()
    elif args.dataset:
        if args.dataset in KAGGLE_DATASETS:
            info = KAGGLE_DATASETS[args.dataset]
            downloader.download_dataset(
                owner=args.dataset,
                dataset_name=info["name"]
            )
        else:
            print(f"✗ 未知数据集: {args.dataset}")
            print("  使用 --list 查看可用数据集")
    else:
        parser.print_help()
        print("\n提示: 首次使用请运行 --setup 查看配置指南")


if __name__ == "__main__":
    main()
