#!/usr/bin/env python3
"""
NASA PCoE Battery Dataset 下载脚本 (最终版)
支持下载NASA Prognostics Center of Excellence的锂电池老化数据集

注意: NASA数据现在托管在AWS S3上:
https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import warnings

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("请先安装requests库:")
    print("  pip install requests")
    sys.exit(1)


# NASA PCoE 电池数据集AWS S3链接
NASA_BATTERY_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"

# 数据集说明
DATASET_INFO: Dict[str, Dict] = {
    "B0005": {"type": "Li-ion 18650", "cycles": 168, "conditions": "Room temperature (24°C)", "discharge_current": "2A"},
    "B0006": {"type": "Li-ion 18650", "cycles": 168, "conditions": "Room temperature (24°C)", "discharge_current": "2A"},
    "B0007": {"type": "Li-ion 18650", "cycles": 168, "conditions": "Room temperature (24°C)", "discharge_current": "2A"},
    "B0018": {"type": "Li-ion 1850", "cycles": 132, "conditions": "Room temperature (24°C)", "discharge_current": "2A"},
    "B0025": {"type": "Li-ion 18650", "cycles": 100, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0026": {"type": "Li-ion 18650", "cycles": 100, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0027": {"type": "Li-ion 18650", "cycles": 100, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0028": {"type": "Li-ion 18650", "cycles": 100, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0029": {"type": "Li-ion 18650", "cycles": 104, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0030": {"type": "Li-ion 18650", "cycles": 120, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0031": {"type": "Li-ion 18650", "cycles": 156, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0032": {"type": "Li-ion 18650", "cycles": 140, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0033": {"type": "Li-ion 18650", "cycles": 160, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0034": {"type": "Li-ion 18650", "cycles": 140, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0036": {"type": "Li-ion 18650", "cycles": 130, "conditions": "Accelerated aging", "discharge_current": "4A"},
    "B0038": {"type": "Li-ion 18650", "cycles": 80, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0039": {"type": "Li-ion 18650", "cycles": 80, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0040": {"type": "Li-ion 18650", "cycles": 80, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0041": {"type": "Li-ion 18650", "cycles": 80, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0042": {"type": "Li-ion 18650", "cycles": 80, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0043": {"type": "Li-ion 18650", "cycles": 80, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0044": {"type": "Li-ion 18650", "cycles": 80, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0045": {"type": "Li-ion 18650", "cycles": 60, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0046": {"type": "Li-ion 18650", "cycles": 60, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0047": {"type": "Li-ion 18650", "cycles": 60, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0048": {"type": "Li-ion 18650", "cycles": 60, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0049": {"type": "Li-ion 18650", "cycles": 50, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0050": {"type": "Li-ion 18650", "cycles": 50, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0051": {"type": "Li-ion 18650", "cycles": 50, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0052": {"type": "Li-ion 18650", "cycles": 50, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0053": {"type": "Li-ion 18650", "cycles": 40, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0054": {"type": "Li-ion 18650", "cycles": 40, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0055": {"type": "Li-ion 18650", "cycles": 40, "conditions": "Different temperatures", "discharge_current": "2A"},
    "B0056": {"type": "Li-ion 18650", "cycles": 40, "conditions": "Different temperatures", "discharge_current": "2A"},
}


class NASADataDownloader:
    """NASA PCoE数据集下载器"""
    
    def __init__(self, save_dir: str = "data/raw/nasa"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建session并配置重试策略
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def download_and_extract(self, cleanup: bool = True) -> bool:
        """
        下载NASA电池数据集并解压所有文件
        
        Args:
            cleanup: 是否清理临时ZIP文件
            
        Returns:
            bool: 下载是否成功
        """
        zip_filename = "NASA_Battery_Dataset.zip"
        zip_filepath = self.save_dir / zip_filename
        
        # 检查是否已下载
        if zip_filepath.exists():
            print(f"✓ {zip_filename} 已存在")
        else:
            print(f"⬇️  正在下载 NASA Battery Dataset...")
            print(f"   URL: {NASA_BATTERY_URL}")
            print(f"   保存到: {zip_filepath}")
            print(f"   注意: 文件较大 (~200MB)，请耐心等待...\n")
            
            try:
                # 发送请求
                response = self.session.get(
                    NASA_BATTERY_URL, 
                    stream=True, 
                    timeout=300,
                    verify=True
                )
                response.raise_for_status()
                
                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))
                if total_size == 0:
                    total_size = 200 * 1024 * 1024  # 估计200MB
                
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                
                # 下载文件
                with open(zip_filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 显示进度
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r   进度: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
                
                print(f"\n✓ 下载完成 ({downloaded} bytes)")
                
            except Exception as e:
                print(f"\n✗ 下载失败: {str(e)}")
                if zip_filepath.exists():
                    zip_filepath.unlink()
                return False
        
        # 解压主ZIP文件
        print(f"\n📦 正在解压主文件...")
        extract_dir = self.save_dir / "temp_extract"
        extract_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✓ 主文件解压完成")
        except Exception as e:
            print(f"✗ 解压失败: {str(e)}")
            return False
        
        # 查找并解压子ZIP文件
        print(f"\n📦 正在解压子数据集...")
        battery_data_dir = extract_dir / "5. Battery Data Set"
        
        if not battery_data_dir.exists():
            # 尝试其他可能的路径
            for subdir in extract_dir.iterdir():
                if subdir.is_dir() and "battery" in subdir.name.lower():
                    battery_data_dir = subdir
                    break
        
        if battery_data_dir.exists():
            mat_count_before = len(list(self.save_dir.glob("*.mat")))
            
            for zip_file in battery_data_dir.glob("*.zip"):
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        # 只解压.mat文件和README文件
                        for name in zf.namelist():
                            if name.endswith('.mat') or 'README' in name:
                                zf.extract(name, self.save_dir)
                    print(f"   ✓ {zip_file.name}")
                except Exception as e:
                    print(f"   ✗ {zip_file.name}: {str(e)}")
            
            mat_count_after = len(list(self.save_dir.glob("*.mat")))
            print(f"\n✓ 子数据集解压完成")
            print(f"   新增 {mat_count_after - mat_count_before} 个MAT文件")
        
        # 清理临时文件
        if cleanup:
            print(f"\n🗑️  正在清理临时文件...")
            shutil.rmtree(extract_dir, ignore_errors=True)
            zip_filepath.unlink(missing_ok=True)
            print(f"   ✓ 临时文件已清理")
        
        # 显示结果
        print(f"\n{'='*60}")
        print("下载完成!")
        print(f"{'='*60}")
        print(f"数据保存位置: {self.save_dir}")
        
        mat_files = sorted(self.save_dir.glob("B*.mat"))
        print(f"MAT文件数量: {len(mat_files)}")
        
        if mat_files:
            total_size = sum(f.stat().st_size for f in mat_files) / (1024 * 1024)
            print(f"总大小: {total_size:.1f} MB")
        
        print(f"{'='*60}\n")
        
        return True
    
    def list_datasets(self):
        """列出可用的数据集信息"""
        print(f"\n{'='*60}")
        print("NASA PCoE Battery Dataset 信息")
        print(f"{'='*60}\n")
        
        for dataset_id, info in DATASET_INFO.items():
            print(f"📊 {dataset_id}")
            for key, value in info.items():
                print(f"   {key}: {value}")
            print()
        
        print(f"{'='*60}")
        print(f"总计: {len(DATASET_INFO)} 个电池数据文件")
        print(f"下载链接: {NASA_BATTERY_URL}")
        print(f"{'='*60}\n")
    
    def verify_data(self) -> bool:
        """验证数据完整性"""
        print(f"\n{'='*60}")
        print("验证NASA数据集完整性")
        print(f"{'='*60}\n")
        
        expected_files = [f"{bid}.mat" for bid in DATASET_INFO.keys()]
        missing = []
        found = []
        
        for mat_file in expected_files:
            filepath = self.save_dir / mat_file
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                found.append(f"{mat_file} ({size_mb:.2f} MB)")
            else:
                missing.append(mat_file)
        
        if found:
            print(f"✓ 找到 {len(found)} 个文件:")
            for f in found:
                print(f"   - {f}")
        
        if missing:
            print(f"\n✗ 缺失 {len(missing)} 个文件:")
            for m in missing:
                print(f"   - {m}")
            return False
        
        print(f"\n{'='*60}")
        print("✅ 所有数据文件完整")
        print(f"{'='*60}\n")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='下载NASA PCoE Battery Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载并解压所有数据（推荐）
  python download_nasa_data_final.py --download
  
  # 下载但保留临时文件
  python download_nasa_data_final.py --download --no-cleanup
  
  # 查看数据集信息
  python download_nasa_data_final.py --list
  
  # 验证数据完整性
  python download_nasa_data_final.py --verify
  
  # 指定保存目录
  python download_nasa_data_final.py --download --save-dir ./my_data
        """
    )
    
    parser.add_argument('--download', action='store_true', help='下载数据集')
    parser.add_argument('--no-cleanup', action='store_true', help='保留临时文件')
    parser.add_argument('--list', action='store_true', help='列出数据集信息')
    parser.add_argument('--verify', action='store_true', help='验证数据完整性')
    parser.add_argument('--save-dir', default='data/raw/nasa', help='数据保存目录')
    
    args = parser.parse_args()
    
    downloader = NASADataDownloader(save_dir=args.save_dir)
    
    if args.list:
        downloader.list_datasets()
    elif args.verify:
        downloader.verify_data()
    elif args.download:
        downloader.download_and_extract(cleanup=not args.no_cleanup)
    else:
        parser.print_help()
        print("\n提示: 使用 --download 下载数据，或使用 --list 查看数据集信息")


if __name__ == "__main__":
    main()
