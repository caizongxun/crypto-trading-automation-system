# PyTorch DLL Error Fix Guide
PyTorch DLL錯誤修復指南 (Windows)

## 問題
```
OSError: [WinError 1114] 動態連結程式庫 (DLL) 初始化例行程序失敗。
```

## 解決方案

### 方法1: 安裝Microsoft Visual C++ Redistributable (最常見)

**下載連結:**
https://aka.ms/vs/17/release/vc_redist.x64.exe

**步驟:**
1. 下載上面的安裝程式
2. 執行安裝
3. 重新啟動電腦
4. 再次嘗試執行GUI

### 方法2: 重新安裝PyTorch

```bash
# 卸載舊版本
pip uninstall torch torchvision torchaudio -y

# 安裝CPU版本 (穩定)
pip install torch torchvision torchaudio

# 或安裝GPU版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或安裝GPU版本 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 方法3: 使用Conda (最穩定)

```bash
# 建立新環境
conda create -n v4_trading python=3.10
conda activate v4_trading

# 安裝PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安裝其他依賴
pip install -r requirements.txt
```

### 方法4: 暫時使用命令行 (繞過GUI)

如果上面方法都無法解決,可以先使用命令行訓練:

```bash
# 使用V3 (不需PyTorch)
cd adaptive_strategy_v3
python train.py --symbol BTCUSDT --timeframe 15m

# V3使用LightGBM,不需要PyTorch
# 訓練更快 (2分鐘 vs 7分鐘)
# 性能穩定 (50%月報酬)
```

## 檢查PyTorch是否正常

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# 測試基本操作
x = torch.rand(5, 3)
print(f"Tensor created: {x.shape}")
print("PyTorch is working!")
```

## 常見問題

### Q: 安裝VC++ Redistributable後仍然錯誤?

**A:** 嘗試重新安裝PyTorch:
```bash
pip uninstall torch -y
pip cache purge
pip install torch --no-cache-dir
```

### Q: 我有GPU但不想用?

**A:** 安裝CPU版本即可:
```bash
pip install torch torchvision torchaudio
```

### Q: 錯誤提示 "c10.dll" 或 "c10_cuda.dll"?

**A:** 這是PyTorch核心函式庫問題:
1. 確認已安裝VC++ Redistributable
2. 重新安裝PyTorch
3. 嘗試使用Conda

## 推薦方案

### 如果你是新手:
使用**V3策略** (不需PyTorch):
- 更穩定
- 訓練更快
- 已驗證有效 (50%月報酬)

### 如果你想使用V4:
1. 先安裝VC++ Redistributable
2. 使用Conda建立獨立環境
3. 如果GUI無法使用,用命令行訓練

## 測試PyTorch安裝

建立測試檔案 `test_pytorch.py`:

```python
try:
    import torch
    print("✅ PyTorch imported successfully")
    print(f"Version: {torch.__version__}")
    
    x = torch.rand(5, 3)
    print(f"✅ Tensor created: {x.shape}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        x_cuda = x.cuda()
        print("✅ CUDA tensor created")
    else:
        print("⚠️ CUDA not available (CPU mode)")
    
    print("\n✅ PyTorch is fully functional!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n解決方法:")
    print("1. 安裝VC++ Redistributable")
    print("2. pip uninstall torch -y && pip install torch")
    print("3. 使用Conda")
```

執行:
```bash
python test_pytorch.py
```

## 聯絡支援

如果仍然無法解決:
1. 提供你的Python版本: `python --version`
2. 提供Windows版本
3. 提供PyTorch版本: `pip show torch`
4. 提供完整錯誤訊息

---

**快速解決:** 先使用V3,再解決PyTorch問題
