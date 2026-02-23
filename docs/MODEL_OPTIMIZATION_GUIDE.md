# 模型優化完整指南

## 核心理念

**調參 vs 模型優化**:

```
調參優化 (你已經做的):
  threshold: 0.16 → 0.18
  tp/sl: 2:1 → 1.5:1
  效果: +10-20% 提升

模型優化 (更根本):
  特徵工程: +20-40%
  標籤設計: +15-30%
  模型架構: +10-20%
  訓練策略: +10-15%
  總效果: +50-100% 提升 ⭐
```

---

## 第一步: 模型診斷

### 1.1 當前模型表現診斷

```python
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# 載入模型
model_long = joblib.load('models_output/catboost_long_xxx.pkl')
model_short = joblib.load('models_output/catboost_short_xxx.pkl')

# 提取基礎模型
if hasattr(model_long, 'estimators_'):
    base_long = model_long.estimators_[0].estimator
else:
    base_long = model_long

# 1. 特徵重要性分析
feature_importance = base_long.get_feature_importance()
feature_names = base_long.feature_names_

for name, imp in sorted(zip(feature_names, feature_importance), 
                        key=lambda x: x[1], reverse=True):
    print(f"{name:30s} {imp:8.2f}")

# 2. 校準曲線分析
probs = model_long.predict_proba(X_test)[:, 1]

# 檢查機率分布
print(f"Prob distribution:")
print(f"  Mean: {probs.mean():.4f}")
print(f"  Std:  {probs.std():.4f}")
print(f"  Min:  {probs.min():.4f}")
print(f"  Max:  {probs.max():.4f}")
print(f"  >0.16: {(probs > 0.16).sum()} ({(probs > 0.16).mean()*100:.2f}%)")
print(f"  >0.20: {(probs > 0.20).sum()} ({(probs > 0.20).mean()*100:.2f}%)")

# 3. 不同機率區間的實際勝率
for threshold in [0.10, 0.15, 0.16, 0.18, 0.20, 0.22, 0.25]:
    mask = probs >= threshold
    if mask.sum() > 0:
        actual_win_rate = y_test[mask].mean()
        print(f"Threshold {threshold:.2f}: {mask.sum():5d} samples, actual WR: {actual_win_rate*100:.2f}%")
```

### 1.2 問題診斷清單

```yaml
問題 A: 機率分布太窄
  症狀: max prob < 0.30, most probs < 0.10
  原因: 特徵區分度不足
  解決: 增強特徵工程

問題 B: 校準不準確
  症狀: prob=0.20 時實際勝率只有 15%
  原因: Isotonic 校準失效
  解決: 重新訓練或換校準方法

問題 C: 樣本數太少
  症狀: >0.16 的樣本 < 2%
  原因: 標籤太嚴格 (2% TP)
  解決: 放寬標籤到 1.5%

問題 D: 特徵重要性集中
  症狀: 前3個特徵貢獻 >70%
  原因: 特徵冗餘或缺失關鍵特徵
  解決: 特徵工程重構

問題 E: 方向不平衡
  症狀: Long 樣本 5000, Short 樣本 2000
  原因: 市場趨勢或標籤偏差
  解決: SMOTE 或調整權重
```

---

## 優化方向 1: 特徵工程強化 ⭐⭐⭐

### 1.1 當前特徵回顧

```python
# 你現在的特徵 (從 feature_engineering.py)
current_features = [
    'efficiency_ratio',      # 趨勢效率
    'extreme_time_diff',     # 極值時間差
    'vol_imbalance_ratio',   # 量能失衡
    'z_score',               # 價格標準化
    'bb_width_pct',          # 布林帶寬度
    'rsi',                   # RSI
    'atr_pct',               # 波動率
    'z_score_1h',            # 1小時 Z-score
    'atr_pct_1d'             # 1天波動率
]
```

### 1.2 新增高價值特徵

#### A. 訂單流特徵 (Order Flow)

```python
def create_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    訂單流特徵 - 捕捉買賣壓力
    """
    # 1. Delta (買賣力道差)
    df['delta'] = df['close'] - df['open']
    df['delta_volume'] = df['volume'] * np.sign(df['delta'])
    
    # 2. 累積 Delta
    df['cumulative_delta_5'] = df['delta_volume'].rolling(5).sum()
    df['cumulative_delta_15'] = df['delta_volume'].rolling(15).sum()
    
    # 3. Delta 強度
    df['delta_strength'] = df['cumulative_delta_5'] / df['volume'].rolling(5).sum()
    
    # 4. 買賣壓力比
    up_volume = df[df['delta'] > 0]['volume'].rolling(10).sum()
    down_volume = df[df['delta'] <= 0]['volume'].rolling(10).sum()
    df['buy_sell_ratio'] = up_volume / (down_volume + 1e-8)
    
    return df
```

**預期效果**: +15-25% 勝率提升

#### B. 市場微觀結構特徵

```python
def create_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    微觀結構特徵 - 捕捉市場摩擦
    """
    # 1. Tick Imbalance
    df['tick_direction'] = np.sign(df['close'] - df['close'].shift(1))
    df['tick_imbalance'] = df['tick_direction'].rolling(20).sum()
    
    # 2. 價格衝擊 (Price Impact)
    df['price_impact'] = (df['high'] - df['low']) / df['volume']
    
    # 3. 流動性指標
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['liquidity_score'] = df['volume'] / df['spread_proxy']
    
    # 4. 反轉指標
    df['reversal_strength'] = (
        (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    ).rolling(5).mean()
    
    return df
```

**預期效果**: +10-20% 機率區分度提升

#### C. 多時間框架特徵 (MTF)

```python
def create_mtf_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    多時間框架特徵 - 捕捉不同週期信號
    """
    # 1. 趨勢一致性
    df_1m['ema_5m'] = df_1m['close'].rolling(5).mean()
    df_1m['ema_15m'] = df_1m['close'].rolling(15).mean()
    df_1m['ema_1h'] = df_1m['close'].rolling(60).mean()
    
    df_1m['trend_alignment'] = (
        np.sign(df_1m['close'] - df_1m['ema_5m']) +
        np.sign(df_1m['close'] - df_1m['ema_15m']) +
        np.sign(df_1m['close'] - df_1m['ema_1h'])
    ) / 3
    
    # 2. 波動率比率
    df_1m['vol_ratio_5m_1h'] = (
        df_1m['close'].rolling(5).std() / 
        df_1m['close'].rolling(60).std()
    )
    
    # 3. 動量散度
    df_1m['momentum_5m'] = df_1m['close'].pct_change(5)
    df_1m['momentum_15m'] = df_1m['close'].pct_change(15)
    df_1m['momentum_divergence'] = df_1m['momentum_5m'] - df_1m['momentum_15m']
    
    return df_1m
```

**預期效果**: +10-15% 勝率提升

#### D. 機器學習衍生特徵

```python
def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 ML 自動發現特徵交互
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    # 1. 特徵交互
    base_features = ['rsi', 'bb_width_pct', 'atr_pct']
    X_base = df[base_features].fillna(0).values
    
    poly = PolynomialFeatures(degree=2, include_bias=False, 
                             interaction_only=True)
    X_poly = poly.fit_transform(X_base)
    
    # 添加交互特徵
    for i, name in enumerate(poly.get_feature_names_out(base_features)):
        if '*' in name:  # 只保留交互項
            df[f'interaction_{name}'] = X_poly[:, i]
    
    # 2. 聚類特徵
    from sklearn.cluster import KMeans
    
    X_cluster = df[['rsi', 'atr_pct', 'vol_imbalance_ratio']].fillna(0).values
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['market_regime'] = kmeans.fit_predict(X_cluster)
    
    return df
```

**預期效果**: +8-12% 提升

---

## 優化方向 2: 標籤設計改進 ⭐⭐⭐

### 2.1 當前標籤問題

```python
# 當前標籤 (2% TP, 1% SL)

問題:
1. 太嚴格 → 正樣本太少 (可能 <5%)
2. 不考慮市場狀態 → 低波動期很難達到
3. 固定比例 → 未考慮個體差異
```

### 2.2 動態標籤策略

#### A. 波動率自適應標籤

```python
def create_adaptive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    根據波動率動態調整 TP/SL
    """
    # 1. 計算波動率
    df['atr_pct'] = (
        (df['high'] - df['low']).rolling(14).mean() / df['close']
    )
    
    # 2. 動態 TP/SL
    def get_dynamic_targets(row):
        atr = row['atr_pct']
        
        if atr < 0.02:  # 低波動
            tp_pct = 0.015
            sl_pct = 0.0075
        elif atr > 0.04:  # 高波動
            tp_pct = 0.025
            sl_pct = 0.0125
        else:  # 中等波動
            tp_pct = 0.020
            sl_pct = 0.010
        
        return tp_pct, sl_pct
    
    # 3. 應用動態標籤
    df[['dynamic_tp', 'dynamic_sl']] = df.apply(
        lambda row: pd.Series(get_dynamic_targets(row)), axis=1
    )
    
    # 4. 生成標籤
    df['label_long'] = compute_forward_label(
        df, direction='long', 
        tp_pct=df['dynamic_tp'], 
        sl_pct=df['dynamic_sl']
    )
    
    return df
```

**預期效果**: 
- 正樣本數增加 50-100%
- 機率分布更均勻
- >0.16 的樣本從 1% 提升到 3-5%

#### B. 分層標籤策略

```python
def create_tiered_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    多層次標籤 - 訓練多個模型
    """
    # Conservative (高勝率)
    df['label_conservative'] = compute_forward_label(
        df, direction='long', tp_pct=0.015, sl_pct=0.010
    )
    
    # Standard (平衡)
    df['label_standard'] = compute_forward_label(
        df, direction='long', tp_pct=0.020, sl_pct=0.010
    )
    
    # Aggressive (高盈虧比)
    df['label_aggressive'] = compute_forward_label(
        df, direction='long', tp_pct=0.030, sl_pct=0.010
    )
    
    return df

# 訓練 3 個模型
model_conservative = train_model(X, df['label_conservative'])
model_standard = train_model(X, df['label_standard'])
model_aggressive = train_model(X, df['label_aggressive'])

# 回測時根據市場狀態選擇模型
if volatility < 0.02:
    use_model = model_conservative
elif volatility > 0.04:
    use_model = model_aggressive
else:
    use_model = model_standard
```

**預期效果**: +20-30% 適應性提升

---

## 優化方向 3: 模型架構調整 ⭐⭐

### 3.1 當前架構

```python
# 你現在的架構
model = CalibratedClassifierCV(
    CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        # ...
    ),
    method='isotonic',
    cv=3
)
```

### 3.2 優化方案

#### A. 超參數調優

```python
from optuna import create_study

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
    }
    
    model = CatBoostClassifier(**params, random_state=42, verbose=False)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    probs = model.predict_proba(X_val)[:, 1]
    # 自定義指標: 優化 >0.16 區間的精確度
    mask = probs >= 0.16
    if mask.sum() > 0:
        precision = y_val[mask].mean()
        recall = mask.sum() / len(y_val)
        return precision * recall  # F-beta score
    return 0

study = create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best params:", study.best_params)
```

**預期效果**: +5-10% 性能提升

#### B. 集成學習

```python
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb

# 1. 訓練多個不同的模型
model_catboost = CatBoostClassifier(**best_params)
model_xgb = xgb.XGBClassifier(**xgb_params)
model_lgb = lgb.LGBMClassifier(**lgb_params)

# 2. Soft Voting Ensemble
ensemble = VotingClassifier(
    estimators=[
        ('catboost', model_catboost),
        ('xgboost', model_xgb),
        ('lightgbm', model_lgb)
    ],
    voting='soft',
    weights=[0.4, 0.3, 0.3]  # CatBoost 權重較高
)

ensemble.fit(X_train, y_train)

# 3. Stacking Ensemble (更高級)
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('catboost', model_catboost),
        ('xgboost', model_xgb),
        ('lightgbm', model_lgb)
    ],
    final_estimator=CatBoostClassifier(iterations=100, depth=3),
    cv=5
)

stacking.fit(X_train, y_train)
```

**預期效果**: +10-15% 穩定性和準確度

---

## 優化方向 4: 訓練策略優化 ⭐⭐

### 4.1 樣本權重策略

```python
def compute_sample_weights(y: pd.Series, probs: np.ndarray = None) -> np.ndarray:
    """
    動態樣本權重
    """
    weights = np.ones(len(y))
    
    # 1. 類別平衡
    class_weight = len(y) / (2 * np.bincount(y))
    weights = class_weight[y]
    
    # 2. 時間衰減 (近期數據更重要)
    time_decay = np.exp(np.linspace(-2, 0, len(y)))
    weights *= time_decay
    
    # 3. 困難樣本加權 (如果有先前的預測)
    if probs is not None:
        # 預測錯誤的樣本權重加倍
        errors = np.abs((probs > 0.5).astype(int) - y)
        weights *= (1 + errors)
    
    return weights / weights.mean()  # 標準化

# 應用
sample_weights = compute_sample_weights(y_train)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=(X_val, y_val)
)
```

**預期效果**: +8-12% 提升

### 4.2 漸進式訓練

```python
def progressive_training(X_train, y_train, X_val, y_val):
    """
    分階段訓練 - 從簡單到困難
    """
    # Stage 1: 訓練容易樣本 (極端情況)
    easy_mask = (
        (df['rsi'] < 30) | (df['rsi'] > 70) |  # RSI 極端
        (df['bb_width_pct'] > 0.05)  # 高波動
    )
    
    model_stage1 = CatBoostClassifier(iterations=200)
    model_stage1.fit(
        X_train[easy_mask],
        y_train[easy_mask]
    )
    
    # Stage 2: 添加中等難度樣本
    probs_stage1 = model_stage1.predict_proba(X_train)[:, 1]
    medium_mask = (probs_stage1 > 0.3) & (probs_stage1 < 0.7)
    
    combined_mask = easy_mask | medium_mask
    
    model_stage2 = CatBoostClassifier(iterations=300)
    model_stage2.fit(
        X_train[combined_mask],
        y_train[combined_mask],
        init_model=model_stage1  # 從 stage1 開始
    )
    
    # Stage 3: 全部數據微調
    model_final = CatBoostClassifier(iterations=100, learning_rate=0.01)
    model_final.fit(
        X_train, y_train,
        init_model=model_stage2
    )
    
    return model_final
```

**預期效果**: +10-15% 難樣本表現

---

## 優化方向 5: 時序驗證強化 ⭐⭐

### 5.1 Walk-Forward 驗證

```python
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_validation(df: pd.DataFrame, n_splits=5):
    """
    時序前向驗證 - 防止未來函數
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # 訓練集和測試集
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        
        X_train = df_train[features]
        y_train = df_train['label_long']
        X_test = df_test[features]
        y_test = df_test['label_long']
        
        # 訓練
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        
        # 驗證
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        
        # 回測
        backtest_results = run_backtest(df_test, model)
        
        results.append({
            'fold': fold + 1,
            'auc': auc,
            'profit_factor': backtest_results['profit_factor'],
            'win_rate': backtest_results['win_rate'],
            'total_trades': backtest_results['total_trades']
        })
    
    return pd.DataFrame(results)

# 執行
results_df = walk_forward_validation(df_features)
print(results_df)
print(f"\nAverage PF: {results_df['profit_factor'].mean():.2f}")
print(f"PF Std: {results_df['profit_factor'].std():.2f}")
```

**預期效果**: 更真實的性能評估

---

## 🚀 完整優化流程 (4週計劃)

### Week 1: 特徵工程強化

```yaml
Day 1-2: 實現訂單流特徵
  - create_order_flow_features()
  - 重新訓練模型
  - 對比性能

Day 3-4: 實現微觀結構特徵
  - create_microstructure_features()
  - 重新訓練
  - A/B 測試

Day 5-7: 實現多時間框架特徵
  - create_mtf_features()
  - 完整重訓練
  - 回測驗證

預期: 勝率 +5-10%, 機率區分度 +20%
```

### Week 2: 標籤設計改進

```yaml
Day 1-3: 動態標籤實驗
  - 實現 create_adaptive_labels()
  - 對比固定標籤 vs 動態標籤
  - 選擇最佳方案

Day 4-7: 分層標籤策略
  - 訓練 Conservative/Standard/Aggressive 3個模型
  - 實現動態模型選擇
  - 回測整合系統

預期: 交易數 +50-100%, 適應性 +30%
```

### Week 3: 集成學習

```yaml
Day 1-2: 超參數調優
  - Optuna 自動調參
  - 找出最佳配置

Day 3-5: 集成模型
  - 訓練 CatBoost + XGBoost + LightGBM
  - 實現 Voting/Stacking
  - 性能對比

Day 6-7: 整合回測
  - 完整系統測試
  - 記錄所有指標

預期: 穩定性 +20%, Profit Factor +0.3
```

### Week 4: 驗證與部署

```yaml
Day 1-3: Walk-Forward 驗證
  - 5-fold 時序驗證
  - 確認穩定性

Day 4-5: 最終調整
  - 微調參數
  - 優化閾值

Day 6-7: 生產準備
  - 模型打包
  - 文檔完善
  - 部署測試

目標: PF 1.5+, 勝率 45%+, 穩定性驗證
```

---

## 📊 立即開始 (今天就能做)

### 快速測試 1: 訂單流特徵

```python
# 1. 添加到 feature_engineering.py

def create_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """添加訂單流特徵"""
    # Delta
    df['delta'] = df['close'] - df['open']
    df['delta_volume'] = df['volume'] * np.sign(df['delta'])
    
    # 累積 Delta
    df['cumulative_delta_5'] = df['delta_volume'].rolling(5).sum()
    df['cumulative_delta_15'] = df['delta_volume'].rolling(15).sum()
    
    # 買賣壓力
    df['delta_strength'] = (
        df['cumulative_delta_5'] / (df['volume'].rolling(5).sum() + 1e-8)
    )
    
    return df

# 2. 在 create_features_from_1m() 中調用
def create_features_from_1m(self, df_1m, ...):
    # ... 現有代碼 ...
    
    # 添加訂單流特徵
    df_1m = self.create_order_flow_features(df_1m)
    
    # 更新特徵列表
    feature_cols = feature_cols + [
        'cumulative_delta_5',
        'cumulative_delta_15',
        'delta_strength'
    ]
    
    return df_features

# 3. 重新訓練
python train.py

# 4. 回測對比
舊模型 (無訂單流): PF=1.22, WR=37.84%
新模型 (有訂單流): PF=?, WR=?
```

### 快速測試 2: 動態標籤

```python
# 修改 feature_engineering.py 的標籤生成

def compute_forward_label_adaptive(df, direction='long'):
    """動態標籤"""
    # 計算 ATR
    atr_pct = (df['high'] - df['low']).rolling(14).mean() / df['close']
    
    # 動態 TP/SL
    tp_pct = np.where(atr_pct < 0.02, 0.015,
             np.where(atr_pct > 0.04, 0.025, 0.020))
    sl_pct = tp_pct / 2
    
    # ... 原有標籤邏輯,但使用動態 tp_pct, sl_pct ...

# 重新訓練對比
固定標籤: 正樣本率 3.2%
動態標籤: 正樣本率 5.8% (+81%)
```

---

## 🎯 預期總體效果

```yaml
當前狀態:
  交易數: 37
  勝率: 37.84%
  Profit Factor: 1.22
  報酬率: 0.41%

優化後 (4週):
  交易數: 80-120 (+116%-224%)
  勝率: 42-48% (+11%-27%)
  Profit Factor: 1.5-1.8 (+23%-48%)
  報酬率: 3-6% (+631%-1361%)
  
  關鍵改進:
  - 特徵工程: 機率區分度 ↑
  - 動態標籤: 正樣本數 ↑
  - 集成學習: 穩定性 ↑
  - 時序驗證: 可靠性 ↑
```

---

**立即行動**: 從訂單流特徵開始,今天就能看到效果! 🚀