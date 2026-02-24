#!/usr/bin/env python3
"""
Model Metadata Verification Script

驗證模型是否正確儲存了特徵名稱
"""

import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).parent))
from utils.agent_backtester import load_model_with_metadata

def verify_model(model_path: str):
    """驗證單個模型"""
    print(f"\n{'='*80}")
    print(f"Verifying: {model_path}")
    print("="*80)
    
    try:
        model, feature_names, version = load_model_with_metadata(model_path)
        
        print(f"✅ Model loaded successfully")
        print(f"\n📊 Model Information:")
        print(f"  Version:        {version}")
        print(f"  Feature Count:  {len(feature_names)}")
        print(f"\n📋 Feature Names:")
        for i, feat in enumerate(feature_names, 1):
            print(f"  {i:2d}. {feat}")
        
        return True, version, feature_names
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False, None, None

def verify_model_pair(long_path: str, short_path: str):
    """驗證 Long/Short 模型對"""
    print("\n" + "#" * 80)
    print("# MODEL PAIR VERIFICATION")
    print("#" * 80)
    
    # 驗證 Long Model
    success_long, version_long, features_long = verify_model(long_path)
    
    # 驗證 Short Model
    success_short, version_short, features_short = verify_model(short_path)
    
    if not (success_long and success_short):
        print("\n❌ One or both models failed to load")
        return False
    
    # 檢查特徵一致性
    print("\n" + "="*80)
    print("FEATURE CONSISTENCY CHECK")
    print("="*80)
    
    if features_long == features_short:
        print("✅ Features match perfectly!")
        print(f"   Both models use {len(features_long)} features")
        return True
    else:
        print("⚠️ Features DO NOT match!")
        print(f"   Long:  {len(features_long)} features")
        print(f"   Short: {len(features_short)} features")
        
        # 找出差異
        only_in_long = set(features_long) - set(features_short)
        only_in_short = set(features_short) - set(features_long)
        
        if only_in_long:
            print(f"\n   Only in Long ({len(only_in_long)}):")
            for feat in only_in_long:
                print(f"     - {feat}")
        
        if only_in_short:
            print(f"\n   Only in Short ({len(only_in_short)}):")
            for feat in only_in_short:
                print(f"     - {feat}")
        
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify model metadata')
    parser.add_argument('--long', type=str, help='Path to Long model')
    parser.add_argument('--short', type=str, help='Path to Short model')
    parser.add_argument('--single', type=str, help='Path to single model')
    
    args = parser.parse_args()
    
    if args.single:
        # 驗證單個模型
        success, version, features = verify_model(args.single)
        sys.exit(0 if success else 1)
    
    elif args.long and args.short:
        # 驗證模型對
        success = verify_model_pair(args.long, args.short)
        sys.exit(0 if success else 1)
    
    else:
        print("⚠️ No model paths provided")
        print("\nUsage:")
        print("  # Verify single model")
        print("  python verify_model_metadata.py --single models_output/catboost_long_v2_xxx.pkl")
        print("\n  # Verify model pair")
        print("  python verify_model_metadata.py --long models_output/catboost_long_v2_xxx.pkl --short models_output/catboost_short_v2_xxx.pkl")
        sys.exit(1)