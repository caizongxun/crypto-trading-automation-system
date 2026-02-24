import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import joblib
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.agent_backtester import load_model_with_metadata

logger = setup_logger('model_management_tab', 'logs/model_management_tab.log')

class ModelManagementTab:
    def __init__(self):
        logger.info("Initializing ModelManagementTab")
        self.models_dir = Path("models_output")
    
    def render(self):
        st.header("📊 模型管理 - Metadata 驗證")
        st.markdown("""
        此標籤用於驗證和管理訓練好的模型,確保模型包含正確的特徵 metadata。
        """)
        
        if not self.models_dir.exists():
            st.warning("📜 模型目錄不存在,請先訓練模型")
            return
        
        st.markdown("---")
        
        # 模式選擇
        tab1, tab2, tab3 = st.tabs(["🔍 模型檢查", "⚖️ 模型比較", "🗑️ 模型管理"])
        
        with tab1:
            self.render_model_inspection()
        
        with tab2:
            self.render_model_comparison()
        
        with tab3:
            self.render_model_management()
    
    def get_all_models(self) -> dict:
        """獲取所有模型檔案"""
        models = {
            'long_v1': list(self.models_dir.glob("catboost_long_[0-9]*.pkl")),
            'short_v1': list(self.models_dir.glob("catboost_short_[0-9]*.pkl")),
            'long_v2': list(self.models_dir.glob("catboost_long_v2_*.pkl")),
            'short_v2': list(self.models_dir.glob("catboost_short_v2_*.pkl"))
        }
        
        # 排序
        for key in models:
            models[key] = sorted(models[key], key=lambda x: x.stat().st_mtime, reverse=True)
        
        return models
    
    def load_and_inspect_model(self, model_path: Path) -> dict:
        """載入並檢查模型"""
        try:
            model, feature_names, version = load_model_with_metadata(str(model_path))
            
            # 獲取檔案信息
            file_stat = model_path.stat()
            file_size_mb = file_stat.st_size / (1024 * 1024)
            modified_time = datetime.fromtimestamp(file_stat.st_mtime)
            
            return {
                'success': True,
                'model': model,
                'feature_names': feature_names,
                'version': version,
                'file_size': file_size_mb,
                'modified_time': modified_time,
                'path': model_path
            }
        
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': model_path
            }
    
    def render_model_inspection(self):
        """渲染模型檢查界面"""
        st.subheader("🔍 模型 Metadata 檢查")
        
        models = self.get_all_models()
        
        # 顯示模型統計
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Long V1 模型", len(models['long_v1']))
        with col2:
            st.metric("Short V1 模型", len(models['short_v1']))
        with col3:
            st.metric("Long V2 模型", len(models['long_v2']))
        with col4:
            st.metric("Short V2 模型", len(models['short_v2']))
        
        st.markdown("---")
        
        # 選擇模型
        all_models = []
        for key, model_list in models.items():
            all_models.extend([(m.name, m) for m in model_list])
        
        if not all_models:
            st.warning("沒有找到任何模型")
            return
        
        selected_model_name = st.selectbox(
            "選擇要檢查的模型",
            [name for name, _ in all_models],
            key="inspect_model_select"
        )
        
        # 找到選中的模型路徑
        selected_model_path = None
        for name, path in all_models:
            if name == selected_model_name:
                selected_model_path = path
                break
        
        if st.button("🔍 檢查模型", use_container_width=True, type="primary"):
            with st.spinner("載入模型..."):
                result = self.load_and_inspect_model(selected_model_path)
            
            if result['success']:
                st.success("✅ 模型載入成功")
                
                # 顯示模型信息
                st.markdown("### 📊 模型信息")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("版本", result['version'].upper())
                    st.metric("檔案大小", f"{result['file_size']:.2f} MB")
                
                with col2:
                    st.metric("特徵數量", len(result['feature_names']))
                    st.metric("修改時間", result['modified_time'].strftime('%Y-%m-%d %H:%M'))
                
                with col3:
                    model_type = "Long" if "long" in selected_model_name else "Short"
                    st.metric("模型類型", model_type)
                
                # 版本判斷
                if result['version'] == 'v2':
                    st.success("✅ 此模型包含完整 metadata,可以正常使用")
                elif result['version'] == 'v1':
                    st.warning("⚠️ 此模型缺少 metadata,使用 V1 fallback 特徵")
                    st.info("💡 建議重新訓練以獲取 V2 metadata")
                
                st.markdown("---")
                
                # 顯示特徵列表
                st.markdown("### 📋 特徵列表")
                
                # 分成兩欄顯示
                features = result['feature_names']
                mid = len(features) // 2
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**前半部分**")
                    for i, feat in enumerate(features[:mid], 1):
                        st.text(f"{i:2d}. {feat}")
                
                with col2:
                    st.markdown("**後半部分**")
                    for i, feat in enumerate(features[mid:], mid+1):
                        st.text(f"{i:2d}. {feat}")
                
                # 匯出 JSON
                st.markdown("---")
                if st.button("💾 匯出 Metadata 為 JSON"):
                    metadata = {
                        'model_name': selected_model_name,
                        'version': result['version'],
                        'feature_count': len(result['feature_names']),
                        'features': result['feature_names'],
                        'file_size_mb': result['file_size'],
                        'modified_time': result['modified_time'].isoformat()
                    }
                    
                    json_str = json.dumps(metadata, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        "⬇️ 下載 JSON",
                        json_str,
                        f"{selected_model_name}_metadata.json",
                        "application/json"
                    )
            
            else:
                st.error(f"❌ 載入失敗: {result['error']}")
    
    def render_model_comparison(self):
        """渲染模型比較界面"""
        st.subheader("⚖️ Long/Short 模型對比較")
        st.markdown("檢查 Long 和 Short 模型是否使用相同的特徵")
        
        models = self.get_all_models()
        
        col1, col2 = st.columns(2)
        
        # 選擇 Long 模型
        with col1:
            st.markdown("#### Long 模型")
            long_models = models['long_v1'] + models['long_v2']
            if not long_models:
                st.warning("沒有 Long 模型")
                return
            
            selected_long = st.selectbox(
                "選擇 Long 模型",
                [m.name for m in long_models],
                key="compare_long"
            )
            long_path = next(m for m in long_models if m.name == selected_long)
        
        # 選擇 Short 模型
        with col2:
            st.markdown("#### Short 模型")
            short_models = models['short_v1'] + models['short_v2']
            if not short_models:
                st.warning("沒有 Short 模型")
                return
            
            selected_short = st.selectbox(
                "選擇 Short 模型",
                [m.name for m in short_models],
                key="compare_short"
            )
            short_path = next(m for m in short_models if m.name == selected_short)
        
        st.markdown("---")
        
        if st.button("⚖️ 比較模型", use_container_width=True, type="primary"):
            with st.spinner("載入模型..."):
                long_result = self.load_and_inspect_model(long_path)
                short_result = self.load_and_inspect_model(short_path)
            
            if long_result['success'] and short_result['success']:
                st.success("✅ 兩個模型載入成功")
                
                # 基本信息比較
                st.markdown("### 📊 基本信息")
                
                comparison_df = pd.DataFrame({
                    'Long 模型': [
                        long_result['version'].upper(),
                        len(long_result['feature_names']),
                        f"{long_result['file_size']:.2f} MB",
                        long_result['modified_time'].strftime('%Y-%m-%d %H:%M')
                    ],
                    'Short 模型': [
                        short_result['version'].upper(),
                        len(short_result['feature_names']),
                        f"{short_result['file_size']:.2f} MB",
                        short_result['modified_time'].strftime('%Y-%m-%d %H:%M')
                    ]
                }, index=['版本', '特徵數', '檔案大小', '修改時間'])
                
                st.dataframe(comparison_df, use_container_width=True)
                
                st.markdown("---")
                
                # 特徵一致性檢查
                st.markdown("### 🔍 特徵一致性檢查")
                
                long_features = set(long_result['feature_names'])
                short_features = set(short_result['feature_names'])
                
                if long_features == short_features:
                    st.success("✅ 特徵完全一致!")
                    st.info(f"兩個模型都使用 {len(long_features)} 個相同特徵")
                else:
                    st.error("❌ 特徵不一致!")
                    
                    only_in_long = long_features - short_features
                    only_in_short = short_features - long_features
                    common = long_features & short_features
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("共同特徵", len(common))
                    with col2:
                        st.metric("僅 Long 有", len(only_in_long))
                    with col3:
                        st.metric("僅 Short 有", len(only_in_short))
                    
                    if only_in_long:
                        st.markdown("**僅 Long 模型有的特徵:**")
                        for feat in sorted(only_in_long):
                            st.text(f"  - {feat}")
                    
                    if only_in_short:
                        st.markdown("**僅 Short 模型有的特徵:**")
                        for feat in sorted(only_in_short):
                            st.text(f"  - {feat}")
                    
                    st.warning("💡 建議重新訓練並使用同一批次的模型")
                
                # 版本一致性檢查
                st.markdown("---")
                st.markdown("### 🏷️ 版本一致性檢查")
                
                if long_result['version'] == short_result['version']:
                    st.success(f"✅ 兩個模型都是 {long_result['version'].upper()} 版本")
                else:
                    st.warning(
                        f"⚠️ 版本不一致: "
                        f"Long={long_result['version'].upper()}, "
                        f"Short={short_result['version'].upper()}"
                    )
                    st.info("💡 建議使用相同版本的模型進行回測")
            
            else:
                if not long_result['success']:
                    st.error(f"Long 模型載入失敗: {long_result['error']}")
                if not short_result['success']:
                    st.error(f"Short 模型載入失敗: {short_result['error']}")
    
    def render_model_management(self):
        """渲染模型管理界面"""
        st.subheader("🗑️ 模型管理")
        st.warning("⚠️ 請謹慎使用刪除功能,刪除後無法恢復")
        
        models = self.get_all_models()
        
        # 顯示所有模型
        all_models = []
        for key, model_list in models.items():
            for m in model_list:
                stat = m.stat()
                all_models.append({
                    '檔案名': m.name,
                    '大小 (MB)': f"{stat.st_size / (1024*1024):.2f}",
                    '修改時間': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                    '路徑': str(m)
                })
        
        if not all_models:
            st.info("沒有模型可管理")
            return
        
        models_df = pd.DataFrame(all_models)
        st.dataframe(models_df.drop(columns=['路徑']), use_container_width=True)
        
        st.markdown("---")
        
        # 刪除興路模型
        st.markdown("#### 🗑️ 刪除模型")
        
        selected_to_delete = st.multiselect(
            "選擇要刪除的模型",
            [m['檔案名'] for m in all_models],
            key="delete_models"
        )
        
        if selected_to_delete:
            st.warning(f"🚨 即將刪除 {len(selected_to_delete)} 個模型")
            
            confirm = st.checkbox("我確定要刪除這些模型", key="confirm_delete")
            
            if confirm and st.button("🗑️ 確定刪除", type="primary"):
                deleted_count = 0
                for filename in selected_to_delete:
                    try:
                        file_path = self.models_dir / filename
                        file_path.unlink()
                        deleted_count += 1
                        st.success(f"✅ 已刪除: {filename}")
                    except Exception as e:
                        st.error(f"❌ 刪除失敗 {filename}: {e}")
                
                if deleted_count > 0:
                    st.success(f"✅ 共刪除 {deleted_count} 個模型")
                    st.rerun()