import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

def calculate_hit_rate_at_k(y_pred, y_true, ranker_ids, k=3):
    # DataFrame으로 결합
    df = pd.DataFrame({
        'ranker_id': ranker_ids,
        'selected': y_true,
        'pred_score': y_pred
    })
    
    hits = 0  # 성공한 세션 수
    valid_queries_count = 0  # 유효한 쿼리 수
    
    for ranker_id, group in df.groupby('ranker_id'):
        # 대회 조건: 10개 이상 항목이 있는 세션만 평가
        if len(group) < 10:
            continue
            
        valid_queries_count += 1  # 유효한 쿼리 카운트 증가
        
        # 예측 점수 기준으로 랭킹 계산 (높은 점수가 1등)
        group = group.sort_values('pred_score', ascending=False).reset_index(drop=True)
        group['predicted_rank'] = range(1, len(group) + 1)
        
        # 실제로 선택된 항목 찾기
        true_selected_item = group[group['selected'] == 1]
        
        if not true_selected_item.empty:
            # 실제 선택된 항목의 예측 순위 가져오기
            rank_of_true_item = true_selected_item.iloc[0]['predicted_rank']
            # 상위 k개 안에 있으면 성공
            if rank_of_true_item <= k:
                hits += 1
            
    if valid_queries_count == 0:
        return 0.0
    return hits / valid_queries_count    # 성공률 반환    

def session_based_cv_ranker_simple_v2(df, n_splits=5):
    sessions = df['ranker_id'].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(sessions)):
        # 데이터 분할
        train_sessions = sessions[train_idx]
        val_sessions = sessions[val_idx]
        
        train_df = df[df['ranker_id'].isin(train_sessions)].sort_values('ranker_id')
        val_df = df[df['ranker_id'].isin(val_sessions)].sort_values('ranker_id')
        
        # 특성/타겟 분리
        X_train, y_train = train_df.drop(columns=['selected', 'ranker_id']), train_df['selected']
        X_val, y_val = val_df.drop(columns=['selected', 'ranker_id']), val_df['selected']
        
        # 자동으로 object 타입 컬럼들을 category로 변환
        cat_features = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                X_train[col] = X_train[col].astype(str).astype('category')
                X_val[col] = X_val[col].astype(str).astype('category')
                cat_features.append(col)
        
        # 그룹 크기
        train_groups = train_df.groupby('ranker_id').size().values
        val_groups = val_df.groupby('ranker_id').size().values
        
        # 모델 학습 - 자동 감지된 카테고리 피처 명시
        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            num_leaves=31,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1,
            n_estimators=100
        )
        
        ranker.fit(X_train, y_train, group=train_groups, categorical_feature=cat_features)
        y_pred = ranker.predict(X_val)
        
        # 평가
        hit_rate = calculate_hit_rate_at_k(y_pred, y_val, val_df['ranker_id'], k=3)
        auc = roc_auc_score(y_val, y_pred)
        scores.append(hit_rate)
        
        print(f"Fold {fold+1}: Hit Rate@3={hit_rate:.4f}, AUC={auc:.4f}")
    
    print(f"Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores), np.std(scores), scores