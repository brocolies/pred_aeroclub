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

def get_cv_score(df, n_splits=5):
    sessions = df['ranker_id'].unique()
    # session 단위로 묶어 평가해야 하기 때문에, ranker_id의 unique값을 session 개수로 파악
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # KFold 설정
    scores = []
    # n_splits 개수의 fold별 점수 저장 리스트 선언
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(sessions)):
        # sessions를 n_splits 개수로 split하고 이를 enumerate로 indexing
        # index는 fold에 배정되고, kf에 의해 나뉜 train set, validation set은 각각 배분 
        train_sessions = sessions[train_idx]
        val_sessions = sessions[val_idx]
        # index 활용하여 list 호출
        
        train_df = df[df['ranker_id'].isin(train_sessions)].sort_values('ranker_id')
        val_df = df[df['ranker_id'].isin(val_sessions)].sort_values('ranker_id')
        # train/val 각자 해당하는 ranker_id에 대한 session을 할당, ranker_id 기준 정렬
        
        X_train, y_train = train_df.drop(columns=['selected', 'ranker_id']), train_df['selected']
        X_val, y_val = val_df.drop(columns=['selected', 'ranker_id']), val_df['selected']
        # X, y로 데이터 분리하여 학습을 위한 데이터셋 마련
        
        cat_features = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                X_train[col] = X_train[col].astype(str).astype('category')
                X_val[col] = X_val[col].astype(str).astype('category')
                cat_features.append(col)
        # lightgbm이 인식하지 못하는 object columns -> category type으로 변경 
        
        train_groups = train_df.groupby('ranker_id').size().values
        val_groups = val_df.groupby('ranker_id').size().values # df/series -> numpy 
        # lgbmRanker session 별로 상대적 지위를 예측하기 때문에, session 별 크기 계산 
        # 이 정보가 있어야 sorted된 정보들 중 어디까지가 같은 session인지 판단 가능

        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            # 모델이 무엇을 학습할지 정하는 목적함수 
            # lambdarank -> 순위함수 
            num_leaves=31,
            # 트리 잎의 개수(31~255)
            learning_rate=0.1,
            # 학습률(0.01~0.3, 낮을수록 안정적)
            random_state=42,
            verbosity=-1,
            # 진행 로그 끄기: -1 
            n_estimators=100
            # 트리 개수(100~10000)
        )
        # 여기부터 잘 모르겠음 일단 lgbmranker 써야 하는 문제는 맞나? 
        # 확실하지 않아서 import도 삭제했음 이 코드도 수정 예정 
        # lgbmranker가 뭔지도 잘 모르겠고, 저런 파라미터에 대한 지식도 전무
        # 질문한 뒤에 이후 주석 작성해야겠다 
        
        ranker.fit(X_train, y_train, group=train_groups, categorical_feature=cat_features)
        y_pred = ranker.predict(X_val)
        
        # 평가
        hit_rate = calculate_hit_rate_at_k(y_pred, y_val, val_df['ranker_id'], k=3)
        auc = roc_auc_score(y_val, y_pred)
        scores.append(hit_rate)
        
        print(f"Fold {fold+1}: Hit Rate@3={hit_rate:.4f}, AUC={auc:.4f}")
    
    print(f"Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores), np.std(scores), scores