"""DataForge Agent 6: Modeling - Train Random Forest classifier/regressor."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

import config
from pipeline_context import PipelineContext
from logger import get_logger

logger = get_logger("Modeling")


def detect_task_type(df: pd.DataFrame, target_column: str) -> str:
    """
    Auto-detect classification vs regression.
    Rule: unique values < 20 AND dtype int → classification
    Otherwise → regression
    """
    unique_count = df[target_column].nunique()
    dtype = df[target_column].dtype
    
    if unique_count < 20 and dtype in ['int64', 'int32', 'int16', 'int8']:
        return "classification"
    else:
        return "regression"


def encode_categoricals(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Encode categorical features using pd.get_dummies.
    Drop high-cardinality categoricals (>50 unique values).
    FIX: Keep all one-hot encoded columns (no drop_first).
    
    Returns:
        tuple: (X, y) - encoded features and target
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Drop high-cardinality categoricals
    high_card_cols = [col for col in cat_cols if X[col].nunique() > config.MAX_CATS_UNIQUE]
    if high_card_cols:
        logger.warning(f"Dropping high-cardinality columns (>{config.MAX_CATS_UNIQUE} unique): {high_card_cols}")
        X = X.drop(columns=high_card_cols)
        cat_cols = [col for col in cat_cols if col not in high_card_cols]
    
    # One-hot encode remaining categoricals (keep ALL encoded columns)
    if cat_cols:
        logger.info(f"One-hot encoding categorical columns: {cat_cols}")
        # FIX: Remove drop_first=True to keep all encoded columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    
    return X, y


def run_modeling(ctx: PipelineContext) -> PipelineContext:
    """
    Train Random Forest model (classifier or regressor) on ctx.clean_df.
    Auto-detects task type, encodes categoricals, evaluates performance.
    Includes 5-fold cross-validation.
    
    Args:
        ctx: PipelineContext with clean_df, target_column, and has_target
        
    Returns:
        PipelineContext with model_results populated
    """
    logger.info("=" * 50)
    logger.info("Starting Agent 6: Modeling")
    logger.info("=" * 50)
    
    ctx.mark_agent("Modeling", "running")
    
    # === Check Prerequisites ===
    if ctx.clean_df is None:
        logger.warning("No clean_df available. Skipping modeling.")
        ctx.mark_agent("Modeling", "skipped")
        ctx.errors.append("Modeling: No clean_df available")
        return ctx
    
    if not ctx.has_target:
        logger.warning("No target variable defined. Skipping modeling.")
        ctx.mark_agent("Modeling", "skipped")
        ctx.errors.append("Modeling: No target variable defined")
        return ctx
    
    if ctx.target_column not in ctx.clean_df.columns:
        logger.warning(f"Target column '{ctx.target_column}' not found in clean_df. Skipping modeling.")
        ctx.mark_agent("Modeling", "skipped")
        ctx.errors.append(f"Modeling: Target column '{ctx.target_column}' not found")
        return ctx
    
    try:
        df = ctx.clean_df.copy()
        logger.info(f"Working with DataFrame shape: {df.shape}")
        
        # === Auto-Detect Task Type ===
        if ctx.task_type is None:
            ctx.task_type = detect_task_type(df, ctx.target_column)
        logger.info(f"Task type: {ctx.task_type}")
        
        # === Prepare Data ===
        logger.info("Preparing data: encoding categoricals...")
        X, y = encode_categoricals(df, ctx.target_column)
        
        # Handle any remaining missing values (drop rows with NaN)
        initial_rows = len(X)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        dropped_rows = initial_rows - len(X)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values")
        
        # Ensure all features are numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.warning(f"Dropping non-numeric columns that couldn't be encoded: {non_numeric}")
            X = X.drop(columns=non_numeric)
        
        feature_names = X.columns.tolist()
        logger.info(f"Feature count after encoding: {len(feature_names)}")
        
        if len(feature_names) == 0:
            logger.error("No features available after encoding. Skipping modeling.")
            ctx.mark_agent("Modeling", "failed")
            ctx.errors.append("Modeling: No features available after encoding")
            return ctx
        
        # === Train/Test Split ===
        logger.info("Performing 80/20 train/test split...")
        try:
            if ctx.task_type == "classification":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=0.2, 
                    random_state=config.RANDOM_SEED,
                    stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=0.2, 
                    random_state=config.RANDOM_SEED
                )
        except ValueError as e:
            # Stratify may fail if class has too few samples
            logger.warning(f"Stratified split failed: {e}. Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=config.RANDOM_SEED
            )
        
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # === Train Model ===
        logger.info(f"Training Random Forest {'Classifier' if ctx.task_type == 'classification' else 'Regressor'}...")
        logger.info(f"Hyperparameters: n_estimators={config.RF_N_ESTIMATORS}, max_depth={config.RF_MAX_DEPTH}, "
                   f"min_samples_leaf={config.RF_MIN_SAMPLES_LEAF}, n_jobs={config.RF_N_JOBS}, "
                   f"random_state={config.RANDOM_SEED}")
        
        if ctx.task_type == "classification":
            # IMPROVEMENT: Add class_weight='balanced' for imbalanced datasets
            model = RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
                n_jobs=config.RF_N_JOBS,
                random_state=config.RANDOM_SEED,
                class_weight='balanced'  # Handle class imbalance
            )
        else:
            model = RandomForestRegressor(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
                n_jobs=config.RF_N_JOBS,
                random_state=config.RANDOM_SEED
            )
        
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
        
        # === 5-Fold Cross-Validation ===
        logger.info("Performing 5-fold cross-validation...")
        try:
            scoring = 'accuracy' if ctx.task_type == 'classification' else 'r2'
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, 
                scoring=scoring
            )
            cv_mean = round(cv_scores.mean(), 4)
            cv_std = round(cv_scores.std(), 4)
            logger.info(f"Cross-Validation {scoring.upper()} (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            cv_mean = None
            cv_std = None
        
        # === Evaluate Model ===
        logger.info("Evaluating model performance...")
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = {}
        
        if ctx.task_type == "classification":
            # Classification metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Handle binary vs multiclass
            average = 'binary' if len(np.unique(y)) == 2 else 'macro'
            
            test_precision = precision_score(y_test, y_test_pred, average=average, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average=average, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average=average, zero_division=0)
            
            conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()
            class_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
            
            # Round all metrics to 4 decimal places
            metrics = {
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "precision": round(test_precision, 4),
                "recall": round(test_recall, 4),
                "f1_score": round(test_f1, 4),
                "classification_report": class_report
            }
            
            # Add CV metrics
            if cv_mean is not None:
                metrics["cv_mean"] = cv_mean
                metrics["cv_std"] = cv_std
            
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Test F1 Score: {test_f1:.4f}")
            logger.info(f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
            
        else:
            # Regression metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # MAPE (avoid division by zero)
            non_zero_mask = y_test != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((y_test[non_zero_mask] - y_test_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
            else:
                mape = None
            
            # Round all metrics to 4 decimal places
            metrics = {
                "train_rmse": round(train_rmse, 4),
                "test_rmse": round(test_rmse, 4),
                "test_mae": round(test_mae, 4),
                "test_r2": round(test_r2, 4),
            }
            if mape is not None:
                metrics["test_mape"] = round(mape, 4)
            
            # Add CV metrics
            if cv_mean is not None:
                metrics["cv_mean"] = cv_mean
                metrics["cv_std"] = cv_std
            
            conf_matrix = None
            
            logger.info(f"Test RMSE: {test_rmse:.4f}")
            logger.info(f"Test R² Score: {test_r2:.4f}")
            logger.info(f"Test MAE: {test_mae:.4f}")
        
        # === Feature Importance ===
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Round importance to 4 decimal places
        top_features = []
        for _, row in importance_df.head(10).iterrows():
            top_features.append({
                'feature': row['feature'],
                'importance': round(row['importance'], 4)
            })
        
        logger.info(f"Top 5 important features: {importance_df.head(5)['feature'].tolist()}")
        
        # === Store Results ===
        ctx.model_results = {
            "task_type": ctx.task_type,
            "model": model,  # Store model for SHAP
            "X_test": X_test,  # Store X_test for SHAP analysis
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "hyperparameters": {
                "n_estimators": config.RF_N_ESTIMATORS,
                "max_depth": config.RF_MAX_DEPTH,
                "min_samples_leaf": config.RF_MIN_SAMPLES_LEAF,
                "n_jobs": config.RF_N_JOBS,
                "random_state": config.RANDOM_SEED,
                "class_weight": "balanced" if ctx.task_type == "classification" else None
            },
            "metrics": metrics,
            "feature_importance": top_features,
        }
        
        if conf_matrix is not None:
            ctx.model_results["confusion_matrix"] = conf_matrix
        
        ctx.mark_agent("Modeling", "done")
        logger.info("Agent 6: Modeling completed successfully")
        
    except Exception as e:
        logger.error(f"Modeling failed with error: {str(e)}", exc_info=True)
        ctx.errors.append(f"Modeling: {str(e)}")
        ctx.mark_agent("Modeling", "failed")
    
    return ctx
