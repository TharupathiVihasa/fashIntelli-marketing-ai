from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .artifacts import ArtifactPaths, save_dataframe, save_metrics, save_model
from .config import ModelConfig, ProjectPaths
from .evaluation import (
    ClassificationMetrics,
    infer_group_from_onehot,
    plot_calibration,
    plot_confusion,
    plot_pr,
    plot_roc,
    subgroup_report,
)
from .explainability import ExplainabilityEngine
from .features import FeatureEngineer
from .io import load_social_data, load_survey_data
from .preprocessing import DataCleaner
from .modeling import ModelTrainer
from .reporting import PDFReportBuilder, ReportSection
from .utils import ensure_dir, get_logger


@dataclass
class TaskRunOutputs:
    task: str
    model_name: str
    metrics: Dict[str, Any]


def _predict_proba(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def run_training(
    *,
    project: ProjectPaths,
    cfg: ModelConfig,
    survey_csv: Optional[Path] = None,
    social_csv: Optional[Path] = None,
    survey_excel: Optional[Path] = None,
    survey_sheet: Optional[str] = None,
    social_excel: Optional[Path] = None,
    social_sheet: Optional[str] = None,
    survey_target: str = "purchase_intention_label",
    social_target: str = "high_engagement_label",
    generate_pdf_report: bool = True,
) -> Dict[str, TaskRunOutputs]:
    logger = get_logger("fashintelli.pipeline", log_dir=project.logs_dir)

    ensure_dir(project.artifacts_dir)
    ensure_dir(project.figures_dir)
    ensure_dir(project.reports_dir)
    ensure_dir(project.outputs_dir)

    # Load
    survey_df = load_survey_data(excel_path=survey_excel, excel_sheet=survey_sheet, csv_path=survey_csv)
    social_df = load_social_data(excel_path=social_excel, excel_sheet=social_sheet, csv_path=social_csv)

    # Clean
    cleaner = DataCleaner()
    survey_df, survey_rep = cleaner.clean_survey(survey_df, target_col=survey_target)
    social_df, social_rep = cleaner.clean_social(social_df, target_col=social_target)

    logger.info(f"Survey cleaned: {survey_rep.n_rows_before} -> {survey_rep.n_rows_after}")
    logger.info(f"Social cleaned: {social_rep.n_rows_before} -> {social_rep.n_rows_after}")

    # Feature engineering
    fe = FeatureEngineer()
    social_df = fe.add_platform_brand_labels(social_df, platform_default="instagram")
    # ML Training
    trainer = ModelTrainer(cfg)
    xai = ExplainabilityEngine()

    outputs: Dict[str, TaskRunOutputs] = {}

    # ---------------------------
    # A) Engagement model
    # ---------------------------
    eng_run = trainer.train_classification(social_df, target_col=social_target)
    eng_model = eng_run.best_estimator

    y_true = eng_run.y_test
    y_pred = eng_model.predict(eng_run.X_test)
    y_proba = _predict_proba(eng_model, eng_run.X_test)

    eng_metrics = ClassificationMetrics.from_predictions(y_true, y_pred, y_proba).__dict__

    plot_confusion(y_true, y_pred, title="Engagement Model — Confusion Matrix", out_path=project.figures_dir / "engagement_confusion.png")
    if y_proba is not None:
        plot_roc(y_true, y_proba, title="Engagement Model — ROC Curve", out_path=project.figures_dir / "engagement_roc.png")
        plot_pr(y_true, y_proba, title="Engagement Model — Precision-Recall", out_path=project.figures_dir / "engagement_pr.png")
        plot_calibration(y_true, y_proba, title="Engagement Model — Calibration", out_path=project.figures_dir / "engagement_calibration.png")

    perm = xai.permutation_importance(eng_model, eng_run.X_test, y_true, scoring="f1")
    perm_df = pd.DataFrame({
        "feature": perm.feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    shap_df = xai.try_shap_global(eng_model, eng_run.X_test)

    ap_eng = ArtifactPaths.for_task(project.artifacts_dir, "engagement")
    save_model(eng_model, ap_eng.model_path)
    save_metrics({
        "task": "engagement",
        "best_model_name": eng_run.best_model_name,
        "metrics": eng_metrics,
        "feature_columns": list(eng_run.X_train.columns),
        "cleaning": {"social": social_rep.__dict__},
    }, ap_eng.metrics_path)
    save_dataframe(eng_run.leaderboard, ap_eng.leaderboard_path)
    save_dataframe(perm_df, ap_eng.permutation_importance_path)
    if shap_df is not None:
        save_dataframe(shap_df, ap_eng.shap_importance_path)

    outputs["engagement"] = TaskRunOutputs(task="engagement", model_name=eng_run.best_model_name, metrics=eng_metrics)

    # Integration: score social posts and build platform index
    social_scored = social_df.copy()
    X_social_all = social_df.drop(columns=[social_target])
    if hasattr(eng_model, "predict_proba"):
        social_scored["engagement_score"] = eng_model.predict_proba(X_social_all)[:, 1]
    else:
        social_scored["engagement_score"] = eng_model.predict(X_social_all)

    platform_index = fe.build_platform_engagement_index(social_scored, score_col="engagement_score", platform_col="platform")

    # ---------------------------
    # B) Purchase intention model (integrated)
    # ---------------------------
    survey_integrated = fe.attach_engagement_index_to_survey(survey_df, platform_index, out_col="platform_engagement_index")

    pur_run = trainer.train_classification(survey_integrated, target_col=survey_target)
    pur_model = pur_run.best_estimator

    y_true2 = pur_run.y_test
    y_pred2 = pur_model.predict(pur_run.X_test)
    y_proba2 = _predict_proba(pur_model, pur_run.X_test)

    pur_metrics = ClassificationMetrics.from_predictions(y_true2, y_pred2, y_proba2).__dict__

    plot_confusion(y_true2, y_pred2, title="Purchase Intention Model — Confusion Matrix", out_path=project.figures_dir / "purchase_intent_confusion.png")
    if y_proba2 is not None:
        plot_roc(y_true2, y_proba2, title="Purchase Intention Model — ROC Curve", out_path=project.figures_dir / "purchase_intent_roc.png")
        plot_pr(y_true2, y_proba2, title="Purchase Intention Model — Precision-Recall", out_path=project.figures_dir / "purchase_intent_pr.png")
        plot_calibration(y_true2, y_proba2, title="Purchase Intention Model — Calibration", out_path=project.figures_dir / "purchase_intent_calibration.png")

    if "age_range" in pur_run.X_test.columns:
        age_group = pur_run.X_test["age_range"].astype(str)
    else:
        age_group = infer_group_from_onehot(pur_run.X_test, prefix="age_range_", default="unknown")
    if "gender" in pur_run.X_test.columns:
        gender_group = pur_run.X_test["gender"].astype(str)
    else:
        gender_group = infer_group_from_onehot(pur_run.X_test, prefix="gender_", default="unknown")
    subgroup_age = subgroup_report(pur_run.X_test, y_true2, y_pred2, group_series=age_group, min_group_size=20)
    subgroup_gender = subgroup_report(pur_run.X_test, y_true2, y_pred2, group_series=gender_group, min_group_size=20)

    perm2 = xai.permutation_importance(pur_model, pur_run.X_test, y_true2, scoring="f1")
    perm_df2 = pd.DataFrame({
        "feature": perm2.feature_names,
        "importance_mean": perm2.importances_mean,
        "importance_std": perm2.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    shap_df2 = xai.try_shap_global(pur_model, pur_run.X_test)

    ap_pur = ArtifactPaths.for_task(project.artifacts_dir, "purchase_intent")
    save_model(pur_model, ap_pur.model_path)
    save_metrics({
        "task": "purchase_intent",
        "best_model_name": pur_run.best_model_name,
        "metrics": pur_metrics,
        "platform_index": platform_index,
        "feature_columns": list(pur_run.X_train.columns),
        "cleaning": {"survey": survey_rep.__dict__},
    }, ap_pur.metrics_path)
    save_dataframe(pur_run.leaderboard, ap_pur.leaderboard_path)
    save_dataframe(perm_df2, ap_pur.permutation_importance_path)
    if shap_df2 is not None:
        save_dataframe(shap_df2, ap_pur.shap_importance_path)

    # Export integrated snapshots
    survey_integrated.to_csv(project.outputs_dir / "survey_integrated_for_modeling.csv", index=False)
    social_scored.to_csv(project.outputs_dir / "social_scored_for_modeling.csv", index=False)
    subgroup_age.to_csv(project.outputs_dir / "subgroup_age_purchase_intent.csv", index=False)
    subgroup_gender.to_csv(project.outputs_dir / "subgroup_gender_purchase_intent.csv", index=False)

    outputs["purchase_intent"] = TaskRunOutputs(task="purchase_intent", model_name=pur_run.best_model_name, metrics=pur_metrics)

    if generate_pdf_report:
        builder = PDFReportBuilder("FashIntelli — ML + XAI Summary Report")
        sections = [
            ReportSection(
                heading="Engagement model (social posts)",
                paragraphs=[
                    f"Selected model: <b>{eng_run.best_model_name}</b>",
                    f"Accuracy: {eng_metrics['accuracy']:.3f} • Precision: {eng_metrics['precision']:.3f} • Recall: {eng_metrics['recall']:.3f} • F1: {eng_metrics['f1']:.3f}",
                ],
                table=(
                    ["Top features (Permutation Importance)", "Importance"],
                    [[r["feature"], f"{r['importance_mean']:.4f}"] for r in perm_df.head(12).to_dict(orient="records")],
                ),
                images=[project.figures_dir / "engagement_confusion.png"],
            ),
            ReportSection(
                heading="Purchase intention model (integrated)",
                paragraphs=[
                    f"Selected model: <b>{pur_run.best_model_name}</b>",
                    f"Accuracy: {pur_metrics['accuracy']:.3f} • Precision: {pur_metrics['precision']:.3f} • Recall: {pur_metrics['recall']:.3f} • F1: {pur_metrics['f1']:.3f}",
                    "Integrated feature: platform_engagement_index (derived from predicted engagement by platform).",
                ],
                table=(
                    ["Top features (Permutation Importance)", "Importance"],
                    [[r["feature"], f"{r['importance_mean']:.4f}"] for r in perm_df2.head(12).to_dict(orient="records")],
                ),
                images=[project.figures_dir / "purchase_intent_confusion.png"],
            ),
        ]
        builder.build(sections, project.reports_dir / "FashIntelli_ML_XAI_Summary.pdf")

    return outputs