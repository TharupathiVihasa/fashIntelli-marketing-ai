from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from fashintelli.artifacts import ArtifactPaths, load_metrics, load_model
from fashintelli.config import ModelConfig, ProjectPaths
from fashintelli.pipeline import run_training


def pretty_brand_label(value: Any) -> str:
    return str(value or '').replace('_', ' ').title()

def clean_json_value(obj):
    if isinstance(obj, dict):
        return {k: clean_json_value(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_json_value(v) for v in obj]

    if isinstance(obj, tuple):
        return [clean_json_value(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    return obj

# Checking Datasets
class AnalyticsService:
    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parents[2]
        self.project = ProjectPaths(root=self.root)
        self.cfg = ModelConfig()
        self.sample_survey_path = self.project.sample_data_dir / 'FashIntelli_Survey_Model_Dataset_v2.csv'
        self.sample_social_path = self.project.sample_data_dir / 'FashIntelli_Social_Engagement_Model_Dataset_v2.csv'
        self.upload_dir = self.root / 'data' / 'uploaded'
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.upload_dir / 'active_dataset_manifest.json'
        self.active_survey_csv = self.upload_dir / 'current_survey.csv'
        self.active_social_csv = self.upload_dir / 'current_social.csv'
        self._reload()

    def _manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding='utf-8'))
            except Exception:
                return {}
        return {}

    def _write_manifest(self, payload: Dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    # Checking which Datasets to use
    def _dataset_paths(self) -> Dict[str, Path]:
        return {
            'survey': self.active_survey_csv if self.active_survey_csv.exists() else self.sample_survey_path,
            'social': self.active_social_csv if self.active_social_csv.exists() else self.sample_social_path,
        }

    def _read_any_dataset(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        name = (filename or '').lower()
        bio = BytesIO(file_bytes)
        if name.endswith('.csv'):
            return pd.read_csv(bio)
        if name.endswith('.xlsx') or name.endswith('.xls'):
            return pd.read_excel(bio)
        raise ValueError('Only CSV or Excel files are supported for dataset upload.')

    # Validate Upload Datasets
    def _validate_columns(self, incoming_df: pd.DataFrame, reference_df: pd.DataFrame, label: str) -> None:
        incoming_cols = {str(c).strip() for c in incoming_df.columns}
        required_cols = {str(c).strip() for c in reference_df.columns}
        missing = sorted(required_cols - incoming_cols)
        if missing:
            preview = ', '.join(missing[:12])
            extra = ' ...' if len(missing) > 12 else ''
            raise ValueError(f'{label} dataset does not match the expected format. Missing columns: {preview}{extra}')
        if len(incoming_df) < 20:
            raise ValueError(f'{label} dataset has too few rows for stable processing. Upload at least 20 rows.')

    # Preparing datasets to reload in backend
    def _reload(self) -> None:
        dataset_paths = self._dataset_paths()
        self.active_dataset_paths = dataset_paths
        self.survey_df = pd.read_csv(dataset_paths['survey'])
        self.social_df = pd.read_csv(dataset_paths['social'])
        self.engagement_paths = ArtifactPaths.for_task(self.project.artifacts_dir, 'engagement')
        self.purchase_paths = ArtifactPaths.for_task(self.project.artifacts_dir, 'purchase_intent')
        if not self.engagement_paths.model_path.exists() or not self.purchase_paths.model_path.exists():
            run_training(
                project=self.project,
                cfg=self.cfg,
                survey_csv=dataset_paths['survey'],
                social_csv=dataset_paths['social'],
                generate_pdf_report=False,
            )
        self.engagement_model = load_model(self.engagement_paths.model_path)
        self.purchase_model = load_model(self.purchase_paths.model_path)
        self.engagement_meta = load_metrics(self.engagement_paths.metrics_path)
        self.purchase_meta = load_metrics(self.purchase_paths.metrics_path)
        self.brand_options = sorted(self.social_df['brand'].dropna().astype(str).unique().tolist()) if 'brand' in self.social_df.columns else []
        self.platforms = ['facebook', 'instagram', 'tiktok']

    # Reload the backend ML environment again
    def refresh(self) -> None:
        self._reload()

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return default
            return float(value)
        except Exception:
            return default

    # Calculate platform-level purchase statistics, load platform engagement indexes,
    # and generate summaries of the currently active survey and social datasets used by the backend.
    def _survey_platform_rates(self) -> Dict[str, float]:
        grouped = self.survey_df.groupby('platform')['purchase_intention_label'].mean().to_dict()
        return {k: self._safe_float(v) for k, v in grouped.items()}

    def _platform_index(self) -> Dict[str, float]:
        return {k: self._safe_float(v) for k, v in self.purchase_meta.get('platform_index', {}).items()}

    def current_dataset_summary(self) -> Dict[str, Any]:
        manifest = self._manifest()
        dataset_paths = self._dataset_paths()
        survey_uploaded = dataset_paths['survey'] == self.active_survey_csv
        social_uploaded = dataset_paths['social'] == self.active_social_csv
        if survey_uploaded and social_uploaded:
            source = 'uploaded'
        elif survey_uploaded or social_uploaded:
            source = 'mixed'
        else:
            source = 'sample'
        survey_name = manifest.get('survey_original_name') if survey_uploaded else self.sample_survey_path.name
        social_name = manifest.get('social_original_name') if social_uploaded else self.sample_social_path.name
        last_updated = manifest.get('updated_at') if source != 'sample' else None
        return {
            'source': source,
            'last_updated': last_updated,
            'survey': {
                'file_name': survey_name,
                'rows': int(len(self.survey_df)),
                'columns': int(len(self.survey_df.columns)),
                'has_brand_field': 'brand' in self.survey_df.columns,
            },
            'social': {
                'file_name': social_name,
                'rows': int(len(self.social_df)),
                'columns': int(len(self.social_df.columns)),
                'has_brand_field': 'brand' in self.social_df.columns,
            },
        }

    # Handles analyst dataset uploads and retrain the model
    def upload_datasets(
        self,
        *,
        survey_bytes: Optional[bytes],
        survey_filename: Optional[str],
        social_bytes: Optional[bytes],
        social_filename: Optional[str],
        train_immediately: bool = False,
        brand: Optional[str] = None,
        generate_pdf_report: bool = False,
    ) -> Dict[str, Any]:
        if not survey_bytes and not social_bytes:
            raise ValueError('Upload at least one dataset file.')

        reference_survey = pd.read_csv(self.sample_survey_path)
        reference_social = pd.read_csv(self.sample_social_path)
        manifest = self._manifest()
        changes: List[str] = []

        if survey_bytes:
            survey_df = self._read_any_dataset(survey_bytes, survey_filename or 'survey.csv')
            self._validate_columns(survey_df, reference_survey, 'Survey')
            survey_df.to_csv(self.active_survey_csv, index=False)
            manifest['survey_original_name'] = survey_filename or 'uploaded_survey.csv'
            changes.append(f'Survey dataset uploaded: {manifest["survey_original_name"]}')
        elif not self.active_survey_csv.exists() and manifest.get('source') == 'uploaded':
            # Ensure both files exist for uploaded mode.
            reference_survey.to_csv(self.active_survey_csv, index=False)

        if social_bytes:
            social_df = self._read_any_dataset(social_bytes, social_filename or 'social.csv')
            self._validate_columns(social_df, reference_social, 'Social')
            social_df.to_csv(self.active_social_csv, index=False)
            manifest['social_original_name'] = social_filename or 'uploaded_social.csv'
            changes.append(f'Social dataset uploaded: {manifest["social_original_name"]}')
        elif not self.active_social_csv.exists() and manifest.get('source') == 'uploaded':
            reference_social.to_csv(self.active_social_csv, index=False)

        manifest.update({
            'source': 'uploaded',
            'updated_at': datetime.now(timezone.utc).isoformat(),
        })
        self._write_manifest(manifest)
        self.refresh()

        result: Dict[str, Any] = {
            'status': 'uploaded',
            'message': 'Active datasets updated successfully.',
            'changes': changes,
            'current_dataset': self.current_dataset_summary(),
        }
        if train_immediately:
            result['training'] = self.train_models(brand, generate_pdf_report=generate_pdf_report)
        return result

    # Compare Facebook, Instagram, and TikTok performance for a selected brand
    # using engagement rate, purchase rate, and ML-derived platform engagement index.
    def platform_comparison(self, brand: Optional[str]) -> Dict[str, Any]:
        brand = brand or (self.brand_options[0] if self.brand_options else 'nolimit_srilanka')
        brand_df = self._brand_platform_frame(brand)
        purchase_rates = self._survey_platform_rates()
        platform_index = self._platform_index()
        max_index = max(platform_index.values()) if platform_index else 1.0
        rows = []
        for rec in brand_df.to_dict(orient='records'):
            platform = rec['platform']
            engagement_rate = self._safe_float(rec['engagement_rate'])
            purchase_rate = self._safe_float(purchase_rates.get(platform, 0.0))
            index_component = self._safe_float(platform_index.get(platform, 0.0)) / (max_index or 1.0)
            estimated_purchase_intent = (0.5 * engagement_rate) + (0.3 * purchase_rate) + (0.2 * index_component)

            # Store platform comparison results
            rows.append({
                'platform': platform,
                'engagement_rate': round(engagement_rate * 100, 2),
                'purchase_rate': round(purchase_rate * 100, 2),
                'platform_index': round(index_component * 100, 2),
                'post_volume': int(rec['post_volume']),
                'avg_followers': round(self._safe_float(rec['avg_followers']), 0),
                'estimated_purchase_intent': round(estimated_purchase_intent * 100, 2),
                'comparison_basis': {
                    'brand_engagement_weight': 50,
                    'platform_purchase_weight': 30,
                    'predicted_platform_index_weight': 20,
                },
            })
        rows.sort(key=lambda x: x['estimated_purchase_intent'], reverse=True)
        leader = rows[0] if rows else None
        return {
            'brand': brand,
            'rows': rows,
            'winner': leader,
            'explanation': (
                'Estimated purchase intention is compared transparently using a weighted mix of '
                'brand-level engagement on each platform, survey-based purchase propensity by platform, '
                'and the ML-derived platform engagement index.'
            ),
            'note': 'Brand-specific purchase retraining is data-readiness dependent because the survey dataset may not contain a brand field.',
        }

    # Create platform-level engagement summary for a selected brand
    def _brand_platform_frame(self, brand: str) -> pd.DataFrame:
        social = self.social_df.copy()
        social['brand'] = social['brand'].astype(str)
        df = social[social['brand'].str.lower() == brand.lower()].copy() if 'brand' in social.columns else social.copy()
        if df.empty:
            rows = [{'platform': p, 'engagement_rate': 0.0, 'avg_followers': 0.0, 'post_volume': 0} for p in self.platforms]
            return pd.DataFrame(rows)
        grp = df.groupby('platform').agg(
            engagement_rate=('high_engagement_label', 'mean'),
            avg_followers=('followers_count', 'mean'),
            post_volume=('platform', 'size'),
        ).reset_index()
        base = pd.DataFrame({'platform': self.platforms})
        merged = base.merge(grp, on='platform', how='left').fillna({'engagement_rate': 0.0, 'avg_followers': 0.0, 'post_volume': 0})
        return merged

    # Generate overall brand ranking using engagement rate,
    # purchase rate, and ML-derived platform engagement index.
    def overall_brand_analysis(self) -> Dict[str, Any]:
        purchase_rates = self._survey_platform_rates()
        platform_index = self._platform_index()
        max_index = max(platform_index.values()) if platform_index else 1.0
        rows = []
        for brand in self.brand_options:
            brand_df = self._brand_platform_frame(brand)
            weighted_estimates = []
            total_posts = max(int(brand_df['post_volume'].sum()), 1)
            overall_engagement = self._safe_float(brand_df['engagement_rate'].mean())
            for rec in brand_df.to_dict(orient='records'):
                platform = rec['platform']
                volume = int(rec['post_volume'])
                purchase_rate = self._safe_float(purchase_rates.get(platform, 0.0))
                idx = self._safe_float(platform_index.get(platform, 0.0)) / (max_index or 1.0)
                estimate = (0.5 * self._safe_float(rec['engagement_rate'])) + (0.3 * purchase_rate) + (0.2 * idx)
                weighted_estimates.append(estimate * (volume if volume > 0 else 1))
            rows.append({
                'brand': brand,
                'overall_engagement_rate': round(overall_engagement * 100, 2),
                'estimated_purchase_intent': round((sum(weighted_estimates) / total_posts) * 100, 2),
                'post_volume': total_posts,
            })
        rows.sort(key=lambda x: x['estimated_purchase_intent'], reverse=True)
        return {'rows': rows}

    # Generate public landing page data including project details,
    # model performance metrics, available brands/platforms, and dataset summary.
    def public_payload(self) -> Dict[str, Any]:
        metrics_purchase = self.purchase_meta.get('metrics', {})
        metrics_engagement = self.engagement_meta.get('metrics', {})
        return {
            'identity': {
                'title': 'FashIntelli',
                'project_name': 'Explainable AI Decision Support System for Fashion Marketing',
                'subtitle': 'A professional platform for fashion campaign planning and explainable prediction support.',
                'tagline': 'Brand comparison, platform recommendation, and user-friendly campaign guidance for fashion marketing.',
            },
            'kpis': {
                'engagement_f1': round(self._safe_float(metrics_engagement.get('f1')), 3),
                'engagement_auc': round(self._safe_float(metrics_engagement.get('roc_auc')), 3),
                'purchase_f1': round(self._safe_float(metrics_purchase.get('f1')), 3),
                'purchase_auc': round(self._safe_float(metrics_purchase.get('roc_auc')), 3),
            },
            'brands': self.brand_options,
            'platforms': self.platforms,
            'current_dataset': self.current_dataset_summary(),
        }

    # Build all data needed for the normal User Dashboard
    def user_dashboard(self, brand: Optional[str] = None) -> Dict[str, Any]:
        brand = brand or (self.brand_options[0] if self.brand_options else 'nolimit_srilanka')
        comparison = self.platform_comparison(brand)
        overall = self.overall_brand_analysis()
        top_brands = overall['rows'][:8]
        recommended_platform = comparison['winner']['platform'] if comparison.get('winner') else 'facebook'
        leader_score = comparison['winner']['estimated_purchase_intent'] if comparison.get('winner') else 0
        return {
            'selected_brand': brand,
            'recommended_platform': recommended_platform,
            'comparison': comparison,
            'overall_brands': top_brands,
            'current_dataset': self.current_dataset_summary(),
            'current_models': {
                'engagement': str(self.engagement_meta.get('best_model_name', '-')).upper(),
                'purchase': str(self.purchase_meta.get('best_model_name', '-')).upper(),
            },
            'hero_metrics': {
                'recommended_platform_score': leader_score,
                'compared_platforms': len(comparison['rows']),
                'brand_rank_position': next((idx + 1 for idx, row in enumerate(overall['rows']) if row['brand'] == brand), None),
            },
            'journey_steps': [
                'Select a brand to see where audience response is strongest.',
                'Review the transparent platform comparison to understand why a platform leads.',
                'Use the simplified prediction lab to test campaign readiness before publishing.',
                'Turn the recommendation into a campaign brief and channel plan.',
            ],
            'experience_cards': [
                {'title': 'Recommended platform', 'text': f'{recommended_platform.title()} currently leads for {pretty_brand_label(brand)} in the transparent decision layer.'},
                {'title': 'How the decision is made', 'text': comparison['explanation']},
                {'title': 'Current data package', 'text': f"Using the {self.current_dataset_summary()['source']} data package with the latest trained outputs."},
            ],
            'campaign_brief': {
                'objective': 'Improve purchase intention for the selected brand using the strongest platform signal.',
                'primary_platform': recommended_platform,
                'secondary_platform': comparison['rows'][1]['platform'] if len(comparison['rows']) > 1 else recommended_platform,
                'message_angle': 'Use trust-building creatives, social proof, and a clear product benefit in the first 3 seconds.',
                'cta': 'Drive product click-through with a short, direct action statement.'
            },
            'action_checklist': [
                'Match creative style to the selected platform before publishing.',
                'Use product-focused visuals and social proof in the opening frame.',
                'Keep the call-to-action visible and easy to understand.',
                'Compare expected purchase intention before and after creative changes.',
            ],
            'explainability_cards': [
                {'title': 'Why this platform leads', 'text': comparison['explanation']},
                {'title': 'Current trained package', 'text': 'User-facing outputs refresh automatically after analyst retraining. so the same decision flow always reflects the latest backend results.'},
                {'title': 'Data-readiness note', 'text': comparison['note']},
            ],
            'chart_urls': self.chart_urls(),
        }

    # Build the complete Analyst Dashboard containing model
    def analyst_dashboard(self) -> Dict[str, Any]:
        # def read_csv(path: Path) -> List[Dict[str, Any]]:
        #     if not path.exists():
        #         return []
        #     return pd.read_csv(path).head(20).replace({np.nan: None}).to_dict(orient='records')
        def read_csv(path: Path) -> List[Dict[str, Any]]:
            if not path.exists():
                return []

            df = pd.read_csv(path).head(20)

            # Convert Infinity values to NaN first
            df = df.replace([np.inf, -np.inf], np.nan)

            # Convert NaN values to None so JSON can return them as null
            df = df.astype(object).where(pd.notna(df), None)

            return df.to_dict(orient='records')





        # return {
        #     'public': self.public_payload(),
        #     'current_dataset': self.current_dataset_summary(),
        #     'overall_brand_analysis': self.overall_brand_analysis(),
        #     'sample_platform_comparison': self.platform_comparison(self.brand_options[0] if self.brand_options else None),
        #     'engagement_metrics': self.engagement_meta,
        #     'purchase_metrics': self.purchase_meta,
        #     'engagement_leaderboard': read_csv(self.engagement_paths.leaderboard_path),
        #     'purchase_leaderboard': read_csv(self.purchase_paths.leaderboard_path),
        #     'engagement_perm_importance': read_csv(self.engagement_paths.permutation_importance_path),
        #     'purchase_perm_importance': read_csv(self.purchase_paths.permutation_importance_path),
        #     'engagement_shap_importance': read_csv(self.engagement_paths.shap_importance_path),
        #     'purchase_shap_importance': read_csv(self.purchase_paths.shap_importance_path),
        #     'subgroup_age_purchase_intent': read_csv(self.project.outputs_dir / 'subgroup_age_purchase_intent.csv'),
        #     'subgroup_gender_purchase_intent': read_csv(self.project.outputs_dir / 'subgroup_gender_purchase_intent.csv'),
        #     'chart_urls': self.chart_urls(),
        #     'dataset_readiness': {
        #         'survey_has_brand_field': 'brand' in self.survey_df.columns,
        #         'social_has_brand_field': 'brand' in self.social_df.columns,
        #         'brand_specific_purchase_training_supported': 'brand' in self.survey_df.columns,
        #     },
        #     'feature_notes': [
        #         'Analyst role can view all metrics, feature importance, subgroup outcomes, and training controls.',
        #         'User role sees only guided insights, simplified predictions, and external decision-support results.',
        #         'Uploaded datasets must follow the same format as the bundled sample templates.',
        #     ],
        # }

        payload = {
            'public': self.public_payload(),
            'current_dataset': self.current_dataset_summary(),
            'overall_brand_analysis': self.overall_brand_analysis(),
            'sample_platform_comparison': self.platform_comparison(
                self.brand_options[0] if self.brand_options else None),
            'engagement_metrics': self.engagement_meta,
            'purchase_metrics': self.purchase_meta,
            'engagement_leaderboard': read_csv(self.engagement_paths.leaderboard_path),
            'purchase_leaderboard': read_csv(self.purchase_paths.leaderboard_path),
            'engagement_perm_importance': read_csv(self.engagement_paths.permutation_importance_path),
            'purchase_perm_importance': read_csv(self.purchase_paths.permutation_importance_path),
            'engagement_shap_importance': read_csv(self.engagement_paths.shap_importance_path),
            'purchase_shap_importance': read_csv(self.purchase_paths.shap_importance_path),
            'subgroup_age_purchase_intent': read_csv(self.project.outputs_dir / 'subgroup_age_purchase_intent.csv'),
            'subgroup_gender_purchase_intent': read_csv(
                self.project.outputs_dir / 'subgroup_gender_purchase_intent.csv'),
            'chart_urls': self.chart_urls(),
            'dataset_readiness': {
                'survey_has_brand_field': 'brand' in self.survey_df.columns,
                'social_has_brand_field': 'brand' in self.social_df.columns,
                'brand_specific_purchase_training_supported': 'brand' in self.survey_df.columns,
            },
            'feature_notes': [
                'Analyst role can view all metrics, feature importance, subgroup outcomes, and training controls.',
                'User role sees only guided insights, simplified predictions, and external decision-support results.',
                'Uploaded datasets must follow the same format as the bundled sample templates.',
            ],
        }

        return clean_json_value(payload)

    # Generate chart image URLs generated during model evaluation for frontend dashboard visualization.
    def chart_urls(self) -> Dict[str, str]:
        base = '/figures'
        return {
            'engagement_confusion': f'{base}/engagement_confusion.png',
            'engagement_roc': f'{base}/engagement_roc.png',
            'engagement_pr': f'{base}/engagement_pr.png',
            'engagement_calibration': f'{base}/engagement_calibration.png',
            'purchase_confusion': f'{base}/purchase_intent_confusion.png',
            'purchase_roc': f'{base}/purchase_intent_roc.png',
            'purchase_pr': f'{base}/purchase_intent_pr.png',
            'purchase_calibration': f'{base}/purchase_intent_calibration.png',
        }

    # Predict purchase intention using user input from the Prediction Studio.
    def predict_purchase_intent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        comparison = self.platform_comparison(payload.get('brand'))
        platform = payload.get('platform', 'facebook').lower()
        lookup = {row['platform']: row for row in comparison['rows']}
        selected = lookup.get(platform, comparison['winner'])
        platform_engagement_index = self._safe_float(selected['estimated_purchase_intent']) / 100.0 if selected else self._safe_float(self._platform_index().get(platform, 0.0))
        row = {
            'age_range': payload.get('age_range', '25-34'),
            'gender': payload.get('gender', 'Female'),
            'occupation': payload.get('occupation', 'Employed'),
            'income': payload.get('income', '25,000-50,000'),
            'platform': platform,
            'platform_usage_count': int(payload.get('platform_usage_count', 2)),
            'ad_exposure_ordinal': int(payload.get('ad_exposure_ordinal', 4)),
            'influencer_trust': int(payload.get('influencer_trust', 4)),
            'influencer_recommend': int(payload.get('influencer_recommend', 4)),
            'ad_appeal': int(payload.get('ad_appeal', 4)),
            'brand_engagement_trust': int(payload.get('brand_engagement_trust', 4)),
            'social_media_presence': int(payload.get('social_media_presence', 4)),
            'social_proof_confidence': int(payload.get('social_proof_confidence', 4)),
            'past_purchase_flag': int(payload.get('past_purchase_flag', 3)),
            'overall_influence': int(payload.get('overall_influence', 4)),
            'app_helpful': np.nan,
            'influencer_credibility_score': np.mean([int(payload.get('influencer_trust', 4)), int(payload.get('influencer_recommend', 4))]),
            'ad_effectiveness_score': np.mean([int(payload.get('ad_exposure_ordinal', 4)), int(payload.get('ad_appeal', 4))]),
            'brand_trust_score': np.mean([int(payload.get('brand_engagement_trust', 4)), int(payload.get('social_media_presence', 4))]),
            'overall_smm_index': np.mean([
                int(payload.get('ad_exposure_ordinal', 4)),
                int(payload.get('influencer_trust', 4)),
                int(payload.get('influencer_recommend', 4)),
                int(payload.get('ad_appeal', 4)),
                int(payload.get('brand_engagement_trust', 4)),
                int(payload.get('social_media_presence', 4)),
                int(payload.get('social_proof_confidence', 4)),
                int(payload.get('overall_influence', 4)),
            ]),
            'trust_x_socialproof': int(payload.get('influencer_trust', 4)) * int(payload.get('social_proof_confidence', 4)),
            'platform_engagement_index': platform_engagement_index,
        }
        X = pd.DataFrame([row])
        prediction = int(self.purchase_model.predict(X)[0])
        probability = float(self.purchase_model.predict_proba(X)[0, 1]) if hasattr(self.purchase_model, 'predict_proba') else float(prediction)
        score = round(probability * 100, 2)
        return {
            'prediction_label': 'High Purchase Intention' if prediction == 1 else 'Low Purchase Intention',
            'probability_score': score,
            'selected_brand': payload.get('brand'),
            'selected_platform': platform,
            'explanations': [
                f"Platform engagement index injected into the model: {round(platform_engagement_index * 100, 2)}",
                f"Top recommendation platform for the selected brand: {comparison['winner']['platform'] if comparison.get('winner') else platform}",
                'Prediction Lab has been simplified so non-technical users can test scenarios using guided sliders and dropdowns.',
            ],
        }

    # Retrain the engagement and purchase intention models using the current datasets.
    # If a brand is selected, the system tries to filter datasets by that brand before training.
    def train_models(self, brand: Optional[str], generate_pdf_report: bool = False) -> Dict[str, Any]:
        survey_df = self.survey_df.copy()
        social_df = self.social_df.copy()
        notes = []
        normalized_brand = (brand or '').strip()
        if normalized_brand:
            subset = social_df[social_df['brand'].astype(str).str.lower() == normalized_brand.lower()].copy() if 'brand' in social_df.columns else pd.DataFrame()
            if len(subset) >= 40:
                social_df = subset
                notes.append(f'Engagement model retrained using the social dataset filtered to brand: {normalized_brand}.')
            else:
                notes.append(f'Brand filter "{normalized_brand}" has too few social rows for stable retraining, so the full social dataset was used.')
            if 'brand' in survey_df.columns:
                survey_subset = survey_df[survey_df['brand'].astype(str).str.lower() == normalized_brand.lower()].copy()
                if len(survey_subset) >= 40:
                    survey_df = survey_subset
                    notes.append(f'Purchase dataset retrained using the survey subset filtered to brand: {normalized_brand}.')
                else:
                    notes.append(f'Brand filter "{normalized_brand}" has too few survey rows for stable purchase retraining, so the full survey dataset was used.')
            else:
                notes.append('Purchase dataset has no brand field, so purchase-intention retraining used the full survey dataset. This is transparent by design.')
        survey_tmp = self.project.outputs_dir / '_api_survey_train.csv'
        social_tmp = self.project.outputs_dir / '_api_social_train.csv'
        survey_df.to_csv(survey_tmp, index=False)
        social_df.to_csv(social_tmp, index=False)
        outputs = run_training(
            project=self.project,
            cfg=self.cfg,
            survey_csv=survey_tmp,
            social_csv=social_tmp,
            generate_pdf_report=generate_pdf_report,
        )
        self.refresh()
        return {
            'status': 'completed',
            'brand': normalized_brand or None,
            'notes': notes,
            'current_dataset': self.current_dataset_summary(),
            'engagement': outputs['engagement'].metrics,
            'purchase_intent': outputs['purchase_intent'].metrics,
        }

    # Generate a downloadable ZIP package containing active datasets,
    # upload templates, dataset summaries, and README instructions for analysts.
    def build_dataset_zip(self) -> bytes:
        buffer = BytesIO()
        current = self.current_dataset_summary()
        with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as zf:
            zf.write(self._dataset_paths()['survey'], arcname='active/current_survey.csv')
            zf.write(self._dataset_paths()['social'], arcname='active/current_social.csv')
            zf.write(self.sample_survey_path, arcname='templates/FashIntelli_Survey_Template.csv')
            zf.write(self.sample_social_path, arcname='templates/FashIntelli_Social_Template.csv')
            zf.writestr('README.txt', (
                'This package contains the current active survey and social datasets used by the app, '\
                'plus sample templates for new uploads. Upload new files using the same format as the templates.\n'
            ))
            zf.writestr('active_dataset_summary.json', json.dumps(current, indent=2))
        buffer.seek(0)
        return buffer.getvalue()

    # Generate a downloadable PDF summary report for the selected brand.
    def build_summary_pdf(self, brand: Optional[str], user_name: str = 'User') -> bytes:
        brand = brand or (self.brand_options[0] if self.brand_options else 'nolimit_srilanka')
        comparison = self.platform_comparison(brand)
        overall = self.overall_brand_analysis()['rows'][:6]
        current = self.current_dataset_summary()
        purchase_metrics = self.purchase_meta.get('metrics', {})
        engagement_metrics = self.engagement_meta.get('metrics', {})
        winner = comparison.get('winner') or {'platform': '-', 'estimated_purchase_intent': 0.0}

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        w, h = A4

        # Background and header
        c.setFillColor(colors.HexColor('#fff9f6'))
        c.rect(0, 0, w, h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#2d1320'))
        c.rect(0, h - 110, w, 110, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#f6d7c3'))
        c.setFont('Helvetica-Bold', 26)
        c.drawString(40, h - 55, 'FashIntelli Summary Report')
        c.setFillColor(colors.white)
        c.setFont('Helvetica', 11)
        c.drawString(40, h - 78, 'Explainable AI decision support for fashion marketing')
        c.drawRightString(w - 40, h - 55, datetime.now().strftime('%d %b %Y'))

        def card(x: float, y: float, width: float, height: float, title: str, value: str, subtitle: str = '') -> None:
            c.setFillColor(colors.white)
            c.roundRect(x, y, width, height, 12, fill=1, stroke=0)
            c.setStrokeColor(colors.HexColor('#e7c8b8'))
            c.roundRect(x, y, width, height, 12, fill=0, stroke=1)
            c.setFillColor(colors.HexColor('#7a274f'))
            c.setFont('Helvetica-Bold', 10)
            c.drawString(x + 14, y + height - 18, title)
            c.setFillColor(colors.HexColor('#2d1320'))
            c.setFont('Helvetica-Bold', 18)
            c.drawString(x + 14, y + height - 42, value)
            if subtitle:
                c.setFont('Helvetica', 9)
                c.setFillColor(colors.HexColor('#5c4b52'))
                c.drawString(x + 14, y + 12, subtitle)

        card(40, h - 200, 160, 72, 'Selected brand', pretty_brand_label(brand), 'Current focus')
        card(215, h - 200, 160, 72, 'Recommended platform', str(winner['platform']).title(), 'Transparent comparison winner')
        card(390, h - 200, 165, 72, 'Platform score', f"{self._safe_float(winner['estimated_purchase_intent']):.2f}%", 'Estimated purchase intention')

        # Short summary
        c.setFillColor(colors.HexColor('#2d1320'))
        c.setFont('Helvetica-Bold', 14)
        c.drawString(40, h - 240, f'Prepared for: {user_name}')
        c.setFont('Helvetica', 11)
        summary_text = (
            f"{str(winner['platform']).title()} currently leads for {pretty_brand_label(brand)} based on a transparent blend of "
            f"brand engagement, platform purchase tendency, and the trained engagement index."
        )
        text = c.beginText(40, h - 258)
        text.setFont('Helvetica', 10)
        text.setFillColor(colors.HexColor('#4d3a43'))
        for line in self._wrap_text(summary_text, 88):
            text.textLine(line)
        c.drawText(text)

        # Comparison bars
        chart_x, chart_y = 40, h - 460
        chart_w, chart_h = 300, 150
        c.setFillColor(colors.white)
        c.roundRect(chart_x, chart_y, chart_w, chart_h, 12, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor('#e7c8b8'))
        c.roundRect(chart_x, chart_y, chart_w, chart_h, 12, fill=0, stroke=1)
        c.setFillColor(colors.HexColor('#2d1320'))
        c.setFont('Helvetica-Bold', 12)
        c.drawString(chart_x + 14, chart_y + chart_h - 18, 'Platform comparison')
        c.setFont('Helvetica', 9)
        c.setFillColor(colors.HexColor('#5c4b52'))
        c.drawString(chart_x + 14, chart_y + chart_h - 32, 'Estimated purchase intention vs engagement rate')
        rows = comparison['rows'][:3]
        max_val = max([max(self._safe_float(r['estimated_purchase_intent']), self._safe_float(r['engagement_rate'])) for r in rows] + [1])
        base_y = chart_y + 28
        for idx, row in enumerate(rows):
            x = chart_x + 36 + (idx * 82)
            scale = (chart_h - 70) / max_val
            pi_h = self._safe_float(row['estimated_purchase_intent']) * scale
            er_h = self._safe_float(row['engagement_rate']) * scale
            c.setStrokeColor(colors.HexColor('#d8c5cf'))
            c.line(x, base_y, x, base_y + chart_h - 70)
            c.setFillColor(colors.HexColor('#7a274f'))
            c.rect(x + 5, base_y, 22, pi_h, fill=1, stroke=0)
            c.setFillColor(colors.HexColor('#d39a62'))
            c.rect(x + 31, base_y, 22, er_h, fill=1, stroke=0)
            c.setFillColor(colors.HexColor('#2d1320'))
            c.setFont('Helvetica', 9)
            c.drawCentredString(x + 28, base_y - 12, str(row['platform']).title())

        c.setFillColor(colors.HexColor('#7a274f'))
        c.rect(chart_x + 16, chart_y + 14, 8, 8, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#2d1320'))
        c.setFont('Helvetica', 8)
        c.drawString(chart_x + 28, chart_y + 14, 'Estimated purchase intention')
        c.setFillColor(colors.HexColor('#d39a62'))
        c.rect(chart_x + 156, chart_y + 14, 8, 8, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#2d1320'))
        c.drawString(chart_x + 168, chart_y + 14, 'Engagement rate')

        # Right-side insights
        box_x, box_y, box_w, box_h = 360, h - 460, 195, 150
        c.setFillColor(colors.white)
        c.roundRect(box_x, box_y, box_w, box_h, 12, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor('#e7c8b8'))
        c.roundRect(box_x, box_y, box_w, box_h, 12, fill=0, stroke=1)
        c.setFillColor(colors.HexColor('#2d1320'))
        c.setFont('Helvetica-Bold', 12)
        c.drawString(box_x + 14, box_y + box_h - 18, 'Current trained setup')
        insights = [
            f"Engagement model: {str(self.engagement_meta.get('best_model_name', '-')).upper()}",
            f"Purchase model: {str(self.purchase_meta.get('best_model_name', '-')).upper()}",
            f"Engagement F1: {self._safe_float(engagement_metrics.get('f1')):.5f}",
            f"Purchase F1: {self._safe_float(purchase_metrics.get('f1')):.5f}",
            f"Dataset source: {current['source'].title()}",
        ]
        t = c.beginText(box_x + 14, box_y + box_h - 38)
        t.setFont('Helvetica', 10)
        t.setFillColor(colors.HexColor('#4d3a43'))
        for line in insights:
            t.textLine(line)
        c.drawText(t)

        # Brand table
        c.setFillColor(colors.HexColor('#2d1320'))
        c.setFont('Helvetica-Bold', 12)
        c.drawString(40, h - 490, 'Overall brand ranking snapshot')
        table_y = h - 515
        headers = ['Brand', 'Estimated PI', 'Engagement', 'Posts']
        col_x = [40, 220, 330, 430]
        c.setFillColor(colors.HexColor('#7a274f'))
        c.roundRect(40, table_y, 515, 22, 8, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont('Helvetica-Bold', 9)
        for hx, label in zip(col_x, headers):
            c.drawString(hx + 8, table_y + 7, label)
        y = table_y - 24
        c.setFont('Helvetica', 9)
        for row in overall[:5]:
            c.setFillColor(colors.white)
            c.roundRect(40, y, 515, 20, 6, fill=1, stroke=0)
            c.setStrokeColor(colors.HexColor('#f0d7cb'))
            c.roundRect(40, y, 515, 20, 6, fill=0, stroke=1)
            c.setFillColor(colors.HexColor('#2d1320'))
            values = [pretty_brand_label(row['brand']), f"{self._safe_float(row['estimated_purchase_intent']):.2f}%", f"{self._safe_float(row['overall_engagement_rate']):.2f}%", str(row['post_volume'])]
            for hx, value in zip(col_x, values):
                c.drawString(hx + 8, y + 6, value)
            y -= 24

        c.setFont('Helvetica', 8)
        c.setFillColor(colors.HexColor('#7b6a72'))
        footer = 'Generated from the current FashIntelli backend outputs. User-facing guidance updates automatically after analyst retraining.'
        c.drawString(40, 24, footer)
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    def _wrap_text(self, text: str, width: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = []
        for word in words:
            candidate = ' '.join(current + [word])
            if len(candidate) <= width:
                current.append(word)
            else:
                lines.append(' '.join(current))
                current = [word]
        if current:
            lines.append(' '.join(current))
        return lines


@lru_cache
def get_analytics_service() -> AnalyticsService:
    return AnalyticsService()
