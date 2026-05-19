from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .utils import get_logger


@dataclass
class Recommendation:
    title: str
    rationale: str
    priority: str = "Medium"


class MarketingRecommendationEngine:
    def __init__(self, logger_name: str = "fashintelli.reco") -> None:
        self.logger = get_logger(logger_name)

    def recommend_for_purchase_intent(
        self,
        features: Dict[str, float],
        purchase_prob: float,
        *,
        top_drivers: Optional[List[str]] = None,
    ) -> List[Recommendation]:
        top_drivers = top_drivers or []
        recos: List[Recommendation] = []

        if purchase_prob < 0.35:
            recos.append(Recommendation(
                title="Strengthen trust & clarity in campaigns",
                rationale="Predicted purchase intention is low. Focus on authenticity, value clarity, and reducing friction to purchase.",
                priority="High",
            ))
        elif purchase_prob < 0.65:
            recos.append(Recommendation(
                title="Optimise creatives and social proof for conversion",
                rationale="Predicted purchase intention is moderate. Improve creatives, targeting, and highlight reviews/UGC.",
                priority="Medium",
            ))
        else:
            recos.append(Recommendation(
                title="Scale what works and segment audiences",
                rationale="Predicted purchase intention is high. Scale successful content and personalise by segment.",
                priority="Medium",
            ))

        infl_trust = features.get("influencer_trust", np.nan)
        infl_reco = features.get("influencer_recommend", np.nan)
        ad_appeal = features.get("ad_appeal", np.nan)
        brand_eng = features.get("brand_engagement_trust", np.nan)
        social_proof = features.get("social_proof_confidence", np.nan)

        def _is_low(x): return np.isfinite(x) and x <= 2.5

        if _is_low(infl_trust) or _is_low(infl_reco):
            recos.append(Recommendation(
                title="Use more authentic influencers (micro / niche creators)",
                rationale="Influencer trust/recommendation is low. Improve authenticity signals via genuine reviews and long-term partnerships.",
                priority="High",
            ))

        if _is_low(ad_appeal):
            recos.append(Recommendation(
                title="Upgrade ad creatives (visuals + informativeness)",
                rationale="Ad appeal is low. Test clearer product benefits, stronger visuals, and short-form video formats.",
                priority="High",
            ))

        if _is_low(brand_eng):
            recos.append(Recommendation(
                title="Increase two-way engagement",
                rationale="Brand engagement trust is low. Respond to comments, encourage UGC, and build community interactions.",
                priority="Medium",
            ))

        if np.isfinite(social_proof) and social_proof >= 4.0:
            recos.append(Recommendation(
                title="Lean into social proof",
                rationale="Audience responds strongly to social proof. Highlight testimonials, ratings, and customer stories.",
                priority="Medium",
            ))

        if top_drivers:
            recos.append(Recommendation(
                title="Prioritise the top model drivers",
                rationale="Top influencing factors detected: " + ", ".join(top_drivers[:5]),
                priority="Medium",
            ))

        return recos

    def recommend_for_post_engagement(
        self,
        features: Dict[str, float],
        engagement_prob: float,
        *,
        top_drivers: Optional[List[str]] = None,
    ) -> List[Recommendation]:
        top_drivers = top_drivers or []
        recos: List[Recommendation] = []

        if engagement_prob < 0.35:
            recos.append(Recommendation(
                title="Rework post composition for higher engagement",
                rationale="Predicted engagement is low. Adjust copy length, timing, and hashtag strategy; test video formats.",
                priority="High",
            ))
        elif engagement_prob < 0.65:
            recos.append(Recommendation(
                title="A/B test timing and content format",
                rationale="Predicted engagement is moderate. Test post hours/day-of-week and vary formats (video vs image).",
                priority="Medium",
            ))
        else:
            recos.append(Recommendation(
                title="Amplify and repurpose high-performing content",
                rationale="Predicted engagement is high. Boost posts, repurpose into stories/reels, and cross-post to other platforms.",
                priority="Medium",
            ))

        hashtag_count = features.get("hashtag_count", np.nan)
        caption_len_words = features.get("caption_len_words", np.nan)

        if np.isfinite(hashtag_count) and hashtag_count < 3:
            recos.append(Recommendation(
                title="Increase relevant hashtags (avoid spam)",
                rationale="Hashtag usage is low. Add 5–12 relevant hashtags aligned with brand + product + occasion.",
                priority="Medium",
            ))

        if np.isfinite(caption_len_words) and caption_len_words > 120:
            recos.append(Recommendation(
                title="Shorten captions or add stronger structure",
                rationale="Captions are long. Use hooks, bullet points, and a clear CTA to improve readability.",
                priority="Low",
            ))

        if top_drivers:
            recos.append(Recommendation(
                title="Prioritise top engagement drivers",
                rationale="Top influencing factors detected: " + ", ".join(top_drivers[:5]),
                priority="Medium",
            ))

        return recos
