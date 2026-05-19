from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from ..services.analytics import get_analytics_service

router = APIRouter(prefix='/api/public', tags=['public'])


@router.get('/landing')
def landing():
    return get_analytics_service().public_payload()

# GET endpoint used to get overall brand-level analysis
@router.get('/brand-overview')
def brand_overview():
    return get_analytics_service().overall_brand_analysis()

# GET endpoint used to compare platforms such as Instagram, Facebook, and TikTok
@router.get('/platform-comparison')
def platform_comparison(brand: str | None = Query(default=None)):
    return get_analytics_service().platform_comparison(brand)

# GET endpoint used to download a summary report as PDF
@router.get('/download/report.pdf')
def download_report(brand: str | None = Query(default=None), user_name: str = Query(default='User')):
    payload = get_analytics_service().build_summary_pdf(brand, user_name=user_name)
    headers = {'Content-Disposition': 'attachment; filename="fashintelli-summary-report.pdf"'}
    return StreamingResponse(iter([payload]), media_type='application/pdf', headers=headers)

# GET endpoint used to download sample datasets as ZIP file
@router.get('/download/datasets.zip')
def download_datasets():
    payload = get_analytics_service().build_dataset_zip()
    headers = {'Content-Disposition': 'attachment; filename="fashintelli-sample-datasets.zip"'}
    return StreamingResponse(iter([payload]), media_type='application/zip', headers=headers)
