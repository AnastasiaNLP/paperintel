from api.app_factory import create_paperintel_service
from api.rest.app import create_rest_app
from config.settings import settings


app = create_rest_app(
    service=create_paperintel_service(),
    auth_token=settings.paperintel_api_auth_token,
    cors_allow_origins=settings.cors_allow_origins_list,
)
