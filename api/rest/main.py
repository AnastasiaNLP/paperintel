from api.app_factory import create_paperintel_service
from api.rest.app import create_rest_app


app = create_rest_app(service=create_paperintel_service())
