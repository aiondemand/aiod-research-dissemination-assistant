from keycloak import KeycloakOpenID
from starlette.requests import Request

from .settings import settings

keycloak_openid = KeycloakOpenID(
    server_url=str(settings.aiod_keycloak.SERVER_URL),
    client_id=settings.aiod_keycloak.CLIENT_ID,
    client_secret_key=settings.aiod_keycloak.CLIENT_SECRET,
    realm_name=settings.aiod_keycloak.REALM,
    verify=True,
)


def get_user(request: Request):
    user = request.session.get("user")
    if user:
        return user["name"]
    return None
