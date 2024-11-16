from django.urls import path
from .views import UserCreateApiView

urlpatterns = [
    path("users/create", UserCreateApiView.as_view())
]
