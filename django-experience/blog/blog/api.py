from ninja import NinjaAPI


api = NinjaAPI()

api.add_router("/users/", "core.api.router.router")