from ninja import Router
from django.contrib.auth.models import User
from rest_fre
from ninja import Schema


router = Router()


class UserIn(Schema):
    username: str
    password: str

class UserOut(Schema):
    id: int
    username: str
    
@router.post("/create", response=UserOut)
def create_user(request, user_data: UserIn) -> UserOut:
    user = User.objects.create(**user_data.dict())
    return UserOut(id=user.id, username=user.username)
