from dataclasses import field
from django.contrib.auth.models import User

from rest_framework import serializers
from rest_framework.generics import CreateAPIView


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    id = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = User
        fields = ['username', 'password', 'id']
        
# Create your views here.
class UserCreateApiView(CreateAPIView):
    serializer_class = UserSerializer
    queryset = User.objects.all()