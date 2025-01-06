from . import views
from django.urls import path 
urlpatterns=[
    path('',views.home,name='home'),
    path('report/',views.report,name='report'),
    path('newreport/',views.newreport,name='newreport'),
]