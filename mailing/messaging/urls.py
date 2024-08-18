from django.urls import path
from . import views

urlpatterns = [
    path('send/', views.send_message, name='send_message'),
    path('inbox/', views.inbox, name='inbox'),
    path('spams/', views.spam_list, name='spam_list'),
    path('sents/', views.sent_messages, name='sents_messages'),
    path('search/', views.search_messages, name='search_messages'),
    path('move-to-spam/<int:message_id>/', views.move_to_spam, name='move_to_spam'),
    path('move-to-inbox/<int:message_id>/', views.move_to_inbox, name='move_to_inbox'),
    path('reply/<int:message_id>/', views.reply_message, name='reply_message'),
    path('train-model/', views.train_model, name='train_model')
]
