from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import Message
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Q
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Charger le modèle et le vectoriseur
model = joblib.load('E:/IA/mail-spam/train-model/modelgasy.joblib')
vectorizer = joblib.load('E:/IA/mail-spam/train-model/vectorizergasy.joblib')

# Envoyer un message
@login_required
def send_message(request):
    if request.method == 'POST':
        recipient_id = request.POST['recipient']
        subject = request.POST['subject']
        body = request.POST['body']
        recipient = User.objects.get(id=recipient_id)
        
        # Vérifier si le message est un spam
        vectorized_message = vectorizer.transform([body])
        is_spam = model.predict(vectorized_message)[0] == 1
        
        # Enregistrer le message avec l'indicateur spam
        message = Message(sender=request.user, recipient=recipient, subject=subject, body=body, is_spam=is_spam)
        message.save()
        
        # Redirection vers la boîte de réception
        return redirect('inbox')
    
    users = User.objects.all()
    return render(request, 'messages/send_message.html', {'users': users})

# Afficher la boîte de réception
@login_required
def inbox(request):
    messages = Message.objects.filter(recipient=request.user, is_spam=False)
    return render(request, 'messages/inbox.html', {'messages': messages})

# Afficher la liste des spams
@login_required
def spam_list(request):
    spams = Message.objects.filter(recipient=request.user, is_spam=True)
    return render(request, 'messages/spam_list.html', {'spams': spams})

# Afficher les messages envoyés
@login_required
def sent_messages(request):
    sent_messages = Message.objects.filter(sender=request.user, is_spam=False)
    return render(request, 'messages/sents.html', {'messages': sent_messages})

# Rechercher des messages
@login_required
def search_messages(request):
    query = request.GET.get('q', '')
    if query:
        results = Message.objects.filter(
            Q(subject__icontains=query) | Q(body__icontains=query),
            recipient=request.user
        )
    else:
        results = []
        
    return render(request, 'messages/searchs.html', {'query': query, 'results': results})

# Déplacer un message dans le spam
@login_required
def move_to_spam(request, message_id):
    message = get_object_or_404(Message, id=message_id, recipient=request.user)
    message.is_spam = True
    message.save()
    return redirect('inbox')

# Déplacer un message dans la boîte de réception
@login_required
def move_to_inbox(request, message_id):
    if not message_id:  # Vérifiez si `message_id` n'est pas vide ou None
        return redirect('spam_list')  # Redirection si `message_id` est invalide
    
    message = get_object_or_404(Message, id=message_id, recipient=request.user)
    message.is_spam = False
    message.save()
    return redirect('spam_list')

# Répondre à un message
@login_required
def reply_message(request, message_id):
    original_message = get_object_or_404(Message, id=message_id, recipient=request.user)
    
    if request.method == 'POST':
        recipient = original_message.sender
        subject = request.POST['subject']
        body = request.POST['body']
        
        # Vérifier si la réponse est un spam
        vectorized_message = vectorizer.transform([body])
        is_spam = model.predict(vectorized_message)[0] == 1
        
        # Enregistrer la réponse
        reply_message = Message(
            sender=request.user, 
            recipient=recipient, 
            subject=subject, 
            body=body, 
            is_spam=is_spam
        )
        reply_message.save()
        
        return redirect('inbox')
    
    # Pré-remplir le sujet avec "Re: [sujet original]"
    initial_subject = f"Re: {original_message.subject}"
    return render(request, 'messages/reply_message.html', {
        'recipient': original_message.sender, 
        'initial_subject': initial_subject,
    })

@login_required
def train_model(request):
    # Récupérer les données de la base de données
    messages = Message.objects.all()
    
    # Préparer les données pour l'entraînement
    data = pd.DataFrame(list(messages.values('subject', 'body', 'is_spam')))
    data['text'] = data['subject'] + " " + data['body']
    X = data['text']
    y = data['is_spam']
    
    # Vérifier s'il y a au moins deux classes dans les données
    if len(set(y)) < 2:
        return HttpResponse("Les données doivent contenir au moins deux classes (spam et non-spam) pour entraîner le modèle. Veuillez ajouter des exemples de spam.")
    
    # Vectorisation des données textuelles
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X).toarray()
    
    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle de régression logistique
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Sauvegarder le modèle et le vectoriseur
    joblib.dump(model, 'E:/IA/mail-spam/train-model/modelgasy.joblib')
    joblib.dump(vectorizer, 'E:/IA/mail-spam/train-model/vectorizergasy.joblib')
    
    return HttpResponse("Modèle entraîné et sauvegardé avec succès.")