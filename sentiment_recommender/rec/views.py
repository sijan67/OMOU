from django.shortcuts import render
from . import models
from .models import ml_util, ml_models

def home(request):
    return render(request, 'home.html') 

def select(request):
    request.session['user_input'] = clean_string = ml_util.clean_input(request.POST.get('journal', None))
    return render(request, 'select.html')

def recommendation(request):
    
    context = {}
    context['journal'] = request.session['user_input']
    show_more = request.POST.get('more', None)

    if show_more == "More":
        request.session['result_start'] += 5
        request.session['result_end'] += 5
        context['result'] = request.session['result']
        context['selection'] = request.session['selection']
    
    else:
        request.session['result_start'] = 0
        request.session['result_end'] = 5
        selection = request.POST.get('selection', None)
        request.session['selection'] = selection
        context['selection'] = selection
        print(type(context['journal']))
        if selection == "Movie":
            context['result'] = ["Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5", "Movie 6", "Movie 7", "Movie 8", "Movie 9", "Movie 10", "Movie 11"]
        
        if selection == "Music":
            context['result'] = ml_models.music_model(context['journal'])

        if selection == "Podcast":
            context['result'] = ml_models.podcast_model(context['journal'])

        if selection == "Anime":
            context['result'] = ml_models.anime_model(context['journal'])
        
        request.session['result'] = context['result']

    context['start'] = request.session['result_start']
    context['end'] = request.session['result_end']

    return render(request, 'recommendation.html', context)
