from django.shortcuts import render
from django.utils.safestring import SafeData, SafeText, mark_safe
from . import models
from .models import ml_util, podcast_rec

def home(request):
    return render(request, 'home.html') 

def select(request):
    request.session['user_input'] = request.POST.get('journal', None)
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

        if selection == "Movies":
            context['result'] = ["Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5", "Movie 6", "Movie 7", "Movie 8", "Movie 9", "Movie 10", "Movie 11"]
        
        if selection == "Music":
            context['result'] = ml_util.music_model(context['journal'])

        if selection == "Podcasts":
            context['result'] = podcast_rec.podcast_model(context['journal'])

        if selection == "Anime":
            context['result'] = ["Show 1", "Show 2", "Show 3", "Show 4", "Show 5", "Show 6", "Show 7", "Show 8"]
        
        request.session['result'] = context['result']

    context['start'] = request.session['result_start']
    context['end'] = request.session['result_end']

    return render(request, 'recommendation.html', context)
