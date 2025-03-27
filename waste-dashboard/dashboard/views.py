from django.shortcuts import render
from django.http import JsonResponse
from . import services

def index(request):
    """Main dashboard view"""
    return render(request, 'dashboard/index.html')

def api_predictions(request):
    """API endpoint for predictions"""
    predictions = services.get_all_predictions()
    return JsonResponse(predictions)

def api_predict(request):
    """API endpoint for making new predictions"""
    if request.method == 'POST':
        # Get data from request
        data = {
            'meal': request.POST.get('meal'),
            # Add other fields as needed
        }
        prediction = services.predict_waste(data)
        return JsonResponse({'prediction': prediction})
    return JsonResponse({'error': 'POST request required'})