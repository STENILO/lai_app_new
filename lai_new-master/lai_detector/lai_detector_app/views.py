from fileinput import filename
from json import JSONDecodeError
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from .detector import detect
import os
from datetime import datetime
from .models import Image
from django.http.response import JsonResponse
from django.contrib.auth import authenticate, login as authlogin, logout as authlogout
from .forms import UserCreationForm
from django.contrib.auth.decorators import login_required


# Create your views here.
@login_required(login_url='/signin/')
@csrf_exempt
def index(request):
    images_count = len(Image.objects.all())

    return render(request, 'lai_detector_app/index.html', context={'images_count': images_count})

@login_required(login_url='/signin/')
@csrf_exempt
def load_more_images(request):
    images_count = len(Image.objects.all())
    images = Image.objects.all()[int(request.GET.get('offset')):int(request.GET.get('offset')) + 5].values()

    return JsonResponse({
        'images': list(images),
        'images_count': images_count
    })

@login_required(login_url='/signin/')
@csrf_exempt
def predict_lai(request):
    fileName, fileExtension = os.path.splitext(request.POST.get('filename'))
    X = int(request.POST.get('X'))
    Y = int(request.POST.get('Y'))
    Latitude = int(request.POST.get('Latitude')) + X
    Longitude = int(request.POST.get('Longitude')) + Y

    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    image = Image(path = "images/%s/"%date_time, image_file=request.FILES['file'])
    image.save()

    uploaded_path = image.image_file.url

    try:
        detected_output_image_path, output_image_path, LAI, FVC, coords = detect(uploaded_path, image.id, X, Y, Latitude, Longitude)

        # image = Image.objects.get(path=detected_path)
        return JsonResponse({
            'imageID': image.id,
            'LAI': LAI,
            'FVC': FVC,
            'error': 'no',
            'destination': uploaded_path,
            'detectedOutputImage': detected_output_image_path,
            'outputImage': output_image_path,
            'X': X,
            'Y': Y,
            'Latitude': Latitude,
            'Longitude': Longitude,
            'meta': list(coords),
        })
    except Exception as e:
        import traceback, sys
        print(traceback.format_exc())
        if type(e).__name__ == "InvalidArgumentError":
            return JsonResponse({'error': 'err', 'message': 'Please, check your X, Y, Latitude, Longitude values and try again'})
            
        # print(sys.exc_info()[2])
        
        return JsonResponse({'error': 'err', 'message': 'Internal server error'})

    return render(request, 'lai_detector_app/index.html')

def signin(request):
    return render(request, 'lai_detector_app/login.html')

def login(request):
    username = request.POST['login']
    password = request.POST['password']
    user = authenticate(request, username=username, password=password)
    
    if user is not None:
        authlogin(request, user)

        return redirect('index')

    return render(request, 'lai_detector_app/login.html', {'error': 'Incorrect username or password'})

def logout(request):
    authlogout(request)

    return redirect('index')

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(request, username=username, password=raw_password)
            authlogin(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()


    return render(request, 'lai_detector_app/signup.html', {'form': form})
