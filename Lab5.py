import requests
import streamlit as st
from openai import OpenAI

# Set up OpenAI API Key
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Page content
st.title("Lab 5: Travel Weather and Suggestion Bot - Tanu Rana")

# Function to get current weather
def get_current_weather(location, API_key):
    if not location:
        location = "Syracuse, NY"  # Default location if none is provided
    if "," in location:
        location = location.split(",")[0].strip()

    urlbase = "https://api.openweathermap.org/data/2.5/"
    urlweather = f"weather?q={location}&appid={API_key}"
    url = urlbase + urlweather
    response = requests.get(url)
    data = response.json()

    # Extract temperatures & Convert Kelvin to Celsius
    temp = data['main']['temp'] - 273.15
    feels_like = data['main']['feels_like'] - 273.15
    temp_min = data['main']['temp_min'] - 273.15
    temp_max = data['main']['temp_max'] - 273.15
    humidity = data['main']['humidity']

    weather_info = {
        "location": location,
        "temperature": round(temp, 2),
        "feels_like": round(feels_like, 2),
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),
        "humidity": round(humidity, 2)
    }
    return weather_info

# Function to get clothing suggestions based on weather using OpenAI's Chat API
def get_clothing_suggestions(weather_info):
    prompt = f"The current weather in {weather_info['location']} is {weather_info['temperature']}°C, with a 'feels like' temperature of {weather_info['feels_like']}°C. The humidity is {weather_info['humidity']}%. Based on this, what kind of clothing would you suggest for someone traveling today?"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify model as per new API
        messages=[
            {"role": "system", "content": "You are a helpful assistant that gives weather-based clothing advice in one line.Provide clothing suggestions and advice on whether it’s a good day for a picnic."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    # Extract the text from the completion response
    return response.choices[0].message.content.strip()

# User input for location
location = st.text_input("Enter a city (or leave blank for default 'Syracuse, NY'):")



# Get clothing suggestion  
if st.button("Get Clothing Suggestion", key="clothing_button"):
    weather_info = get_current_weather(location, st.secrets["weather_api_key"])
    # st.write(f"Weather information for {weather_info['location']}:")
    # st.write(f"Temperature: {weather_info['temperature']}°C")
    # st.write(f"Feels Like: {weather_info['feels_like']}°C")
    # st.write(f"Humidity: {weather_info['humidity']}%")

    # Fetch clothing suggestions
    suggestion = get_clothing_suggestions(weather_info)
    st.write(f"Suggested clothing: {suggestion}")
