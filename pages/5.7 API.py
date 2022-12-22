import matplotlib.pyplot as plt
from pyparsing import srange
import streamlit as st
import seaborn as sns

st.set_page_config(
    page_icon=":blue_heart:"
)

st.markdown("<h1 style='text-align: center;'>API</h1>", unsafe_allow_html=True)



st.markdown(r"""
To call an API using Python, you will need to use a library that can send HTTP requests. There are several options available, but a popular choice is the requests library.

Here is an example of how you can use the requests library to call an API in Python:

""")

code = r"""
import requests

# Set the API endpoint URL
url = "https://api.example.com/endpoint"

# Set the request parameters
params = {
    "param1": "value1",
    "param2": "value2"
}

# Send a GET request to the API
response = requests.get(url, params=params)

# Check the status code of the response
if response.status_code == 200:
    # If the status code is 200, the request was successful
    # You can access the response data as a JSON object
    data = response.json()
else:
    # If the status code is not 200, the request was not successful
    print("An error occurred:", response.status_code)



"""
st.code(code, language='python')




st.markdown(r"""
This example sends a GET request to the API endpoint specified in the url variable, with the request parameters specified in the params dictionary. The `requests.get()` function returns a response object, which contains information about the API's response, including the status code and the response data.

You can also use the requests library to send other types of HTTP requests, such as POST, PUT, and DELETE, using the `requests.post()`, `requests.put()`, and `requests.delete()` functions, respectively.



""")

st.markdown("---")
st.header("The Lord of the Rings example")


st.markdown("""
To use the `the-one-api` API, you will need to sign up for a API key and then include the key in your API requests. You can find more information about how to use the API, including documentation for the different API endpoints, on the API's documentation page.

Here is an example of how you can use the requests library to call an endpoint of the `the-one-api` API:
""")

code = r"""

import requests

# Set the API endpoint URL
url = "https://the-one-api.dev/v2/book"

# Set the request headers
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

# Set the request parameters
params = {
    "name": "The Lord of the Rings"
}

# Send a GET request to the API
response = requests.get(url, headers=headers, params=params)

# Check the status code of the response
if response.status_code == 200:
    # If the status code is 200, the request was successful
    # You can access the response data as a JSON object
    data = response.json()
    print("API response:", data)
else:
    # If the status code is not 200, the request was not successful
    print("An error occurred:", response.status_code)
"""

st.code(code, language='python')


st.markdown("""
This example sends a GET request to the `/v2/book` endpoint of the `the-one-api API`, with the request `headers` specified in the headers dictionary and the request parameters specified in the params dictionary. The `requests.get()` function returns a response object, which contains information about the API's response, including the status code and the response data.

Keep in mind that this is just one example of how you can use the `requests` library to call an endpoint of the `the-one-api API`. Depending on the endpoint you are working with, the details of the request (such as the request headers and the request parameters) may be different.
""")