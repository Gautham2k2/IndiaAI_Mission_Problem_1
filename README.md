Here, in this project we have developed a PDF reading application using LLMs.
The LLM used here is called "**google/flan-t5-base**" which can be used free of cost with no subsciption needed for calling it's API.

The following libraries were used for running this application:
------>langchain==0.0.154
------>PyPDF2==3.0.1
------>python-dotenv==1.0.0
------>streamlit==1.18.1
------>faiss-cpu==1.7.4
------>streamlit-extras
------>altair<5
------>transformers==4.41.2
------>sentence_transformers==3.0
------>torch==2.1.1

The chatbot interface was designed using streamlit which makes the job fairly easy for any python coder.

First, run the requirements.txt file using the command **"pip install -r requirements.txt"** in the terminal of visual studio code.
Then the run the application using the command **"streamlit run app.py"**.
Then a localhost link will pop up in our default browser where we can upload any PDFs and start asking our questions.
