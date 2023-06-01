import os, tempfile, ssl, openai, re
from faster_whisper import WhisperModel
from aiohttp import web
from dotenv import load_dotenv


### Load variables with keys - file would be .env
load_dotenv()

# Create an SSL context
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('ldn1whisper.crt', 'ldn1whisper.key')

# OpenAI keys
openai.api_type = "azure"
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_version = "2023-05-15"
openai.api_key = os.getenv('AZURE_OPENAI_KEY')

model = WhisperModel('large-v2', device="cuda", compute_type="float32")  ### Change to int8 or int8_float16 if you have a more capable GPU (AWS)



async def transcribe(request):
    try:
        reader = await request.multipart()
        field = await reader.next()

        temp_file_descriptor, temp_filepath = tempfile.mkstemp(suffix='.wav')

        with open(temp_filepath, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        segments, info = model.transcribe(temp_filepath, beam_size=5, best_of=5)

        transcription = ''
        for segment in segments:
            transcription += f"{segment.txt}\n"



        return web.json_response({"transcription":  transcription})

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def gpt_process(transcript):

    try:

        transcript = await transcript.json()
        transcript = transcript['transcript']

        ### Set up the system prompts to instruct our bot how to behave. This happens before any questions are submitted
        messages = []
        messages.append({"role": "system", "content": "You are about to converse with General Hannibal, a renowned Tunisian military strategist and historian. He possesses extensive knowledge of Tunisian history, archaeology, and general knowledge up until 2021. He can understand and respond to queries in English, French, and Arabic. Feel free to ask him anything related to Tunisia or seek his insights on military tactics and strategies."})
        messages.append({"role": "assistant", "content": "As-salamu alaykum! I am General Hannibal, at your service. How can I assist you today? Feel free to ask me anything about Tunisia's history, archaeology, or any other topic you'd like to discuss. Vous pouvez également me poser des questions en français ou en arabe. Alors, que puis-je faire pour vous?"})

        messages.append({"role": "user", "content": f"{transcript}"})  # Add the chat history to the messages after the system prompts

        # Call the OpenAI API to generate a chat completion
        response = openai.ChatCompletion.create(   
            engine="eit_gptturbo",   
            messages=messages     
        )

        # Get the generated text from the API response
        text_output = response["choices"][0]["message"]["content"]

        return web.json_response({"summary": text_output}, status=200)

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    
    
async def index(request):
    return web.FileResponse('templates/index.html')



app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/transcribe', transcribe)
app.router.add_post('/talkwithme', gpt_process)


web.run_app(app, host='0.0.0.0', port=5000, ssl_context=ssl_context)
