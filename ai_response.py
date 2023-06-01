import os, tempfile, ssl, openai, re
from faster_whisper import WhisperModel
from aiohttp import web
from dotenv import load_dotenv


### Load variables with keys - file would be .env
load_dotenv()

# Create an SSL context
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('ldn1whisper.crt', 'ldn1whisper.key')  #### Sort you a self-signed or CA signed cert. You cannot use the microphone transcription without it because HTTPS is a requirement

# OpenAI keys
openai.api_type = "azure"
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')  ### Remove this if you're using OpenAI
openai.api_version = "2023-05-15"  ### Remove this if you're using OpenAI
openai.api_key = os.getenv('AZURE_OPENAI_KEY')    ### Change this to your OpenAI Key

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
            transcription += f"{segment.text}\n"



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
            engine="eit_gptturbo",     ### Change this to the OpenAI engine of: gpt-3.5-turbo - https://platform.openai.com/docs/guides/chat/introduction
            messages=messages
            #temperature=0     
        )
        
        """
        For the temperature setting, it depends on the specific requirements of your conversation and the desired trade-off between creativity and sticking closely to the given context. Here are a few recommendations:

        Temperature of 0.2-0.5: Use a lower temperature if you want the responses to be more focused and deterministic. This will make the model generate more precise and conservative responses, sticking closely to the knowledge and context provided.

        Temperature of 0.8-1.0: Increase the temperature if you want more creativity and variability in the responses. This can lead to more exploratory and imaginative answers, but it may also cause the model to generate less accurate or coherent responses at times.

        Experiment and fine-tune: Feel free to adjust the temperature within the recommended range and see what works best for your specific use case. It's often helpful to try different settings and evaluate the results to find the right balance between accuracy and creativity.
        """

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
app.router.add_post('/talktome', gpt_process)


web.run_app(app, host='0.0.0.0', port=5000, ssl_context=ssl_context)
