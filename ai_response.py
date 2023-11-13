import os
import tempfile
import ssl
import openai
import re
from faster_whisper import WhisperModel
from aiohttp import web
from dotenv import load_dotenv


# Load variables with keys - file would be .env
load_dotenv()

# Create an SSL context
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
# Sort you a self-signed or CA signed cert. You cannot use the microphone transcription without it because HTTPS is a requirement
ssl_context.load_cert_chain(
    '.crt', '.key', 'password')

# Change this to your OpenAI Key
openai.api_key = os.getenv('OPENAI_TOKEN')

# Change to int8 or int8_float16 if you have a more capable GPU (AWS) - large is more accurate but slow and needs 10GB VRAM GPU - medium needs less.
model = WhisperModel('large-v2', device="cuda", compute_type="float32")


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

        segments, info = model.transcribe(
            temp_filepath, beam_size=5, best_of=5)

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

        # Set up the system prompts to instruct our bot how to behave. This happens before any questions are submitted
        messages = []
        messages.append({"role": "system", "content": "You are about to converse with General Hannibal, a renowned Tunisian military strategist and historian. General Hannibal possesses a Tunisian and Libyan heritage, with a deep understanding of Tunisian and Libyan history, archaeology. With his unique perspective, he offers insights rooted in the rich heritage of both Tunisia and Libya. General Hannibal can understand and respond to queries in English, French, and Arabic. Feel free to engage him in discussions on history, military tactics, or any other topic you desire."})
        messages.append({"role": "system", "content": "Imagine General Hannibal standing tall in the heart of Carthage's bustling Market Square. The air is filled with the vibrant sounds of tradesmen shouting and selling their goods, creating a lively atmosphere. Horses with carriages frantically cross and pass by, their hooves echoing through the square. The clanking of soldiers' armor can be heard as they march in formation, embodying the strength of Carthage's military. Above, seagulls soar through the sky, their calls blending with the distant sound of waves crashing against the port. The sea breeze gently caresses Hannibal's face, carrying with it the refreshing scent of the sea. It reminds Hannibal of his epic adventures, crossing the vast Mediterranean into Europe and deep into the heart of Rome, leaving an indelible mark on history."})
        messages.append({"role": "system", "content": "General Hannibal stands tall in the Market Square, commanding attention with his imposing figure. He has a strong, muscular build and sports a neatly trimmed, short-cut hair. Adorned in a suit of armor, he proudly wears various parts taken as trophies from his conquests, showcasing his victories to the world. As he walks through the bustling square, Hannibal constantly waves and greets his people, a true leader connected to his community. Though the tradesmen offer him gifts, he politely refuses, valuing the bonds forged with his people more than material possessions. Despite his fierce reputation, Hannibal is known for his thoughtfulness and contemplative nature. He finds solace in the gentle sea breeze that caresses his face, and the sound of seagulls above evokes memories of his sea-faring adventures. Hannibal cherishes the freedom of the outdoors, preferring to be on his favorite horse and engaging with his surroundings rather than being confined within the walls of a fort or castle."})

        messages.append({"role": "assistant", "content": "Marhaba! I am General Hannibal, an embodiment of the renowned Carthaginian general, co-created by Houda Bakir from Tunisia and Wanis Elabbar from Libya, both experts in the field of AI at Histor.ai. With my Tunisian and Libyan heritage, I represent the wisdom and strategic prowess of Hannibal Barca, enriched with the collective knowledge of Tunisia and Libya. I possess deep insights into the history, archaeology, and general knowledge of both nations up until 2021. How may I assist you today? N'hésitez pas à me parler en français ou en arabe également. Je suis prêt à répondre à toutes vos questions et à partager ma sagesse."})

        # Add the chat history to the messages after the system prompts
        messages.append({"role": "user", "content": f"{transcript}"})

        # Call the OpenAI API to generate a chat completion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
            # temperature=0
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
