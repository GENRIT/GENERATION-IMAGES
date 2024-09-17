import os
import telebot
from diffusers import StableDiffusionPipeline
import torch

# Вставьте сюда токен вашего Telegram-бота
API_TOKEN = '7218060489:AAEx4jhciHiBh1Vxpo-MVkHHkHXObcR2dxg'

# Инициализация бота
bot = telebot.TeleBot(API_TOKEN)

# Загрузка модели генерации изображений
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Функция для генерации изображения по текстовому запросу
def generate_image(prompt):
    image = pipe(prompt).images[0]
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path

# Обработчик команд start и help
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь мне текст, и я сгенерирую изображение на его основе.")

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_text(message):
    prompt = message.text
    bot.reply_to(message, f"Генерирую изображение для текста: {prompt}")
    image_path = generate_image(prompt)

    with open(image_path, 'rb') as img:
        bot.send_photo(message.chat.id, img)

# Запуск бота
bot.polling()