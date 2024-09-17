import os
import telebot
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Установите свой токен Telegram бота здесь
TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'

# Инициализация бота
bot = telebot.TeleBot(TOKEN)

# Инициализация модели Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я бот для генерации изображений. Отправь мне текстовое описание, и я создам изображение.")

@bot.message_handler(func=lambda message: True)
def generate_image(message):
    # Получаем текст от пользователя
    prompt = message.text
    
    # Отправляем сообщение о начале генерации
    bot.reply_to(message, "Начинаю генерацию изображения. Это может занять некоторое время...")
    
    try:
        # Генерируем изображение
        image = pipe(prompt).images[0]
        
        # Преобразуем изображение в байты
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Отправляем изображение пользователю
        bot.send_photo(message.chat.id, img_byte_arr, caption=f"Изображение по запросу: {prompt}")
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка при генерации изображения: {str(e)}")

# Запуск бота
if __name__ == '__main__':
    bot.polling()