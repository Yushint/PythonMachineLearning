import telebot
from machine_learning import build_model, convert_image_to_appropriate_format, predict_model

from telebot import types


bot = telebot.TeleBot("%...%")

@bot.message_handler(commands=["start"])
def start(message):
    msg = f"Привет, {message.from_user.first_name}! Я - бот, использующий нейронные сети."
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    programmer = types.KeyboardButton("Связь с разработчиком.🐍")
    neural_prediction = types.KeyboardButton("Распознавание чисел.🤖")
    donations = types.KeyboardButton("Пожертвования.💵")
    markup.add(programmer, neural_prediction, donations)
    bot.send_message(message.chat.id, msg, parse_mode="html", reply_markup=markup)

@bot.message_handler(content_types=["text"])
def text_message_handler(message):
    if message.text == "Распознавание чисел.🤖":
        bot.send_message(message.from_user.id, text="Пришлите картинку в формате .jpg или .png.")
    elif message.text == "Связь с разработчиком.🐍":
        bot.send_message(message.from_user.id, text="Telegram: @felix_nightingale.")
    elif message.text == "Пожертвования.💵":
        bot.send_message(message.from_user.id, text="Пожертвования не реализованы.")

@bot.message_handler(content_types=["photo"])
def photo_message_handler(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("current_image.jpg", "wb") as new_file:
        new_file.write(downloaded_file)
    bot.send_message(message.from_user.id, text="Картинка получена.")
    bot_prediction = process_image("./current_image.jpg")
    bot.send_message(message.from_user.id, text=f"Результат предсказания: {bot_prediction}.")
    
def process_image(image):
    model = build_model()
    image_list = convert_image_to_appropriate_format(image)
    result = predict_model(model, image_list, mode="bot")
    return result
    
bot.polling(none_stop=True, interval=0)
