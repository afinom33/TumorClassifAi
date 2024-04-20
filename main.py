import asyncio
import aiohttp
import io
from PIL import Image
import numpy as np

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.enums import ChatAction

from config import TOKEN
from ultralytics import YOLO

bot = Bot(token=TOKEN)
dp = Dispatcher()

model = YOLO('best.pt')

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        '👋 Привет! Я - бот, который может анализировать <b>снимки головного мозга</b>. '
        'Просто отправьте мне снимок, и я скажу вам, есть ли на нем опухоль, и если да, то какая. '
        '\n\n⚠️ Пожалуйста, помните, что я могу помочь в <b>предварительной оценке</b>, но окончательный диагноз всегда должен ставить <b>врач</b>.',
        parse_mode='HTML'
    )


@dp.message(F.photo)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_path = file_info.file_path

    async with aiohttp.ClientSession() as session:
        async with session.get(f'https://api.telegram.org/file/bot{TOKEN}/{file_path}') as resp:
            photo_bytes = await resp.read()
    
    image = Image.open(io.BytesIO(photo_bytes))
    image_np = np.asarray(image)

    results = model(image_np)
    names = results[0].names
    translation_dict = {
        'glioma': 'глиома',
        'meningioma': 'менингиома',
        'notumor': 'нет опухоли',
        'pituitary': 'опухоль гипофиза'
    }
    translated_result = translation_dict[names[results[0].probs.top1]]

    await message.answer("На данном снимке с большей вероятности: {}".format(translated_result))


async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Exit')
