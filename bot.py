"""Telegram-бот поверх rag.py: вопрос -> поиск -> reranking -> LLM -> ответ с источниками."""
import asyncio
import logging
import os
import time

from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from rag import RAG

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit("Не задан TELEGRAM_BOT_TOKEN (см. .env.example)")

MAX_QUESTION_LEN = 500        # длиннее не обрабатываем
MIN_INTERVAL_SEC = 5          # не чаще одного вопроса в 5 секунд от одного пользователя
MAX_PARALLEL = 2              # одновременно обрабатываемых вопросов (защита CPU и лимитов API)

HELP_TEXT = (
    "Я отвечаю на вопросы по аннотациям научных статей arXiv (cs.CL, cs.LG) "
    "и показываю, на какие статьи опираюсь.\n\n"
    "Напишите вопрос по машинному обучению, на русском или английском.\n"
    "Например: «Какие есть методы сжатия нейросетей?»\n\n"
    "Ответ строится только по аннотациям, поэтому деталей из полных текстов в нём нет."
)

dp = Dispatcher()
rag: RAG | None = None
sem = asyncio.Semaphore(MAX_PARALLEL)
last_request: dict[int, float] = {}


def format_reply(result: dict) -> str:
    text = result["answer"]
    if result.get("sources"):
        lines = ["", "Источники:"]
        for s in result["sources"]:
            lines.append(f"[{s['n']}] {s['title']} ({s['categories']})")
            lines.append(s["url"])
        text += "\n" + "\n".join(lines)
    return text


async def send_long(m: Message, text: str) -> None:
    """Telegram ограничивает сообщение 4096 символами."""
    for i in range(0, len(text), 4000):
        await m.answer(text[i:i + 4000], disable_web_page_preview=True)


@dp.message(CommandStart())
async def on_start(m: Message):
    await m.answer("Привет! " + HELP_TEXT)


@dp.message(Command("help"))
async def on_help(m: Message):
    await m.answer(HELP_TEXT)


@dp.message()
async def on_question(m: Message):
    if not m.text:
        await m.answer("Я понимаю только текстовые вопросы.")
        return
    if len(m.text) > MAX_QUESTION_LEN:
        await m.answer(f"Вопрос слишком длинный, уложитесь в {MAX_QUESTION_LEN} символов.")
        return

    uid = m.from_user.id if m.from_user else m.chat.id
    now = time.monotonic()
    if now - last_request.get(uid, 0) < MIN_INTERVAL_SEC:
        await m.answer("Слишком часто, подождите несколько секунд.")
        return
    last_request[uid] = now

    await m.bot.send_chat_action(m.chat.id, "typing")
    try:
        async with sem:
            result = await asyncio.to_thread(rag.answer, m.text)
        await send_long(m, format_reply(result))
    except Exception:
        log.exception("Не удалось ответить на вопрос")
        await m.answer("Что-то пошло не так, попробуйте ещё раз позже.")


async def main():
    global rag
    log.info("Загружаю модели и строю индекс (может занять несколько минут)...")
    rag = await asyncio.to_thread(RAG)
    log.info("Готово, бот запущен.")
    bot = Bot(TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
