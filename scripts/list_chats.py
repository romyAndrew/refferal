import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import Channel, Chat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_config():
    api_id = os.getenv("TG_API_ID", "").strip()
    api_hash = os.getenv("TG_API_HASH", "").strip()
    session_value = os.getenv("TG_SESSION_PATH", "").strip()

    if not api_id or not api_hash or not session_value:
        raise SystemExit(
            "Не заданы TG_API_ID, TG_API_HASH или TG_SESSION_PATH в .env"
        )

    if not api_id.isdigit():
        raise SystemExit("TG_API_ID должен быть числом.")

    session_path = Path(
        os.path.expandvars(session_value)
    ).expanduser().resolve()

    if session_path.is_relative_to(PROJECT_ROOT.resolve()):
        raise SystemExit(
            "Файл сессии должен храниться за пределами проекта."
        )

    session_path.parent.mkdir(parents=True, exist_ok=True)

    return int(api_id), api_hash, session_path


def get_chat_type(entity):
    if isinstance(entity, Channel):
        return "супергруппа" if entity.megagroup else "канал"
    if isinstance(entity, Chat):
        return "группа"
    return "личный чат или другой диалог"


async def main():
    api_id, api_hash, session_path = get_config()

    client = TelegramClient(
        str(session_path),
        api_id,
        api_hash,
    )

    try:
        print("Подключение к Telegram...")
        await client.start()

        print("Подключение успешно.")
        print("Получаем список последних 200 диалогов...")
        dialogs = await client.get_dialogs(limit=200)

        print(f"Получено диалогов: {len(dialogs)}")

        while True:
            query = input(
                "\nВведи часть названия нужного чата "
                "(или q для выхода): "
            ).strip()

            if query.lower() == "q":
                break

            if not query:
                print("Введи часть названия.")
                continue

            matches = [
                dialog for dialog in dialogs
                if query.casefold() in dialog.name.casefold()
            ]

            if not matches:
                print("Совпадений нет. Попробуй другую часть названия.")
                continue

            print("\nСовпадения:")
            for dialog in matches:
                print(
                    f"ID: {dialog.id} | "
                    f"Тип: {get_chat_type(dialog.entity)} | "
                    f"Название: {dialog.name}"
                )

    finally:
        await client.disconnect()
        print("Соединение закрыто.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nРабота остановлена.")