import logging
from io import BytesIO
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
    AIORateLimiter
)
import os

# === ENV VARS ===
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
WEBHOOK_DOMAIN = os.environ["RENDER_EXTERNAL_URL"]
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_DOMAIN}{WEBHOOK_PATH}"

# === Logging ===
logging.basicConfig(level=logging.INFO)

# === Load model once ===
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# === Image handler ===
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = None

    if update.message.photo:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
    elif update.message.document and update.message.document.mime_type.startswith("image/"):
        file = await context.bot.get_file(update.message.document.file_id)

    if file:
        img_bytes = BytesIO()
        await file.download_to_memory(out=img_bytes)
        img_bytes.seek(0)

        try:
            image = Image.open(img_bytes).convert("RGB")
            inputs = feature_extractor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                pred_class = logits.argmax(-1).item()
                label = model.config.id2label[pred_class]

            await update.message.reply_text(f"Predicted class: {label}")
        except Exception as e:
            logging.exception("Error processing image")
            await update.message.reply_text("Sorry, I couldn't process that image.")
    else:
        await update.message.reply_text("Please send a photo or image file.")

# === Main app with webhook ===
async def main():
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))

    # Set Telegram webhook to your Render HTTPS URL
    await app.bot.set_webhook(WEBHOOK_URL)

    # Start webhook server (Render forwards HTTP requests)
    await app.run_webhook(
        listen="0.0.0.0",
        port=10000,
        webhook_path=WEBHOOK_PATH,
        use_app=True  # use aiohttp
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())