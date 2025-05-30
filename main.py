import logging
import os
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
from aiohttp import web

# === Logging ===
logging.basicConfig(level=logging.INFO)

# === Environment Variables ===
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
RENDER_EXTERNAL_URL = os.environ["RENDER_EXTERNAL_URL"]
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{RENDER_EXTERNAL_URL.rstrip('/')}{WEBHOOK_PATH}"

# === Load Model and Feature Extractor ===
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# === Telegram Message Handler ===
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

# === Optional Healthcheck Route ===
async def healthcheck(request):
    return web.Response(text="OK")

# === Main Function ===
async def main():
    # Create Telegram Application
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    # Add handler for photo or image document
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))

    # Delete any old webhook and set new one (required for Render Free Tier)
    await app.bot.delete_webhook(drop_pending_updates=True)
    success = await app.bot.set_webhook(WEBHOOK_URL)

    if success:
        logging.info(f"✅ Webhook set to: {WEBHOOK_URL}")
    else:
        logging.warning("❌ Failed to set webhook!")

    # Define aiohttp app for optional healthcheck
    aio_app = web.Application()
    aio_app.router.add_get("/", healthcheck)

    # Run the webhook server
    await app.run_webhook(
        listen="0.0.0.0",
        port=10000,
        webhook_path=WEBHOOK_PATH,
        use_app=aio_app
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())