"""Shared SDK clients (cheap to construct, no network on init)."""
from groq import AsyncGroq

from app.config import settings

groq_client = AsyncGroq(api_key=settings.groq_api_key)
