"""Compatibility backend entrypoint for the segregated web app layout."""
from __future__ import annotations

from web.main_app import app


def create_app():
    """Return the configured Flask application."""
    return app
