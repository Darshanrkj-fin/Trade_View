"""Compatibility entrypoint for the Flask terminal UI."""

from app import app
from terminal_app.settings import settings


if __name__ == "__main__":
    if settings.debug:
        app.run(debug=True, host=settings.host, port=settings.port)
    else:
        try:
            from waitress import serve

            serve(app, host=settings.host, port=settings.port)
        except ImportError:
            app.run(debug=False, host=settings.host, port=settings.port)
