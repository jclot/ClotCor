def main():  # pragma: no cover
    try:
        from clotcor.ui import Window
    except (ModuleNotFoundError, ImportError) as error:
        message = str(error)
        if (
            "PySide6" in message
            or "Qt binding modules" in message
            or getattr(error, "name", "") == "PySide6"
        ):
            raise SystemExit(
                "Qt dependencies are missing. Install runtime dependencies with:\n"
                "pip install -r requirements.txt"
            ) from error
        raise

    app_window = Window()
    app_window.run()
