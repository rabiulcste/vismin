import logging
import os


class Logger:
    @staticmethod
    def get_logger(name, log_file=None, log_level=logging.INFO):
        # Create or get the logger with the specified name
        logger = logging.getLogger(name)

        # Check if the logger has handlers set up already
        if not logger.handlers:
            # Set the log level
            logger.setLevel(log_level)

            # Create a formatter
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s")

            # Create and add the console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Optionally add a file handler
            if log_file:
                file_handler = logging.FileHandler(os.path.abspath(log_file))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            # Ensure the logger doesn't propagate messages to the root logger
            logger.propagate = False

        return logger
