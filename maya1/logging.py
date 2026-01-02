import logging

def get_logging_config(log_level_str):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'loggers': {
            '': {  # Root logger
                'level': 'WARNING',
                'handlers': ['console'],
            },
            'api_v2': {  # Specific level for your code
                'level': log_level_str,
                'propagate': True,
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
        },
        'formatters': {
            'standard': {
                'format': '%(name)s - %(levelname)s - %(message)s',
            },
        },
    }

def setup_child_logging(queue):
    """Configures a child process to send all logs to the main process via a Queue."""
    handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

def start_logging_listener(queue):
    """Starts a listener in the main process to handle logs coming from children."""
    # Use the existing Uvicorn/FastAPI logger as the destination
    main_logger = logging.getLogger("uvicorn.error")
    
    # This listener runs in a background thread of the main process
    listener = logging.handlers.QueueListener(queue, *main_logger.handlers)
    listener.start()
    return listener