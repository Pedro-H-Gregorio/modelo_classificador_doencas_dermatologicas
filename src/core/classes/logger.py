import datetime

class Logger:
    def __init__(self, log_to_file=False, filename='app.log'):
        self.log_to_file = log_to_file
        self.filename = filename

    def _log(self, level: str, message: str):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] [{level}] {message}"

        print(full_message)  # Sempre imprime no console

        if self.log_to_file:
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(full_message + '\n')

    def info(self, message: str):
        self._log('INFO', message)

    def warning(self, message: str):
        self._log('WARNING', message)

    def error(self, message: str):
        self._log('ERROR', message)

logger = Logger()