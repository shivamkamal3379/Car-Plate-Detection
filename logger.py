from datetime import datetime

class Logger:
    gui_log_callback = None   # will be set by GUI when needed

    @classmethod
    def set_gui_callback(cls, callback):
        """GUI will call this once to enable live log updates."""
        cls.gui_log_callback = callback

    @classmethod
    def log(cls, message):
        """
        Logs messages :
        1. Writes to Functionality_Logs.txt
        2. Prints to console (optional hai )
        3. Sends log to GUI if callback exists
        """

        # Write to log file
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("Functionality_Logs.txt", "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except:
            pass

        # If GUI is allows then , update GUI label will hapen ise hum GUI mai hi set kr sakte hain 
        try:
            if cls.gui_log_callback:
                cls.gui_log_callback(message)
        except:
            pass
