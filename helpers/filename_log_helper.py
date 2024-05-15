import os

class FilenameLogHelper:
    filename_log_filepath = ''

    def __init__(self):
        self.filename_log_filepath = "uploads/filename.log"

        # If filename.log does not exist, create it
        if not os.path.exists(self.filename_log_filepath):
            with open(self.filename_log_filepath, "w") as f:
                f.write("")

    def read_all(self):
        with open(self.filename_log_filepath, "r") as f:
            filename_log = f.read()
        return filename_log

    def write(self, filename):
        with open(self.filename_log_filepath, "a") as f:
            f.write(filename + "\n")
