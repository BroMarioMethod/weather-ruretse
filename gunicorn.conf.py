import multiprocessing

# Render free tier has limited RAM — keep it lean
workers = 2
worker_class = "sync"
bind = "0.0.0.0:10000"  # Render uses port 10000
timeout = 120            # chart generation can take a moment
accesslog = "-"          # log to stdout
errorlog = "-"
loglevel = "info"
preload_app = False      # let each worker init independently