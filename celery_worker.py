from app import app, celery

if __name__ == '__main__':
    with app.app_context():
        celery.worker_main(['worker', '--loglevel=info', '--pool=solo']) # Use solo pool for Windows compatibility
