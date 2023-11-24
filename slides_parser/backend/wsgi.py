from flsite import app, create_db

if __name__ == "__main__":
    try:
        create_db()
    finally:
        # Run the app on localhost and specify a port (e.g., 5000)
        app.run(host='localhost', port=5000, debug=True)
