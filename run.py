'''
SMART COMPOST - MODEL PROJECT.

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
'''


# Import required modules
from app import create_app, socketio
from app.config import  DevelopmentConfig

# ---


# Create app instance for the development environment
app = create_app(DevelopmentConfig)

# Driver code
if __name__ == "__main__":

    # Serve the app with websockets
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        allow_unsafe_werkzeug=True,
        manage_session=False, # not to overide the app session
        
    )
