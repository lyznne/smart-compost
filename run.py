'''
SMART COMPOST - MODEL PROJECT.

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 22 Nov 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
'''


# Import required modules
from livereload import Server
from app import create_app
from app.config import  DevelopmentConfig

# ---


# Create app instance for the development environment
app = create_app(DevelopmentConfig)

# Driver code
if __name__ == "__main__":
    # Initialize livereload server
    server = Server(app.wsgi_app)

    server.watch("./app/templates/*.html")  # Watch HTML templates
    server.watch("./app/static/css/*.css")  # Watch CSS files
    server.watch("./app/static/js/*.js")  # Watch JavaScript files

    # Serve the app
    server.serve(port=5000, host="0.0.0.0", debug=True)
