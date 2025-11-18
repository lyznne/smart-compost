SERVER_API  = 'http://localhost:5000/notification';
const userId = document.body.getAttribute("data-user-id");

document.addEventListener("DOMContentLoaded", function() {
    const socket  = io.connect(SERVER_API)

    socket.on("notification", function (data) {
        if (data.user_id  === parseInt(userId)) {
            console.log("notification", data.message)
            alertNotification(data)
        }
    })


    socket.on("new_notification", function (data) {
        const userId = document.body.getAttribute("data-user-id");
        if (data.user_id === parseInt(userId)) {
            alert("New Notification: " + data.message);
        }
    });

    socket.on("disconnect", function () {
        console.log("Disconnected from /notification namespace");
    });

    
    // alert botification
    function alertNotification(data) {
        // show the message on ui
    }
})
