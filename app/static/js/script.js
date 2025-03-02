document.addEventListener('DOMContentLoaded', function () {
    // Auto-dismiss flash messages after 5 seconds
    setTimeout(function () {
        const flashMessages = document.querySelectorAll('.flash-message');
        flashMessages.forEach(message => {
            message.style.opacity = '0';
            setTimeout(() => {
                message.remove();
            }, 500);
        });
    }, 5000);

    // Handle tab navigation
    const navButtons = document.querySelectorAll('.nav-button');
    navButtons.forEach(button => {
        button.addEventListener('click', function (e) {
            if (this.getAttribute('href') === '#') {
                e.preventDefault();
                if (this.querySelector('.fa-sign-out-alt')) {
                    if (confirm('Are you sure you want to log out?')) {
                        window.location.href = '/signin';
                    }
                }
            }
        });
    });

    // Initialize charts if they exist
    initializeCharts();
});

function initializeCharts() {
    // Temperature chart
    const tempChart = document.querySelector('.temperature-chart');
    if (tempChart) {
        const tempBars = tempChart.querySelectorAll('.chart-bar');
        tempBars.forEach(bar => {
            const temp = parseFloat(bar.getAttribute('data-temp'));
            // Scale height based on temperature (assuming range 20-30°C)
            const height = ((temp - 20) * 10) + '%';
            bar.style.height = height;
        });
    }

    // Moisture chart
    const moistureChart = document.querySelector('.moisture-chart');
    if (moistureChart) {
        const moistureBars = moistureChart.querySelectorAll('.chart-bar');
        moistureBars.forEach(bar => {
            const moisture = parseFloat(bar.getAttribute('data-moisture'));
            // Use moisture value directly as percentage
            const height = moisture + '%';
            bar.style.height = height;
        });
    }
}

// Form validation
const forms = document.querySelectorAll('form');
forms.forEach(form => {
    form.addEventListener('submit', function (e) {
        const passwordFields = form.querySelectorAll('input[type="password"]');

        // If this is a password change form with multiple password fields
        if (passwordFields.length > 1) {
            const newPassword = passwordFields[1].value;
            const confirmPassword = passwordFields[2].value;

            if (newPassword !== confirmPassword) {
                e.preventDefault();
                alert('New passwords do not match!');
                return false;
            }

            if (newPassword.length < 8) {
                e.preventDefault();
                alert('Password must be at least 8 characters long!');
                return false;
            }
        }
    });
});

// Network connection simulation
const connectButtons = document.querySelectorAll('.networks-list .btn:not(.btn-connected)');
connectButtons.forEach(button => {
    button.addEventListener('click', function () {
        const networkItem = this.closest('.network-item');
        const networkName = networkItem.querySelector('.network-name').textContent;

        this.textContent = 'Connecting...';
        this.disabled = true;

        // Simulate connection process
        setTimeout(() => {
            alert(`Connected to ${networkName}!`);
            window.location.reload();
        }, 2000);
    });
});

// Profile image change simulation
const changePhotoBtn = document.querySelector('.change-photo-btn');
if (changePhotoBtn) {
    changePhotoBtn.addEventListener('click', function () {
        alert('This feature would open a file picker in a real application.');
    });
}

// Save changes button simulation
const saveChangesBtn = document.querySelector('.profile-form .btn-primary');
if (saveChangesBtn) {
    saveChangesBtn.addEventListener('click', function () {
        // Simulate saving
        this.textContent = 'Saving...';
        this.disabled = true;

        setTimeout(() => {
            this.textContent = 'Save Changes';
            this.disabled = false;

            // Create a flash message
            const flashContainer = document.querySelector('#flash-messages .flash-messages');
            if (flashContainer) {
                const flashMessage = document.createElement('li');
                flashMessage.className = 'flash-message success';
                flashMessage.innerHTML = `
                    <div class="message-icon">
                        <i class="fas fa-check-circle"></i>
                    </div>
                    <span>Profile updated successfully!</span>
                    <button class="close-btn" onclick="this.parentElement.remove()">
                        <i class="fas fa-times"></i>
                    </button>
                `;
                flashContainer.appendChild(flashMessage);

                // Auto-dismiss after 5 seconds
                setTimeout(() => {
                    flashMessage.style.opacity = '0';
                    setTimeout(() => {
                        flashMessage.remove();
                    }, 500);
                }, 5000);
            }
        }, 1500);
    });
}

// Reset button simulation
const resetBtn = document.querySelector('.profile-form .btn-secondary');
if (resetBtn) {
    resetBtn.addEventListener('click', function () {
        if (confirm('Reset all changes?')) {
            window.location.reload();
        }
    });
}

// Delete account button simulation
const deleteAccountBtn = document.querySelector('.btn-danger');
if (deleteAccountBtn) {
    deleteAccountBtn.addEventListener('click', function () {
        if (confirm('Are you sure you want to delete your account? This action cannot be undone!')) {
            if (confirm('Really delete your account and all associated data?')) {
                alert('Account deletion would be processed in a real application.');
                window.location.href = '/signin';
            }
        }
    });
}


