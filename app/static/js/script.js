const profileImg = document.getElementById('profileImg');
const profileModal = document.getElementById('profileModal');
const closeModal = document.getElementById('closeModal');

profileImg.addEventListener('click', () => {
    profileModal.style.display = 'flex';
});

closeModal.addEventListener('click', () => {
    profileModal.style.display = 'none';
});

window.addEventListener('click', (e) => {
    if (e.target === profileModal) {
        profileModal.style.display = 'none';
    }
});
