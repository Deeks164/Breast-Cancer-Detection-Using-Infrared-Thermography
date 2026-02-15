document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('upload-form');
  const fileInput = document.getElementById('fileInput');
  const resultDiv = document.getElementById('result');
  const progressBar = document.getElementById('progressBar');
  const progressBarContainer = document.getElementById('progressBarContainer');

  // Image preview elements (optional)
  const previewContainer = document.createElement('div');
  previewContainer.id = 'imagePreviewContainer';
  previewContainer.className = 'text-center mt-4';
  previewContainer.style.display = 'none';

  const previewImage = document.createElement('img');
  previewImage.id = 'imagePreview';
  previewImage.alt = 'Image Preview';
  previewImage.className = 'img-fluid rounded animate__animated animate__zoomIn';
  previewImage.style.maxHeight = '300px';

  previewContainer.appendChild(previewImage);
  form.parentNode.appendChild(previewContainer);

  // Image preview on file select
  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        previewImage.classList.remove('animate__zoomOut');
        previewImage.classList.add('animate__zoomIn');
      };
      reader.readAsDataURL(file);

      resultDiv.classList.add('d-none');
      resultDiv.textContent = '';
    } else {
      previewContainer.style.display = 'none';
    }
  });

  // Form submit and analysis
  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Reset progress and results
    progressBarContainer.style.display = 'block';
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';
    resultDiv.classList.add('d-none');
    resultDiv.textContent = '';
    resultDiv.className = 'alert mt-4';

    // Animate preview image out
    previewImage.classList.remove('animate__zoomIn');
    previewImage.classList.add('animate__zoomOut');

    try {
      // Simulate progress bar animation (optional)
      let progress = 0;
      const fakeProgress = setInterval(() => {
        progress += 10;
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = `${progress}%`;
        if (progress >= 90) clearInterval(fakeProgress);
      }, 100);

      // Send request to backend
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
      });

      clearInterval(fakeProgress);
      progressBar.style.width = '100%';
      progressBar.textContent = '100%';

      const data = await response.json();

      if (response.ok) {
        resultDiv.classList.remove('alert-danger');
        resultDiv.classList.add('alert-success', 'animate__animated', 'animate__fadeInUp');
        resultDiv.innerHTML = `<strong>${data.prediction}</strong><br>Confidence: ${(data.confidence * 100).toFixed(2)}%`;
      } else {
        throw new Error(data.error || 'Prediction failed');
      }
    } catch (error) {
      resultDiv.classList.remove('alert-success');
      resultDiv.classList.add('alert-danger', 'animate__animated', 'animate__shakeX');
      resultDiv.textContent = error.message;
    } finally {
      resultDiv.classList.remove('d-none');
      progressBarContainer.style.display = 'none';
    }
  });
});
