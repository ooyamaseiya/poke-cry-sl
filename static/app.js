document.addEventListener('DOMContentLoaded', () => {
    const audioFileInput = document.getElementById('audioFile');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const predictBtn = document.getElementById('predictBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorArea = document.getElementById('errorArea');
    const resultsArea = document.getElementById('resultsArea');
    const predictionsList = document.getElementById('predictionsList');

    // 1. Handle File Selection
    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            fileNameDisplay.textContent = `選択済み: ${file.name}`;
            predictBtn.disabled = false;
            errorArea.classList.add('hidden');
            resultsArea.classList.add('hidden');
        } else {
            fileNameDisplay.textContent = "ファイルが選択されていません";
            predictBtn.disabled = true;
        }
    });

    // 2. Handle Prediction Click
    predictBtn.addEventListener('click', async () => {
        const file = audioFileInput.files[0];
        if (!file) {
            alert("ファイルを選択してください。");
            return;
        }

        // UI Reset: Show loading, disable button, hide previous results/errors
        predictBtn.disabled = true;
        loadingIndicator.classList.remove('hidden');
        errorArea.classList.add('hidden');
        resultsArea.classList.add('hidden');
        predictionsList.innerHTML = '';

        // Prepare FormData for file upload
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                // Handle errors from server (e.g., 400 or 500)
                const errorData = await response.json();
                throw new Error(errorData.detail || '推論中にエラーが発生しました。');
            }

            const data = await response.json();

            if (data.success && data.predictions) {
                displayResults(data.predictions);
            } else {
                throw new Error('予期しないレスポンス形式を受け取りました。');
            }

        } catch (error) {
            console.error("Error:", error);
            showError(error.message);
        } finally {
            // Cleanup: Hide loading and re-enable button
            loadingIndicator.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    function displayResults(predictions) {
        resultsArea.classList.remove('hidden');
        predictionsList.innerHTML = '';

        if (predictions.length === 0) {
            predictionsList.innerHTML = '<li>結果が見つかりませんでした。</li>';
            return;
        }

        predictions.forEach((pred, index) => {
            const li = document.createElement('li');
            li.className = 'prediction-item';

            // Create info row (Name and Percentage)
            const infoDiv = document.createElement('div');
            infoDiv.className = 'prediction-info';
            infoDiv.innerHTML = `
                <span class="prediction-name">${index + 1}. ${pred.name}</span>
                <span>${pred.probability}%</span>
            `;

            // Create progress bar container and fill
            const progressBarContainer = document.createElement('div');
            progressBarContainer.className = 'progress-bar-container';
            
            const progressBarFill = document.createElement('div');
            progressBarFill.className = 'progress-bar-fill';
            progressBarFill.style.width = `${pred.probability}%`;

            progressBarContainer.appendChild(progressBarFill);
            li.appendChild(infoDiv);
            li.appendChild(progressBarContainer);
            predictionsList.appendChild(li);
        });
    }

    function showError(msg) {
        errorArea.textContent = msg;
        errorArea.classList.remove('hidden');
    }
});
